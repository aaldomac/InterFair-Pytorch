"""Experiment-specific disparities; uncertainty is computed by the ensemble evaluator."""
import numpy as np
from .data import GROUPS, LOG2

def audit_uncertainties(ensemble, groups, alpha=.05):
    """
    Args:
        ensemble: dict of ensemble outputs, including aleatoric_uncertainty,
            epistemic_uncertainty, predictive_entropy, ensemble_probs.
        groups: array-like of group identifiers, one per observation.
        alpha: failure probability parameter for the simultaneous Hoeffding bounds; 0.05 corrsponds to 95% coverage under their assumptions.
    Returns:
        dict: Group means, all pair disparities, maxima, and simultaneous Hoeffding radii.

    F_U uses max(component gaps)/log(2). Hidden maxima are computed pairwise,
    never by combining component maxima attained by different group pairs.
    """
    u = np.column_stack([ensemble[k] for k in (
        'aleatoric_uncertainty', 'epistemic_uncertainty', 'predictive_entropy')])
    # Check u values are finite and satisfy decomposition
    if not np.isfinite(u).all():
        raise ValueError('Nonfinite uncertainty estimates.')
    np.testing.assert_allclose(u[:, 0]+u[:, 1], u[:, 2], atol=2e-6)
    groups = np.asarray(groups)
    # Compute group counts and mean uncertainties
    if groups.shape != (len(u),) or not np.isin(groups, range(4)).all():
        raise ValueError('Expected one valid group id per observation.')
    counts = np.array([(groups == g).sum() for g in range(4)])
    if min(counts) == 0 or not 0 < alpha < 1:
        raise ValueError('All groups must be present and alpha in (0,1).')
    means = np.array([u[groups == g].mean(axis=0) for g in range(4)])
    # Compute pairwise disparities
    pairs = {}
    for a in range(4):
        for b in range(a+1,4):
            da,de,dt = (means[a]-means[b]).tolist()
            pairs[f'{GROUPS[a]}-{GROUPS[b]}'] = dict(
                F_alea=abs(da), F_epis=abs(de), F_tot=abs(dt),
                F_U=max(abs(da),abs(de))/LOG2,
                hidden_normalized=max(0., abs(da)+abs(de)-abs(dt))/(2*LOG2))
    # Compute interaction contrast
    interaction = np.array([1,-1,-1,1]) @ means
    return dict(component_order=['alea','epis','tot'], means=means.tolist(),
                counts=counts.tolist(), pairs=pairs,
                maxima={k:max(v[k] for v in pairs.values()) for k in next(iter(pairs.values()))},
                interaction=interaction.tolist(),
                F_U_int=float(max(abs(interaction[:2]))/(2*LOG2)),
                hoeffding_component_radii=(LOG2*np.sqrt(np.log(16/alpha)/(2*counts))).tolist())

def audit_stripe_regions(ensemble, groups, X, S, metadata):
    """Descriptive regional means; empty regions use null, never zero.
    These are not simultaneous intervals or additional fairness maxima.
    Each condition's region follows its own width; compare group-wide metrics 
    on the fixed audit population when comparing different stripe widths.
    """
    from .data import stripe_region_mask
    if 'stripe' not in metadata:
        return []
    info = metadata['stripe']
    g = GROUPS.index(info['group'])
    inside = stripe_region_mask(X, S, info['half_width'], info['slope'])
    groups = np.asarray(groups)
    oracle = metadata['oracle_entropy_bits'][g]*LOG2
    rows = []
    for region, mask in [('inside', inside), ('outside', ~inside)]:
        mask = mask & (groups == g)
        count = int(mask.sum())
        row = dict(group=info['group'], region=region, support=count,
                   half_width=info['half_width'], slope=info['slope'],
                   expected_retention=info['expected_retention'], oracle_alea=oracle)
        for name, key in [('alea','aleatoric_uncertainty'), ('epis','epistemic_uncertainty'),
                          ('tot','predictive_entropy')]:
            if count > 0:
                row[name] = float(np.asarray(ensemble[key][mask].mean()))
            else:
                row[name] = None
        row['alea_error'] = row['alea']-oracle if count else None
        rows.append(row)
    return rows