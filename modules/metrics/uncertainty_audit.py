"""Dataset-independent uncertainty disparities; entropy units are nats."""
import numpy as np


def audit_uncertainties(ensemble, groups, alpha=.05, *, group_names=None,
                        interaction_weights=None, num_classes=None):
    u = np.column_stack([ensemble[k] for k in (
        'aleatoric_uncertainty', 'epistemic_uncertainty', 'predictive_entropy')])
    groups = np.asarray(groups)
    if num_classes is None:
        num_classes = np.asarray(ensemble.get('mean_probs', np.empty((0, 2)))).shape[-1]
    if num_classes < 2 or not 0 < alpha < 1:
        raise ValueError('Require at least two classes and alpha in (0, 1).')
    bound = float(np.log(num_classes))
    if groups.shape != (len(u),) or not len(u) or not np.isfinite(u).all():
        raise ValueError('Require finite uncertainties and one group per observation.')
    if (u < -2e-6).any() or (u > bound + 2e-6).any():
        raise ValueError('Uncertainties must lie in [0, log(num_classes)].')
    if not np.allclose(u[:, 0] + u[:, 1], u[:, 2], atol=2e-6):
        raise ValueError('Uncertainty decomposition does not hold.')
    ids = np.unique(groups)
    if len(ids) < 2:
        raise ValueError('Disparities require at least two observed groups.')
    names = [str(group_names.get(int(g), g)) if group_names else str(g) for g in ids]
    if len(set(names)) != len(names):
        raise ValueError('Group names must be unique.')
    counts = np.array([(groups == g).sum() for g in ids])
    means = np.array([u[groups == g].mean(axis=0) for g in ids])
    pairs = {}
    for a in range(len(ids)):
        for b in range(a+1, len(ids)):
            da, de, dt = means[a] - means[b]
            pairs[f'{names[a]}-{names[b]}'] = dict(
                F_alea=float(abs(da)), F_epis=float(abs(de)), F_tot=float(abs(dt)),
                F_U=float(max(abs(da), abs(de))/bound),
                hidden_normalized=float(max(0., abs(da)+abs(de)-abs(dt))/(2*bound)))
    interaction, f_int = None, None
    if interaction_weights is not None:
        weights = {str(k): float(v) for k, v in interaction_weights.items()}
        if set(weights) != {str(g) for g in ids}:
            raise ValueError('Interaction weights must specify every observed group ID exactly.')
        w = np.array([weights[str(g)] for g in ids])
        scale = np.abs(w).sum()/2
        if not np.isfinite(w).all() or not np.isclose(w.sum(), 0) or scale == 0:
            raise ValueError('Interaction weights must be finite, nonzero and sum to zero.')
        contrast = w @ means
        interaction = contrast.tolist()
        f_int = float(np.max(np.abs(contrast[:2]))/(scale*bound))
    return dict(component_order=['alea','epis','tot'], group_ids=ids.tolist(),
                group_order=names, means=means.tolist(), counts=counts.tolist(), pairs=pairs,
                maxima={k:max(p[k] for p in pairs.values()) for k in next(iter(pairs.values()))},
                interaction=interaction, F_U_int=f_int, alpha=alpha,
                entropy_units='nats', normalization=bound,
                # Union bound for TWO independent components per group; total is their sum.
                hoeffding_component_radii=(bound*np.sqrt(np.log(4*len(ids)/alpha)/(2*counts))).tolist(),
                hoeffding_scope='aleatoric and epistemic group means; iid audit observations, fixed trained model')
