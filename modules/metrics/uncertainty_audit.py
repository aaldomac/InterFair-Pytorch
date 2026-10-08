"""Dataset-independent uncertainty disparities; entropy units are nats."""
import numpy as np
from typing import Dict, List, Union

def audit_uncertainties(ensemble: Dict[str, np.ndarray], groups: Union[List, np.ndarray], alpha=.05, *, group_names=None,
                        interaction_weights=None, num_classes=None):
    """Audit disparities in aleatoric, epistemic, and total predictive uncertainty across groups.
    Args:
        ensemble: Dictionary of row-aligned ensemble results, including keys 'ensemble_probs', 'mean_probs', 'predictions', 'aleatoric_uncertainty', 'epistemic_uncertainty', and 'predictive_entropy'.
        groups: Row-aligned group labels for each observation in the ensemble.
        alpha: Significance level for the audit.
        group_names: Optional mapping from group IDs to names.
        interaction_weights: Optional mapping from group IDs to weights for computing interaction metrics.
        num_classes: Number of classes in the classification problem.
    """
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
    means = np.array([u[groups == g].mean(axis=0) for g in ids])  # shape: (n_groups, 3) -> [mean U ale, mean U epi, mean U tot] per group
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
        w = np.array([weights[str(g)] for g in ids])  # shape: (n_groups,) -> weight for each group
        scale = np.abs(w).sum()/2
        if not np.isfinite(w).all() or not np.isclose(w.sum(), 0) or scale == 0:
            raise ValueError('Interaction weights must be finite, nonzero and sum to zero.')
        contrast = w @ means  # shape: (3,) -> weighted contrast of group means for aleatoric, epistemic, total
        interaction = contrast.tolist()  # shape: (3,) -> Interaction metrics for aleatoric, epistemic, total
        f_int = float(np.max(np.abs(contrast[:2]))/(scale*bound))
    return dict(component_order=['alea','epis','tot'], group_ids=ids.tolist(),
                group_order=names, means=means.tolist(), counts=counts.tolist(), pairs=pairs,
                maxima={k:max(p[k] for p in pairs.values()) for k in next(iter(pairs.values()))},
                interaction=interaction, F_U_int=f_int, alpha=alpha,
                entropy_units='nats', normalization=bound,
                # Union bound for TWO independent components per group; total is their sum.
                hoeffding_component_radii=(bound*np.sqrt(np.log(4*len(ids)/alpha)/(2*counts))).tolist(),
                hoeffding_scope='aleatoric and epistemic group means; iid audit observations, fixed trained model')
