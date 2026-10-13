"""Predictive quality of the ensemble mean, for binary and multiclass tasks."""
import numpy as np
import pandas as pd


def ensemble_quality(ensemble, labels, groups, *, threshold=.5, group_names=None, binary=None):
    p = np.asarray(ensemble['mean_probs'], dtype=float)
    y, groups = np.asarray(labels, dtype=int), np.asarray(groups)
    if p.ndim != 2 or y.shape != (len(p),) or groups.shape != y.shape:
        raise ValueError('Probabilities, labels and groups must be row aligned.')
    if not len(p) or p.shape[1] < 2 or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        raise ValueError('Require nonempty, finite class probabilities in [0,1].')
    if not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise ValueError('Class probabilities must sum to one.')
    if (y < 0).any() or (y >= p.shape[1]).any() or not 0 <= threshold <= 1:
        raise ValueError('Invalid labels or threshold.')
    binary = p.shape[1] == 2 if binary is None else binary
    if binary and p.shape[1] != 2:
        raise ValueError('Binary evaluation requires two probability columns.')
    pred = (p[:, 1] >= threshold).astype(int) if binary else p.argmax(axis=1)
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
    brier = ((p[:, 1]-y)**2 if p.shape[1] == 2 else
             ((p-np.eye(p.shape[1])[y])**2).sum(axis=1))
    def quality(mask):
        return dict(support=int(mask.sum()), accuracy=float((pred[mask] == y[mask]).mean()),
                    nll=float(nll[mask].mean()), brier=float(brier[mask].mean()))
    return quality(np.ones(len(y), dtype=bool)), pd.DataFrame([
        dict(group_id=int(g), group=str((group_names or {}).get(int(g), g)), **quality(groups == g))
        for g in np.unique(groups)])
