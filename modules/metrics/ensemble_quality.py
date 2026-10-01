"""Predictive quality of the ensemble mean, for binary and multiclass tasks."""
import numpy as np
import pandas as pd


def ensemble_quality(ensemble, labels, groups, *, threshold=.5, group_names=None):
    p = np.asarray(ensemble['mean_probs'], dtype=float)
    y, groups = np.asarray(labels, dtype=int), np.asarray(groups)
    if p.ndim != 2 or y.shape != (len(p),) or groups.shape != y.shape:
        raise ValueError('Probabilities, labels and groups must be row aligned.')
    pred = (p[:, 1] >= threshold).astype(int) if p.shape[1] == 2 else p.argmax(axis=1)
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
    brier = ((p[:, 1]-y)**2 if p.shape[1] == 2 else
             ((p-np.eye(p.shape[1])[y])**2).sum(axis=1))
    def quality(mask):
        return dict(support=int(mask.sum()), accuracy=float((pred[mask] == y[mask]).mean()),
                    nll=float(nll[mask].mean()), brier=float(brier[mask].mean()))
    return quality(np.ones(len(y), dtype=bool)), pd.DataFrame([
        dict(group_id=int(g), group=str((group_names or {}).get(int(g), g)), **quality(groups == g))
        for g in np.unique(groups)])
