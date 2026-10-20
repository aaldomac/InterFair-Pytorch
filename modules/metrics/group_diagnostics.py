"""Prediction-quality, class-composition and paired-contrast diagnostics.

All summaries use one estimate per run. This module never trains or changes audits.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

UNCERTAINTIES = ('U_alea', 'U_epis', 'U_tot')


def _entropy(p):
    return -(p * np.log(np.clip(p, 1e-300, 1))).sum(axis=-1)


def error_auroc(errors, scores):
    """AUROC for detecting errors with larger scores; NaN for one outcome only."""
    errors = np.asarray(errors, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    n1, n0 = errors.sum(), (~errors).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = rankdata(scores, method='average')
    return float((ranks[errors].sum() - n1*(n1+1)/2)/(n1*n0))


def diagnose_predictions(ensemble, labels, groups, *, group_names=None, threshold=.5,
                         positive_class=1, high_confidence=.9, bins=10):
    """Return wide subgroup/stratum metrics and per-group reliability bins.

    Binary decisions follow threshold on column 1, matching ensemble_quality;
    positive_class only selects the outcome for TPR/FPR. Multiclass uses argmax.
    Confidence is the probability assigned to the actual chosen decision, even
    when the binary decision threshold is not .5. No calibration is performed.
    """
    p = np.asarray(ensemble['mean_probs'], dtype=float)
    y, g = np.asarray(labels), np.asarray(groups)
    # Check validity of inputs and parameters.
    if p.ndim != 2 or p.shape[1] < 2 or not len(p):
        raise ValueError('Require nonempty [N, C] probabilities.')
    if y.shape != (len(p),) or g.shape != y.shape:
        raise ValueError('Labels/groups must align with probability rows.')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-6):
        raise ValueError('Invalid probability rows. Either infinite, negative, >1 or do not sum to 1.')
    c = p.shape[1]  # number of classes
    if not np.isfinite(y).all() or (y != y.astype(int)).any() or (y < 0).any() or (y >= c).any():
        raise ValueError('Invalid class labels.')
    if not 0 <= threshold <= 1 or not 0 <= positive_class < c or not .5 <= high_confidence <= 1:
        raise ValueError('Invalid decision/confidence parameters.')
    if not isinstance(bins, int) or bins < 2:
        raise ValueError('bins must be an integer >= 2.')
    y = y.astype(int)
    # Compute predictions, confidence, errors, high-confidence mask, and uncertainty components for all samples.
    pred = ((p[:, 1] >= threshold).astype(int) if c == 2 else p.argmax(1))
    confidence = p[np.arange(len(p)), pred]
    error = pred != y
    high = confidence >= high_confidence
    u = {'U_tot': _entropy(p)}
    for key, name in [('aleatoric_uncertainty', 'U_alea'), ('epistemic_uncertainty', 'U_epis')]:
        if key not in ensemble:
            raise ValueError(f'Missing saved {key}; run the ensemble evaluation first.')
        values = np.asarray(ensemble[key], dtype=float)
        if values.shape != y.shape or not np.isfinite(values).all() or (values < -2e-6).any():
            raise ValueError(f'Invalid {key}.')
        u[name] = values
    if not np.allclose(u['U_alea'] + u['U_epis'], u['U_tot'], atol=2e-6):
        raise ValueError('Saved uncertainty components do not sum to predictive entropy.')
    nll = -np.log(np.clip(p[np.arange(len(p)), y], 1e-12, 1))
    brier = (p[:, 1]-y)**2 if c == 2 else ((p-np.eye(c)[y])**2).sum(1)
    bin_id = np.minimum((confidence*bins).astype(int), bins-1)
    rows, reliability = [], []
    linear = ['accuracy', 'error_rate', 'nll', 'brier', 'mean_confidence',
              'high_conf_coverage', 'high_conf_error_rate', *UNCERTAINTIES]

    def summarize(mask):
        """Return a dict of summary metrics for the selected samples."""
        n = int(mask.sum())
        result = dict(support=n, error_count=int(error[mask].sum()),
                      high_conf_count=int(high[mask].sum()),
                      high_conf_error_count=int((high[mask] & error[mask]).sum()))
        def mean(v):
            return float(v[mask].mean()) if n else np.nan
        result.update(accuracy=mean(~error), error_rate=mean(error), nll=mean(nll),
            brier=mean(brier), mean_confidence=mean(confidence),
            high_conf_coverage=mean(high), high_conf_error_rate=mean(high & error))
        for name, eligible, event in [
            ('TPR', y == positive_class, pred == positive_class),
            ('FPR', y != positive_class, pred == positive_class),
            ('PPV', pred == positive_class, y == positive_class),
        ]:
            selected = mask & eligible
            result[name] = float(event[selected].mean()) if selected.any() else np.nan
        selected = mask & high
        result['high_conf_risk'] = float(error[selected].mean()) if selected.any() else np.nan
        selected = mask & error
        result['confidence_on_errors'] = float(confidence[selected].mean()) if selected.any() else np.nan
        for key, values in u.items():
            result[key] = mean(values)
            result['error_auroc_' + key] = error_auroc(error[mask], values[mask])
        result['error_auroc_low_confidence'] = error_auroc(error[mask], -confidence[mask])
        result['U_mix'] = float(_entropy(p[mask].mean(0))) if n else np.nan
        result['U_het'] = result['U_mix'] - result['U_tot']
        for k in range(c):
            result[f'prevalence_class_{k}'] = mean(y == k)
            result[f'mean_prob_class_{k}'] = mean(p[:, k])
        ece = 0.
        for b in range(bins):
            bm = mask & (bin_id == b)
            if bm.any():
                ece += bm.sum() * abs(float((~error[bm]).mean()) - float(confidence[bm].mean()))
        result['ece'] = ece/n if n else np.nan
        return result

    for gid in np.unique(g):
        mask = g == gid
        name = str((group_names or {}).get(str(gid), (group_names or {}).get(int(gid), gid)))
        identity = dict(group_id=str(gid), group=name)
        class_rows = []
        scopes = [('overall', mask), ('correct', mask & ~error), ('incorrect', mask & error)]
        scopes += [(f'true_class={k}', mask & (y == k)) for k in range(c)]
        scopes += [(f'predicted_class={k}', mask & (pred == k)) for k in range(c)]
        for scope, selected in scopes:
            stats = summarize(selected)
            rows.append(dict(**identity, scope=scope, **stats))
            if scope.startswith('true_class='):
                class_rows.append(stats)
        # Equal class weights: remove prevalence differences, retain within-class behavior.
        standardized = dict(**identity, scope='class_standardized', support=int(mask.sum()))
        complete = all(r['support'] > 0 for r in class_rows)
        for metric in linear:
            standardized[metric] = float(np.mean([r[metric] for r in class_rows])) if complete else np.nan
        rows.append(standardized)
        for b in range(bins):
            selected = mask & (bin_id == b)
            n = int(selected.sum())
            reliability.append(dict(**identity, bin=b, lower=b/bins, upper=(b+1)/bins,
                support=n, confidence=float(confidence[selected].mean()) if n else np.nan,
                accuracy=float((~error[selected]).mean()) if n else np.nan))
    return pd.DataFrame(rows), pd.DataFrame(reliability)


def read_run_diagnostics(run, **kwargs):
    """Reuse source-run routing for re-audits; honor the selected audit threshold."""
    from modules.utils.loading_utils import load_saved_outputs, load_audit_config
    run = Path(run)
    def read(path):
        return json.loads(path.read_text()) if path.exists() else {}
    context = read(run/'audit/audit_context.json')
    config = read(run/'config.json')
    options = load_audit_config(run)
    ensemble, y, g = load_saved_outputs(run)
    names = context.get('group_names')
    if names is None:
        info = read(run/'data/data_info.json')
        names = {str(value): key for key, value in info.get('group_id', {}).items()}
    quality = run/'audit/group_predictive_metrics.csv'
    if quality.exists():
        frame = pd.read_csv(quality, dtype={'group_id':str, 'group':str})
        names = {str(r.group_id): str(r.group) for r in frame.itertuples()}
    threshold = context.get('threshold', config.get('train', {}).get('threshold', .5))
    positive = options.get('positive_class', 1)
    diagnostics, reliability = diagnose_predictions(ensemble, y, g, group_names=names,
        threshold=threshold, positive_class=positive, **kwargs)
    identity = dict(condition=run.parent.name, seed=run.name, run=str(run.resolve()))
    if (run/'summary.csv').exists():
        summary = pd.read_csv(run/'summary.csv', dtype={'condition':str})
        if len(summary) != 1:
            raise ValueError('Expected one run summary row.')
        identity.update(condition=str(summary.iloc[0]['condition']),
                        seed=str(summary.iloc[0].get('data_seed', run.name)))
    for frame in (diagnostics, reliability):
        for key, value in identity.items():
            frame[key] = value
    meta = dict(**identity, threshold=threshold, positive_class=positive,
                classes=ensemble['mean_probs'].shape[1], group_names=names,
                decision_policy='Binary column-1 threshold (ensemble_quality); multiclass argmax',
                provenance=context.get('provenance', {}))
    return diagnostics, reliability, meta


def summarize_runs(frame, keys, metrics, run_counts):
    """Strict complete-run pointwise t intervals; never pool individual rows."""
    from modules.utils.reporting_utils import mean_ci95
    out = []
    if frame.duplicated([*keys, 'run']).any():
        raise ValueError('Duplicate run/summary cells.')
    for identity, part in frame.groupby(keys, sort=False, dropna=False):
        if not isinstance(identity, tuple):
            identity = (identity,)
        record = dict(zip(keys, identity))
        n = run_counts[record['condition']]
        for metric in metrics:
            values = pd.to_numeric(part[metric], errors='raise').to_numpy(float)
            values = values[np.isfinite(values)]
            complete = len(values) == n
            stats = mean_ci95(values) if complete and n > 1 else {}
            out.append(dict(**record, metric=metric, n_runs=n, n_valid=len(values),
                mean=float(values.mean()) if complete else np.nan, std=stats.get('sd', np.nan),
                ci95=stats.get('ci_half_width', np.nan), ci_low=stats.get('ci_low', np.nan),
                ci_high=stats.get('ci_high', np.nan)))
    return pd.DataFrame(out)


def compute_contrasts(frame, specifications, metrics):
    """Signed sums of group metrics within a run/scope; missing cells stay NaN.

    Specifications map contrast name to {exact saved group label: weight}.
    Require finite, nonzero, zero-sum weights. No absolute values are applied.
    """
    available = set(frame.group)
    if not isinstance(specifications, dict) or not specifications:
        raise ValueError('Contrasts must be a nonempty JSON object.')
    for name, weights in specifications.items():
        if not isinstance(weights, dict) or len(weights) < 2:
            raise ValueError(f'{name}: require at least two group weights.')
        w = np.asarray(list(weights.values()), dtype=float)
        if not np.isfinite(w).all() or not np.any(w != 0) or not np.isclose(w.sum(), 0):
            raise ValueError(f'{name}: weights must be finite, nonzero and sum to zero.')
        if set(weights) - available:
            raise ValueError(f'{name}: unknown group labels {set(weights)-available}.')
    output = []
    for identity, part in frame.groupby(['condition', 'run', 'seed', 'scope'], sort=False):
        if part.group.duplicated().any():
            raise ValueError('Group display labels must be unique within each run.')
        table = part.set_index('group')
        for name, weights in specifications.items():
            record = dict(zip(['condition','run','seed','scope'], identity), contrast=name)
            selected = table.reindex(list(weights))
            for metric in metrics:
                values = selected[metric].to_numpy(float)
                record[metric] = float(np.dot(list(weights.values()), values)) if np.isfinite(values).all() else np.nan
            output.append(record)
    return pd.DataFrame(output)