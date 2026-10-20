"""Dataset-independent run discovery, seed aggregation and table formatting."""
from pathlib import Path
import pandas as pd
import numpy as np

from numpy.typing import ArrayLike
from scipy.stats import t


def completed_runs(folder):
    runs = sorted(run for run in (Path(folder)/'runs').glob('*/seed_*')
                  if (run/'COMPLETE').is_file())
    validate_audit_policies(runs)
    return runs


def read_run_summaries(folder):
    paths = [run/'summary.csv' for run in completed_runs(folder)]
    if not paths:
        raise ValueError('No completed run summaries found')
    frames = [pd.read_csv(path) for path in paths]
    if any(len(frame) != 1 for frame in frames):
        raise ValueError('Expected exactly one summary row per completed run')
    return pd.concat(frames, ignore_index=True)


def mean_ci95(values: ArrayLike) -> dict[str, float]:
    """One scalar metric value per independent experiment run."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError("Provide at least two finite run-level values.")
    n = x.size
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / np.sqrt(n)
    half_width = float(t.ppf(0.975, df=n-1) * se)
    return {"mean": mean, "sd": sd, "se": se, "ci_low": mean - half_width, "ci_high": mean + half_width, "ci_half_width":half_width}

def aggregate(frame, keys, metrics, *, missing="skip"):
    """Aggregate once; skip missing values for reports or reject them for plots.

    Each input row is one equally weighted run. SD uses ddof=1.
    """
    if missing not in ("skip", "raise"):
        raise ValueError("missing must be skip or raise")
    frame = frame.copy()
    if 'audit_id' in frame and (frame.groupby(keys, dropna=False)['audit_id'].nunique(dropna=False) > 1).any():
        raise ValueError('Mixed audit settings; select one audit_id before aggregation')
    for metric in metrics:
        frame[metric] = pd.to_numeric(frame[metric], errors="raise")
        if missing == "raise" and not np.isfinite(frame[metric].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite values in {metric}")
    rows=[]
    for identity, part in frame.groupby(keys, sort=True, dropna=False):
        if not isinstance(identity, tuple):
            identity=(identity,)
        row=dict(zip(keys,identity))
        row['runs']=len(part)
        for metric in metrics:
            row[metric+'_n']=int(part[metric].notna().sum())
            row[metric+'_mean']=part[metric].mean()
            row[metric+'_std']=mean_ci95(part[metric])["sd"]
            row[metric+'_ci95']=mean_ci95(part[metric])["ci_half_width"]
        rows.append(row)
    columns = list(keys) + ['runs'] + [m+suffix for m in metrics for suffix in ('_n', '_mean', '_std', '_ci95')]
    return pd.DataFrame(rows, columns=columns)


def summary_to_long(stats, keys, metrics):
    """Reshape already computed statistics; never aggregate them again."""
    rows = []
    for row in stats.to_dict('records'):
        for metric in metrics:
            rows.append({**{key: row[key] for key in keys}, 'metric': metric,
                         'mean': row[metric+'_mean'], 'std': row[metric+'_std'], 'ci95': row[metric+'_ci95'],
                         'n': row[metric+'_n']})
    return pd.DataFrame(rows, columns=list(keys)+['metric', 'mean', 'std', 'ci95', 'n'])


def mean_std_table(stats, keys, metrics, digits):
    out=stats[keys+['runs']].copy()
    for m in metrics:
        out[m]=[f'{mean:.{digits}f} ± {sd:.{digits}f}' if pd.notna(sd)
                else f'{mean:.{digits}f} ± NA'
                for mean,sd in zip(stats[m+'_mean'],stats[m+'_std'])]
    return out


def latex_escape(value):
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$',
             '#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}',
             '^':r'\textasciicircum{}','±':r'$\pm$'}
    return ''.join(mapping.get(ch,ch) for ch in str(value))


def latex_table(frame):
    """Small booktabs tabular; escape user labels without a Jinja2 dependency."""
    lines=[r'% Mean +/- sample SD across runs; NA = SD undefined for one run.',
           r'\begin{tabular}{'+'l'*len(frame.columns)+'}',r'\toprule',
           ' & '.join(latex_escape(x) for x in frame.columns)+r' \\', r'\midrule']
    for row in frame.itertuples(index=False,name=None):
        lines.append(' & '.join(latex_escape(x) for x in row)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    return '\n'.join(lines)+'\n'

def validate_audit_policies(runs):
    """Prevent reports/plots from pooling incompatible settings within a condition."""
    import json
    policies = {}
    for run in runs:
        path = Path(run)/'audit/audit_context.json'
        context = json.loads(path.read_text()) if path.exists() else {}
        identity = context.get('audit_id', 'legacy')
        policies.setdefault(Path(run).parent.name, set()).add(identity)
    if any(len(values) > 1 for values in policies.values()):
        raise ValueError('Mixed audit settings within a condition; report each audit policy separately')
