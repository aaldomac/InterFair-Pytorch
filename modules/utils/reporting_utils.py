"""Dataset-independent run discovery, seed aggregation and table formatting."""
from pathlib import Path
import pandas as pd
import numpy as np

def completed_runs(folder):
    return sorted(run for run in (Path(folder)/'runs').glob('*/seed_*')
                  if (run/'COMPLETE').is_file())


def read_run_summaries(folder):
    paths = [run/'summary.csv' for run in completed_runs(folder)]
    if not paths:
        raise ValueError('No completed run summaries found')
    frames = [pd.read_csv(path) for path in paths]
    if any(len(frame) != 1 for frame in frames):
        raise ValueError('Expected exactly one summary row per completed run')
    return pd.concat(frames, ignore_index=True)


def aggregate(frame, keys, metrics, *, missing="skip"):
    """Aggregate once; skip missing values for reports or reject them for plots.

    Each input row is one equally weighted run. SD uses ddof=1.
    """
    if missing not in ("skip", "raise"):
        raise ValueError("missing must be skip or raise")
    frame = frame.copy()
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
            row[metric+'_std']=part[metric].std(ddof=1)
        rows.append(row)
    columns = list(keys) + ['runs'] + [m+suffix for m in metrics for suffix in ('_n', '_mean', '_std')]
    return pd.DataFrame(rows, columns=columns)


def summary_to_long(stats, keys, metrics):
    """Reshape already computed statistics; never aggregate them again."""
    rows = []
    for row in stats.to_dict('records'):
        for metric in metrics:
            rows.append({**{key: row[key] for key in keys}, 'metric': metric,
                         'mean': row[metric+'_mean'], 'std': row[metric+'_std'],
                         'n': row[metric+'_n']})
    return pd.DataFrame(rows, columns=list(keys)+['metric', 'mean', 'std', 'n'])


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