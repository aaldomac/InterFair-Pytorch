"""One subplot per metric, one bar per intersectional group; no model inference."""
from pathlib import Path
import argparse
import json
import math
import numpy as np
import pandas as pd

ALIASES = {
    'U_alea': 'alea', 'U_epis': 'epis', 'U_tot': 'tot',
    'U_mix': 'group_mix_entropy', 'U_het': 'group_het_entropy',
    'aleatoric_uncertainty': 'alea', 'epistemic_uncertainty': 'epis',
    'predictive_entropy': 'tot', 'Accuracy': 'accuracy',
}
NATS = {'alea', 'epis', 'tot', 'group_mix_entropy', 'group_het_entropy', 'nll'}


def _json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def _scalars(mapping, prefix=''):
    """Flatten scalar fields; vectors get explicit component indices."""
    result = {}
    for key, value in mapping.items():
        name = prefix + str(key)
        if isinstance(value, dict):
            result.update(_scalars(value, name + '.'))
        elif isinstance(value, list):
            result.update(_scalars({str(i): v for i, v in enumerate(value)}, name + '.'))
        elif value is None or isinstance(value, (int, float)) and not isinstance(value, bool):
            result[name] = value
    return result


def read_group_run(run):
    """Join saved per-group artifacts by group ID, never by row position."""
    run = Path(run)
    audit = run / 'audit'
    rows = {}
    names = _json(audit / 'audit_context.json').get('group_names', {})

    def row(gid):
        key = str(gid)
        return rows.setdefault(key, {'group_id': key, 'group': str(names.get(key, key))})

    path = audit / 'group_predictive_metrics.csv'
    if path.exists():
        frame = pd.read_csv(path, dtype={'group_id': str, 'group': str})
        if frame.group_id.duplicated().any():
            raise ValueError(f"Duplicate group IDs in {path}")
        for record in frame.to_dict('records'):
            row(record['group_id']).update(record)

    uncertainty = _json(audit / 'uncertainty_audit.json')
    ids = uncertainty.get('group_ids', [])
    if ids:
        components = uncertainty['component_order']
        means = np.asarray(uncertainty['means'], dtype=float)
        if means.shape != (len(ids), len(components)):
            raise ValueError(f"Invalid uncertainty means in {run}.")
        for i, gid in enumerate(ids):
            target = row(gid)
            target.update(zip(components, means[i]))
            if 'group_order' in uncertainty:
                target['group'] = str(uncertainty['group_order'][i])
        for key in ('counts', 'oracle_entropy_nats', 'alea_minus_oracle'):
            if key in uncertainty:
                values = uncertainty[key]
                if len(values) != len(ids):
                    raise ValueError(f"Invalid {key} length in {run}.")
                for gid, value in zip(ids, values):
                    row(gid)['uncertainty_' + key] = value
        # Group-keyed maps are per-group metrics, unlike maxima or interactions.
        for key, values in uncertainty.items():
            if isinstance(values, dict) and set(values) == {str(g) for g in ids}:
                for gid in ids:
                    row(gid).update(_scalars({key: values[str(gid)]}))

    performance = _json(audit / 'performance_fairness.jdon')
    for record in performance.get('groups', []):
        target = row(record['group_id'])
        target.update(_scalars({k: v for k, v in record.items() if k != 'group_id'}))
        # Keep the two accuracy sources distinct (positive_class=0 can matter).
        if 'Accuracy' in record:
            target['performance_accuracy'] = record['Accuracy']
            target.setdefault('accuracy', record['Accuracy'])
                                     
    fairness = _json(audit / 'fairness.json')
    def collect(node, prefix=''):
        if not isinstance(node, dict):
            return
        if isinstance(node.get('groups'), list):
           group_ids = node['groups']
           for field in ('group_rates', 'group_probs', 'group_uncertainties',
                          'subgroup_group_uncertainties', 'subgroup_values', 'values'):
               values = node.get(field)
               if isinstance(values, list) and len(values) == len(group_ids):
                   for gid, value in zip(group_ids, values):
                       row(gid).update(_scalars({prefix + field: value}))
        for key, value in node.items():
            if isinstance(value, dict):
                collect(value, prefix + key + '.')
    collect(fairness)
    # Fallback when uncertainty auditing was disabled but fairness was saved.
    for source, dest in (('aleatoric_uncertainty', 'alea'),
                         ('epistemic_uncertainty', 'epis'), ('predictive_entropy', 'tot')):
        for target in rows.values():
            key = f'uncertainty.{source}.group_uncertainties'
            if key in target:
                target.setdefault(dest, target[key])
    distributions = _json(audit / 'distribution.json')
    for gid, values in distributions.get('group_entropy_decompositions', {}).items():
        row(gid).update(_scalars(values, 'distribution.'))

    if not rows:
        raise ValueError(f'No saved group metrics found in {audit}')
    identity = {}
    summary = run / 'summary.csv'
    if summary.exists():
        identities = pd.read_csv(summary, dtype={'condition': str})
        if len(identities) != 1:
            raise ValueError(f'Expected one summary row in {run}')
        identity = identities.iloc[0].to_dict()
    for target in rows.values():
        target.update(condition=str(identity.get('condition', run.parent.name)),
                      seed=str(identity.get('data_seed', run.name)), run=str(run.resolve()))
    return pd.DataFrame(rows.values())


def read_group_metrics(source):
    """Accept a run, experiment root, or wide per-run/per-group CSV."""
    source = Path(source)
    if source.is_file():
        frame = pd.read_csv(source, dtype={'condition': str, 'group_id': str,
                                          'group': str, 'seed': str})
        if 'group_id' not in frame and 'group' in frame:
            frame['group_id'] = frame['group']
        if 'group' not in frame and 'group_id' in frame:
            frame['group'] = frame['group_id']
        if 'condition' not in frame:
            frame['condition'] = 'results'
        if 'run' not in frame:
            if 'seed' in frame:
                frame['run'] = frame['condition'] + '/' + frame['seed']
            else:
                frame['run'] = 'single_run'
        return frame
    if (source / 'audit').is_dir():
        return read_group_run(source)
    from modules.utils.reporting_utils import completed_runs
    runs = completed_runs(source)
    if not runs:
        raise ValueError('Input must be a run, an experiment with completed runs, or a group CSV.')
    return pd.concat([read_group_run(run) for run in runs], ignore_index=True)


def summarize_group_metrics(frame, metrics):
    """Equal-run means within each condition/group. Missing runs yield NaN.

    n_valid and n_runs are exported. SD is across runs, not across individuals.
    A missing/undefined value in any selected run makes that bar unavailable.
    """
    metrics = [ALIASES.get(m, m) for m in metrics]
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError('Select distinct metrics, including after alias resolution.')
    required = {'condition', 'run', 'group_id', 'group'}
    if not required <= set(frame) or frame.empty:
        raise ValueError(f'Require nonempty data with {sorted(required)}.')
    if frame[list(required)].isna().any().any():
        raise ValueError('Missing group/run identifiers or labels.')
    if frame.duplicated(['condition', 'run', 'group_id']).any():
        raise ValueError('Multiple rows per condition/run/group; filter the CSV first.')
    missing = set(metrics) - set(frame)
    if missing:
        raise ValueError(f'Metrics not saved per group: {sorted(missing)}. Use --list-metrics.')
    result = []
    for condition, part in frame.groupby('condition', sort=False):
        total = part.run.nunique()
        for gid, group in part.groupby('group_id', sort=False):
            if group.group.nunique() != 1:
                raise ValueError(f'Inconsistent group label for {condition}/{gid}.')
            for metric in metrics:
                values = pd.to_numeric(group[metric], errors='raise').to_numpy(dtype=float)
                valid = values[np.isfinite(values)]
                complete = len(valid) == total
                result.append(dict(condition=condition, group_id=gid, group=group.group.iloc[0],
                    metric=metric, mean=float(valid.mean()) if complete else np.nan,
                    std=float(valid.std(ddof=1)) if complete and total > 1 else np.nan,
                    n_valid=len(valid), n_runs=total))
    return pd.DataFrame(result)


def _error_values(part, errorbars):
    """Select sample SD or the two-sided Student-t 95% CI half-width."""
    if errorbars == 'none':
        return np.full(len(part), np.nan)
    column = 'std' if errorbars == 'sd' else 'ci95'
    if column not in part:
        raise ValueError(f'Missing {column}; regenerate with summarize_group_metrics().')
    return part[column].to_numpy(dtype=float)


def _error_note(errorbars):
    if errorbars == 'none':
        return ''
    label = 'sample SD across runs' if errorbars == 'sd' else '95% Student-t CI of the run mean'
    return f'\nError bars: {label} (unavailable for one run)'


def plot_group_metrics(summary, *, ncols=3, errorbars='sd', annotate=False, title=None):
    """Plot one condition as N independently scaled metric panels; return fig, axes."""
    import matplotlib.pyplot as plt
    if summary.empty or summary.condition.nunique() != 1:
        raise ValueError('Pass exactly one nonempty condition to plot_group_metrics.')
    if ncols < 1 or errorbars not in ('sd', 'ci95', 'none'):
        raise ValueError('Require ncols >= 1 and errorbars sd, ci95 or none.')
    metrics = list(dict.fromkeys(summary.metric))
    groups = list(dict.fromkeys(summary.group_id))
    ncols = min(ncols, len(metrics))
    fig, axes = plt.subplots(math.ceil(len(metrics)/ncols), ncols,
        figsize=(5*ncols, 4.2*math.ceil(len(metrics)/ncols)), squeeze=False,
        constrained_layout=True)
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(groups)))
    for ax, metric in zip(axes.flat, metrics):
        part = summary[summary.metric == metric].set_index('group_id').loc[groups]
        values = part['mean'].to_numpy(dtype=float)
        x = np.arange(len(groups))
        good = np.isfinite(values)
        ax.bar(x[good], values[good], color=colors[good], width=.72)
        sd = _error_values(part, errorbars)
        eligible = good & np.isfinite(sd)
        if errorbars != 'none' and eligible.any():
            ax.errorbar(x[eligible], values[eligible], yerr=sd[eligible], fmt='none',
                        color='#333333', capsize=3, linewidth=1)
        for i in x:
            if not good[i]:
                ax.text(i, .03, 'NA', transform=ax.get_xaxis_transform(), ha='center', color='#a33')
            elif annotate:
                edge = values[i]
                if errorbars != 'none' and np.isfinite(sd[i]):
                    edge += sd[i] if values[i] >= 0 else -sd[i]
                ax.annotate(f'{values[i]:.3g}', (i, edge), xytext=(0, 5 if values[i]>=0 else -12),
                            textcoords='offset points', ha='center', fontsize=9)
        ax.set_xticks(x, part['group'], rotation=35, ha='right')
        ax.set_xlim(-.6, len(groups)-.4)
        ax.set_title(metric)
        ax.set_xlabel('Intersectional group')
        ax.set_ylabel('nats' if metric in NATS else 'Value')
        ax.axhline(0, color='#555555', linewidth=.7)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.margins(y=.18)
        ax.spines[['top', 'right']].set_visible(False)
    for ax in list(axes.flat)[len(metrics):]:
        ax.set_visible(False)
    note = 'Each panel has its own scale; NA = missing/undefined in at least one run'
    note += _error_note(errorbars)
    fig.suptitle((title or str(summary.condition.iloc[0])) + '\n' + note, fontsize=11)
    return fig, axes


def plot_group_metrics_combined(summary, *, ncols=1, errorbars='sd',
                                annotate=False, title=None):
    """One metric per panel; condition blocks contain consistently ordered groups.

    All conditions share a y scale within a metric. Different metrics have their
    own scales. Missing condition/group cells retain their position and show NA.
    Returns fig, axes; accepts summarize_group_metrics() output unchanged.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    if summary.empty or ncols < 1 or errorbars not in ('sd', 'ci95', 'none'):
        raise ValueError('Require nonempty summary, ncols >= 1, and errorbars sd, ci95 or none.')
    if summary.duplicated(['condition', 'group_id', 'metric']).any():
        raise ValueError('Duplicate condition/group/metric cells.')
    metrics = list(dict.fromkeys(summary.metric))
    conditions = list(dict.fromkeys(summary.condition))
    groups = list(dict.fromkeys(summary.group_id))
    labels = {}
    for gid, part in summary.groupby('group_id', sort=False):
        if part.group.nunique() != 1:
            raise ValueError(f'Group {gid} has different meanings across conditions.')
        labels[gid] = str(part.group.iloc[0])
    cmap = plt.get_cmap('tab10' if len(groups) <= 10 else 'tab20' if len(groups) <= 20 else 'viridis')
    colors = cmap(np.linspace(0, 1, len(groups)))
    ncols = min(ncols, len(metrics))
    nrows = math.ceil(len(metrics) / ncols)
    panel_width = max(7, len(conditions) * max(1.7, .28*len(groups)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_width*ncols, 3.8*nrows),
                             squeeze=False, constrained_layout=True)
    centers = np.arange(len(conditions), dtype=float)
    width = .78 / len(groups)
    for ax, metric in zip(axes.flat, metrics):
        for index in range(len(conditions)):
            if index % 2 == 0:
                ax.axvspan(index-.46, index+.46, color='#f1f3f5', zorder=0)
        table = summary[summary.metric == metric].set_index(['condition', 'group_id'])
        for j, gid in enumerate(groups):
            part = table.reindex(pd.MultiIndex.from_tuples(
                [(condition, gid) for condition in conditions], names=['condition', 'group_id']))
            values = part['mean'].to_numpy(dtype=float)
            sd = _error_values(part, errorbars)
            x = centers + (j-(len(groups)-1)/2)*width
            good = np.isfinite(values)
            ax.bar(x[good], values[good], width=width*.9, color=colors[j], zorder=3)
            eligible = good & np.isfinite(sd)
            if errorbars != 'none' and eligible.any():
                ax.errorbar(x[eligible], values[eligible], yerr=sd[eligible],
                            fmt='none', color='#333333', capsize=2, linewidth=.9, zorder=4)
            for i in range(len(conditions)):
                if not good[i]:
                    ax.text(x[i], .025, 'NA', transform=ax.get_xaxis_transform(),
                            ha='center', fontsize=8, color='#a33')
                elif annotate:
                    edge = values[i]
                    if errorbars != 'none' and np.isfinite(sd[i]):
                        edge += sd[i] if values[i] >= 0 else -sd[i]
                    ax.annotate(f'{values[i]:.3g}', (x[i], edge),
                                xytext=(0, 4 if values[i]>=0 else -12),
                                textcoords='offset points', ha='center', fontsize=8)
        ax.set_xticks(centers, conditions, rotation=20, ha='right')
        ax.set_xlim(-.55, len(conditions)-.45)
        ax.set_title(metric, loc='left')
        ax.set_xlabel('Condition')
        ax.set_ylabel('nats' if metric in NATS else 'Value')
        ax.axhline(0, color='#555555', linewidth=.7)
        ax.grid(axis='y', alpha=.2, zorder=0)
        ax.set_axisbelow(True)
        ax.margins(y=.2)
        ax.spines[['top', 'right']].set_visible(False)
    for ax in list(axes.flat)[len(metrics):]:
        ax.set_visible(False)
    # Axis legend is accounted for by constrained layout, including older Matplotlib.
    axes.flat[0].legend(handles=[Patch(facecolor=colors[j], label=labels[gid])
                                for j, gid in enumerate(groups)],
                        title='Intersectional group', loc='lower left',
                        bbox_to_anchor=(0, 1.12), ncol=min(8, len(groups)), frameon=False)
    note = 'Shared scale across conditions within each metric; NA = missing/undefined'
    note += _error_note(errorbars)
    fig.suptitle((title+'\n' if title else '') + note, fontsize=11)
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--metrics', nargs='+', default=['U_epis', 'U_alea', 'U_tot'])
    parser.add_argument('--conditions', nargs='+')
    parser.add_argument('--seeds', nargs='+')
    parser.add_argument('--groups', nargs='+', help='Group IDs in desired order')
    parser.add_argument('--list-metrics', action='store_true')
    parser.add_argument('--out', type=Path, default=Path('figures/group_metrics'))
    parser.add_argument('--formats', nargs='+', choices=['png', 'pdf', 'svg'], default=['png', 'pdf'])
    parser.add_argument('--layout', choices=['separate', 'combined'], default='separate',
                        help='Separate figures per condition, or one figure with condition blocks')
    parser.add_argument('--ncols', type=int, default=None,
                        help='Default: 3 for separate, 1 for combined')
    parser.add_argument('--errorbars', choices=['sd', 'ci95', 'none'], default='sd')
    parser.add_argument('--annotate', action='store_true')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        frame = read_group_metrics(args.input)
        for field, selection in (('condition', args.conditions), ('seed', args.seeds),
                                 ('group_id', args.groups)):
            if selection is not None:
                if field not in frame or set(selection) - set(frame[field].astype(str)):
                    raise ValueError(f'Unknown {field} selection: {selection}')
                frame = frame[frame[field].astype(str).isin(selection)]
        if args.conditions:
            order = {c: i for i, c in enumerate(args.conditions)}
            frame = frame.assign(_condition_order=frame.condition.map(order)).sort_values(
                '_condition_order', kind='stable').drop(columns='_condition_order')
        if args.groups:
            frame = frame.assign(_order=frame.group_id.astype(str).map(
                {g:i for i,g in enumerate(args.groups)})).sort_values('_order', kind='stable').drop(columns='_order')
        if args.list_metrics:
            excluded = {'condition', 'seed', 'run', 'group_id', 'group'}
            print('\n'.join(str(c) for c in frame if c not in excluded))
            print('\nAliases:', ALIASES)
            return
        summary = summarize_group_metrics(frame, args.metrics)
        if args.conditions:
            summary = summary.assign(_order=summary.condition.map(
                {c: i for i, c in enumerate(args.conditions)})).sort_values(
                '_order', kind='stable').drop(columns='_order')
        args.out.mkdir(parents=True, exist_ok=True)
        if args.layout == 'combined':
            fig, _ = plot_group_metrics_combined(summary,
                ncols=args.ncols if args.ncols is not None else 1,
                errorbars=args.errorbars, annotate=args.annotate)
            for fmt in args.formats:
                fig.savefig(args.out / ('combined.' + fmt), dpi=200)
            plt.close(fig)
            print(f'Combined conditions: {args.out / "combined"}')
        else:
            for i, (condition, part) in enumerate(summary.groupby('condition', sort=False), 1):
                fig, _ = plot_group_metrics(part,
                    ncols=args.ncols if args.ncols is not None else 3,
                    errorbars=args.errorbars, annotate=args.annotate)
                stem = args.out / f'condition_{i:02d}'
                for fmt in args.formats:
                    fig.savefig(str(stem)+'.'+fmt, dpi=200)
                plt.close(fig)
                print(f'{condition}: {stem}')
        frame.to_csv(args.out/'group_runs.csv', index=False)
        summary.to_csv(args.out/'group_metric_values.csv', index=False)
        settings = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
        settings['conditions_in_file_order'] = list(dict.fromkeys(summary.condition))
        settings['aggregation'] = 'Equal run weights within condition/group; incomplete bars NA'
        settings['ci_method'] = 'Pointwise two-sided Student-t, df=n_runs-1, 95%; across run means'
        settings['ci_confidence'] = 0.95
        (args.out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    except (ValueError, KeyError, OSError, ImportError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()