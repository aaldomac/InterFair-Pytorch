"""Display saved synthetic audits and export CSV/LaTeX tables. No PyTorch needed.

From project root:
  python -m experiments.synthetic.report experiments/synthetic/results/pilot
  python -m experiments.synthetic.report experiments/synthetic/results/pilot --pairs

Requires NumPy and pandas. Reads completed runs directly, not summarize.py output.
Exports derived reports only; never edits models, audits or original summaries.
Sample SD uses ddof=1 across ensemble runs (not across ensemble members).
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

QUALITY = ['accuracy', 'nll', 'brier']
DISPARITY = ['F_alea', 'F_epis', 'F_tot', 'F_U', 'hidden_normalized', 'F_U_int']
GROUPS = ['00', '01', '10', '11']
COMPONENTS = ['alea', 'epis', 'tot']


def read_json(path):
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def finite(values, label):
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError(f'{label}: missing or nonfinite values')
    return arr


def load_results(folder, conditions=None):
    """Read each completed run once; reject malformed completed artifacts."""
    runs, groups, pairs, skipped = [], [], [], []
    found = set()
    for path in sorted((folder/'runs').glob('*/seed_*')):
        if not path.is_dir():
            continue
        condition = path.parent.name
        if conditions and condition not in conditions:
            continue
        found.add(condition)
        if not (path/'COMPLETE').exists():
            skipped.append(str(path))
            continue
        try:
            audit = read_json(path/'synthetic_audit'/'uncertainty_audit.json')
            quality = read_json(path/'synthetic_audit'/'predictive_metrics.json')
            meta = read_json(path/'synthetic_data'/'metadata.json')
            seed = int(path.name.removeprefix('seed_'))
            if seed != audit['data_seed'] or seed != meta['seed']:
                raise ValueError('Data seed does not match directory')
            if meta['group_order'] != GROUPS:
                raise ValueError('Unexpected group order')
            cols = audit['component_order']
            if sorted(cols) != sorted(COMPONENTS):
                raise ValueError('Unexpected uncertainty components')
            means = finite(audit['means'], 'means')
            counts = finite(audit['counts'], 'counts')
            if means.shape != (4, 3) or counts.shape != (4,) or (counts <= 0).any():
                raise ValueError('Expected four nonempty groups and three uncertainty components')
            oracle = finite(audit['oracle_entropy_nats'], 'oracle')
            interaction = finite(audit['interaction'], 'interaction')
            radii = finite(audit['hoeffding_component_radii'], 'radii')
            if any(a.shape != (4,) for a in (oracle, radii)) or interaction.shape != (3,):
                raise ValueError('Invalid oracle/interaction/radius shape')
            row = dict(condition=condition, seed=seed, support=int(quality['support']),
                       **{k:quality[k] for k in QUALITY},
                       **{k:audit['maxima'][k] for k in DISPARITY if k != 'F_U_int'},
                       F_U_int=audit['F_U_int'])
            if row['support'] != counts.sum():
                raise ValueError('Predictive support differs from audit counts')
            for i, c in enumerate(cols):
                row['I_'+c] = interaction[i]
            row['I_oracle'] = meta['oracle_interaction_bits']*np.log(2.)
            finite([row[k] for k in QUALITY+DISPARITY+['I_alea','I_epis','I_tot','I_oracle']], 'metrics')
            group_quality = pd.read_csv(path/'synthetic_audit'/'group_predictive_metrics.csv')
            if set(group_quality.group_id) != set(range(4)) or len(group_quality) != 4:
                raise ValueError('Invalid group quality IDs')
            group_quality = group_quality.set_index('group_id')
            local_groups = []
            for i, g in enumerate(GROUPS):
                q = group_quality.loc[i]
                finite([q[k] for k in QUALITY], 'group quality')
                if q['support'] != counts[i]:
                    raise ValueError('Group supports disagree')
                gr = dict(condition=condition, seed=seed, group=g, support=int(counts[i]),
                          **{c:means[i, cols.index(c)] for c in COMPONENTS},
                          **{k:float(q[k]) for k in QUALITY}, oracle_alea=oracle[i],
                          alea_error=means[i, cols.index('alea')]-oracle[i],
                          hoeffding_radius=radii[i])
                local_groups.append(gr)
            expected_pairs = {f'{a}-{b}' for i,a in enumerate(GROUPS) for b in GROUPS[i+1:]}
            if set(audit['pairs']) != expected_pairs:
                raise ValueError('Expected six group pairs')
            local_pairs = []
            for pair, metrics in audit['pairs'].items():
                values = {k:metrics[k] for k in DISPARITY if k != 'F_U_int'}
                finite(list(values.values()), 'pair metrics')
                local_pairs.append(dict(condition=condition, seed=seed, pair=pair, **values))
            runs.append(row)
            groups.extend(local_groups)
            pairs.extend(local_pairs)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            raise ValueError(f'Cannot report completed run {path}: {exc}') from exc
    if conditions and set(conditions)-found:
        raise ValueError(f'Conditions not found: {sorted(set(conditions)-found)}')
    print(f"Runs in report: {runs}")
    if not runs:
        raise ValueError('No completed runs found under folder/runs/<condition>/seed_*')
    return pd.DataFrame(runs), pd.DataFrame(groups), pd.DataFrame(pairs), skipped


def aggregate(frame, keys, metrics):
    """One equally weighted observation per run; never pool audit individuals."""
    rows=[]
    for identity, part in frame.groupby(keys, sort=True, dropna=False):
        if not isinstance(identity, tuple):
            identity=(identity,)
        row=dict(zip(keys,identity))
        row['runs']=len(part)
        for metric in metrics:
            row[metric+'_mean']=part[metric].mean()
            row[metric+'_std']=part[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


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


def build_report(runs, groups, pairs, digits=4, show_pairs=False):
    texts=[]
    exports={}
    def show(title, frame):
        texts.append(title+'\n'+frame.to_string(index=False, float_format=lambda x:f'{x:.{digits}f}'))
    show('COMPLETED RUNS', runs.groupby('condition').agg(
        runs=('seed','size'), seeds=('seed',lambda s:', '.join(map(str, sorted(s))))).reset_index())
    show('PREDICTIVE QUALITY — each ensemble run', runs[['condition','seed','support']+QUALITY])
    show('UNCERTAINTY DISPARITY — each ensemble run',runs[['condition','seed']+DISPARITY])
    specs=[('predictive',runs,['condition'],QUALITY),
           ('disparity',runs,['condition'],DISPARITY),
           ('interaction',runs,['condition'],['I_alea','I_epis','I_tot','I_oracle']),
           ('group_uncertainty',groups,['condition','group'],['alea','epis','tot']),
           ('group_quality',groups,['condition','group'],QUALITY),
           ('oracle',groups,['condition','group'],['oracle_alea','alea_error'])]
    if show_pairs:
        specs.append(('pairs',pairs,['condition','pair'],DISPARITY[:-1]))
    for name,frame,keys,metrics in specs:
        stats=aggregate(frame,keys,metrics)
        display=mean_std_table(stats,keys,metrics,digits)
        show(name.upper().replace('_',' ')+' — mean ± sample SD across runs',display)
        exports[name+'_stats.csv']=stats.to_csv(index=False)
        exports[name+'.tex']=latex_table(display)
    exports['runs.csv']=runs.to_csv(index=False)
    exports['groups.csv']=groups.to_csv(index=False)
    exports['pairs.csv']=pairs.to_csv(index=False)
    # Always retain full pair statistics numerically, even with concise terminal output.
    exports['pairs_stats.csv']=aggregate(pairs,['condition','pair'],DISPARITY[:-1]).to_csv(index=False)
    notes='''INTERPRETATION
Accuracy is a fraction (higher is better); NLL and class-1 Brier are lower-is-better.
All predictive metrics describe the ensemble mean, not member zero.
U means, F_alea/F_epis/F_tot and signed interactions I_* are in nats.
F_U, hidden_normalized and F_U_int are normalized, dimensionless disparities.
Each worst-pair metric is maximized BEFORE averaging runs; different metrics may select different pairs.
I_oracle is the DGP entropy interaction; alea_error is learned mean aleatoric minus oracle entropy.
SD uses n-1 and measures variability across complete ensemble runs, not members or audit rows.
One run gives SD=NA, not zero. These are not confidence intervals or significance tests.
Seeds change both generated data and training; the SD includes both sources of variability.
Group order: 00, 01, 10, 11. Audit counts and Hoeffding radii remain in groups.csv.
Hoeffding radii apply to aleatoric/epistemic group means conditional on a fitted ensemble;
they are not the between-run SD and are not averaged into confidence intervals here.
No outcome-fairness baselines or bootstrap intervals are computed by this reporter.
Only existing completed runs are counted; unstarted runs cannot be inferred from directories.
Use the same experimental settings across seeds within each condition.
LaTeX tables require \\usepackage{booktabs}; CSV files preserve unrounded numeric values.'''
    texts.append(notes)
    return '\n\n'.join(texts)+'\n',exports

def stripe_region_report(folder, conditions=None, digits=4):
    """Additional outputs; keep load_results/build_report public APIs unchanged."""
    rows = []
    for path in sorted((folder/'runs').glob('*/seed_*')):
        if not (path/'COMPLETE').exists():
            continue
        condition = path.parent.name
        if conditions and condition not in conditions:
            continue
        audit = read_json(path/'synthetic_audit'/'uncertainty_audit.json')
        for region in audit.get('stripe_regions', []):
            rows.append(dict(condition=condition, seed=audit['audit_seed'], **region))
    if not rows:
        return '', {}
    frame = pd.DataFrame(rows)
    exports = {'stripe_regionscsv': frame.to_csv(index=False)}
    # Exclude empty regions explicitly: runs counts nonempty run-level estimates.
    valid = frame[frame.support > 0]
    keys = ['condition', 'group', 'region']
    metrics = ['alea', 'epis', 'tot', 'oracle_alea', 'alea_error']
    stats = aggregate(valid, keys, metrics)
    display = mean_std_table(stats, keys, metrics, digits)
    exports['stripe_regions_stats.csv'] = stats.to_csv(index=False)
    exports['stripe_regions.tex'] = latex_table(display)
    text = ('STRIPE REGIONS — mean +/- sample SD across nonempty runs\n'
            + display.to_string(index=False)
            + '\nRegion boundaries vary with the condition width. Empty regions are '
              'omitted from this table; supports for every run are in stripe_regions.csv.\n')
    return text, exports 

def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folder',type=Path,help='Experiment root, e.g. experiments/synthetic/results/pilot')
    parser.add_argument('--condition',action='append',help='Filter condition (repeatable)')
    parser.add_argument('--digits',type=int,default=4,help='Display/LaTeX decimal places, default 4')
    parser.add_argument('--pairs',action='store_true',help='Also display/export LaTeX for all pairwise comparisons')
    parser.add_argument('--no-export',action='store_true',help='Terminal only; write nothing')
    parser.add_argument('--out',type=Path,help='Derived report directory; default <folder>/report')
    args=parser.parse_args()
    if not 0<=args.digits<=10:
        parser.error('--digits must be between 0 and 10')
    try:
        runs,groups,pairs,skipped=load_results(args.folder,args.condition)
        report,exports=build_report(runs,groups,pairs,args.digits,args.pairs)
        regional_text, regional_exports = stripe_region_report(args.folder, args.condition, args.digits)
        report += '\n' + regional_text
        exports.update(regional_exports)
        preamble=f'SYNTHETIC EXPERIMENT REPORT\nSource: {args.folder.resolve()}\n'
        if skipped:
            preamble+='Skipped incomplete runs:\n'+'\n'.join(skipped)+'\n'
        report=preamble+'\n'+report
        print(report)
        if not args.no_export:
            out=args.out or args.folder/'report'
            # Protect source artifacts even if an unsafe custom destination is supplied.
            source=(args.folder/'runs').resolve()
            if out.resolve()==source or source in out.resolve().parents:
                raise ValueError('--out must not be inside the source runs directory')
            out.mkdir(parents=True,exist_ok=True)
            exports['report.txt']=report
            for name,content in exports.items():
                (out/name).write_text(content,encoding='utf-8')
            print(f'Reports saved to {out.resolve()} (derived files replaced on rerun).')
    except (ValueError,OSError,KeyError) as exc:
        parser.exit(1,f'Error: {exc}\n')


if __name__=='__main__':
    main()