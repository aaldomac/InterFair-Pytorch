"""Post-hoc classical fairness reports from saved row-aligned ensemble predictions."""
import argparse
import json
from pathlib import Path
import pandas as pd


def render_report(folder, *, out=None, conditions=None):
    """Use saved classical audits; no training or prediction recomputation."""
    from modules.utils.reporting_utils import aggregate, mean_std_table, latex_table, completed_runs
    runs,groups,pairs=[],[],[]
    for run in completed_runs(folder):
        if not (run/'COMPLETE').exists() or not (run/'audit/classical_summary.json').exists():
            continue
        if conditions and run.parent.name not in conditions:
            continue
        identity=dict(condition=run.parent.name,seed=int(run.name.removeprefix('seed_')))
        scalar=json.loads((run/'audit/classical_summary.json').read_text())
        detail=json.loads((run/'audit/performance_fairness.json').read_text())
        runs.append(dict(identity,**scalar))
        groups.extend(dict(identity,**row) for row in detail['groups'])
        original=json.loads((run/'audit/fairness.json').read_text())
        pair_keys={'differential_fairness':'DF_epsilon','statistical_parity':'SP_gap',
                   'disparate_impact':'DI_ratio','equal_opportunity':'EO_gap','equalized_odds':'EOdds_gap'}
        for item in detail['pairs']:
            row=dict(identity,**item)
            for key,label in pair_keys.items():
                metric=original[key]; order=metric['groups']
                i,j=order.index(item['group_a']),order.index(item['group_b'])
                row[label]=metric['matrix'][i][j]
            if item['TPR_gap'] is None: row['EO_gap']=None
            if item['TPR_gap'] is None or item['FPR_gap'] is None: row['EOdds_gap']=None
            pairs.append(row)
    if not runs:
        raise ValueError('No completed classical audits; enable fairness and run scripts.audit first.')
    frame=pd.DataFrame(runs); metrics=[c for c in frame if c not in ('condition','seed')]
    stats=aggregate(frame,['condition'],metrics)
    display=mean_std_table(stats,['condition'],metrics,4)
    output=Path(out) if out else Path(folder)/'classical_report'
    if output.resolve().is_relative_to((Path(folder)/'runs').resolve()):
        raise ValueError('Report destination must not be inside training runs')
    output.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output/'runs.csv',index=False)
    pd.DataFrame(groups).to_csv(output/'groups.csv',index=False)
    pd.DataFrame(pairs).to_csv(output/'pairs.csv',index=False)
    stats.to_csv(output/'classical_stats.csv',index=False)
    (output/'classical.tex').write_text(latex_table(display))
    notes=('DF_epsilon is original binary outcome differential fairness (both outcomes).\n'
           'TPR/Accuracy/etc epsilon compares only the named smoothed rate; max_ratio=exp(epsilon).\n'
           'Smoothing alpha is recorded in each performance_fairness.json; gaps use raw rates.\n'
           'Lower gaps/epsilon are better; DI_ratio is better closer to 1.\n'
           'Undefined conditional rates remain missing; *_n columns count valid runs.\n'
           'Positive-class conventions are recorded in each performance_fairness.json.\n'
           'Mean +/- sample SD is across runs, not a confidence interval.\n')
    (output/'report.txt').write_text(notes+'\n'+display.to_string(index=False)+'\n')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    args=parser.parse_args()
    print(f'Report: {render_report(args.folder)}')

if __name__=='__main__':
    main()