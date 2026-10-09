"""Reports for arbitrary observed groups, without synthetic oracle assumptions."""
import argparse
import json
from pathlib import Path
import pandas as pd


def _json(path):
    return json.loads(path.read_text())


def statistics(frame, keys, metrics):
    rows=[]
    for identity,part in frame.groupby(keys,dropna=False,sort=True):
        if not isinstance(identity,tuple): identity=(identity,)
        row=dict(zip(keys,identity)); row['runs']=len(part)
        for m in metrics:
            values=pd.to_numeric(part[m],errors='raise')
            row[m+'_n']=int(values.notna().sum())
            row[m+'_mean']=values.mean()
            row[m+'_std']=values.std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def report(folder, out=None):
    from modules.reporting.synthetic_report import latex_table
    folder = Path(folder)
    output = Path(out) if out else folder/'report'
    if output.resolve().is_relative_to((folder/'runs').resolve()):
        raise ValueError('Report destination must not be inside the training runs')
    runs=[]; groups=[]; pairs=[]; coverages=[]
    for run in sorted((folder/'runs').glob('*/seed_*')):
        if not (run/'COMPLETE').exists(): 
            continue
        identity=dict(condition=run.parent.name,seed=int(run.name.removeprefix('seed_')))
        frame=pd.read_csv(run/'summary.csv')
        if len(frame)!=1: 
            raise ValueError(f'Expected one summary row: {run}')
        row=frame.iloc[0].to_dict()
        row.update(identity)
        row.pop('data_seed',None)
        scalar=run/'audit/classical_summary.json'
        if scalar.exists(): 
            row.update(_json(scalar))
        else:
            legacy=run/'audit/fairness.json'
            if legacy.exists():
                f=_json(legacy)
                for key,name in {'differential_fairness':'DF_epsilon','statistical_parity':'SP_gap',
                    'disparate_impact':'DI_ratio','equal_opportunity':'EO_gap','equalized_odds':'EOdds_gap'}.items():
                    row[name]=f[key]['aggregate']
        config=_json(run/'config.json')
        row['positive_label']=config.get('dataset_kwargs',{}).get('positive_label','unspecified')
        runs.append(row)
        quality=pd.read_csv(run/'audit/group_predictive_metrics.csv',dtype={'group':str})
        audit_path=run/'audit/uncertainty_audit.json'
        audit=_json(audit_path) if audit_path.exists() else None
        if audit:
            component=dict(zip(audit['group_ids'],audit['means']))
            for idx,item in quality.iterrows():
                for col,value in zip(['alea','epis','tot'],component[int(item.group_id)]):
                    quality.loc[idx,col]=value
            pairs.extend(dict(identity,pair=name,**values) for name,values in audit['pairs'].items())
        detail=run/'audit/performance_fairness.json'
        if detail.exists():
            performance=_json(detail)
            rates=pd.DataFrame(performance['groups']).drop(columns=['support'],errors='ignore')
            quality=quality.merge(rates,on='group_id',how='left',validate='one_to_one')
        groups.extend(dict(identity,**item) for item in quality.to_dict('records'))
        coverage=run/'audit/group_coverage.csv'
        if coverage.exists():
            coverages.extend(dict(identity,**item) for item in pd.read_csv(coverage,dtype={'group':str}).to_dict('records'))
    if not runs: 
        raise ValueError('No completed runs under folder/runs/<condition>/seed_*')
    frame=pd.DataFrame(runs)
    if frame['positive_label'].nunique()>1:
        raise ValueError('Runs have different positive-label conventions; report these separately')
    output.mkdir(parents=True,exist_ok=True)
    texts=['REAL-DATA EXPERIMENT REPORT',f'Completed runs: {len(runs)}',
           f"Positive label: {frame['positive_label'].iloc[0]}",
           'Means and sample SDs are across runs. *_n records nonmissing values.',
           'Undefined scores are missing, never zero. One run has undefined SD.',
           'Fairness scores apply only to observed test groups; inspect group_coverage.csv.',
           'No oracle entropies or synthetic interaction contrasts are inferred.']
    if coverages:
        absent=sum(not x['evaluated'] for x in coverages)
        texts.append(f'Group/run combinations absent from test: {absent}')
    else:
        texts.append('Group coverage unavailable in older runs; absence cannot be assessed from observed groups alone.')
    tables=[('runs',frame,['condition'],{'seed','model_seed_start','support'}),
            ('groups',pd.DataFrame(groups),['condition','group_id','group'],{'seed','group_id'})]
    if pairs: 
        tables.append(('pairs',pd.DataFrame(pairs),['condition','pair'],{'seed'}))
    for name,table,keys,exclude in tables:
        table.to_csv(output/f'{name}.csv',index=False)
        metrics=[c for c in table.select_dtypes('number') if c not in exclude and c not in keys]
        # Null interactions absent on real data are excluded rather than fabricated
        # TODO: Apply the formula from our paper to compute interaction metrics from observed group means, if desired.
        metrics=[m for m in metrics if table[m].notna().any()]
        stats=statistics(table,keys,metrics)
        stats.to_csv(output/f'{name}_stats.csv',index=False)
        display=stats[keys+['runs']].copy()
        for m in metrics:
            display[m]=[('NA' if pd.isna(mean) else f'{mean:.4g} ± '+('NA' if pd.isna(sd) else f'{sd:.4g}'))
                        for mean,sd in zip(stats[m+'_mean'],stats[m+'_std'])]
        (output/f'{name}.tex').write_text(latex_table(display))
        if name=='runs': 
            texts.append(display.to_string(index=False))
    if coverages: pd.DataFrame(coverages).to_csv(output/'group_coverage.csv',index=False)
    (output/'report.txt').write_text('\n\n'.join(texts)+'\n')
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    print(f'Report saved to {report(a.folder,a.out)}')

if __name__=='__main__': main()
