"""Combine completed per-run summary CSVs; descriptive seed variability only."""
import argparse
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path,help='e.g. experiments/synthetic/results/main')
    args=p.parse_args()
    files=sorted(path for path in (args.folder/'runs').glob('*/*/summary.csv')
                 if (path.parent/'COMPLETE').exists())
    if not files:
        p.error('No completed training runs found')
    frame=pd.concat([pd.read_csv(path) for path in files],ignore_index=True)
    frame.to_csv(args.folder/'all_runs.csv',index=False)
    columns=[c for c in frame.select_dtypes('number').columns
             if c not in ('data_seed','model_seed_start','support')]
    frame.groupby('condition')[columns].agg(['count','mean','std']).to_csv(args.folder/'by_condition.csv')
    print(f'Combined {len(files)} runs. Standard deviations describe seed variability, not audit confidence intervals.')

if __name__=='__main__':
    main()