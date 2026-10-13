"""Combine completed per-run summary CSVs; descriptive seed variability only."""
import argparse
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path,help='e.g. experiments/main')
    args=p.parse_args()
    from modules.utils.reporting_utils import read_run_summaries
    try:
        frame = read_run_summaries(args.folder)
    except ValueError as exc:
        p.error(str(exc))
    frame.to_csv(args.folder/'all_runs.csv',index=False)
    columns=[c for c in frame.select_dtypes('number').columns
             if c not in ('data_seed','model_seed_start','support')]
    frame.groupby('condition')[columns].agg(['count','mean','std']).to_csv(args.folder/'by_condition.csv')
    print(f'Combined {len(frame)} runs. Standard deviations describe seed variability, not audit confidence intervals.')

if __name__=='__main__':
    main()