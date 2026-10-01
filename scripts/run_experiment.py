"""Run real or synthetic experiments from YAML; never overwrite existing runs."""
import argparse
import csv
import json
from pathlib import Path
import yaml
from modules.pipelines.experiment_config import ROOT, read_config, pipeline_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/synthetic/smoke.yaml')
    parser.add_argument('--condition')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--generate-only',action='store_true')
    args = parser.parse_args()
    cfg, output = read_config(args.config)
    conditions, seeds = cfg['conditions'], cfg['data_seeds']
    if args.condition is not None:
        if args.condition not in conditions:
            parser.error('Unknown condition')
        conditions = {args.condition:conditions[args.condition]}
    if args.seed is not None:
        if args.seed not in seeds:
            parser.error('--seed must occur in data_seeds')
        seeds = [args.seed]
    synthetic = cfg['dataset']['name'] == 'synthetic_uncertainty'
    if args.generate_only and not synthetic:
        parser.error('--generate-only applies to synthetic_uncertainty')
    jobs = []
    for name, overrides in conditions.items():
        for seed in seeds:
            kwargs = dict(cfg['dataset']['kwargs'], **overrides)
            if synthetic:
                kwargs['seed'] = seed
            model_seed = cfg['model_seed_base'] + seed*cfg['pipeline'].get('n_models',5)
            dest = output/('generated' if args.generate_only else 'runs')/name/f'seed_{seed}'
            jobs.append((name, seed, model_seed, kwargs, dest))
            print(f'{name}: data_seed={seed}, model_seed_start={model_seed}, output={dest}')
    if args.dry_run:
        return
    for *_, dest in jobs:
        if dest.exists():
            raise FileExistsError(f'{dest} exists; choose a new output in YAML')
    for name, seed, model_seed, kwargs, dest in jobs:
        if args.generate_only:
            import numpy as np
            from modules.data.synthetic_uncertainty import generate
            splits, meta = generate(**kwargs)
            dest.mkdir(parents=True)
            for split, values in splits.items():
                np.savez_compressed(dest/f'{split}.npz',**values)
            (dest/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        else:
            from modules.pipelines.experiment_pipeline import run_experiment
            config = pipeline_config(cfg, kwargs, model_seed, data_seed=seed)
            result = run_experiment(config,out=dest,audit_config=cfg['audit'])
            row = dict(condition=name, data_seed=seed, model_seed_start=model_seed,
                       **result['predictive_metrics'])
            row.update(result['additional_metrics'].get('classical_summary', {}))
            audit = result['uncertainty_audit']
            if audit is not None:
                row.update(audit['maxima'], F_U_int=audit['F_U_int'])
                if audit['interaction'] is not None:
                    row.update(zip(['I_alea','I_epis','I_tot'],audit['interaction']))
            with (dest/'summary.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
        (dest/'experiment.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        (dest/'resolved_run.yaml').write_text(yaml.safe_dump(dict(condition=name,
            data_seed=seed, model_seed=model_seed, dataset_kwargs=kwargs),sort_keys=False))
        (dest/'COMPLETE').write_text('Completed successfully\n')

if __name__ == '__main__':
    main()
