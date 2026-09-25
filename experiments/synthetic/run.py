"""Run from project root: python -m experiments.synthetic.run --config ..."""
import argparse
import csv
import json
import re
from pathlib import Path
import yaml
from .data import generate

HERE = Path(__file__).resolve().parent


def read_config(path):
    with Path(path).open() as f:
        cfg = yaml.safe_load(f)
    required = {'output', 'data', 'data_seeds', 'model_seed_base', 'conditions', 'pipeline'}
    if not isinstance(cfg, dict) or set(cfg) != required:
        raise ValueError(f'Config must contain exactly {sorted(required)}')
    if not isinstance(cfg['data_seeds'], list) or not cfg['data_seeds']:
        raise ValueError('data_seeds must be a nonempty list')
    if len(set(cfg['data_seeds'])) != len(cfg['data_seeds']):
        raise ValueError('Duplicate data seeds')
    if any(type(s) is not int or s < 0 for s in cfg['data_seeds']):
        raise ValueError('Data seeds must be nonnegative integers')
    if not cfg['conditions']:
        raise ValueError('At least one condition is required')
    allowed = {'loader', 'model', 'optimizer', 'train', 'n_models',
               'append_protected_to_predictor', 'device'}
    if set(cfg['pipeline']) - allowed:
        raise ValueError('Unknown pipeline configuration keys')
    n_models = cfg['pipeline'].get('n_models', 10)
    if type(n_models) is not int or n_models < 1:
        raise ValueError('n_models must be a positive integer')
    if type(cfg['model_seed_base']) is not int or cfg['model_seed_base'] < 0:
        raise ValueError('model_seed_base must be a nonnegative integer')
    if not isinstance(cfg['output'], str) or not cfg['output']:
        raise ValueError('output must be a nonempty relative path')
    output = (HERE / cfg['output']).resolve()
    if not output.is_relative_to(HERE / 'results'):
        raise ValueError('output must be inside this experiment folder: results/<name>')
    for name, condition in cfg['conditions'].items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', str(name)):
            raise ValueError(f'Invalid condition name: {name}')
        if set(condition) - {'scenario', 'rho', 'eta', 'strength', 'stripe_half_width',
                             'stripe_slope', 'stripe_group', 'stripe_validation'}:
            raise ValueError(f'Unknown condition parameters: {name}')
        # Validate the actual DGP arguments cheaply, before training anything.
        generate(seed=cfg['data_seeds'][0], **cfg['data'], **condition)
    return cfg, output


def pipeline_config(cfg, kwargs, seed):
    """Converts YAML dictionaries into our existing configuration objects."""
    from modules.pipelines.train_predictive_pipeline import (
        PipelineConfig, SplitConfig, LoaderConfig, ModelConfig, OptimizerConfig)
    from modules.predictive.trainer import TrainConfig
    options = dict(cfg['pipeline'])
    for key, cls in [('loader', LoaderConfig), ('model', ModelConfig),
                     ('optimizer', OptimizerConfig), ('train', TrainConfig)]:
        if key in options:
            options[key] = cls(**options[key])
    options.setdefault('n_models', 10)
    options.setdefault('append_protected_to_predictor', False)
    return PipelineConfig(dataset_name='synthetic_uncertainty',
                          dataset_kwargs=kwargs,
                          split=SplitConfig(seed=seed), **options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE/'configs'/'smoke.yaml')
    parser.add_argument('--condition', help='Run only this named condition')
    parser.add_argument('--seed', type=int, help='Run only this configured data seed')
    parser.add_argument('--dry-run', action='store_true', help='Validate and list runs, without torch')
    parser.add_argument('--generate-only', action='store_true', help='Save raw DGP splits without training')
    args = parser.parse_args()
    cfg, output = read_config(args.config)
    conditions = cfg['conditions']
    if args.condition:
        if args.condition not in conditions:
            parser.error('Unknown condition')
        conditions = {args.condition: conditions[args.condition]}
    seeds = cfg['data_seeds']
    if args.seed is not None:
        if args.seed not in seeds:
            parser.error('--seed must occur in data_seeds')
        seeds = [args.seed]
    jobs = []
    for name, condition in conditions.items():
        for seed in seeds:
            kwargs = dict(cfg['data'], **condition, seed=seed)
            # Pair model seed schedules across conditions; disjoint across data seeds.
            model_seed = cfg['model_seed_base'] + seed*cfg['pipeline'].get('n_models', 10)
            dest = output/('generated' if args.generate_only else 'runs')/name/f'seed_{seed}'
            jobs.append((name, seed, model_seed, kwargs, dest))
    for name, seed, model_seed, _, dest in jobs:
        print(f'{name}: data_seed={seed}, model_seed_start={model_seed}, output={dest}')
    if args.dry_run:
        return
    # Refuse all conflicting destinations before beginning a potentially long run.
    for *_, dest in jobs:
        if dest.exists():
            raise FileExistsError(f'{dest} already exists. Choose a new output in YAML.')
    for name, seed, model_seed, kwargs, dest in jobs:
        if args.generate_only:
            import numpy as np
            splits, meta = generate(**kwargs)
            dest.mkdir(parents=True)
            for split, values in splits.items():
                np.savez_compressed(dest/f'{split}.npz', **values)
            (dest/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
        else:
            from .pipeline import run_experiment
            config = pipeline_config(cfg, kwargs, model_seed)
            result = run_experiment(config, out=dest)
            row = dict(condition=name, data_seed=seed, model_seed_start=model_seed,
                       **result['predictive_metrics'],
                       **result['uncertainty_audit']['maxima'],
                       F_U_int=result['uncertainty_audit']['F_U_int'])
            with (dest/'summary.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
        (dest/'experiment.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        (dest/'COMPLETE').write_text('Completed successfully\n')


if __name__ == '__main__':
    main()