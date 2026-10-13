"""Strict YAML experiment configuration, independent of PyTorch for dry runs."""
from pathlib import Path
import re
import yaml
import math
ROOT = Path(__file__).resolve().parents[2]


def read_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    required = {'output','dataset','data_seeds','model_seed_base','conditions','pipeline'}
    if not isinstance(cfg, dict) or not required <= cfg.keys() or set(cfg)-required-{'audit'}:
        raise ValueError(f'Expected {sorted(required)} and optional audit')
    seeds = cfg['data_seeds']
    if not isinstance(seeds,list) or not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('data_seeds must contain distinct nonnegative integers')
    if type(cfg['model_seed_base']) is not int or cfg['model_seed_base'] < 0:
        raise ValueError('model_seed_base must be a nonnegative integer')
    dataset = cfg['dataset']
    if not isinstance(dataset,dict) or set(dataset)-{'name','kwargs'} or not isinstance(dataset.get('name'),str) or not dataset['name'].isidentifier():
        raise ValueError('dataset requires a valid module name and optional kwargs')
    dataset.setdefault('kwargs',{})
    if not isinstance(dataset['kwargs'],dict):
        raise ValueError('dataset.kwargs must be a mapping')
    allowed = {'split','loader','model','optimizer','train','n_models','append_protected_to_predictor','store_member_probs','device'}
    pipeline = cfg['pipeline']
    if not isinstance(pipeline,dict) or set(pipeline)-allowed:
        raise ValueError('Unknown pipeline configuration keys')
    n = pipeline.get('n_models',5)
    if type(n) is not int or n < 1:
        raise ValueError('n_models must be a positive integer')
    fields = {'split':{'test_size','val_size','stratify','rare_group_policy'},
              'loader':{'batch_size','eval_batch_size','shuffle_train'},
              'model':{'hidden_dims','dropout'}, 'optimizer':{'lr','weight_decay'},
              'train':{'epochs','patience','binary','threshold','grad_clip_norm','restore_best','verbose'}}
    for key, permitted in fields.items():
        if key in pipeline and (not isinstance(pipeline[key],dict) or set(pipeline[key])-permitted):
            raise ValueError(f'Unknown {key} settings')
    validate_pipeline_options(pipeline)
    if not isinstance(cfg['output'],str) or Path(cfg['output']).is_absolute():
        raise ValueError('output must be relative to the repository')
    output = (ROOT/cfg['output']).resolve()
    if not output.is_relative_to(ROOT/'experiments') or output == ROOT/'experiments':
        raise ValueError('output must be experiments/<name>')
    if not isinstance(cfg['conditions'],dict) or not cfg['conditions']:
        raise ValueError('conditions must be a nonempty mapping of dataset overrides')
    for name, overrides in cfg['conditions'].items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', str(name)) or not isinstance(overrides,dict):
            raise ValueError(f'Invalid condition: {name}')
        if 'seed' in overrides or 'seed' in dataset['kwargs']:
            raise ValueError('Use data_seeds rather than seed in dataset kwargs')
        if dataset['name'] == 'synthetic_data':
            from modules.data.synthetic_data import validate_generation_options
            validate_generation_options(**dict(dataset['kwargs'], **overrides), seed=seeds[0])
    audit = cfg.setdefault('audit',{})
    if not isinstance(audit, dict):
        raise ValueError('audit must be a mapping')
    from modules.pipelines.audit_pipeline import AuditConfig
    AuditConfig.from_mapping(audit)
    return cfg, output


def pipeline_config(cfg, kwargs, seed, data_seed=0):
    from modules.pipelines.train_predictive_pipeline import (
        PipelineConfig, SplitConfig, LoaderConfig, ModelConfig, OptimizerConfig)
    from modules.predictive.trainer import TrainConfig
    options = dict(cfg['pipeline'])
    split = SplitConfig(seed=data_seed, **options.pop('split',{}))
    for key, cls in [('loader',LoaderConfig),('model',ModelConfig),
                     ('optimizer',OptimizerConfig),('train',TrainConfig)]:
        if key in options:
            options[key] = cls(**options[key])
    options.setdefault('n_models',5)
    options.setdefault('append_protected_to_predictor',False)
    return PipelineConfig(dataset_name=cfg['dataset']['name'], dataset_kwargs=kwargs,
                          split=split, model_seed=seed, **options)


def validate_training_options(options):
    """Shared validation for TrainConfig and YAML; patience=0 stops on first non-improvement."""
    for key, minimum in (('epochs', 1), ('patience', 0)):
        value = options.get(key, 50 if key == 'epochs' else 5)
        if type(value) is not int or value < minimum:
            raise ValueError(f'train.{key} must be an integer >= {minimum}')
    threshold = options.get('threshold', .5)
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('train.threshold must be in [0, 1]')
    clip = options.get('grad_clip_norm')
    if clip is not None and (not isinstance(clip, (int, float)) or not math.isfinite(clip) or clip <= 0):
        raise ValueError('train.grad_clip_norm must be finite and positive')
    for key in ('binary', 'restore_best', 'verbose'):
        if key in options and type(options[key]) is not bool:
            raise ValueError(f'train.{key} must be boolean')


def validate_pipeline_options(options):
    validate_training_options(options.get('train', {}))
    n = options.get('n_models', 5)
    if type(n) is not int or n < 1:
        raise ValueError('n_models must be a positive integer')
    for key in ('append_protected_to_predictor', 'store_member_probs'):
        if key in options and type(options[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    loader = options.get('loader', {})
    for key in ('batch_size', 'eval_batch_size'):
        if key in loader and (type(loader[key]) is not int or loader[key] < 1):
            raise ValueError(f'loader.{key} must be a positive integer')
    if 'shuffle_train' in loader and type(loader['shuffle_train']) is not bool:
        raise ValueError('loader.shuffle_train must be boolean')
    model = options.get('model', {})
    dropout = model.get('dropout', .2)
    if not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError('model.dropout must be in [0, 1)')
    if 'hidden_dims' in model and (not isinstance(model['hidden_dims'], (list, tuple)) or
            any(type(n) is not int or n < 1 for n in model['hidden_dims'])):
        raise ValueError('model.hidden_dims must contain positive integers')
    optimizer = options.get('optimizer', {})
    for key, default in (('lr', 1e-3), ('weight_decay', 1e-4)):
        value = optimizer.get(key, default)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (key == 'lr' and value == 0):
            raise ValueError(f'optimizer.{key} has an invalid value')
    split = options.get('split', {})
    for key in ('test_size', 'val_size'):
        if key in split and (not isinstance(split[key], (int, float)) or not 0 < split[key] < 1):
            raise ValueError('split fractions must be in (0, 1)')
    if split.get('rare_group_policy') not in (None, 'train_only'):
        raise ValueError('rare_group_policy must be null or train_only')