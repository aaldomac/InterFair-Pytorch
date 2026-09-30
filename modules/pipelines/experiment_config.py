"""Strict YAML experiment configuration, independent of PyTorch for dry runs."""
from pathlib import Path
import re
import yaml
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
    allowed = {'split','loader','model','optimizer','train','n_models','append_protected_to_predictor','device'}
    pipeline = cfg['pipeline']
    if not isinstance(pipeline,dict) or set(pipeline)-allowed:
        raise ValueError('Unknown pipeline configuration keys')
    n = pipeline.get('n_models',5)
    if type(n) is not int or n < 1:
        raise ValueError('n_models must be a positive integer')
    fields = {'split':{'test_size','val_size','stratify'},
              'loader':{'batch_size','eval_batch_size','shuffle_train'},
              'model':{'hidden_dims','dropout'}, 'optimizer':{'lr','weight_decay'},
              'train':{'epochs','patience','binary','threshold','grad_clip_norm','restore_best','verbose'}}
    for key, permitted in fields.items():
        if key in pipeline and (not isinstance(pipeline[key],dict) or set(pipeline[key])-permitted):
            raise ValueError(f'Unknown {key} settings')
    for section, keys in {'loader':('batch_size','eval_batch_size'), 'train':('epochs','patience')}.items():
        for key in keys:
            value = pipeline.get(section,{}).get(key)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f'{section}.{key} must be a positive integer')
    for section,key in [('model','dropout'),('train','threshold')]:
        value = pipeline.get(section,{}).get(key)
        if value is not None and not 0 <= value < 1:
            raise ValueError(f'{section}.{key} must be in [0,1)')
    split = pipeline.get('split',{})
    if any(not 0 < split[k] < 1 for k in ('test_size','val_size') if k in split):
        raise ValueError('split fractions must be in (0,1)')
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
        if dataset['name'] == 'synthetic_uncertainty':
            from Projects.InterFairPytorch.modules.data.synthetic_data import generate
            generate(**dict(dataset['kwargs'], **overrides), seed=seeds[0])
    audit = cfg.setdefault('audit',{})
    if not isinstance(audit,dict) or set(audit)-{'enabled','alpha','interaction_weights','fairness','distribution'}:
        raise ValueError('Unknown audit settings')
    if not 0 < audit.get('alpha',.05) < 1:
        raise ValueError('audit.alpha must be in (0,1)')
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
