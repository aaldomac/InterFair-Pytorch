"""Synthetic DGP and fixed-split adapter to your PreparedData contract.

X contains strong protected-attribute proxies. Oracle columns never enter X.
Validation/audit/reference remain fixed across paired scarcity conditions.
"""
import json
from pathlib import Path
import numpy as np
GROUPS = ['00', '01', '10', '11']
LOG2 = float(np.log(2.0))

def h(p):
    """Binary entropy, including exact endpoints."""
    p = np.asarray(p, dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError('Probabilities must be finite and in [0,1].')
    q = np.clip(p, np.finfo(float).tiny, 1)
    r = np.clip(1-p, np.finfo(float).tiny, 1)
    return -p*np.log(q)-(1-p)*np.log(r)

def inverse_h_bits(target):
    """ Find a label-flip probability whose binary entropy equals 'target' bits."""
    target = np.asarray(target, dtype=float)
    if np.any(~np.isfinite(target)) or np.any((target < 0) | (target > 1)):
        raise ValueError('Entropy targets must lie in [0,1] bits.')
    lo, hi = np.zeros_like(target), np.full_like(target, .5)
    for _ in range(60):
        mid = (lo+hi)/2
        low = h(mid)/LOG2 < target
        lo = np.where(low, mid, lo)
        hi = np.where(low, hi, mid)
    return (lo+hi)/2

def parameters(scenario='baseline', rho=1., eta=.2, strength=.15):
    """Converts a condition into four group noise rates and four training-retention fractions."""
    if not np.isfinite(rho) or not 0 < rho <= 1:
        raise ValueError('rho must be in (0,1].')
    if not np.isfinite(eta) or not 0 <= eta <= .5:
        raise ValueError('eta must be in [0,.5].')
    if not np.isfinite(strength) or strength < 0:
        raise ValueError('strength must be nonnegative.')
    noise = np.full(4, .05)
    retain = np.ones(4)
    if scenario == 'scarcity':
        retain[3] = rho
    elif scenario == 'noise':
        noise[3] = eta
    elif scenario in ('additive', 'interaction'):
        pattern = np.array([0,1,1,2] if scenario == 'additive' else [0,0,0,1])
        noise = inverse_h_bits(.30 + strength*pattern)
    elif scenario == 'cancellation':
        noise[0], retain[3] = eta, rho
    elif scenario != 'baseline':
        raise ValueError('Unknown scenario.')
    return noise, retain

def sample_split(n_per_group, seed, stream, noise, retain=None):
    """Generates one partition with features, labels, group identifiers and oracle values.
    Fixed stratified counts sample an equal-group audit target population."""
    if n_per_group < 1:
        raise ValueError('Split sizes must be positive.')
    rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
    n = 4*n_per_group
    group = np.repeat(np.arange(4), n_per_group)
    S = np.column_stack((group//2, group % 2))
    z = rng.uniform(-1, 1, (n, 2))
    nuisance = rng.normal(size=(n, 4))
    # Disjoint rectangles: proxy feature support reveals group, although S is
    # excluded from X. Translation symmetry gives equal baseline difficulty.
    centers = 3*(2*S-1)
    X = np.column_stack((centers+z, nuisance)).astype(np.float32)
    score = z[:, 0] + .5*z[:, 1] + .35*np.sin(np.pi*z[:, 1])
    clean = (score >= 0).astype(np.int64)
    uniforms = rng.random(n)
    y = np.bitwise_xor(clean, (uniforms < noise[group]).astype(np.int64))
    p = noise[group] + (1-2*noise[group])*clean
    result = dict(X=X, y=y, S=S.astype(np.int8), group=group,
                  y_clean=clean, p_true=p, U_true=h(p), row_id=np.arange(n))
    # Permutation prefixes guarantee nested retained training observations.
    index = []
    for g in range(4):
        perm = rng.permutation(np.flatnonzero(group == g))
        count = n_per_group if retain is None else max(1, int(np.floor(n_per_group*retain[g])))
        index.extend(perm[:count])
    index = np.asarray(index, dtype=int)
    return {k: v[index] for k, v in result.items()}

def generate(seed=0, scenario='baseline', rho=1., eta=.2, strength=.15,
             train=2000, validation=500, audit=5000, reference=25000):
    """Calls 'parameters()' and generates train, validation, audit and reference partitions. Returns partitions plus metadata."""
    noise, retain = parameters(scenario, rho, eta, strength)
    sizes = dict(train=train, validation=validation, audit=audit, reference=reference)
    splits = {name: sample_split(n, seed, stream, noise,
              retain if name == 'train' else None)
              for stream, (name, n) in enumerate(sizes.items())}
    meta = dict(seed=seed, scenario=scenario, rho=rho, eta=eta, strength=strength,
                group_order=GROUPS, requested_per_group=sizes,
                flip_probability=noise.tolist(), retention=retain.tolist(),
                oracle_entropy_bits=(h(noise)/LOG2).tolist(),
                oracle_interaction_bits=float(np.dot([1,-1,-1,1], h(noise)/LOG2)),
                feature_names=['proxy_1','proxy_2','nuisance_1','nuisance_2','nuisance_3','nuisance_4'],
                entropy_units='nats', numpy_version=np.__version__,
                counts={name: np.bincount(d['group'], minlength=4).tolist()
                        for name,d in splits.items()})
    return splits, meta

def get_spec():
    """Constructs our 'DatasetSpec', identifying protected columns, label and columns excluded from predictors.
    Exclude all identifiers, split labels and oracle fields from predictors."""
    from modules.utils.dataset_utils import DatasetSpec
    return DatasetSpec(
        name='synthetic_uncertainty', protected_cols=('S1', 'S2'), label_col='y',
        drop_feature_cols=('split', 'row_id', 'y_clean', 'p_true', 'U_true'),
    )

def load_dataset(*, folder=None, **generation_kwargs):
    """Return your LoadedDataset with fixed splits in metadata['split_indices'].

    Called locally by this experiment; no dataset registration is needed.
    Or pass folder pointing to the original four NPZ files and metadata.json.
    Do NOT call split_df on this result: it would redistribute scarcity and put
    reference observations into training. Use prepare_pipeline_data instead.
    """
    import pandas as pd
    from modules.utils.dataset_utils import LoadedDataset, compute_pg_dirichlet
    if folder is None:
        splits, meta = generate(**generation_kwargs)
    else:
        if generation_kwargs:
            raise ValueError('Use folder OR generation arguments, not both.')
        folder = Path(folder)
        meta = json.loads((folder / 'metadata.json').read_text())
        splits = {}
        for name in ('train', 'validation', 'audit', 'reference'):
            with np.load(folder / f'{name}.npz', allow_pickle=False) as data:
                splits[name] = {key: data[key] for key in data.files}
    frames, indices, offset = [], {}, 0
    for name, data in splits.items():
        df = pd.DataFrame(data['X'], columns=meta['feature_names'])
        df['S1'], df['S2'] = data['S'][:, 0], data['S'][:, 1]
        df['y'] = data['y']
        df['group_id'] = data['group']
        df['group'] = pd.Categorical(
            np.asarray(GROUPS)[data['group']], categories=GROUPS)
        df['split'] = name
        for key in ('row_id', 'y_clean', 'p_true', 'U_true'):
            df[key] = data[key]
        indices[name] = np.arange(offset, offset + len(df), dtype=np.int64)
        offset += len(df)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    # Training proportions differ from the fixed equal-group audit target.
    pg_table, pg = compute_pg_dirichlet(df.iloc[indices['train']])
    meta = dict(meta, spec=get_spec(), split_indices=indices, pg_source='train',
                population_pg={g: .25 for g in GROUPS},
                row_id_scope='within split; identify rows by (split, row_id)')
    return LoadedDataset(df=df, original_columns=list(df.columns),
                         group_id={g: i for i, g in enumerate(GROUPS)},
                         pg_table=pg_table, pg=pg, metadata=meta)

def save_prepared_dataset(prepared, folder):
    """Save raw splits plus preprocessing using your saving_utils conventions.

    Reject an existing destination. Load raw data via load_dataset(folder=...).
    For exact saved preprocessing use joblib.load on the trusted local file
    preprocessing/predictor/predictor_preprocessing_schema.joblib.
    loading_utils currently imports missing legacy dataset utility names; this
    module therefore does not import it or depend on its legacy JSON loader.
    """
    from modules.utils.saving_utils import (
        save_all_preprocessing_artifacts, save_split_indices,
    )
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    loaded = prepared['loaded']
    meta = {k: v for k, v in loaded.metadata.items() if k not in ('split_indices', 'spec')}
    for name, df in prepared['frames'].items():
        np.savez_compressed(folder / f'{name}.npz',
            X=df[meta['feature_names']].to_numpy(dtype=np.float32),
            y=df.y.to_numpy(), S=df[['S1', 'S2']].to_numpy(dtype=np.int8),
            group=df.group_id.to_numpy(),
            **{k: df[k].to_numpy() for k in ('row_id', 'y_clean', 'p_true', 'U_true')})
    (folder / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    save_all_preprocessing_artifacts(prepared['predictor_schema'], folder / 'preprocessing')
    idx = loaded.metadata['split_indices']
    save_split_indices(idx['train'], idx['validation'], idx['audit'], folder / 'splits')
    np.save(folder / 'splits' / 'reference_idx.npy', idx['reference'])
    return folder

def prepare_pipeline_data(config):
    """Build the existing pipeline's PreparedData using the DGP's fixed splits.

    config.dataset_kwargs controls DGP seed and sizes. config.split.seed controls
    model seeds only; test_size/val_size/stratify are intentionally not applied.
    Audit is the pipeline's test split. Reference never enters fitting/evaluation.
    """
    from torch.utils.data import TensorDataset
    from modules.pipelines.train_predictive_pipeline import PreparedData
    from modules.utils.dataset_utils import (
        fit_predictor_schema, transform_predictor_schema, make_loader,
    )
    if config.dataset_name != 'synthetic_uncertainty':
        raise ValueError("Set dataset_name='synthetic_uncertainty'.")
    if config.build_px_loaders:
        raise ValueError('This synthetic adapter does not build px loaders.')
    loaded = load_dataset(**dict(config.dataset_kwargs))
    spec = loaded.metadata['spec']
    frames = {name: loaded.df.iloc[idx].copy().reset_index(drop=True)
              for name, idx in loaded.metadata['split_indices'].items()}
    if set(frames['train']['y'].unique()) != {0, 1}:
        raise ValueError('Training needs both labels; increase train size or change seed.')
    schema = fit_predictor_schema(frames['train'], spec,
                                 append_protected=config.append_protected_to_predictor)
    tensors, loaders = {}, {}
    for name in ('train', 'validation', 'audit'):
        tensors[name] = transform_predictor_schema(frames[name], schema)
        loaders[name] = make_loader(
            TensorDataset(*tensors[name]),
            batch_size=(config.loader.batch_size if name == 'train'
                        else config.loader.eval_batch_size),
            shuffle=config.loader.shuffle_train if name == 'train' else False,
        )
    return PreparedData(
        loaded=loaded, spec=spec, train_df=frames['train'],
        val_df=frames['validation'], test_df=frames['audit'],
        predictor_schema=schema, train_loader=loaders['train'],
        val_loader=loaders['validation'], test_loader=loaders['audit'],
        num_features=int(tensors['train'][0].shape[1]), num_classes=2,
        binary=bool(config.train.binary),
    )