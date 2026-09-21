"""Controlled intersectional classification data. NumPy generation; project preparation uses your pandas/sklearn/PyTorch utilities.

Examples:
  python -m modules.data.synthetic_uncertainty --self-test
  python -m modules.data.synthetic_uncertainty --scenario scarcity --rho .125 --out data/scarce
  python -m modules.data.synthetic_uncertainty --scenario additive --strength .15 --out data/add

Train ONLY on X and y. S, group, y_clean, p_true, U_true, row_id are audit/oracle
metadata, not predictors. Group order: 00,01,10,11. Entropies use natural logs.
Same seed and split sizes couple covariates, label uniforms and nested training
subsets across conditions. Validation/audit/reference streams are independent.
Reference is for integration after fitting, NEVER model selection.
run_synthetic_experiment delegates training to your existing predictive pipeline.
The command-line interface remains data generation only.

Training from your project root (existing user modules are left unchanged):
    from modules.data.synthetic_uncertainty import run_synthetic_experiment
    from modules.pipelines.train_predictive_pipeline import (
        PipelineConfig, SplitConfig, ModelConfig)
    from modules.predictive.trainer import TrainConfig
    config = PipelineConfig(
        dataset_name="synthetic_uncertainty",
        dataset_kwargs={"scenario": "scarcity", "rho": .125, "seed": 0},
        split=SplitConfig(seed=1000),
        model=ModelConfig(hidden_dims=(128, 64), dropout=0.0),
        train=TrainConfig(epochs=300, patience=30, binary=True),
        n_models=10, append_protected_to_predictor=False)
    result = run_synthetic_experiment(config, out="experiments/scarcity_seed0")

Use this runner instead of run_predictive_pipeline for synthetic data: the latter
always randomly resplits its input. All model training is delegated unchanged to
its train_models function. This implements deep ensembles, not MC dropout/Laplace.
"""
import argparse
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
    """Fixed stratified counts sample an equal-group audit target population."""
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


def uncertainty(member_p):
    """Input shape (M,N): class-1 probabilities from coherent model draws."""
    p = np.asarray(member_p, dtype=float)
    if p.ndim != 2 or min(p.shape) < 1:
        raise ValueError('Expected nonempty (M,N) array.')
    alea = h(p).mean(axis=0)
    total = h(p.mean(axis=0))
    return np.column_stack((alea, total-alea, total))


def audit_predictions(member_p, groups, alpha=.05):
    """Group means, all pair disparities, maxima, and simultaneous Hoeffding radii.

    F_U uses max(component gaps)/log(2). Hidden maxima are computed pairwise,
    never by combining component maxima attained by different group pairs.
    """
    u = uncertainty(member_p)
    groups = np.asarray(groups)
    if groups.shape != (len(u),) or not np.isin(groups, range(4)).all():
        raise ValueError('Expected one valid group id per observation.')
    counts = np.array([(groups == g).sum() for g in range(4)])
    if min(counts) == 0 or not 0 < alpha < 1:
        raise ValueError('All groups must be present and alpha in (0,1).')
    means = np.array([u[groups == g].mean(axis=0) for g in range(4)])
    pairs = {}
    for a in range(4):
        for b in range(a+1,4):
            da,de,dt = means[a]-means[b]
            pairs[f'{GROUPS[a]}-{GROUPS[b]}'] = dict(
                F_alea=abs(da), F_epis=abs(de), F_tot=abs(dt),
                F_U=max(abs(da),abs(de))/LOG2,
                hidden_normalized=max(0., abs(da)+abs(de)-abs(dt))/(2*LOG2))
    interaction = np.array([1,-1,-1,1]) @ means
    return dict(component_order=['alea','epis','tot'], means=means.tolist(),
                counts=counts.tolist(), pairs=pairs,
                maxima={k:max(v[k] for v in pairs.values()) for k in next(iter(pairs.values()))},
                interaction=interaction.tolist(),
                F_U_int=float(max(abs(interaction[:2]))/(2*LOG2)),
                hoeffding_component_radii=(LOG2*np.sqrt(np.log(16/alpha)/(2*counts))).tolist())


def self_test():
    # Exact entropy-scale null and alternative, not an additive noise surrogate.
    for scenario, expected in [('additive',0.), ('interaction',.15)]:
        n,_ = parameters(scenario)
        np.testing.assert_allclose(np.dot([1,-1,-1,1], h(n)/LOG2), expected, atol=1e-12)
    a,_ = generate(seed=7, train=100, validation=20, audit=20, reference=20)
    b,_ = generate(seed=7, scenario='scarcity', rho=.25, train=100, validation=20, audit=20, reference=20)
    for split in ['validation','audit','reference']:
        for k in a[split]:
            np.testing.assert_array_equal(a[split][k],b[split][k])
    assert len(b['train']['y']) == 325
    assert set(b['train']['row_id']) <= set(a['train']['row_id'])
    # Population Bayes probabilities repeated across members have zero disagreement.
    oracle = audit_predictions(np.tile(a['audit']['p_true'], (3,1)), a['audit']['group'])
    np.testing.assert_allclose(np.array(oracle['means'])[:,1], 0., atol=1e-14)
    # Exact theorem fixture, deliberately NOT learned predictions.
    p = np.array([[.5,.5,.5,0.],[.5,.5,.5,1.]])
    fixture = audit_predictions(p, np.arange(4))
    assert np.isclose(fixture['pairs']['00-11']['hidden_normalized'],1.)
    rng = np.random.default_rng(8)
    u = uncertainty(rng.random((10,100)))
    np.testing.assert_allclose(u[:,0]+u[:,1],u[:,2],atol=1e-14)
    assert u[:,1].min() >= -1e-14
    print('PASS: entropy targets, independent fixed splits, nested scarcity, oracle, cancellation, decomposition')


# Project integration. Imports are lazy so NumPy-only generation still works.
def get_spec():
    """Exclude all identifiers, split labels and oracle fields from predictors."""
    from modules.utils.dataset_utils import DatasetSpec
    return DatasetSpec(
        name='synthetic_uncertainty', protected_cols=('S1', 'S2'), label_col='y',
        drop_feature_cols=('split', 'row_id', 'y_clean', 'p_true', 'U_true'),
    )


def load_dataset(*, folder=None, **generation_kwargs):
    """Return your LoadedDataset with fixed splits in metadata['split_indices'].

    Use load_dataset_by_name('synthetic_uncertainty', scenario=..., seed=...).
    Or pass folder pointing to the original four NPZ files and metadata.json.
    Do NOT call split_df on this result: it would redistribute scarcity and put
    reference observations into training. Use prepare_dataset below instead.
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


def prepare_dataset(loaded=None, *, batch_size=128, device=None,
                    append_protected=False, **load_kwargs):
    """Fit your predictor schema on train; return fixed tensors and loaders.

    result['tensors'][split] is (X, y, group_ids).
    result['loaders'][split] yields (X, y); only training is shuffled.
    result['audit_loaders'][split] yields dicts X/y/group_id in original order.
    Reference data is only for integration of a frozen fitted predictor.
    """
    from modules.utils.dataset_utils import (
        fit_predictor_schema, transform_predictor_schema,
        to_tensor_dataset_predictive, TabularDataset, make_loader,
    )
    if loaded is not None and load_kwargs:
        raise ValueError('Pass loaded OR loading arguments.')
    if loaded is None:
        loaded = load_dataset(**load_kwargs)
    frames = {name: loaded.df.iloc[idx].copy()
              for name, idx in loaded.metadata['split_indices'].items()}
    schema = fit_predictor_schema(frames['train'], get_spec(),
                                  append_protected=append_protected)
    tensors, loaders, audit_loaders = {}, {}, {}
    for name, frame in frames.items():
        X, y, groups = transform_predictor_schema(frame, schema, device=device)
        tensors[name] = (X, y, groups)
        loaders[name] = make_loader(to_tensor_dataset_predictive(X, y),
                                    batch_size, shuffle=name == 'train')
        audit_loaders[name] = make_loader(
            TabularDataset(X=X, y=y, group_id=groups), batch_size, shuffle=False)
    return dict(loaded=loaded, spec=get_spec(), frames=frames,
                predictor_schema=schema, tensors=tensors, loaders=loaders,
                audit_loaders=audit_loaders)


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


def run_synthetic_experiment(config=None, *, out=None,
                             regularizer=None, regularizer_weight=1.0,
                             fair_loader=None):
    """Use your existing training/evaluation pipeline; add an uncertainty audit.

    Returns dict(pipeline_result=PipelineResult, uncertainty_audit=dict,
                 predictive_metrics=dict, group_predictive_metrics=DataFrame,
                 output_folder=Path|None).
    Your PipelineResult.test_metrics/group_metrics describe model 0, as in your
    pipeline. The separate predictive_metrics below describe the MEAN predictor.
    Deep ensembles are implemented; MC dropout and Laplace are not added here.
    Passing dropout>0 trains with dropout; evaluate_ensemble turns it off at test.

    Supply fair_loader explicitly if using a regularizer. No monkeypatching or
    replacement of user pipeline functions is performed.
    """
    import pandas as pd
    from modules.pipelines.train_predictive_pipeline import (
        PipelineConfig, ModelConfig, PipelineResult, train_models, evaluate_models,
    )
    from modules.predictive.ensemble import evaluate_ensemble
    if config is None:
        config = PipelineConfig(
            dataset_name='synthetic_uncertainty', dataset_kwargs={'seed': 0},
            n_models=10, append_protected_to_predictor=False,
            model=ModelConfig(hidden_dims=(128, 64), dropout=0.0),
        )
    if out is not None and Path(out).exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    if regularizer is not None and fair_loader is None:
        raise ValueError('Pass an explicit fair_loader when using a regularizer.')
    data = prepare_pipeline_data(config)
    models, histories = train_models(
        data, config, regularizer=regularizer,
        regularizer_weight=regularizer_weight, fair_loader=fair_loader,
    )
    test_metrics, group_metrics, ensemble = evaluate_models(models, data, config)
    # Your pipeline only evaluates ensembles when M>1; also support a single
    # predictor as a useful zero-disagreement control.
    if ensemble is None:
        ensemble = evaluate_ensemble(models, data.test_loader,
                                     binary=data.binary, device=config.device)
    result = PipelineResult(
        config=config, data=data, models=models, histories=histories,
        test_metrics=test_metrics, group_metrics=group_metrics,
        ensemble_metrics=ensemble,
    )
    member_p = np.asarray(ensemble['ensemble_probs'])[:, :, 1]
    groups = data.test_df['group_id'].to_numpy(dtype=np.int64)
    labels = data.test_df['y'].to_numpy(dtype=np.int64)
    audit = audit_predictions(member_p, groups)
    # Verify that the native evaluator and our audit use the same definitions.
    u = uncertainty(member_p)
    for col, key in enumerate(('aleatoric_uncertainty', 'epistemic_uncertainty',
                               'predictive_entropy')):
        np.testing.assert_allclose(u[:, col], ensemble[key], atol=2e-6, rtol=2e-5)
    mean_p = member_p.mean(axis=0).astype(float)
    predicted = ((mean_p >= config.train.threshold).astype(int) if data.binary
                 else np.asarray(ensemble['predictions']))
    clipped = np.clip(mean_p, 1e-12, 1-1e-12)
    def quality(mask):
        y, p, q = labels[mask], mean_p[mask], clipped[mask]
        return dict(support=int(mask.sum()),
                    accuracy=float(np.mean(predicted[mask] == y)),
                    nll=float(np.mean(-y*np.log(q)-(1-y)*np.log1p(-q))),
                    brier=float(np.mean((p-y)**2)))
    metrics = quality(np.ones(len(labels), dtype=bool))
    by_group = pd.DataFrame([
        dict(group=GROUPS[g], group_id=g, **quality(groups == g)) for g in range(4)])
    audit['oracle_entropy_nats'] = (np.array(data.loaded.metadata['oracle_entropy_bits'])*LOG2).tolist()
    audit['alea_minus_oracle'] = (np.array(audit['means'])[:, 0]
                                -np.array(audit['oracle_entropy_nats'])).tolist()
    audit['data_seed'] = data.loaded.metadata['seed']
    audit['model_seeds'] = [config.split.seed+i for i in range(config.n_models)]
    audit['method'] = 'deep_ensemble' if len(models) > 1 else 'single_model'
    audit['brier_convention'] = 'mean squared error of class-1 probability'
    output = None
    if out is not None:
        from modules.utils.saving_utils import save_pipeline_result
        output = save_pipeline_result(result, exact_path=out)
        extra = output / 'synthetic_audit'
        extra.mkdir()
        (extra / 'uncertainty_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
        (extra / 'predictive_metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
        by_group.to_csv(extra / 'group_predictive_metrics.csv', index=False)
        np.savez_compressed(extra / 'audit_rows.npz',
                            y=labels, group_id=groups,
                            row_id=data.test_df.row_id.to_numpy(),
                            p_true=data.test_df.p_true.to_numpy())
        frames = {name: data.loaded.df.iloc[idx].copy()
                  for name, idx in data.loaded.metadata['split_indices'].items()}
        save_prepared_dataset(dict(loaded=data.loaded, frames=frames,
                                   predictor_schema=data.predictor_schema),
                              output / 'synthetic_data')
    return dict(pipeline_result=result, uncertainty_audit=audit,
                predictive_metrics=metrics, group_predictive_metrics=by_group,
                output_folder=output)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scenario', choices=['baseline','scarcity','noise','additive','interaction','cancellation'], default='baseline')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--rho', type=float, default=1.)
    p.add_argument('--eta', type=float, default=.2)
    p.add_argument('--strength', type=float, default=.15)
    p.add_argument('--train', type=int, default=2000, help='Base observations PER GROUP')
    p.add_argument('--validation', type=int, default=500)
    p.add_argument('--audit', type=int, default=5000)
    p.add_argument('--reference', type=int, default=25000)
    p.add_argument('--out', type=Path, default=Path('synthetic_data'))
    p.add_argument('--self-test', action='store_true')
    args = vars(p.parse_args())
    if args.pop('self_test'):
        self_test()
        return
    out = args.pop('out')
    splits, meta = generate(**args)
    out.mkdir(parents=True, exist_ok=False)
    for name, data in splits.items():
        np.savez_compressed(out/f'{name}.npz', **data)
    (out/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()