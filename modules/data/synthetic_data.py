"""Shared adapter: one Kanubala generator, optional uncertainty interventions.

All split sizes are TOTAL rows before intervention, never per-group counts.
The module name is retained so existing pipeline imports need not change.
"""
import json
import warnings
from pathlib import Path
import numpy as np
from modules.data.synthetic_generators.loan_data import GROUPS, generate_loan_splits
from modules.data.synthetic_generators.uncertainty_conditions import (
    LOG2, h, inverse_h_bits, parameters, apply_uncertainty,
)


def generate(seed=0, scenario='loan_no_bias', train=6000, validation=1500,
             audit=2500, reference=10000, loan_overrides=None, uncertainty=None,
             **legacy):
    """Generate a loan scenario, then optionally transform its row-aligned splits.

    Canonical example: scenario='loan_no_bias',
    uncertainty={'kind':'scarcity', 'rho':.25, 'baseline_noise':.05}.
    No intervention means exact preservation of original loan arrays/metadata.
    """
    if not isinstance(scenario,str):
        raise ValueError('scenario must be a string')
    if not scenario.startswith('loan_'):
        if uncertainty is not None:
            raise ValueError('Use a loan_* scenario when uncertainty is explicit')
        if scenario not in ('baseline','scarcity','noise','cancellation','additive','interaction','stripe'):
            raise ValueError(f'Unknown scenario: {scenario}')
        warnings.warn('Legacy uncertainty scenario now uses loan_no_bias and TOTAL split sizes; '
                      'use scenario=loan_no_bias and uncertainty={kind: ...}.', FutureWarning, stacklevel=2)
        aliases={'stripe_half_width':'half_width','stripe_slope':'slope',
                 'stripe_group':'target_group','stripe_validation':'validation'}
        uncertainty={'kind':scenario, **{aliases.get(k,k):v for k,v in legacy.items()}}
        scenario='loan_no_bias'
    elif legacy:
        raise ValueError('Place uncertainty arguments inside the uncertainty mapping')
    splits, meta = generate_loan_splits(seed=seed,scenario=scenario,train=train,
        validation=validation,audit=audit,reference=reference,loan_overrides=loan_overrides)
    if uncertainty is None:
        return splits, meta
    return apply_uncertainty(splits,meta,uncertainty,seed=seed)


def get_spec():
    """Constructs our 'DatasetSpec', identifying protected columns, label and columns excluded from predictors.
    Exclude all identifiers, split labels and oracle fields from predictors."""
    from modules.utils.dataset_utils import DatasetSpec
    return DatasetSpec(
        name='synthetic_data', protected_cols=('S1', 'S2'), label_col='y',
        drop_feature_cols=('split', 'row_id', 'y_clean', 'p_true', 'U_true'),
    )

def load_dataset(*, folder=None, **generation_kwargs):
    """Return your LoadedDataset with fixed splits in metadata['split_indices'].

    Loaded through the shared dataset module registry.
    Or pass folder pointing to the original four NPZ files and metadata.json.
    Do NOT call split_df on this result: it would redistribute scarcity and put
    reference observations into training. The shared prepare_data pipeline honors these fixed indices.
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
    # Group proportions are inherited from the loan population and interventions.
    pg_table, pg = compute_pg_dirichlet(df.iloc[indices['train']])
    meta = dict(meta, spec=get_spec(), split_indices=indices, pg_source='train',
                population_pg=meta.get('population_pg', {g: .25 for g in GROUPS}),
                row_id_scope='within split; identify rows by (split, row_id)')
    return LoadedDataset(df=df, original_columns=list(df.columns),
                         group_id={g: i for i, g in enumerate(GROUPS)},
                         pg_table=pg_table, pg=pg, metadata=meta)

def save_prepared_dataset(prepared, folder):
    """Save raw splits plus preprocessing using your saving_utils conventions.

    Reject an existing destination. Load raw data via load_dataset(folder=...).
    For exact saved preprocessing use joblib.load on the trusted local file
    preprocessing/predictor/predictor_preprocessing_schema.joblib.
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
            X=df[meta['feature_names']].to_numpy(dtype=np.float64 if meta.get('generator')=='kanubala_2025_repository' else np.float32),
            y=df.y.to_numpy(), S=df[['S1', 'S2']].to_numpy(dtype=np.int8),
            group=df.group_id.to_numpy(),
            **{k: df[k].to_numpy() for k in ('row_id', 'y_clean', 'p_true', 'U_true')})
    (folder / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    save_all_preprocessing_artifacts(prepared['predictor_schema'], folder / 'preprocessing')
    idx = loaded.metadata['split_indices']
    save_split_indices(idx['train'], idx['validation'], idx['audit'], folder / 'splits')
    np.save(folder / 'splits' / 'reference_idx.npy', idx['reference'])
    return folder
