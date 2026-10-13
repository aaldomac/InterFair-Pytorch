"""Readers for current artifacts and explicitly supported legacy layouts."""
import json
from pathlib import Path
import numpy as np


def load_ensemble(run, device='cpu'):
    """Compatibility export; prediction-only readers do not PyTorch"""
    from modules.utils.checkpoint_utils import load_ensemble as load
    return load(run, device=device)


def load_config_from_experiment(folder):
    return json.loads((Path(folder)/'config.json').read_text())


def load_split_indices(folder):
    folder = Path(folder)
    for path in (folder/'splits/split_indices.npz', folder/'split_indices.npz'):
        if path.exists():
            with np.load(path, allow_pickle=False) as values:
                return tuple(values[f'{name}_idx'].copy() for name in ('train','val','test'))
    return tuple(np.load(folder/f'{name}_idx.npy', allow_pickle=False)
                 for name in ('train','val','test'))


def load_saved_outputs(folder, *, _seen=None):
    """Load predictions; re-audit bundles refer to their immutable source run."""
    folder = Path(folder).resolve()
    seen = set() if _seen is None else _seen
    if folder in seen:
        raise ValueError('Cyclic source_run references in audit artifacts')
    seen.add(folder)
    paths = sorted((folder/'results').glob('ensemble_*.npy'))
    if not paths:
        context_path = folder/'audit/audit_context.json'
        context = json.loads(context_path.read_text()) if context_path.exists() else {}
        source = context.get('source_run')
        if source:
            source = Path(source)
            return load_saved_outputs(source if source.is_absolute() else folder/source, _seen=seen)
        raise FileNotFoundError(f'No saved ensemble predictions in {folder}')
    ensemble = {p.stem.removeprefix('ensemble_'):np.load(p,allow_pickle=False) for p in paths}
    with np.load(folder/'audit/audit_rows.npz',allow_pickle=False) as rows:
        labels, groups = rows['y'].copy(), rows['group_id'].copy()
    if 'mean_probs' not in ensemble or labels.shape != groups.shape or len(labels) != len(ensemble['mean_probs']):
        raise ValueError('Saved predictions, labels and group IDs are not row aligned')
    return ensemble, labels, groups


def load_audit_inputs(folder):
    """Recover audit context, with fallbacks for pre-refactor training runs."""
    from modules.pipelines.audit_pipeline import AuditInputs
    folder = Path(folder)
    ensemble, labels, groups = load_saved_outputs(folder)
    config = load_config_from_experiment(folder)
    path = folder/'audit/audit_context.json'
    context = json.loads(path.read_text()) if path.exists() else {}
    metadata = context.get('metadata', {})
    features = protected = None
    arrays = folder/'audit/diagnostic_rows.npz'
    if arrays.exists():
        with np.load(arrays, allow_pickle=False) as rows:
            features = rows['features'].copy() if 'features' in rows else None
            protected = rows['protected'].copy() if 'protected' in rows else None
    elif (folder/'synthetic_data/metadata.json').exists():
        metadata = json.loads((folder/'synthetic_data/metadata.json').read_text())
        if 'stripe' in metadata:
            with np.load(folder/'synthetic_data/audit.npz', allow_pickle=False) as rows:
                if not np.array_equal(rows['group'], groups) or not np.array_equal(rows['y'], labels):
                    raise ValueError('Synthetic diagnostic rows do not match saved audit rows')
                features, protected = rows['X'].copy(), rows['S'].copy()
    names = context.get('group_names')
    if names is None:
        info_path = folder/'data/data_info.json'
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        names = {int(g):name for name,g in info.get('group_id', {}).items()}
    seed = config.get('model_seed')
    if seed is None:
        seed = config.get('split', {}).get('seed', 42)
    n_models = config.get('n_models', 1)
    provenance = context.get('provenance', dict(
        data_seed=metadata.get('seed', config.get('split', {}).get('seed', 42)),
        model_seeds=[seed+i for i in range(n_models)],
        method='deep_ensemble' if n_models > 1 else 'single_model'))
    return AuditInputs(ensemble=ensemble, labels=labels, groups=groups,
        group_names={int(g):str(name) for g,name in names.items()},
        binary=context.get('binary', config.get('train', {}).get('binary', True)),
        threshold=context.get('threshold', config.get('train', {}).get('threshold', .5)),
        num_classes=context.get('num_classes', ensemble['mean_probs'].shape[1]),
        metadata=metadata, provenance=provenance, features=features, protected=protected)


def load_audit_config(folder):
    """Prefer resolved audit settings; support the former experiment YAML."""
    folder = Path(folder)
    path = folder/'audit/audit_config.json'
    if path.exists():
        return json.loads(path.read_text())
    path = folder/'experiment.yaml'
    if path.exists():
        import yaml
        return (yaml.safe_load(path.read_text()) or {}).get('audit', {})
    return {}