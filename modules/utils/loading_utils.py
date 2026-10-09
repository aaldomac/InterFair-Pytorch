"""Readers for artifacts produced by the current shared pipeline."""
import json
from pathlib import Path
import numpy as np
from modules.utils.checkpoint_utils import load_ensemble


def load_config_from_experiment(folder):
    return json.loads((Path(folder)/'config.json').read_text())


def load_split_indices(folder):
    return tuple(np.load(Path(folder)/f'{name}_idx.npy', allow_pickle=False)
                 for name in ('train','val','test'))


def load_saved_outputs(folder):
    """Load row-aligned predictions for re-auditing without training or raw data."""
    folder = Path(folder)
    ensemble = {p.stem.removeprefix('ensemble_'):np.load(p,allow_pickle=False)
                for p in (folder/'results').glob('ensemble_*.npy')}
    with np.load(folder/'audit/audit_rows.npz',allow_pickle=False) as rows:
        labels, groups = rows['y'].copy(), rows['group_id'].copy()
    return ensemble, labels, groups
