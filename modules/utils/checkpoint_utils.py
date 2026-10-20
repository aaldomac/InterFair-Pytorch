"""Reload current shared-pipeline models without fitting preprocessing again.

Only load joblib schemas from trusted runs (pickle format).
"""
import json
from pathlib import Path
import joblib
import torch
from modules.predictive.models import MLPClassifier, ImageClassifier


def load_ensemble(run, device='cpu'):
    run = Path(run)
    config = json.loads((run/'config.json').read_text())
    meta = json.loads((run/'models/model_metadata.json').read_text())
    schema = joblib.load(run/'preprocessing/predictor/predictor_preprocessing_schema.joblib')
    models = []
    for i in range(meta['n_models']):
        kwargs = dict(num_outputs=1 if meta['binary'] else meta['num_classes'],
                      dropout=config['model']['dropout'])
        if schema.get('modality') == 'image':
            model = ImageClassifier(**kwargs)
        else:
            model = MLPClassifier(input_dim=meta['num_features'],
                hidden_dims=tuple(config['model']['hidden_dims']), **kwargs)
        model.load_state_dict(torch.load(run/'predictive_ensemble'/f'predictor_{i}.pt',
                                        map_location='cpu', weights_only=True))
        models.append(model.to(device).eval())
    return models, schema, meta['binary']


def infer_saved_run(run, device='cpu'):
    """Inference only, using fitted preprocessing and exact saved audit rows.

    New tabular runs retain transformed inputs. Legacy runs require the original
    dataset (synthetic runs retain raw splits). Never fit preprocessing or train.
    """
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset
    from modules.predictive.ensemble import evaluate_ensemble
    from modules.utils.loading_utils import load_split_indices
    run = Path(run)
    models, schema, binary = load_ensemble(run, device=device)
    config = json.loads((run/'config.json').read_text())
    with np.load(run/'audit/audit_rows.npz', allow_pickle=False) as rows:
        y, g = rows['y'].copy(), rows['group_id'].copy()
    cached = run/'data/audit_inputs.npz'
    batch_size = config.get('loader', {}).get('eval_batch_size', 256)
    if cached.exists():
        with np.load(cached, allow_pickle=False) as rows:
            x = torch.tensor(rows['X'].copy(), dtype=torch.float32)
            actual_y, actual_g = rows['y'].copy(), rows['group_id'].copy()
        loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    elif schema.get('modality') == 'image':
        from dataclasses import asdict
        from modules.data.CelebA import CelebAConfig, CelebAAttributeDataset, SupervisedImages
        kwargs = asdict(CelebAConfig(**schema['dataset_kwargs']))
        for key in ('download', 'batch_size', 'num_workers'):
            kwargs.pop(key, None)
        dataset = CelebAAttributeDataset(split='test', download=False, **kwargs)
        actual_y, actual_g = dataset.labels.numpy(), dataset.group_ids.numpy()
        loader = DataLoader(SupervisedImages(dataset), batch_size=batch_size, shuffle=False)
    else:
        from modules.utils.dataset_utils import transform_predictor_schema
        from modules.pipelines.train_predictive_pipeline import load_dataset_by_name
        if (run/'synthetic_data/metadata.json').exists():
            loaded = load_dataset_by_name('synthetic_data', folder=str(run/'synthetic_data'))
        else:
            loaded = load_dataset_by_name(config['dataset_name'], **config.get('dataset_kwargs', {}))
        indices = load_split_indices(run)[2]
        frame = loaded.df.loc[indices]
        x, actual_y, actual_g = transform_predictor_schema(frame, schema)
        actual_y, actual_g = actual_y.numpy(), actual_g.numpy()
        loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    if not np.array_equal(actual_y, y) or not np.array_equal(actual_g, g):
        raise ValueError('Inference data does not match saved audit rows; refusing to audit')
    outputs = evaluate_ensemble(models, loader, binary=binary, device=device,
                               store_member_probs=False)
    return outputs, y, g
