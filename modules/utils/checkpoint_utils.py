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
