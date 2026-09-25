"""Thin experiment orchestration: delegates training, evaluation and saving."""
import json
from pathlib import Path
import numpy as np
from .data import GROUPS, LOG2, prepare_pipeline_data, save_prepared_dataset
from .audit import audit_uncertainties, audit_stripe_regions

def run_experiment(config=None, *, out=None,
                             regularizer=None, regularizer_weight=1.0,
                             fair_loader=None):
    """Use your existing training/evaluation pipeline; add an uncertainty audit.

    Args:
        config: PipelineConfig, or None to use a default.
        out: Path to save results, or None to skip saving.
        regularizer: Optional regularizer to pass to your pipeline.
        regularizer_weight: Weight of the regularizer in your pipeline.
        fair_loader: Optional DataLoader for fairness regularization. Must be passed if regularizer is not None.

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
    audit = audit_uncertainties(ensemble, groups)
    if 'stripe' in data.loaded.metadata:
        audit['stripe_regions'] = audit_stripe_regions(
            ensemble, groups,
            data.test_df[data.loaded.metadata['feature_names']].to_numpy(),
            data.test_df[['S1', 'S2']].to_numpy(), data.loaded.metadata)
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
                                -np.array(audit['oracle_entropy_nats'])).tolist()  # Compares learned mean aleatoric uncertainty with the generator's known conditional entropy
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