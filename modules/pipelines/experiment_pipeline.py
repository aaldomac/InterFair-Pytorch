"""Shared train/evaluate/audit/save orchestration for registered datasets."""
from pathlib import Path
import numpy as np
from modules.metrics.uncertainty_audit import audit_uncertainties
from modules.metrics.ensemble_quality import ensemble_quality
from modules.pipelines.train_predictive_pipeline import run_predictive_pipeline


def run_experiment(config, *, out=None, audit_config=None, regularizer=None,
                   regularizer_weight=1., fair_loader=None):
    options = dict(audit_config or {})
    if out is not None and Path(out).exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    result = run_predictive_pipeline(config, regularizer=regularizer,
        regularizer_weight=regularizer_weight, fair_loader=fair_loader)
    data, ensemble = result.data, result.ensemble_metrics
    labels = data.test_df[data.spec.label_col].to_numpy(dtype=int)
    groups = data.test_df[data.spec.group_id_col].to_numpy(dtype=int)
    names = {int(v):str(k) for k,v in data.loaded.group_id.items()}
    metrics, by_group = ensemble_quality(ensemble, labels, groups,
        threshold=config.train.threshold, group_names=names)
    audit = None
    if options.get('enabled', True):
        audit = audit_uncertainties(ensemble, groups, alpha=options.get('alpha', .05),
            group_names=names, interaction_weights=options.get('interaction_weights'),
            num_classes=data.num_classes)
        meta = data.loaded.metadata
        if 'oracle_entropy_bits' in meta:
            oracle = np.asarray(meta['oracle_entropy_bits'])*np.log(2)
            selected = oracle[np.asarray(audit['group_ids'], dtype=int)]
            audit['oracle_entropy_nats'] = selected.tolist()
            audit['alea_minus_oracle'] = (np.asarray(audit['means'])[:,0]-selected).tolist()
        if 'stripe' in meta:
            from modules.metrics.synthetic_diagnostics import audit_stripe_regions
            audit['stripe_regions'] = audit_stripe_regions(ensemble, groups,
                data.test_df[meta['feature_names']].to_numpy(),
                data.test_df[['S1','S2']].to_numpy(), meta)
        audit.update(data_seed=meta.get('seed', config.split.seed),
            model_seeds=[(config.model_seed if config.model_seed is not None else config.split.seed)+i
                         for i in range(config.n_models)],
            method='deep_ensemble' if config.n_models > 1 else 'single_model',
            brier_convention='class-1 squared error' if data.num_classes == 2 else 'sum of classwise squared errors')
    additional = {}
    if options.get('fairness', False):
        from modules.metrics.fairness_metrics import evaluate_ensemble_fairness_from_loader
        additional['fairness'] = evaluate_ensemble_fairness_from_loader(
            ensemble, data.test_loader, binary=data.binary, threshold=config.train.threshold,
            positive_class=1, alpha=1.)
    if options.get('distribution', False):
        from modules.metrics.distribution_decomposition_metrics import group_distribution_analysis_from_ensemble_outputs
        additional['distribution'] = group_distribution_analysis_from_ensemble_outputs(ensemble, group_ids=groups)
    output = None
    if out is not None:
        from modules.utils.saving_utils import save_audited_result
        output = save_audited_result(result, out, audit, metrics, by_group, additional)
    return dict(pipeline_result=result, uncertainty_audit=audit, predictive_metrics=metrics,
                group_predictive_metrics=by_group, additional_metrics=additional, output_folder=output)
