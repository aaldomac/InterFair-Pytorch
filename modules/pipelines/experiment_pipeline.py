"""Shared train/evaluate/audit/save orchestration for registered datasets."""
from pathlib import Path
from modules.pipelines.audit_pipeline import AuditConfig, audit_inputs_from_result, run_audit
from modules.pipelines.train_predictive_pipeline import run_predictive_pipeline


def run_experiment(config, *, out=None, audit_config=None, regularizer=None,
                   regularizer_weight=1., fair_loader=None):
    options = dict(audit_config or {})
    if out is not None and Path(out).exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    result = run_predictive_pipeline(config, regularizer=regularizer,
        regularizer_weight=regularizer_weight, fair_loader=fair_loader)
    inputs = audit_inputs_from_result(result)
    audited = run_audit(inputs, options)
    output = None
    if out is not None:
        from modules.utils.saving_utils import save_audited_result
        output = save_audited_result(result, out, audited['uncertainty_audit'],
            audited['predictive_metrics'], audited['group_predictive_metrics'],
            audited['additional_metrics'], audit_inputs=inputs, audit_config=options)
    return dict(pipeline_result=result, **audited, output_folder=output)