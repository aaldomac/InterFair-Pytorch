"""Re-audit saved predictions through the training audit API; never overwrite a run."""
import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path
import yaml
from modules.pipelines.audit_pipeline import AuditConfig, run_audit
from modules.utils.loading_utils import load_audit_inputs, load_audit_config, load_config_from_experiment


def audit_saved_run(run, out, *, overrides=None, threshold=None):
    from modules.utils.saving_utils import save_audit_bundle, save_run_summary, _json_dump
    run, out = Path(run).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    if out.is_relative_to(run):
        raise ValueError('Re-audit destination must be outside its source run')
    options = load_audit_config(run)
    updates = dict(overrides or {})
    if 'alpha' in updates:
        if 'confidence_alpha' in updates and updates['confidence_alpha'] != updates['alpha']:
            raise ValueError('alpha and confidence_alpha disagree')
        updates['confidence_alpha'] = updates.pop('alpha')
    if 'confidence_alpha' in updates:
        options.pop('alpha', None)
    options.update(updates)
    options = AuditConfig.from_mapping(options)
    inputs = load_audit_inputs(run)
    if threshold is not None:
        inputs = replace(inputs, threshold=threshold)
    audited = run_audit(inputs, options)
    out.mkdir(parents=True, exist_ok=False)
    save_audit_bundle(out, audited, inputs, options, source_run=run)
    config = load_config_from_experiment(run)
    config.setdefault('train', {})['threshold'] = inputs.threshold
    _json_dump(config, out/'config.json')
    for name in ('experiment.yaml', 'resolved_run.yaml'):
        if (run/name).exists():
            shutil.copyfile(run/name, out/name)
    # The synthetic reporter needs only metadata, not another copy of raw splits.
    synthetic_meta = run/'synthetic_data/metadata.json'
    if synthetic_meta.exists():
        (out/'synthetic_data').mkdir()
        shutil.copyfile(synthetic_meta, out/'synthetic_data/metadata.json')
    coverage = run/'audit/group_coverage.csv'
    if coverage.exists():
        shutil.copyfile(coverage, out/'audit/group_coverage.csv')
    identity = dict(condition=run.parent.name, data_seed=inputs.provenance.get('data_seed'),
                    model_seed_start=(inputs.provenance.get('model_seeds') or [None])[0])
    if (run/'summary.csv').exists():
        import csv
        with (run/'summary.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1:
            raise ValueError('Expected exactly one source summary row')
        identity.update({key:rows[0][key] for key in identity if key in rows[0]})
    save_run_summary(out, audited, identity)
    (out/'COMPLETE').write_text('Re-audit completed successfully; predictions linked to source run\n')
    return audited


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run', type=Path)
    source.add_argument('--experiment', type=Path, help='Audit all completed runs into a new experiment tree')
    parser.add_argument('--config', type=Path, help='YAML audit settings override the saved settings')
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    overrides, threshold = {}, args.threshold
    if args.config:
        cfg = yaml.safe_load(args.config.read_text())
        overrides = cfg.get('audit', {})
        if threshold is None:
            threshold = cfg.get('pipeline', {}).get('train', {}).get('threshold')
    if args.run:
        audit_saved_run(args.run, args.out, overrides=overrides, threshold=threshold)
    else:
        from modules.utils.reporting_utils import completed_runs
        runs = completed_runs(args.experiment)
        if not runs:
            parser.error('No completed training runs found')
        if args.out.resolve().is_relative_to(args.experiment.resolve()):
            parser.error('Re-audit output must be outside the source experiment')
        for run in runs:
            audit_saved_run(run, args.out/'runs'/run.parent.name/run.name,
                            overrides=overrides, threshold=threshold)
    print(f'Audit saved to {args.out}')

if __name__ == '__main__':
    main()
