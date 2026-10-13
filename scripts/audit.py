"""Re-audit saved predictions through the training audit API; never overwrite a run."""
import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path
import yaml
from modules.pipelines.audit_pipeline import AuditConfig, run_audit
from modules.utils.loading_utils import load_audit_inputs, load_audit_config, load_config_from_experiment


def overwrite_audit_in_place(run, *, overrides=None, threshold=None):
    import tempfile
    import numpy as np

    run = Path(run).resolve()
    context_path = run / 'audit' / 'audit_context.json'
    old_context = (json.loads(context_path.read_text()) if context_path.exists() else {})

    # Preserve source row identifiers when the original audit retained them.
    source_index = None
    rows_path = run / 'audit' / 'audit_rows.npz'
    if rows_path.exists():
        with np.load(rows_path, allow_pickle=False) as rows:
            if 'source_index' in rows:
                source_index = rows['source_index'].copy()

    with tempfile.TemporaryDirectory(prefix='.reaudit-',dir=run.parent,) as temporary:
        staged = Path(temporary) / 'result'

        # Existing function writes a complete result to a fresh destination.
        audited = audit_saved_run(run,staged,overrides=overrides,threshold=threshold,)

        staged_audit = staged / 'audit'
        staged_context_path = staged_audit / 'audit_context.json'
        context = json.loads(staged_context_path.read_text())

        # The staging operation references `run`. Do not preserve that
        # reference when moving the results back into `run` itself.
        context.pop('source_run', None)

        has_local_predictions = any((run / 'results').glob('ensemble_*.npy'))
        if not has_local_predictions:
            # This run may itself be a previously generated re-audit.
            source = old_context.get('source_run')
            if source is None:
                raise ValueError('Cannot preserve the source prediction reference.')
            source = Path(source)
            if not source.is_absolute():
                source = run / source
            context['source_run'] = str(source.resolve())

        staged_context_path.write_text(json.dumps(context, indent=2) + '\n')

        if source_index is not None:
            staged_rows = staged_audit / 'audit_rows.npz'
            with np.load(staged_rows, allow_pickle=False) as rows:
                arrays = {key: rows[key].copy() for key in rows.files}
            if source_index.shape != arrays['y'].shape:
                raise ValueError('Source indices do not match audit rows.')
            arrays['source_index'] = source_index
            np.savez_compressed(staged_rows, **arrays)

        destination = run / 'audit'
        destination.mkdir(parents=True, exist_ok=True)

        # Remove old optional results if the new audit no longer produces them.
        # Keep unrelated files and diagnostics such as group_coverage.csv.
        optional_outputs = {
            'uncertainty_audit.json',
            'fairness.json',
            'performance_fairness.json',
            'classical_summary.json',
            'distribution.json',
            'diagnostic_rows.npz',
        }

        # A partially installed update must not be marked complete.
        (run / 'COMPLETE').unlink(missing_ok=True)

        for name in optional_outputs:
            if not (staged_audit / name).exists():
                (destination / name).unlink(missing_ok=True)

        for path in staged_audit.iterdir():
            if path.is_file():
                path.replace(destination / path.name)

        for name in ('summary.csv', 'artifact_manifest.json'):
            (staged / name).replace(run / name)

        (run / 'COMPLETE').write_text('In-place re-audit completed successfully\n')

    return audited

def audit_saved_run(run, out, *, overrides=None, threshold=None, overwrite=False):
    from modules.utils.saving_utils import save_audit_bundle, save_run_summary, _json_dump
    run, out = Path(run).resolve(), Path(out).resolve()
    if out == run and overwrite:
        return overwrite_audit_in_place(run, overrides=overrides, threshold=threshold)
    if out.exists():
        raise FileExistsError(f'{out} already exists. ''For an in-place update, use the source path as --out and pass --overwrite.')
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
    parser.add_argument('--overwrite', action='store_true', help='Replace audit outputs in place; requires --out equal to source.')
    args = parser.parse_args()
    source_path = args.run if args.run is not None else args.experiment
    same_destination = args.out.resolve() == source_path.resolve()

    if args.overwrite and not same_destination:
        parser.error('--overwrite requires --out to equal the source path.')

    if args.out.exists() and not args.overwrite:
        parser.error(f'{args.out} already exists. Pass --overwrite to re-audit the source in place.')

    overrides, threshold = {}, args.threshold
    if args.config:
        cfg = yaml.safe_load(args.config.read_text())
        overrides = cfg.get('audit', {})
        if threshold is None:
            threshold = cfg.get('pipeline', {}).get('train', {}).get('threshold')
    if args.run:
        audit_saved_run(args.run, args.out, overrides=overrides, threshold=threshold, overwrite=args.overwrite)
    else:
        from modules.utils.reporting_utils import completed_runs
        runs = completed_runs(args.experiment)
        if not runs:
            parser.error('No completed training runs found')
        if args.out.resolve() != args.experiment.resolve() and args.out.resolve().is_relative_to(args.experiment.resolve()):
            parser.error('A separate re-audit output must be outside the source experiment')
        for run in runs:
            audit_saved_run(run, args.out/'runs'/run.parent.name/run.name,
                            overrides=overrides, threshold=threshold, overwrite=args.overwrite,)
    print(f'Audit saved to {args.out}')

if __name__ == '__main__':
    main()
