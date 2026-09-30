> Updated by the single-generator migration: see [UNIFIED_SYNTHETIC.md](UNIFIED_SYNTHETIC.md). The original refactor decisions below are retained as history.

# Refactoring decisions and migration

The source branch already delegated synthetic training to shared functions, but its
experiment folder owned preparation, auditing, output writing and plotting. A second
older synthetic generator in `modules/data/` made ownership ambiguous. The shared
pipeline randomly re-split datasets, so calling it directly changed the synthetic study.

## Changes

1. The experiment generator is now the canonical `modules/data/synthetic_uncertainty.py`.
   Loan generation and its JSON presets live beside it. Generator sampling equations,
   scenarios, paired streams and raw field names are retained.
2. Shared preparation honors `LoadedDataset.metadata['split_indices']`, validates disjoint
   partitions, fits preprocessing on training rows only, and excludes reference rows.
   Data split seeds and model seeds have separate fields. A single model also produces
   an ensemble-output contract, so zero-disagreement controls use the same metrics.
3. `uncertainty_audit.py` owns general metrics. Four-group assumptions are removed;
   class-count normalization is explicit. Interaction contrasts require specified
   weights. Stripe/oracle diagnostics remain in a separate scientific metrics module.
   Binary four-group definitions, pairwise hidden maxima and Hoeffding radii are retained.
4. `experiment_pipeline.py` composes the shared trainer, predictive metrics, uncertainty
   audit, existing outcome-fairness metrics and distribution decompositions. Saving is
   centralized in `saving_utils.py`; output audit folders are named `audit/`.
5. YAML configuration and command-line entry points live outside `experiments/`.
   Existing smoke, pilot, main and Kanubala configurations have been migrated.
6. Generic bars are exposed by `plotting_utils.py`, with implementations in a small
   `plots/` package to avoid enlarging the existing plotting module further. Reporting
   and plotting scripts are thin entry points. The original scientific report supports
   both `audit/` and older `synthetic_audit/` folders.
7. Current checkpoints reload with the exact saved preprocessing. Broken old loading
   utilities and the old hard-coded root testing script are replaced with current APIs.
8. Adult supports a local CSV, and arbitrary tabular CSVs share the dataset contract.
   CelebA's missing attributes, misspelled normalization call and obsolete imports are
   fixed. Its image batches adapt to the same trainer and audit flow via a compact CNN.

## API migration

| Previous usage | Replacement |
|---|---|
| `experiments.synthetic.data` | `modules.data.synthetic_uncertainty` |
| `experiments.synthetic.audit.audit_uncertainties` | `modules.metrics.uncertainty_audit.audit_uncertainties` |
| `experiments.synthetic.pipeline.run_experiment` | `modules.pipelines.experiment_pipeline.run_experiment` |
| `python -m experiments.synthetic.run` | `python -m scripts.run_experiment` |
| `experiments/synthetic/configs/*.yaml` | `configs/synthetic/*.yaml` |
| YAML `data: {...}` | YAML `dataset: {name: synthetic_uncertainty, kwargs: {...}}` |
| YAML `output: results/name` | YAML `output: experiments/name` |
| old `loading_utils` model reconstruction | `checkpoint_utils.load_ensemble` |
| hard-coded `test_models.py` | `scripts.audit` or its root compatibility entry point |

This is an API migration: old import paths under `experiments/` are intentionally removed.
Do not overlay the ZIP onto an old checkout without removing those files. Use a clean
folder or apply the included patch to the exact source commit. Existing results need
not be deleted. Point plotting/report tools to their original folders as appropriate.
The general bar reader uses completed per-run summaries; the specialized report can
read the old synthetic artifacts directly.

Legacy `modules/models/predictive_models.py` and `fairness_metrics_old.py` remain for
reference and old external notebooks; the new workflow does not depend on them.
Generative-model helpers are not integrated: `build_px_loaders` now raises an explicit
error instead of reaching a nonexistent field. Custom regularizers are supported by
calling `run_experiment` with an explicit regularizer and fairness loader in a pipeline
script; arbitrary Python objects are not deserialized from YAML.

## Scientific interpretation

- Entropy and raw component gaps are in nats. Normalized scores divide by log(C).
- Interactions are signed contrasts; F_U_int divides the largest component contrast
  by log(C) times half the sum of absolute contrast weights.
- Hoeffding radii cover aleatoric and epistemic group means under independent audit
  observations conditional on a fixed fitted ensemble. They are not seed variability.
- Audits compare observed groups. A configured interaction requires all its groups.
- The existing pipeline's `test_metrics` describes member zero; the `audit/` predictive
  metrics and `summary.csv` describe the ensemble mean. Training histories retain
  validation loss; the audit uses the held-out test/audit partition.
- Binary Brier score is class-1 squared error; multiclass Brier is the sum across classes.
- The loan generator's oracle conditions on its original predictors, including protected
  attributes. The original outcome and predictor definitions are not changed here.
- A model seed schedule is paired across conditions, while data splits use data seeds.
  CelebA's official partitions stay fixed regardless of the configured data seed.

## Validation

Validation details and environment are recorded in `VALIDATION.md`. Small smoke runs
check software integration only, not scientific performance. Full Adult/CelebA datasets,
GPU execution, and long paper experiments were not run. Dataset adapters are checked
using local tabular fixtures and mocked CelebA images, with actual training and checkpoint
reload. Original pinned `requirements.txt` is retained as provenance; `requirements-core.txt`
is a portable dependency specification, not a reproducibility lockfile.
