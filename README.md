# InterFair-Pytorch

Real and synthetic experiments share data preparation, training, ensemble evaluation,
uncertainty auditing and artifact saving. `experiments/` contains generated outputs only.
Refactored from branch `refactor-organize`, commit `8fb2f6a6144974774596ec543a45cb5f011cc93f`.

## Run

Use Python 3.10+ and install a PyTorch build suitable for your machine, then:

```bash
pip install -r requirements-core.txt
python -m scripts.run_experiment --config configs/synthetic/smoke.yaml --dry-run
python -m scripts.run_experiment --config configs/synthetic/smoke.yaml
python -m scripts.summarize experiments/smoke
python -m scripts.report experiments/smoke
python -m scripts.plot_uncertainty_bars --input experiments/smoke --out experiments/smoke/figures/bars
```

`train_predictive.py --config ...` is an equivalent training entry point.
Choose a new `output` value before rerunning: existing runs are never overwritten.
`--condition NAME` and `--seed INTEGER` select configured runs.
`--generate-only` exports synthetic raw splits without training.

## Organization

| Location | Responsibility |
|---|---|
| `configs/` | Dataset arguments, conditions, seeds, model/training settings and audit switches |
| `modules/data/` | Adult, local CSV, synthetic and loan generation, CelebA adapter |
| `modules/predictive/` | Canonical MLP/CNN models, losses, trainer and ensemble evaluator |
| `modules/pipelines/` | Configuration, preparation and shared experiment orchestration |
| `modules/metrics/` | Predictive quality, uncertainty disparities, outcome fairness, distribution analysis |
| `modules/utils/` | Saving, checkpoint loading, preprocessing and plotting |
| `modules/reporting/` | Cross-seed aggregation and synthetic oracle reports |
| `scripts/` | Thin command-line entry points |
| `tests/` | Mathematical and pipeline regression checks |
| `experiments/` | Training histories, checkpoints, predictions, validation losses, audits and derived reports |

## Configure an experiment

Start with an existing YAML and edit it. `conditions` contains dataset-argument overrides;
model/training settings live under `pipeline`. Synthetic examples retain the original conditions.

```yaml
output: experiments/my_study
model_seed_base: 1000
data_seeds: [0, 1, 2]
dataset:
  name: synthetic_uncertainty
  kwargs: {train: 2000, validation: 500, audit: 5000, reference: 25000}
conditions:
  baseline: {scenario: baseline}
  scarcity: {scenario: scarcity, rho: 0.25}
pipeline:
  n_models: 5
  append_protected_to_predictor: false
  model: {hidden_dims: [128, 64], dropout: 0.0}
  train: {epochs: 100, patience: 10, binary: true}
audit:
  enabled: true
  alpha: 0.05
  interaction_weights: {'0': 1, '1': -1, '2': -1, '3': 1}
  fairness: true
  distribution: true
```

`audit.enabled` controls uncertainty disparities. `fairness` and `distribution` are
independent switches for the existing shared metrics. Ensemble predictive quality and
per-group quality are always saved. Interaction weights are explicit: four groups alone
do not establish a two-attribute interaction design. Omit weights for general datasets;
`F_U_int` is then null. Group disparities support arbitrary observed groups and use
`log(number_of_classes)` normalization for multiclass tasks.

Synthetic split sizes are **per group**, except `loan_*` sizes, which are **total rows**.
The original loan predictors include Gender and Race even when
`append_protected_to_predictor: false`; that flag prevents appending S1/S2, not removal
of the loan model's original predictors. The generator's outcome coding is preserved.

## Real datasets

```bash
pip install kagglehub
python -m scripts.run_experiment --config configs/adult.yaml
```

For a local Adult CSV, add `csv_path: /path/to/adult.csv` to `dataset.kwargs`.
The source branch's income coding is retained: `<=50K` maps to 1.

For another real tabular dataset, set:

```yaml
dataset:
  name: csv_tabular
  kwargs:
    path: /path/to/data.csv
    label_col: outcome
    protected_cols: [sex, ethnicity]
    drop_feature_cols: [record_id]
```

Labels must be 0/1 for binary training or contiguous integers starting at zero for
multiclass training (`pipeline.train.binary: false`). File paths are relative to your
working directory; run commands at repository root. No interaction weights are inferred.

For images, install a `torchvision` version compatible with your PyTorch build, place
CelebA locally in torchvision's expected layout, edit `configs/celeba.yaml`, then use the
same runner. The adapter uses official train/valid/test partitions and a compact CNN.
The CNN is a new baseline, not a reproduction of an established image experiment.

## Artifacts and reuse

Each run is under `experiments/<study>/runs/<condition>/seed_<seed>/`:

- `config.json`, `experiment.yaml`, `resolved_run.yaml`: model settings and resolved run arguments.
- `predictive_ensemble/`, `preprocessing/`, `splits/`, training histories: reproducibility artifacts.
- `results/ensemble_*.npy`: row-aligned predictions and uncertainties, including a one-member ensemble.
- `audit/`: uncertainty, predictive, group, optional fairness and distribution metrics.
- `synthetic_data/`: raw generated splits and oracle metadata, only for synthetic runs.
- `summary.csv`, `COMPLETE`: aggregation input and successful completion marker.

Re-audit saved predictions with changed audit settings, without retraining:

```bash
python -m scripts.audit --run experiments/smoke/runs/baseline/seed_0 --config configs/synthetic/smoke.yaml --out experiments/smoke/re_audit
```

That command recalculates general uncertainty/predictive metrics; synthetic oracle
reports and optional outcome-fairness/distribution metrics are produced during training.
Load trained models and fitted preprocessing with
`modules.utils.checkpoint_utils.load_ensemble(run, device='cpu')`. Load only trusted
joblib preprocessing artifacts.

```bash
python -m scripts.plot_synthetic --run experiments/smoke/runs/baseline/seed_0 --ensemble
python -m unittest discover -s tests -v
```

Synthetic spatial plots apply to the four-square generator, not loan data.
General condition bar plots and summaries work for both real and synthetic results.
`report` is the specialized four-group synthetic oracle report.

See [docs/REFACTOR.md](docs/REFACTOR.md) for migration decisions and validation limits.
