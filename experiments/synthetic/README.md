# Synthetic uncertainty experiments

Extract this archive at your existing project's root. It adds only
`experiments/synthetic/`. Run commands from that project root, in the Python
environment where your existing training pipeline works.

The previous `modules/data/synthetic_uncertainty.py` is not imported or needed.
Do not copy or modify your existing modules for this experiment.

## Files

| File | Responsibility |
|---|---|
| `data.py` | DGP, oracle values, fixed splits, PreparedData adaptation, raw data saving |
| `audit.py` | Group/pair disparities and interactions from your evaluator's uncertainties |
| `pipeline.py` | Calls existing train_models, evaluate_models and save_pipeline_result |
| `run.py` | YAML configuration, condition/seed selection, output management |
| `configs/smoke.yaml` | Two tiny training runs to check integration |
| `configs/main.yaml` | 16 conditions, 5 data seeds, 10 ensemble members |
| `tests.py` | DGP tests and optional real training/save/reload test |
| `summarize.py` | Combines completed results into two CSVs |

## First run

```bash
python -m experiments.synthetic.run --config experiments/synthetic/configs/smoke.yaml --dry-run
python -m unittest experiments.synthetic.tests -v
python -m experiments.synthetic.run --config experiments/synthetic/configs/smoke.yaml
```

`--dry-run` validates generation settings and lists destinations without training.
The integration test runs only when PyTorch is installed; otherwise it is explicitly
skipped. The smoke configuration is not suitable for scientific conclusions.

Existing dependencies: NumPy, pandas, scikit-learn (>=1.2 for your
`sparse_output` API), PyTorch, joblib. YAML reading additionally needs PyYAML:
`python -m pip install pyyaml`. No Laplace or tree packages are required.

## Main experiment

Start with one condition and seed:

```bash
python -m experiments.synthetic.run --config experiments/synthetic/configs/main.yaml --condition baseline --seed 0
```

For the full sweep, use a fresh `output` in YAML and omit both filters:

```bash
python -m experiments.synthetic.run --config experiments/synthetic/configs/main.yaml
python -m experiments.synthetic.summarize experiments/synthetic/results/main
```

The full supplied configuration fits **800 networks**, potentially a long run.
Already existing run folders are rejected, including incomplete runs. There is
no automatic resume or checkpoint continuation. Select only unrun condition/seed
pairs or set a new output directory. Do not run concurrent writers to one destination.

To generate raw data without training:

```bash
python -m experiments.synthetic.run --config experiments/synthetic/configs/smoke.yaml --generate-only
```

Generated-only outputs use `generated/`, distinct from trained `runs/`.
Relative YAML `output` paths resolve under `experiments/synthetic/` and must be
inside its `results/` directory. All sizes in `data` are observations PER GROUP
before training thinning. The retained count has a minimum of one per group.

## How it fits your project

Expected existing imports are:

- `modules.utils.dataset_utils` and `modules.utils.saving_utils`
- `modules.utils.tensor_utils` (used by your existing modules)
- `modules.pipelines.train_predictive_pipeline`
- `modules.predictive.models`, `losses`, `trainer`, `metrics`, `ensemble`

The adapter creates your `LoadedDataset` with `metadata['spec']` and your
`PreparedData` with `(X, y, group_id)` loaders. It calls `train_models` and
`evaluate_models` unchanged. Your trainer, losses, MLP, AdamW, early stopping and
probability/entropy implementation are reused, not reimplemented.

The standard pipeline's `prepare_data`/`run_predictive_pipeline` is not called:
it always randomly resplits data. The experiment preserves train/validation/audit
partitions, mapping audit to pipeline test. Reference is retained only for later
integration/coverage studies. Preprocessing is fitted on training data only.
Oracle values, row IDs and split labels are excluded from predictors.

`loading_utils.py` is not imported: the supplied version imports legacy names
absent from the supplied dataset utilities. Raw splits reload through this
folder's `data.load_dataset(folder=...)`. Trusted saved preprocessing reloads via
joblib from `preprocessing/predictor/predictor_preprocessing_schema.joblib`.
Model state dicts can be loaded into your MLP using the saved pipeline config.

## Outputs per condition/seed

`results/<name>/runs/<condition>/seed_<n>/` contains:

- Your usual `predictive_ensemble/`, `preprocessing/`, `histories/`, `results/`,
  `models/`, `data/`, `rng/` and `environment/` artifacts.
- `synthetic_data/`: four raw NPZ splits, metadata, saved global split indices
  and preprocessing. Indices refer to the concatenated loaded DataFrame;
  `row_id` alone is only unique within a split.
- `synthetic_audit/`: uncertainty audit JSON, ensemble predictive-quality JSON,
  group quality CSV and audit-row labels/group IDs/oracle probabilities.
- `summary.csv`, complete experiment YAML, and `COMPLETE` on success.

Your native ensemble arrays are saved by your saving utility in `results/`.
Your native pipeline metrics refer to member zero. The extra quality metrics
refer to the mean ensemble predictor. Brier uses the binary class-1 convention
`mean((p1-y)**2)`; NLL clips probabilities at 1e-12.

`all_runs.csv` and `by_condition.csv` are produced by summarize. The latter gives
count, mean and standard deviation across seeds, not confidence intervals.

## Experimental meaning

Four groups occupy separate proxy-feature regions; explicit protected inputs are
excluded by default, but group information remains recoverable from X. Toggle
`append_protected_to_predictor` in YAML to compare explicit binary S1/S2 inputs.
Use the same data seeds and conditions for paired comparisons.

Data seed controls DGP generation. Model seed starts at
`model_seed_base + data_seed * n_models`; your pipeline increments it for members.
Conditions share model seed schedules and paired covariates. Scarcity retains
nested training subsets; validation/audit/reference are unchanged across scarcity.

Noise experiments change label-flip probabilities. Additive/interaction scenarios
set true entropy targets in bits and invert binary entropy to obtain noise rates.
Their zero-strength baseline is 0.30 bits, slightly different from the ordinary
baseline eta=0.05. Compare these sweeps against their own zero-strength condition.
Cancellation is a candidate construction, not a guaranteed learned-model result.

Reported component gaps and interaction contrasts use nats. F_U, F_U_int and
hidden_normalized are normalized; hidden disparity is calculated per pair before
maximizing. Hoeffding radii simultaneously bound the eight aleatoric/epistemic
means conditional on fitted predictors and iid sampling within groups. A total-
entropy radius can be obtained by summing the two component radii; the returned
radius is not a simultaneous bound for all twelve means. Bootstrap coverage,
MC dropout and last-layer Laplace are not implemented in this folder.

## Validation and references

During delivery, DGP pairing, entropy interaction and cancellation tests passed;
YAML dry-run, data-only generation, compilation and user API field checks passed.
End-to-end training could not be executed in the delivery environment because
PyTorch was unavailable. Run the integration test above in your project environment.
No user modules were modified.

YAML uses `safe_load` per https://pyyaml.org/wiki/PyYAMLDocumentation.
Audit loaders keep deterministic row order (`shuffle=False`), consistent with
https://docs.pytorch.org/docs/stable/data.html.