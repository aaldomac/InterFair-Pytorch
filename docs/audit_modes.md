# Audit modes and subgroup metric log ratios

The same ensemble can be audited repeatedly without training. This change was
based on `refactor-organize` commit `38b96284f8401e68442d62d6189672d96997be40`.

## Confirmed diagnosis

* `scripts.audit.main` tested `args.soft is None`, then compared it with `'false'`.
  Omitting the flag overwrote saved/YAML settings with False; supplying either
  flag value did nothing. The repaired precedence is CLI > override YAML > saved
  `audit/audit_config.json` (or legacy experiment YAML) > dataclass defaults.
* `run_audit` correctly forwarded `df_soft` to `evaluate_ensemble_fairness`.
  `_class_probabilities` already supported multiclass matrices and hardened
  **after** aggregation. The suspected member-wise hardening was not present.
* `evaluate_ensemble_fairness` applied the mode only to DF. SP, DI, EO, EOdds and
  SSP always used hard decisions, whereas per-class fairness always used soft
  probabilities. Multiclass evaluation could reuse stale saved labels. These
  paths now use the selected representation of the ensemble mean.
* Binary `positive_class=0` previously changed the decision threshold convention
  in fairness, unlike quality/diagnostics. Decisions now consistently use column
  1; positive_class selects which outcome a rate measures.
* `load_saved_outputs` had no checkpoint inference fallback. Saved probabilities
  are still preferred; missing mean probabilities now trigger inference only.
* `read_group_run` tried to read `performance_fairness.jdon`; the corrected reader
  reads `.json` and exposes the new metric's subgroup values.
* Fairness debug prints exposed entire prediction tensors; removed.

## Commands

Run from the repository root with `requirements-core.txt` installed. For a small
new example (skip training if you already have an experiment):

```bash
python -m scripts.run_experiment --config configs/synthetic/smoke.yaml --condition baseline --seed 0
```

Then audit that **same** experiment twice:

```bash
python -m scripts.audit --experiment experiments/loan_uncertainty_smoke --soft True --disparity-metric class_probability --out experiments/audit_soft_class
python -m scripts.audit --experiment experiments/loan_uncertainty_smoke --soft False --disparity-metric class_probability --out experiments/audit_hard_class
```

Select the other metrics without retraining:

```bash
python -m scripts.audit --experiment experiments/loan_uncertainty_smoke --soft True --disparity-metric tpr --out experiments/audit_soft_tpr
python -m scripts.audit --experiment experiments/loan_uncertainty_smoke --soft True --disparity-metric fpr --out experiments/audit_soft_fpr
python -m scripts.audit --experiment experiments/loan_uncertainty_smoke --soft True --disparity-metric accuracy --out experiments/audit_soft_accuracy
```

Replace `True` with `False` and use a fresh output directory for hard rates.
Use `--threshold 0.7` to change binary decisions. `--run
experiments/loan_uncertainty_smoke/runs/baseline/seed_0` replaces `--experiment`
for one ensemble. `--disparity-metric` enables fairness auditing; `--fairness`
also enables it without changing the metric. Outputs are never reused as a
mode-specific metric cache. `--overwrite` remains available **only for identical
audit settings**; changed settings must use a new destination.

Override YAML, e.g. `--config configs/audit_example.yaml`:

```yaml
audit:
  fairness: true
  df_soft: true
  positive_class: 1
  smoothing_alpha: 0.0
  disparity_metric: tpr
  disparity_offset: 0.0
pipeline:
  train:
    threshold: 0.5
```

`soft` is a mapping alias for `df_soft`; conflicting aliases in one mapping are
an error. Python booleans, including explicit False, are preserved. The default
is `df_soft=False`, binary threshold 0.5 (otherwise inherited from the saved
context/training config), `smoothing_alpha=1`, `disparity_metric=class_probability`,
and `disparity_offset=0`. `--threshold` beats the override YAML threshold and the
saved threshold. The threshold is recorded as audit metadata; source training
configuration and checkpoint contents are unchanged.

```python
from scripts.audit import audit_saved_run
from modules.pipelines.audit_pipeline import run_audit
from modules.utils.loading_utils import load_audit_inputs

result = audit_saved_run("path/to/trained_run", "path/to/new_audit",
    overrides={"soft": False, "fairness": True, "disparity_metric": "tpr"},
    threshold=0.7)
# In-memory alternative, with the same configuration semantics:
result = run_audit(load_audit_inputs("path/to/trained_run"),
                   {"df_soft": True, "fairness": True})
```

## What changes with mode

The ensemble evaluator computes the arithmetic mean of member probabilities.
Hardening never happens member by member. Binary decisions are class 1 iff
`mean_probs[:,1] >= threshold`, including ties. Multiclass uses argmax; a tie
chooses the smallest class index. The positive_class option selects the class
for TPR/FPR/selection; it does not change the decision rule.

* DF, per-class log gaps, SP, DI, EO, EOdds, SSP, performance fairness and the
  selected generalized metric use probabilities in soft mode and one-hot
  ensemble decisions in hard mode.
* Soft TPR is the mean positive-class probability among true positives; soft FPR
  is the same among all other true labels. Multiclass uses one-vs-rest for the
  chosen positive_class. Soft accuracy is mean probability assigned to the true
  class: expected accuracy of a categorical draw, **not** argmax accuracy.
* Soft confusion counts are expected counts: TP=sum(p_k for y=k),
  FP=sum(p_k for y!=k), FN=sum(1-p_k for y=k), TN=sum(1-p_k for y!=k).
  PPV=TP/(TP+FP), selection=(TP+FP)/n, TNR=TN/(TN+FP). Hard mode gives ordinary
  counts/rates. Supports can therefore be fractional for soft PPV.
* Predictive quality accuracy, correctness/error strata, high-confidence errors
  and reliability diagrams concern actual hard decisions. They keep that
  definition in both audit modes. NLL, Brier, AUROC, calibration, predictive
  entropy, uncertainty decomposition, U_mix/U_het and distribution comparisons
  retain the original continuous probabilities. One-hot inputs would destroy
  their intended interpretation. True-label prevalence also stays unchanged.
* Diagnostics retain `mean_prob_class_k` and add
  `audit_prediction_frequency_class_k`, which follows the selected mode. Metadata
  states both the audit mode and the diagnostic policy. Plot readers consume
  the selected saved bundle; they do not recompute a different audit mode.

## Conventional DF versus generalized ratios

`fairness.json/differential_fairness` remains conventional class-probability DF:

`q[g,k] = (sum selected_probability[k] + alpha)/(n[g] + C*alpha)`.

Its epsilon is the maximum absolute natural-log gap across groups and matching
classes. `smoothing_alpha` is the Dirichlet pseudo-count per class; zero disables
it. The same option applies Beta(alpha,alpha) smoothing to performance rates and
Dirichlet smoothing to SSP group weights. `per_class_fairness` now uses the
selected representation, zero pseudo-count and no clipping in the audit wrapper.
Its standalone function retains its legacy explicit `eps` parameter/default.

`metric_log_ratio.json` is deliberately separate: arbitrary metric ratios are
**not** the same formal fairness guarantee. By default it compares raw subgroup
values. It records metric, raw and effective values, groups, epsilon, maximizing
ordered pair (numerator, denominator), component, offset, mode and threshold.
The summary label is `log_ratio_<metric>`, distinct from `DF_epsilon`.

For a vector M, compare matching components across groups, then take the largest
component disparity. Class probability returns all C components; rates and
accuracy return scalars. Both reduction and pair identification consider all
ordered distinct groups. Ties are deterministic by group identifier. The new
class_probability result matches conventional DF when both are unsmoothed and
all declared groups are observed; their defaults differ because legacy DF keeps
alpha=1. Hard binary wrapper DF now returns [G,2] rates, matching soft and
multiclass output. Direct legacy binary-vector calls retain their [G] return.

`--disparity-offset delta` compares `(M(g)+delta)/(M(h)+delta)`; this is additive
metric smoothing, not a count prior. It changes the estimand, shrinks ratios
and is recorded. It cannot repair an undefined conditional rate.

### Numerical policy

| Input | Generic reducer / selected metric result |
|---|---|
| Strictly positive finite values | log(max) - log(min), natural logs |
| Equal positive values | 0 |
| Positive versus zero, offset=0 | +infinity |
| All-zero component | 0 by explicit shared-zero convention |
| Negative or infinite input | ValueError; no clipping or dropping |
| NaN / undefined conditional rate | epsilon=null, status=undefined, reason recorded; all groups retained |
| Declared empty group | undefined value, invalidates aggregate |
| Fewer than two groups | epsilon=null, status=undefined |
| Vector with any undefined component | Entire result undefined |

Raw/effective undefined subgroup values are serialized as null. Infinite
**output** disparity is retained as Python JSON `Infinity`, following the existing
artifact writer's nonfinite-number convention (not strict RFC JSON). Consumers
requiring strict JSON must explicitly encode infinity rather than discard it.
The legacy `differential_fairness` tuple API keeps zero for zero/one observed
subgroups and excludes unobserved groups for compatibility. The generalized
API retains every declared group from audit group_names, including empty ones.

### Python extension points

```python
from modules.metrics.log_ratio import maximum_log_ratio, metric_log_ratio

maximum_log_ratio([0.8, 0.4], groups=["A", "B"])  # epsilon=log(2)
# p_selected is [N,C], already ensemble-averaged and optionally hardened.
metric_log_ratio(p_selected, y, group_ids,
                 metric=lambda p, y, k: p[:, k].mean(), positive_class=1)
```

`subgroup_metric_values` separates subgroup computation from reduction.
Callables receive selected probabilities, integer labels and positive_class for
one group, and return a scalar or fixed-length vector. Register a callable in
`SUBGROUP_METRICS[name]` before `run_audit` to use its name in Python config.
The stock CLI exposes the four built-ins. The generic reducer itself has no
metric-specific branches.

## Persistence and inference

Schema v3 writes a deterministic audit_id from all effective audit settings,
threshold, class count and binary policy into the context/manifest and summary.
This is a **settings identity**, not a checkpoint/data content hash. Existing
outputs cannot be overwritten across settings, and live report/plot readers
reject mixed identities within a condition. Generic aggregation also rejects
mixed audit_id values. Keep separate mode/threshold trees; previously exported
CSV files without audit_id cannot establish their provenance.

Raw probabilities remain reusable across modes. If absent, inference loads the
saved ensemble and fitted preprocessing without training or fitting a schema.
New tabular runs save `data/audit_inputs.npz` with transformed held-out inputs.
Older synthetic runs use retained raw splits; older other tabular runs require
the original dataset and saved indices. Labels/groups are checked against audit
rows. Images require the original local CelebA files and torchvision. No download
is attempted on the image fallback. Neither checkpoints nor source probability
files are modified. A missing dataset/checkpoint raises its loading error;
there is no silent retraining. Legacy feature-content changes cannot be detected
from label/group equality alone; retain original data or the new input archive.

## Validation and limitations

`python -m pytest tests/test_audit_modes.py -q` covers hand calculations, all
reducer edge policies, vector/class regression, ensemble aggregation order,
ties, rate definitions, CLI/config precedence, loader wrapper propagation,
metadata/identity, plot/diagnostic reading, and real synthetic checkpoint reuse
with `PredictiveTrainer.fit` forbidden during every re-audit. It also removes
saved probabilities and exercises both transformed-input and legacy synthetic
inference, verifying unchanged checkpoint bytes.

The original repository contains no trained user checkpoints or local Adult /
CelebA datasets. Validation uses a tiny ensemble trained once on synthetic data
and existing repository fixtures. Real CelebA inference is untested here.
See `audit_validation.md` for executed suite results and baseline failures.
