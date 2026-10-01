# Classical subgroup fairness alongside uncertainty disparities

This update reuses `modules/metrics/fairness_metrics.py`. The existing outcome differential
fairness, statistical parity, disparate impact, equal opportunity, equalized odds and
subgroup statistical parity implementations remain the source of those metrics. New
performance-rate comparisons use the existing `_smoothed_positive_rate` helper. Training,
data generation and model architectures are unchanged.

## Report names and definitions

| Column | Definition | Ideal |
|---|---|---|
| `DF_epsilon` | Existing binary outcome DF: maximum absolute log-ratio across groups and both predicted outcomes | 0 |
| `SP_gap` | Maximum absolute predicted-positive-rate difference | 0 |
| `DI_ratio` | Minimum pairwise min(selection rate)/max(selection rate), existing zero-rate convention | 1 |
| `EO_gap` | Maximum absolute TPR difference | 0 |
| `EOdds_gap` | Maximum of subgroup TPR and FPR differences | 0 |
| `SSP_gap` | Existing weighted subgroup statistical-parity aggregate | 0 |
| `TPR_epsilon` | Maximum absolute log-ratio of smoothed TPRs | 0 |
| `Accuracy_epsilon` | Maximum absolute log-ratio of smoothed subgroup accuracies | 0 |
| `FPR_epsilon`, `TNR_epsilon`, `PPV_epsilon`, `Selection_epsilon` | Same log-ratio definition for the named rate | 0 |

The added rates also expose `_gap` (raw absolute rate disparity) and `_max_ratio`
(the largest smoothed ratio over both orderings of each subgroup pair).

For a rate with k successes among n eligible observations, smoothing is exactly the
existing Beta-style formula `(k + alpha)/(n + 2*alpha)`, with alpha=1 in the shared
pipeline and post-hoc command. Supports are:

- TPR: actual positives; success is predicted positive.
- FPR/TNR: actual negatives; success is predicted positive/negative respectively.
- PPV: predicted positives; success is actual positive.
- Accuracy: all subgroup observations; success is a correct prediction.
- Selection: all subgroup observations; success is predicted positive.

The epsilon convention is `exp(-epsilon) <= rate(g)/rate(h) <= exp(epsilon)` for every
pair. Thus a maximum rate ratio of 2 corresponds to epsilon=log(2), not epsilon=2.
The rate-only TPR/accuracy epsilons are labelled separately from classical outcome DF:
outcome DF considers both predicted-positive and predicted-negative probabilities.
Accuracy_epsilon alone does not also compare error rates; TPR_epsilon alone does not
also compare false-negative rates.

Null handling is intentional. If a subgroup has no eligible observations (e.g. no actual
positives for TPR), its raw and smoothed conditional rate are null. A worst-pair aggregate
is null if any observed group lacks support, rather than certifying fairness from only
the supported groups. Valid-run counts appear as `*_n` in report statistics. Pair tables
still retain comparisons with support. Plotting rejects missing requested scores; inspect
groups.csv and choose a supported metric/condition rather than replacing missing values
with zero.

These are empirical smoothed scores, not population fairness certificates. Smoothing can
matter substantially for small subgroups. Epsilon/gaps are lower-is-better; DI_ratio is
higher-is-better. Good fairness does not imply good predictive accuracy, and disparate
metric scales should not be read as interchangeable measures of magnitude.

## Already trained models: no retraining

Run from the repository root, after applying `classical-fairness.patch` to the previously
delivered unified-loan version:

```bash
python -m scripts.classical_report experiments/kanubala
python -m scripts.summarize experiments/kanubala
python -m scripts.report experiments/kanubala --pairs

python -m scripts.plot_uncertainty_bars \
  --input experiments/kanubala \
  --metrics DF_epsilon TPR_epsilon Accuracy_epsilon \
  --errorbars sd --formats png pdf \
  --out experiments/kanubala/figures/classical_epsilons

python -m scripts.plot_uncertainty_bars \
  --input experiments/kanubala \
  --metrics SP_gap EO_gap EOdds_gap Accuracy_gap \
  --errorbars sd --formats png pdf \
  --out experiments/kanubala/figures/classical_gaps
```

The post-hoc command reads `results/ensemble_*.npy`, `audit/audit_rows.npz` and the saved
`config.json`. It never loads a training dataset or calls the trainer. It updates only
classical audit outputs and classical columns of each completed run's `summary.csv`,
retaining uncertainty columns and run identity. Existing `audit/fairness.json` is recomputed
with the saved threshold. It can be rerun to refresh these derived outputs. The summary
and report commands then refresh aggregate tables.

## Future training

Keep the existing `audit.fairness: true` setting (already present in kanubala.yaml).
The new scores are automatically saved in each run's `summary.csv`, plus:

- `audit/fairness.json`: the existing full fairness result, including pairwise matrices.
- `audit/performance_fairness.json`: raw subgroup confusion counts, raw/smoothed rates,
  supports, pairwise gaps/epsilons/ratios and computation settings.
- `audit/classical_summary.json`: flattened classical scores.

No YAML migration is required. `scripts.report` now adds classical mean/SD tables alongside
uncertainty tables whenever these files exist, and exports classical details too.

## Outputs

- `report/runs.csv`: uncertainty and classical metrics together, one row per condition/seed.
- `report/classical_stats.csv` and `report/classical.tex`: classical mean/sample-SD tables.
- `classical_report/runs.csv`: per-run classical values.
- `classical_report/groups.csv`: TP/FN/FP/TN, supports, raw and smoothed subgroup rates.
- `classical_report/pairs.csv`: pairwise performance comparisons and existing DF/SP/DI/EO/EOdds values.
- `classical_report/classical_stats.csv`, `classical.tex`, `report.txt`: aggregated reports.
- `all_runs.csv`, `by_condition.csv`: updated by scripts.summarize; include classical columns.

Worst-pair disparities are computed within each ensemble run before aggregating across
seeds. Sample SD is across runs, not across ensemble members or individual audit rows.
The original loan class-1 coding is preserved; do not relabel it as approval without
checking the dataset's label convention. In this dataset IDs 0/1/2/3 correspond to
Gender/Race groups 00/01/10/11.

## Small correctness fix and validation

The original evaluator reused stored argmax labels even if a different binary threshold
was requested. It now recomputes binary predictions from the requested threshold, so EO
and EOdds agree with the threshold used for DF/SP and the new performance metrics.

Focused tests cover hand-calculated TPR ratios, accuracy parity, exact reuse of existing
DF, non-default thresholds, undefined conditional supports, saving, and report integration.
Post-hoc reporting and plotting were also exercised on the five completed smoke runs.
Full scientific training is not required for this reporting update.

Validation result: all 25 tests passed; post-hoc updates, standard reports, summaries and
classical epsilon bar plots completed for five existing smoke runs. CPU validation only.
