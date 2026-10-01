# One generator, two independent scenario axes

The only covariate/clean-label generator is now `modules/data/loan_data.py`.
Its original implementation and `loan_presets.json` are unchanged.
`synthetic_uncertainty.py` is the shared pipeline adapter; it contains no alternate
four-square generator. `uncertainty_conditions.py` applies row-aligned interventions.

## Configuration

Keep discrimination in `scenario` and uncertainty in a separate mapping:

```yaml
dataset:
  name: synthetic_uncertainty
  kwargs:
    scenario: loan_no_bias
    train: 8000
    validation: 2000
    audit: 20000
    reference: 100000
conditions:
  baseline:
    uncertainty: {kind: baseline, baseline_noise: 0.05}
  scarcity:
    uncertainty: {kind: scarcity, rho: 0.25, baseline_noise: 0.05}
  noise:
    uncertainty: {kind: noise, eta: 0.30, baseline_noise: 0.05}
  cancellation:
    uncertainty: {kind: cancellation, eta: 0.30, rho: 0.125}
  stripe:
    uncertainty:
      kind: stripe
      rho: 0.5
      features: [Income, LoanAmount]
      slope: -0.5
      validation: true
```

This is the dataset/conditions section of a full experiment YAML. The existing runner,
model, optimizer, audit settings and result layout remain in use. Full runnable examples
are in `configs/synthetic/`.

The two axes are deliberately distinct:

| Axis | Options | Meaning |
|---|---|---|
| `scenario` | `loan_no_bias`, `loan_single`, `loan_additive`, `loan_intersectional`, `loan_compounded` | Original loan structural parameters and discrimination effects |
| `uncertainty.kind` | `baseline`, `scarcity`, `noise`, `cancellation`, `additive`, `interaction`, `stripe`, `custom` | Intervention applied after clean loan generation |

`scenario: loan_additive` is NOT the same experiment as
`uncertainty: {kind: additive}`. The latter preserves the old additive pattern of **label
entropy**, not the loan generator's additive discrimination mechanism.

The runner merges each condition over `dataset.kwargs` at the top level. Each condition's
`uncertainty` mapping replaces that whole mapping; it is not recursively merged.

## Exact intervention definitions

Let `y_clean` be the original deterministic loan label, and let `q_g` be a group-specific
symmetric flip probability. For every split:

```text
y = y_clean XOR Bernoulli(q_g)
p_true = q_g + (1 - 2*q_g) * y_clean
U_true = binary_entropy(p_true)      # natural logarithms
```

The sigmoid score returned by the original loan generator is not treated as a Bernoulli
probability. Its labels were thresholded deterministically. Likewise, the upstream
`loan_overrides.eta` controls draws unused in the original label rule; use
`uncertainty.eta` to introduce the label noise defined here.

Group order is `00, 01, 10, 11` = `(Gender, Race)` in upstream numeric coding.

| Intervention | Label noise probabilities | Row changes |
|---|---|---|
| Omitted `uncertainty` or `null` | Original deterministic labels | None; exact original loan arrays and metadata |
| `baseline` | 0.05 in all groups by default | None |
| `scarcity` | Background noise, default 0.05 | Retain rho of training group 11 |
| `noise` | Group 11 gets eta; others background rate | None |
| `cancellation` | Group 00 gets eta; others background rate | Retain rho of training group 11 |
| `additive` | Entropies in bits: entropy_base + strength*[0,1,1,2] | None |
| `interaction` | Entropies in bits: entropy_base + strength*[0,0,0,1] | None |
| `stripe` | Background noise | Remove a training support region of group 11; optionally same region from validation |
| `custom` | Explicit group flip mapping over background rate | Explicit group training-retention mapping |

`baseline_noise` is configurable in [0, 0.5]; set it to 0 for a clean-label baseline.
The original uncertainty study's 0.05 background is preserved explicitly in migrated
configs. Additive/interaction entropy patterns instead use `entropy_base` (default 0.30
bits) and `strength`; their entropy targets must stay in [0,1].

`target_group` changes the affected group for scarcity, noise or stripe. Cancellation
uses `target_group` for scarcity (default 11) and `noise_group` for label noise (default 00).
Symmetric noise is applied to train, validation, audit and reference: this represents a
changed outcome distribution, rather than training-only annotation corruption.

Scarcity is group-dependent random subsampling of training rows, retaining
`max(1, floor(group_count*rho))`. Audit, validation and reference rows stay unchanged.
The permutations and label-flip uniforms are paired across conditions with the same seed
and split sizes. Thus smaller rho values give nested subsets, and larger q values give
nested label flips. Increasing split sizes is not guaranteed to preserve row identities.

## Stripe adaptation

The old stripe relied on bounded uniform coordinates in four disjoint squares. That
geometry and its analytic retention formula do not apply to loan covariates.

The replacement fits means and standard deviations on **unmodified training covariates
only**, using two configurable continuous loan features (default Income and LoanAmount).
The removed region is:

```text
abs(z_feature_1 - slope*z_feature_2 - center) < half_width
```

Provide either `rho` or `half_width`, never both. With `rho`, a target-group training
order statistic chooses the width to approximate the desired retained fraction; boundary
ties are retained. With `half_width`, the width is in standardized projection units.
The same fitted geometry is frozen for validation and audit diagnostics. No test or
reference information calibrates the intervention. Validation filtering defaults to true,
matching the old stripe protocol; set `validation: false` for training-only scarcity.

Metadata stores coordinates, fitted mean/scale, threshold, requested retention and
realized per-split retention. Population `expected_retention` is null: the old uniform
population formula would be incorrect for loan data. Empty post-intervention groups
are rejected rather than silently removing an audited group.

## Combining fairness and uncertainty

```yaml
conditions:
  intersectional_original:
    scenario: loan_intersectional
    uncertainty: null
  intersectional_noise:
    scenario: loan_intersectional
    uncertainty: {kind: noise, eta: 0.30, baseline_noise: 0.0}
  intersectional_noise_scarcity:
    scenario: loan_intersectional
    uncertainty:
      kind: custom
      baseline_noise: 0.0
      flip_probability: {'11': 0.30}
      retention: {'11': 0.25}
```

The original covariates and deterministic outcome rule stay intact; each intervention
changes only observed labels and/or selected rows. `y_clean`, `p_true`, `U_true`, split
labels and row IDs remain excluded from predictors by the existing preprocessing schema.
The original seven predictors still include Gender and Race, as before.

## Run and report

```bash
# Original discrimination experiments, unchanged configuration/data
python -m scripts.run_experiment --config configs/synthetic/kanubala.yaml

# Small end-to-end check for the migrated uncertainty experiments
python -m scripts.run_experiment --config configs/synthetic/smoke.yaml
python -m scripts.report experiments/loan_uncertainty_smoke
python -m scripts.plot_uncertainty_bars --input experiments/loan_uncertainty_smoke --out experiments/loan_uncertainty_smoke/figures/bars
python -m scripts.plot_synthetic --run experiments/loan_uncertainty_smoke/runs/noise/seed_0 --split audit --ensemble

# Longer uncertainty-only or combined studies
python -m scripts.run_experiment --config configs/synthetic/main.yaml --dry-run
python -m scripts.run_experiment --config configs/synthetic/combined.yaml --dry-run
```

Plot colors are evaluated on full original predictor vectors. The horizontal and vertical
axes project observed loan rows onto two selected features, with panels for the four
groups. They are not model evaluations on an artificial 2D grid. Aleatoric and epistemic
plots have independent scales. All saved rows are evaluated; max-points only limits
rendered points.

## Migration and interpretation

- All sizes now mean TOTAL rows before filtering. Migrated uncertainty YAMLs multiply
  previous per-group counts by four to retain the same nominal sample totals. Groups are
  sampled with the loan generator's probabilities, not forced to equal counts.
- `kanubala.yaml`, `loan_data.py` and `loan_presets.json` are byte-for-byte unchanged.
- Legacy names such as `scenario='scarcity', rho=.25` still map to loan_no_bias with that
  intervention, with a warning about total-count semantics. Prefer explicit YAML axes.
- Existing saved four-square data can still be read by the dataset adapter, but new
  generation always uses loan data. Historical square plots require the previous plotter.
- Baseline label entropy is equal across groups when background rates are equal, but
  learned epistemic uncertainty need not be equal. Group counts and feature distributions
  come from the loan process.
- Scarcity is designed to probe epistemic effects; noise changes conditional label entropy.
  Neither guarantees the model's learned decomposition will behave ideally. In particular,
  cancellation is a hypothesis to inspect with the audit, not a guaranteed outcome.
- These studies are not numerical reproductions of the old square-data experiments.
  Do not pool their results under the same condition name or reuse old model checkpoints.

Apply `loan-unification.patch` on top of the previous delivered refactor. The full project
ZIP also includes a cumulative patch for the original source commit. Never apply both.
