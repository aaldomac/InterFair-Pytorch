# Adult with the existing shared pipeline

This is an additive update to the shared/unified-loan project. Keep your existing classical
fairness extension. No changes to its functions, training orchestration, synthetic reports,
or run summary writer are included here.

## Three commands

From your repository, activate the existing Python environment, then:

```bash
# Only needed for the default automatic Adult download:
python -m pip install kagglehub

bash scripts/train_adult.sh
bash scripts/summarize_adult.sh
bash scripts/report_adult.sh
```

The wrappers invoke the existing `scripts.run_experiment`, existing `scripts.summarize`,
and the new `scripts.report_real` respectively. They locate the repository root themselves.
Use PYTHON_BIN to select a different interpreter if necessary. No chmod is required when
invoking the scripts with bash.

Equivalent Python commands:

```bash
python -m scripts.run_experiment --config configs/adult.yaml
python -m scripts.summarize experiments/adult
python -m scripts.report_real experiments/adult
```

The default configuration trains 5 ensemble members for each of 3 split seeds: 15 models.
It uses the existing Adult Kaggle loader, MLP, trainer, ensemble evaluator, uncertainty audit,
and classical metrics when the previously delivered classical extension is installed.
Model seeds and split seeds remain separate. Existing run folders are not overwritten.

## Local CSV or custom experiment

Uncomment `dataset.kwargs.csv_path` in configs/adult.yaml to use an existing CSV. The CSV
must have headers; raw headerless UCI adult.data/adult.test files need conversion first.
The loader accepts `gender` or its `sex` alias, strips surrounding whitespace, recognizes
`?` as missing, and accepts income labels with trailing periods. Default drop_na removes
incomplete rows. Keep drop_na=true unless you add a missing-value strategy for your study.

```bash
bash scripts/train_adult.sh configs/adult.yaml --dry-run
bash scripts/train_adult.sh configs/adult.yaml --seed 0

# For a separate YAML whose output is experiments/adult_v2:
bash scripts/train_adult.sh configs/adult_v2.yaml
bash scripts/summarize_adult.sh experiments/adult_v2
bash scripts/report_adult.sh experiments/adult_v2
```

Dry-run validates configuration and job enumeration; it does not download the full Adult
source or prove that every source row is valid.

The supplied positive_label is `<=50K`, preserving your original class-1 coding.
If TPR should refer to high income, explicitly set `positive_label: '>50K'` before training
and choose a fresh output directory. This changes label semantics and requires retraining.
The general report records the positive-label convention and rejects mixed conventions
within one report. Older runs lacking the setting are marked unspecified.

All three existing protected attributes are retained: gender, race, native-country
(the latter is US vs non-US). With append_protected_to_predictor=false they are excluded
from predictors, but used for intersectional groups and audits. No interaction_weights
are configured: F_U_int and signed two-attribute contrasts cannot be inferred for these
arbitrary multi-category groups. F_alea, F_epis, F_tot, F_U and hidden disparity still apply.
Oracle entropy is unknown for real observations; the report does not request it.

## Small-group policy and saved splits

Adult intersectional groups can be small. The supplied YAML opts into:

```yaml
pipeline:
  split:
    test_size: 0.25
    val_size: 0.15
    rare_group_policy: train_only
```

This performs reproducible group-stratified allocation. Groups of at least 3 rows receive
at least one training, one validation and one test row. Test counts approximate 25% within
each group; validation counts approximate 15% of the remaining rows. Overall fractions may
change slightly because of rounding/minimums. Groups of 1-2 rows stay in training only;
they are not silently merged or treated as having zero test unfairness. This policy does
not stratify on the label; conditional rates such as TPR can be undefined in a test group.
Their supports must be inspected before interpreting fairness.

Set rare_group_policy to null (or omit it) to retain the existing sklearn splitting
algorithm, which may reject very rare groups. This opt-in does not change fixed synthetic
or official image partitions.

A correction to the shared splitter preserves preprocessed dataset row indices instead
of resetting each partition to 0..N. Saved train/validation/test indices now identify the
original preprocessed dataset rows and remain disjoint. This does not repair older saved
split indices retroactively. Retain the same CSV/preprocessing to reconstruct source rows.

Every new run exports audit/group_coverage.csv: group names/IDs, train/validation/test counts,
positive-class counts and whether the group was evaluated. The general report combines it.
Audit maxima refer only to observed test groups, never a guarantee over unobserved groups.

## Reports and figures

The existing synthetic-specific scripts.report is retained for synthetic oracle tables.
Use scripts.report_real for Adult or any dataset with the shared artifact contract.
It also works without synthetic metadata and without PyTorch for reading saved reports.

Outputs include:

- all_runs.csv and by_condition.csv from the unchanged summary command.
- report/report.txt: run-level metrics and interpretation notes.
- report/runs.csv, runs_stats.csv, runs.tex: per-run and aggregate metrics, including any
  classical summary fields saved by your previous extension.
- report/groups.csv, groups_stats.csv, groups.tex: observed group quality, uncertainties,
  and performance rates when performance_fairness.json is available.
- report/pairs.csv, pairs_stats.csv, pairs.tex: pairwise uncertainty disparities.
- report/group_coverage.csv: all known group/run combinations, including groups absent from test.

Statistics include valid-run counts (`*_n`), mean and sample SD; missing values are not zero.
Without the classical extension the reader can still extract existing outcome DF/SP/DI/EO
scores from fairness.json. The extension is needed for added TPR/accuracy log-ratios and
subgroup confusion-count tables. To enrich old completed runs, your previous command remains:

```bash
python -m scripts.classical_report experiments/adult
bash scripts/summarize_adult.sh
bash scripts/report_adult.sh
```

Existing bar plotting accepts the resulting summary columns:

```bash
python -m scripts.plot_uncertainty_bars --input experiments/adult \
  --metrics F_alea F_epis F_tot F_U \
  --out experiments/adult/figures/uncertainty

# With the classical extension installed:
python -m scripts.plot_uncertainty_bars --input experiments/adult \
  --metrics DF_epsilon TPR_epsilon Accuracy_epsilon \
  --out experiments/adult/figures/classical
```

Undefined requested rates are rejected by the existing plotter; inspect subgroup supports
rather than replacing missing measurements with zero.

## Manual installation

Copy the new files and merge the small edits in the existing files. The archive contains
ONLY this add-on, not a complete repository, so it does not overwrite your classical
reporting implementation. Modified files:

1. modules/data/adult.py: robust CSV cleaning, explicit positive label, metadata.
2. modules/utils/dataset_utils.py: pandas string dtype recognition, optional rare-group
   allocation, preservation of original split indices.
3. modules/pipelines/train_predictive_pipeline.py: one SplitConfig field and forwarding it.
4. modules/pipelines/experiment_config.py: accepts the new split field.
5. modules/utils/saving_utils.py: writes group coverage beside existing audits.
6. configs/adult.yaml: explicit settings for Adult, labels and the small-group policy.

New files: modules/reporting/real_report.py, scripts/report_real.py, the three Bash wrappers,
tests/test_adult_workflow.py, and this document. An optional patch is included for convenience;
manual file editing requires no Git commit.

Validation uses an Adult-format CSV fixture with numerical/categorical predictors and a
rare group: loading, training, auditing, saving, source-index recovery and report generation.
This validates software integration, not results on the full Adult population. Full data
downloads, long scientific training and GPU execution were not performed for this update.
