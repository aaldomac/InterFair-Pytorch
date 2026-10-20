# Validation record

Base: `38b96284f8401e68442d62d6189672d96997be40`, branch `refactor-organize`.
Environment: Python 3.12, CPU PyTorch 2.14.1+cpu, pytest 9.1.1.

Executed from the checkout using `.venv/bin/python`:

| Command | Result |
|---|---|
| `python -m pytest tests/test_audit_modes.py -q` | 9 passed |
| `python -m pytest tests -q` | 48 passed, 9 failed, 1 skipped; 13 subtests passed |
| Same full-suite command in an untouched detached base worktree | 39 passed, 9 failed, 1 skipped; 13 subtests passed |
| `python -m compileall -q modules scripts tests/test_audit_modes.py` | passed |
| `git diff --check` | passed |
| `python -m scripts.audit --help` | CLI exposes both modes and all four metric choices |

The nine failing tests are identical on the untouched base and refactor:

* `test_adult_workflow.py::AdultWorkflowTests::test_adult_training_and_general_report`
* `test_classical_fairness.py::ClassicalFairnessTests::test_pipeline_save_and_report`
* `test_loan_interventions.py::LoanInterventionTests::test_report_stripe_saved_and_reload`
* `test_refactor.py::RefactorTests::test_training_saved_audit_and_reports_round_trip`
* `test_summary_aggregation.py::SummaryAggregationTests::test_cli_preserves_old_csv_and_exports_long_once`
* `test_summary_aggregation.py::SummaryAggregationTests::test_missing_policy_and_duplicate_checks`
* `test_summary_aggregation.py::SummaryAggregationTests::test_plot_raw_and_saved_summary_match`
* `test_summary_aggregation.py::SummaryAggregationTests::test_report_groups_and_empty_results`

These eight fail because `reporting_utils.aggregate` calls `mean_ci95` on
singleton or nonfinite samples; `mean_ci95` requires at least two finite samples.
The ninth failure is
`test_refactor.py::RefactorTests::test_no_regularizer_and_context_regularizer`,
which references undefined variable `zero`. These unrelated baseline problems
were deliberately left unchanged, not marked xfail or suppressed.

The skipped image test requires torchvision, which is not installed. Actual
CelebA image inference and a user's existing full-scale checkpoints are untested.
The Adult workflow uses repository fixtures, not a downloaded production dataset.

New checkpoint integration trains a tiny two-member synthetic ensemble once,
then forbids `PredictiveTrainer.fit` for all audits. It checks both mode values,
CLI/YAML/saved precedence, changing threshold, output identities and metadata,
mode-sensitive plots/diagnostic columns, rejection of cross-mode overwrites,
checkpoint byte preservation, and two inference fallbacks after deleting saved
probabilities. Numerical tests cover all documented reducer policies and
conventional binary/multiclass class-probability regression.
