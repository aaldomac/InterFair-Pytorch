# Experiment workflow wrappers

Place all four .sh files together in bash_scripts/. Run them with Bash;
they locate and switch to the repository root automatically. Activate your
environment first, or set PYTHON=/absolute/path/to/python.

Each wrapper accepts --help and --dry-run. Default dataset: kanubala; modes:
both; formats: png,pdf; errorbars: ci95. Lists use commas without spaces.

## Typical commands

bash bash_scripts/audit.sh --datasets kanubala,adult_age,compas --modes both
bash bash_scripts/audit.sh --datasets kanubala --source experiments/kanubala_no_protected --tag no_protected
bash bash_scripts/bars.sh --datasets kanubala --modes soft --preset prediction
bash bash_scripts/bars.sh --datasets kanubala --modes soft --preset interaction
bash bash_scripts/bars.sh --datasets adult_age,compas --preset epsilon
bash bash_scripts/bars.sh --datasets adult_age --metrics DF_epsilon,TPR_epsilon,Accuracy_epsilon --name epsilon_without_fpr
bash bash_scripts/diagnose.sh --datasets kanubala --modes hard --class0 APROVED --class1 "NOT APROVED"
bash bash_scripts/diagnose.sh --datasets all --modes both
bash bash_scripts/audit.sh --datasets kanubala --existing refresh
bash bash_scripts/audit.sh --datasets kanubala --tag threshold07 -- --threshold 0.7
bash bash_scripts/bars.sh --datasets kanubala --tag threshold07 -- --conditions loan_compounded
bash bash_scripts/diagnose.sh --datasets kanubala --tag threshold07 -- --high-confidence 0.95

Run --dry-run first to inspect resolved paths. --datasets all includes all five
registered names; missing inputs cause an error rather than being skipped.
Edit the mappings in common.sh once to match your local training directories:
kanubala -> experiments/kanubala; adult -> experiments/adult;
adult_age -> experiments/adult_age_25_60; compas -> experiments/compas;
compas_black_white -> experiments/compas_black_white.
--source replaces this mapping for one dataset, audit only.

Audits use audits/<dataset>_<mode>[_tag]. Bar plots use
figures/<dataset>/<name>_<mode>[_tag]. Diagnoses use
figures/<dataset>_diagnostics_<mode>[_tag][_name].
The optional --name distinguishes plot selections or diagnostic variants.
A common --tag must be passed to downstream wrappers to read tagged audits.

## Existing outputs and policy

Auditing defaults to error on an existing destination. --existing skip skips it;
--existing refresh uses that destination as BOTH audit source and output with
--overwrite, preserving original prediction routing. Refresh does not pull new
runs from a different training source. Use a fresh --tag to change settings,
thresholds, smoothing, disparity metric, training source, or label convention;
the Python audit identity checks still apply.
Plot and diagnosis commands reuse their output paths. Use --name or --tag to
preserve previous variants. Scripts stop on the first failed Python command.

Arguments after -- are forwarded unchanged to the underlying CLI. Output/source
and mode overrides are blocked to prevent cross-mode overwrites. Use wrapper
options for these. Other Python arguments such as --config, --threshold,
--contrasts, --conditions, --figsize and --high-confidence remain available
where supported by the underlying command.

## Metric and label semantics

prediction preset: F_tot F_alea F_epis F_hidden_raw
interaction preset: I_het I_alea I_epis F_hidden
epsilon preset: DF_epsilon TPR_epsilon FPR_epsilon Accuracy_epsilon

These presets reproduce the user's commands. Raw hidden and interaction columns
require the corresponding metric changes and regenerated audits. No fallback
silently substitutes another metric. The all-races COMPAS setup has no binary
interaction contrast. Nonfinite FPR disparity still needs explicit treatment or
omission; the wrappers do not discard undefined results.

Diagnostic --metrics is forwarded to the current Python script: it controls
selected metrics/contrasts, not the script's hardcoded standard panel lists.

Class labels are display strings only; they never invert labels/probabilities.
Adult defaults assume class 1 means income <=50K. COMPAS defaults assume class 1
means recorded recidivism. Kanubala defaults use Favorable (0)/Unfavorable (1),
matching the user's chosen convention for newly trained swapped-label runs.
Verify those semantics for older runs. Override with --class0/--class1 as needed.

Validation: Bash syntax, command previews for all datasets/modes, and a mock
Python argument-capture execution passed. Real audits and plotting were not run.
