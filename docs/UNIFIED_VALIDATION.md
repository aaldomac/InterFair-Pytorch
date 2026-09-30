# Single-generator validation

- 21 unittest tests passed with no skips.
- Exact array/metadata equality for all five original loan scenarios versus direct
  `generate_loan_splits` calls, with no uncertainty intervention.
- `loan_data.py`, `loan_presets.json`, and `configs/synthetic/kanubala.yaml` checked
  byte-for-byte against the previous delivered ZIP: unchanged.
- Checks cover nested scarcity, untouched held-out rows, paired label flips, oracle
  probabilities/entropies, zero-noise identity, input immutability, noise/scarcity
  combinations on loan_intersectional, and training-only stripe calibration.
- Invalid/inapplicable intervention settings are rejected. Entirely removed groups fail
  explicitly. Saved stripe diagnostics and dataset reload were checked end-to-end.
- All five synthetic YAML files validated; they define 172 jobs in total. Configuration
  validation generates each condition at its first configured seed, rather than executing
  every job.
- Five-condition smoke study completed: baseline, scarcity, noise, cancellation and stripe,
  each with two small ensemble members trained for two epochs on CPU.
- Synthetic CSV/LaTeX report completed, including stripe-region outputs.
- Loan row projections with saved-ensemble predictions and separate uncertainty scales,
  plus general condition bars, rendered successfully.

Test environment: Python 3.12, CPU torch 2.14.0+cpu, torchvision 0.29.0+cpu and the
available scientific Python runtime. A CPU feature-detection warning did not block runs.
The earlier transient PyTorch installation failed to import; a fresh local installation
was used for this validation. Dependencies and verification results are excluded from
shipping source archives.

These are software and mathematical contract tests, not evidence of scientific model
performance. Long main/pilot/combined studies and GPU training were not run. The loan
population replaces the former square distribution, so numerical equality with historical
controlled-uncertainty experiments is neither expected nor claimed.
