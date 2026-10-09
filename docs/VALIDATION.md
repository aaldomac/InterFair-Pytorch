# Validation record

Source: `refactor-organize`, commit `8fb2f6a6144974774596ec543a45cb5f011cc93f`.

- `python -m unittest discover -s tests -v`: **12 passed**, no skips.
- Two-condition synthetic smoke YAML: completed training, ensemble evaluation, uncertainty,
  fairness/distribution audits, artifact saving and completion markers.
- Kanubala YAML dry run: all 50 configured jobs validated and listed.
- Synthetic oracle report: CSV/LaTeX reporting completed from shared audit artifacts.
- General uncertainty bars: PNG generation completed.
- Four-square data and ensemble maps: PNG/PDF generation completed, with separate
  aleatoric and epistemic scales and reloaded checkpoints.
- Saved-prediction re-audit: completed without retraining.
- Python compilation: passed; pre-existing legacy docstring escape warnings remain.

The tests cover paired scarcity observations, entropy interactions, cancellation metrics,
loan/stripe generators, fixed-split isolation, train/save/reload, arbitrary group IDs,
multiclass normalization, single-model zero disagreement, local Adult/CSV inputs,
CNN training, and CelebA adapter/checkpoint integration using mocked image data.

Environment: Python 3.12, CPU PyTorch 2.14.0+cpu, torchvision 0.29.0+cpu and the available
scientific Python environment. CPU feature probing emitted a `/proc/cpuinfo` warning,
which did not prevent execution. These versions differ from the source branch's original
pinned environment; bitwise training reproducibility across environments is not claimed.

Full Adult/CelebA data downloads, CUDA, and long scientific runs were not executed.
The smoke datasets are tiny and their predictive quality is not a research result.
Generated verification outputs and installed dependencies are excluded from delivery.
