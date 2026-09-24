# Changelog

All notable changes to SPARSH are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/). Versions
below 1.0.0 may introduce breaking changes in any minor release.

## [0.1.0] — 2026-06-15

First public release. This version consolidates prior internal development into
a single, reproducible training and evaluation pipeline for methylation-based
subtype classification under sparse (ONT-like) input.

### Added

- **Model.** `SparseNN`, a feed-forward classifier
  (`Linear → BatchNorm → GELU → Dropout` blocks + linear head) with progressive
  input masking utilities for training under sparsity.
- **Sparsity training.** Anti-curriculum progressive masking schedule
  (`mask_start` → `mask_end`) so the model learns to predict from partial CpG
  coverage.
- **Class-imbalance handling.**
  - Focal loss with per-class inverse-frequency weighting, computed per fold,
    with optional label smoothing.
  - A balanced batch sampler giving each class equal representation per
    mini-batch.
  - Optional masking-based augmentation of minority classes.
- **Leak-free cross-validation.** Feature selection and minority augmentation
  are fitted inside each fold on the training split only; validation folds hold
  real, unaugmented samples and are scored without masking.
- **Hyperparameter search.** Optuna integration optimizing mean macro-F1 across
  folds over learning rate, dropout, weight decay, focal gamma, label smoothing,
  input noise, and masking schedule.
- **Reproducibility.** Deterministic seeding (including deterministic cuBLAS via
  `CUBLAS_WORKSPACE_CONFIG`), run IDs, environment metadata capture, and saved
  fold indices.
- **Evaluation and reporting.** Per-fold and aggregate metrics, confusion
  matrices (raw and normalized), ROC and PR curves with AUC summaries, per-class
  and per-fold metric exports, predictions tagged by fold, and per-sample
  uncertainty estimates (max probability, predictive entropy, normalized
  entropy, top-2 margin). Training-curve tracking with CSV and plot output.
- **Inference.** `scripts/infer_and_evaluate.py` runs prediction on a directory
  of ONT sample CSVs and produces predictions, a coverage report, and (when
  labels are available) a classification report and confusion matrix.
- **Tooling.** CPU/GPU support, a PBS job template, `.gitignore`,
  `requirements.txt`, and this changelog.
- **License.** Released under the MIT License.

### Design notes

- **Evaluation honesty over headline accuracy.** All data-derived preprocessing
  is fitted within the training split of each fold. Synthetic/augmented samples
  never enter validation, and validation is performed under inference-matched
  (unmasked) conditions. Cross-validation estimates from this design are
  intended to reflect generalization rather than in-sample fit.
- **Imbalance is addressed at three levels.** Per-example gradient weighting
  (focal loss), per-batch class frequency (balanced sampler), and optional
  per-class augmentation are complementary and can be enabled together.
- **Sparsity as the bridge to ONT.** Rather than assuming dense array-quality
  input at inference, the model is trained under masking that emulates partial
  CpG coverage, and minority augmentation reuses the same masking mechanism.
- **Tuning targets macro-F1**, not accuracy, so rare subtypes carry weight in
  model selection under heavy imbalance.

### Known limitations

- Research use only; see the disclaimer in `README.md`. No clinical or
  diagnostic performance is claimed.
- Evaluation in the default training entry point is via stratified 5-fold
  cross-validation. A locked train/hold-out split utility is provided in
  `data/dataset.py` but is not part of the default pipeline; wire it in if you
  require a held-out test set separate from cross-validation.
- Trained model weights are not distributed in this repository.
- Label-merging rules in `normalize_and_filter_labels` are dataset-specific and
  should be reviewed before training on a new cohort.

[0.1.0]: https://semver.org/
