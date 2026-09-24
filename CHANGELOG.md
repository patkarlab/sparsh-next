# Changelog

All notable changes to SPARSH and SPARSH-next are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/). Versions
below 1.0.0 may introduce breaking changes in any minor release.

## [0.2.0.dev0] — unreleased (SPARSH-next)

Development line in a separate repository. Changes are listed against v0.1.0.

### Changed

- **Class imbalance.** Balanced batch sampler without class weights by default.
  In v0.1.0, `--upsample_minority` masked minority copies twice (sparsity became a
  class signal) and made the focal-loss weights uniform; without it, the weights and
  the sampler corrected imbalance twice. Both variants remain selectable with
  `--imbalance legacy_upsample` and `--imbalance sampler_alpha`.
- **Sparsity simulation.** Read-level simulation (`--train_sim reads`): Poisson reads
  per CpG and the methylated fraction of the reads as the value, which is what a
  1-2x ONT run reports. Coverage drawn per sample, log-uniform 2-50%
  (`--coverage_mode random`). The v0.1.0 masking and schedule remain available.
- **Missing values** stay NaN until the model's input encoding (`--input_encoding
  midpoint`, the v0.1.0 fill of 0.5, or `scaled`).
- **Nested cross-validation.** An inner validation split of each training fold drives
  the learning-rate schedule, early stopping, checkpoint choice and temperature. The
  outer fold is scored once.
- **Evaluation.** Outer folds are scored on dense arrays and on simulated ONT reads and
  masking at 3, 5, 10, 20 and 30% coverage. Corruption is seeded per Sample_ID, so all
  runs are scored on identical inputs.
- **Calibration.** Temperature scaling fitted on the inner validation split.
- **Deployment.** Fold models are saved and `scripts/predict.py` averages them, so the
  deployed predictor is the one cross-validation evaluated.
- **Grouped CV.** `--groups_file` or `--group_col`; stratified and reproducible on every
  scikit-learn version.
- **Labels.** Merges live in `configs/label_map.json`, are saved with each model and
  applied to ONT ground truth. `--exclude_classes` is exact; `--exclude_prefixes` is explicit.
- **ONT prediction** (`scripts/predict.py`, replaces `infer_and_evaluate.py`): one
  loader that rejects multi-row files, percentages, missing row names and duplicated
  probes; out-of-scheme truth labels reported and counted as errors; accuracy,
  balanced accuracy, top-2, callable share and accuracy at the threshold, by coverage.

### Added

- `scripts/check_data.py`: duplicate clusters, Source_Dataset confounding,
  platform-specific missingness, class list after the label map.
- `scripts/compare_runs.py`: runs side by side, with a comparability check.
- `jobs/settings.sh`, `jobs/train.pbs` (RECIPE=default|legacy|scaled|mask_sim|schedule),
  `jobs/check_data.pbs`, `jobs/predict.pbs`, `jobs/submit_experiments.sh`.
- `tests/smoke_test.sh`: the whole pipeline on synthetic data in about a minute.
- `docs/SETUP.md`, `docs/EXPERIMENTS.md`.
- `jobs/check_settings.sh`: checks the data paths, conda environment, free space and quota
  before anything is submitted; `jobs/submit_experiments.sh` runs it first. Jobs stop at
  once with a clear message if an input file is missing or the environment does not activate.
- `EXTRA_ARGS` in `jobs/train.pbs` passes further `scripts/train.py` options.
- Job settings and `#PBS` lines filled in from the lab's working job (`a40` queue,
  8 CPUs, 1 GPU, 48 GB, 12 hours; `module load cuda/12.3`; conda in `~/miniconda3`).
- Exclusion lists: blank lines and `#` comments are skipped and the first field of each
  line is the sample ID, so a CSV with IDs in the first column also works; unmatched IDs
  are listed with examples.

### Removed

- Optuna tuning, binary mode and feature selection (not used in the reference runs).
- Hard-coded status lines in the training log.

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
