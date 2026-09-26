# Changelog

All notable changes to SPARSH and SPARSH-next are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/). Versions
below 1.0.0 may introduce breaking changes in any minor release.

## [Unreleased]

### Sixth round: label clean-up (25 September 2026)

- `jobs/prepare_relabel.pbs` takes `TAG` (default `25Sep2026`), which names the round's drop list, patient list, pickle,
  exclusion list and groups file. The job stops if any output exists, so one round cannot overwrite the files another
  round's runs use.
- `docs/EXPERIMENTS.md`: sixth round, which checks labels against independent genetics and against the methylation.
  It records the approved rule, the lists, the commands and the rule for keeping the clean-up. The lists now hold
  12 relabels and 82 samples left out, after the check of the Beat AML and TCGA labels against their sequencing.
  The clean-up is compared with the base run `locked_nsd1`, which has the same class scheme.
- `scripts/compare_class_schemes.py`:
  - a failing shared class now counts only against the new class it lost samples to (the amendment of 25 September);
  - `--allowed_samples` (default 1) sets how many samples any shared class may lose. The sixth round uses 2.
  - a round without new classes gets a keep or fail verdict.
- `docs/EXPERIMENTS.md`: fifth-round result. AML_NUP98-NSD1 is kept and T-ALL_TAL1 is renamed T-ALL_TAL1-like.
  AML_NPM1_IDH and AML_ETV6-MNX1 are merged back. The amendment is recorded as adopted after the first result.

## [0.2.0] — 2026-09-25 (SPARSH-next; strategy locked, see docs/STRATEGY.md)

Development line in a separate repository. Changes are listed against v0.1.0.

### Locked strategy (25 September 2026)

- `docs/STRATEGY.md` fixes the training and reporting strategy: the fourth-round run `dil` with AML-MR and
  AML_MECOM-r left out, now the recipe `locked` in `jobs/train.pbs`. It records the evidence (cross-validation
  on training arrays; the user's scoring of the validation cohort, for reporting only) and the rules for later
  changes.
- `docs/EXPERIMENTS.md`: the nanopore samples are the validation cohort, not a development set; the dilution
  rule no longer weights rows by properties of the validation cohort; fourth-round result added.

### Added in the fifth round (25 September 2026)

- `scripts/make_training_pickle.py`: writes a new training pickle with per-sample label changes from CSV files
  (Sample_ID,new_label,reason) and whole-class renames (`--rename OLD=NEW`), keeps the original label in
  `ANNOTATION_ORIGINAL`, never overwrites the input, and writes a change log.
- `scripts/merge_groups.py`: merges groups files (near-identical arrays) and patient lists (diagnosis and relapse
  samples, repeat arrays) into one groups file for `--groups_file`.
- `scripts/npm1_idh_check.py`: IDH1/2-mutated against wild-type NPM1 AML on training arrays (differential CpGs,
  cross-validated and cross-cohort AUC, TET2 check, signature score); with `--island` it decides which candidates
  move to a new class. Kept for reference; class membership is now decided by genotype.
- `scripts/label_audit.py` and `jobs/label_audit.pbs`: label audit of training arrays after Lamprey's label
  cleaning (agreement with the nearest neighbours in a PCA, and the cross-validation probability for the own
  label from a finished run); flags samples for review.
- `scripts/compare_class_schemes.py`: compares a run with the old labels and one with the new labels on the samples
  whose label is the same in both, and applies the fifth-round rule.
- `jobs/prepare_relabel.pbs` applies every `relabel_*.csv` in `RELABEL_DIR` and the class renames, and writes the
  exclusion list and the patient groups. Class scheme (docs/EXPERIMENTS.md): new classes AML_NPM1_IDH,
  AML_NUP98-NSD1 and AML_ETV6-MNX1; T-ALL_TAL1 renamed T-ALL_TAL1-like. `configs/class_hierarchy.json`:
  AML_NPM1_IDH and AML_NUP98-NSD1 join the HOX-related family.

### Added in the fourth round (24 September 2026)

- Dilution by normal marrow (`models/dilution.py`; `--dilution_prob`, `--blast_min`, `--normal_class`,
  `--val_dilution`, `--eval_blasts`): leukaemia array profiles are mixed with a normal-marrow array before
  the reads are simulated; normal marrows are never mixed. Training dilutes a share of samples to a random
  blast fraction with normals of the fit set, the inner validation sets likewise (once per sample, normals of
  the inner split), and the outer folds are scored at fixed blast fractions (conditions such as
  `binary-blast30_0.30`) with normals of the same outer fold, which the network has not seen. With
  dilution off, training and scoring are unchanged bit for bit. Recipes `scaled_wide_dil_base`,
  `scaled_wide_dil`.
- `scripts/platform_check.py` and `jobs/platform_check.pbs`: mean nanopore call against array beta per CpG, on
  CpGs whose array means hardly depend on class; mapping table and discordant CpGs (label-free).
- Evaluation conditions are named tuples (`training.trainer.Condition`); `evaluation.conditions.parse_condition`
  returns simulation, call error, blast fraction and coverage.

### Added in the third round (24 September 2026)

- Per-read call errors (`--call_error_max`, `--val_call_error`, `--eval_call_errors`): each read call is
  wrong with probability e, so a read is methylated with probability b(1 - 2e) + e; drawn per training
  sample, fixed for the inner validation sets, and scored as extra conditions such as `binary-err10_0.30`.
  With e = 0 the inputs, seeds and condition names are those of earlier runs. `compare_runs.py` shows the
  new rows with their own mean and cohort-expected rows.
- `configs/label_map_aml_other.json`: AML-MR and AML_MECOM-r trained as one class, `AML_other`, instead of
  being dropped. Recipes `scaled_wide_err`, `scaled_wide_aml_other`, `scaled_wide_aml_other_err`.
- Reported call (`evaluation/hierarchy.py`, `configs/class_hierarchy.json`): when no subtype reaches the
  threshold, the summed probability of a family of related subtypes, then of a lineage, is tried; background
  classes such as `AML_other` are reported as "AML, no specific subtype". `predict.py` adds the columns
  `reported_call`, `reported_level`, `family`, `lineage` and their probabilities, copies the hierarchy file
  into its output, and scores the reported call when a truth file is given (`--hierarchy`).
- `scripts/hierarchy_report.py`: the fallback on cross-validation predictions (share called and accuracy
  at each level, expected on a cohort's coverage mix), per-class outcomes and the most confused class pairs.
- `scripts/score_excluded.py` and `jobs/score_excluded.pbs`: how a run calls the training arrays of classes
  it was not trained on, under the same simulated reads as cross-validation.
- `scripts/ont_call_error.py` and `jobs/ont_call_error.pbs`: label-free per-read call error of nanopore
  samples, from CpGs that are methylated (or unmethylated) in almost every training array, with the
  simulated rate that matches it.
- `models/ensemble.py` (run loading and ensemble probabilities shared by `predict.py` and
  `score_excluded.py`) and `evaluation/conditions.py` (condition names and summary rows).
- `tests/smoke_test_round3.sh`: these additions end to end on synthetic data (about 30 seconds).

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
- `scripts/gpu_benchmark.py` and `jobs/probe_gpus.sh`: GPU model, PyTorch support and the time of one
  full-size training epoch on each GPU queue, with random numbers (no data read, nothing saved).
- `TRAIN_QUEUES` in `jobs/settings.sh`: `jobs/submit_experiments.sh` sends the runs to these queues in
  turn (the `a40` queue runs at most 2 GPU jobs per user). Data checks and ONT prediction run on the
  CPU queue `short`. `DATA_PATH` has no default; it is set when the training pickle is chosen.
- Job settings and `#PBS` lines filled in from the lab's working job (`a40` queue,
  8 CPUs, 1 GPU, 48 GB, 12 hours; `module load cuda/12.3`; conda in `~/miniconda3`).
- Exclusion lists: blank lines and `#` comments are skipped and the first field of each
  line is the sample ID, so a CSV with IDs in the first column also works; unmatched IDs
  are listed with examples.
- `scripts/compare_runs.py` shows only the simulated nanopore rows (`reads_*`) unless
  `--all_conditions` is given, and adds the share of samples callable at a fixed accuracy
  of calls (`--target_accuracy`, default 0.98) with the confidence cut-off that gives it,
  computed from the saved out-of-fold predictions and independent of calibration.
- `scripts/ont_coverage.py`: coverage of real nanopore samples on a run's CpGs (median, range,
  samples per band, implied reads per CpG, share of values exactly 0 or 1), with the loader
  checks of `predict.py`, before any prediction.
- Repeated probe IDs in ONT files (the same CpG in more than one column) are combined into the
  mean of the copies that have a value, instead of refusing the file; `--duplicate_probes first`
  keeps the first column (what a pandas reader does implicitly) and `error` refuses the file.
  The loader reports how many probes were repeated and whether their copies agree;
  `ont_coverage.py` summarises this per folder. `jobs/predict.pbs` accepts `EXTRA_ARGS`.
- Simulations `binary` (one 0/1 call per covered CpG: the majority of its reads, a random call on
  an even split) and `oneread` (the call of a single read per covered CpG), for nanopore files whose
  values are always 0 or 1. `--eval_sims` chooses the simulations scored on the outer folds (default
  `reads mask`, as before); `--coverage_dist uniform` draws the training coverage uniformly instead of
  log-uniformly; `--primary_condition auto`. The `reads` and `mask` simulations are unchanged.
- Recipes `wide`, `scaled_wide` and `scaled_wide_log` (binary values, coverage 2-95%, scored at 5-90%);
  `jobs/submit_experiments.sh` takes a list of recipes.
- `scripts/compare_runs.py --ont_coverage`: the value of each metric expected on a nanopore cohort,
  interpolated at every sample's coverage (coverage only, no labels); a mean row per simulation.

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
