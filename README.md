# SPARSH

**S**ubtype **P**rediction via **A**daptive **R**ecognition from **S**parse **H**ematological Data

SPARSH is a deep-learning classifier for hematologic malignancy subtypes (WHO
classification) from DNA-methylation profiles. It is designed to remain accurate
when the input is **sparse** — for example, the partial CpG coverage produced by
Oxford Nanopore (ONT) sequencing — by training under simulated sparsity rather
than assuming dense array-quality input at inference time.

> **Version 0.1.0** — first public release. This is pre-1.0 research software:
> interfaces, defaults, and behavior may change between minor versions.

---

## ⚠️ Research use only

**This software is for research purposes only. It is not a medical device, has
not been cleared or approved by any regulatory authority, and must not be used
for clinical decision-making, diagnosis, or patient management.** Model outputs
are predictions from a statistical model and require independent confirmation by
validated orthogonal methods and qualified professionals. No clinical or
diagnostic performance is claimed.

No patient data is included in this repository.

---

## What it does

Given DNA-methylation beta values, SPARSH predicts a malignancy subtype and
returns calibrated class probabilities and per-sample uncertainty estimates.
The implementation focuses on two recurring difficulties in this setting:

- **Severe class imbalance** across many subtypes, and
- **A domain gap** between dense training data (methylation arrays) and sparse
  inference data (ONT).

## How it works

- **Model.** A feed-forward network (`Linear → BatchNorm → GELU → Dropout`
  blocks followed by a linear classifier; Xavier initialization). See
  `models/sparse_nn.py`.
- **Sparsity training (adaptive recognition).** During training, a large,
  scheduled fraction of CpG inputs is masked to an uninformative value
  (anti-curriculum schedule from `mask_start` to `mask_end`), so the model
  learns to predict from whatever subset of CpGs is present at inference.
- **Class imbalance.** Training uses focal loss with per-class
  inverse-frequency weighting (computed per fold), a balanced batch sampler that
  gives each class equal representation per mini-batch, and optional masking-based
  augmentation of minority classes.
- **Leak-free evaluation.** All preprocessing that learns from data — feature
  selection and minority augmentation — is fitted **inside each
  cross-validation fold on the training split only**; validation folds contain
  real, unaugmented samples and are scored **without masking**, matching
  inference conditions.
- **Reproducibility.** Deterministic seeding (including a deterministic cuBLAS
  configuration), run IDs, environment capture, and saved fold indices. See
  `training/reproducibility.py`.
- **Reporting.** Per-fold and aggregate metrics, confusion matrices, ROC/PR
  curves, per-class metrics, training curves, and per-sample uncertainty
  (max probability, predictive entropy, top-2 margin). See `evaluation/metrics.py`.

## Repository structure

```
.
├── data/
│   ├── __init__.py
│   └── dataset.py            # loading, label normalization, augmentation, feature selection
├── models/
│   ├── __init__.py
│   └── sparse_nn.py          # SparseNN model + progressive masking
├── training/
│   ├── __init__.py
│   ├── trainer.py            # focal loss, balanced sampler, leak-free cross-validation
│   ├── tuning.py             # Optuna hyperparameter search (macro-F1 objective)
│   ├── reproducibility.py    # determinism, run IDs, environment metadata
│   └── training_curves.py    # per-epoch curve tracking and plots
├── evaluation/
│   ├── __init__.py
│   └── metrics.py            # metrics + CSV/JSON exporters
├── scripts/
│   ├── train.py              # training entry point
│   └── infer_and_evaluate.py # inference on ONT samples + evaluation
├── jobs/
│   └── train.pbs             # PBS job template (no site-specific paths)
├── requirements.txt
├── VERSION
├── CHANGELOG.md
└── README.md
```

## Installation

Python 3.10+ is recommended.

```bash
git clone <your-repo-url> sparsh
cd sparsh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA toolkit
(https://pytorch.org/get-started/locally/). The reference job uses CUDA 12.3.

## Data format

Training data is a pickled pandas DataFrame (`pd.read_pickle`) with:

- a `Sample_ID` column (falls back to the DataFrame index if absent);
- one column per CpG probe, named with the `cg...` prefix, holding beta values
  in `[0, 1]`;
- an `ANNOTATION` column with the subtype label per sample;
- an optional `Source_Dataset` column (batch information; **not** used in
  training).

An optional `--junk_path` text file lists sample IDs to exclude. Label cleanup
(e.g. merging closely related subtypes) is applied in
`data/dataset.py::normalize_and_filter_labels`; review it before training on a
new dataset.

## Usage

### Train (with cross-validation)

```bash
python scripts/train.py \
  --data_path /path/to/methylation.pkl \
  --output_dir ./outputs/exp1 \
  --epochs 300 \
  --upsample_minority
```

Key options: `--hidden_dims`, `--dropout`, `--focal_gamma`, `--label_smoothing`,
`--samples_per_class_per_batch`, `--mask_start`, `--mask_end`,
`--exclude_classes`, `--min_samples`, `--n_features` (0 = use all CpGs),
`--binary_features`. Run `python scripts/train.py --help` for the full list.

### Hyperparameter search

```bash
python scripts/train.py --data_path /path/to/methylation.pkl \
  --output_dir ./outputs/tuned --tune --n_trials 50
```

Optuna maximizes mean macro-F1 across folds and searches learning rate, dropout,
weight decay, focal gamma, label smoothing, noise, and masking schedule.

### Inference on ONT samples

```bash
python scripts/infer_and_evaluate.py \
  --model_dir ./outputs/exp1 \
  --ont_dir   /path/to/ont_csvs \
  --output_dir ./outputs/exp1_inference
```

`--model_dir` must contain the trained `config.json`, `class_mapping.json`, and
`selected_cpgs.json`. `--ont_dir` is a directory of per-sample CSV files; see
`scripts/infer_and_evaluate.py` for the expected columns. Outputs include
`predictions.csv`, a coverage report, and (when labels are available) a
classification report and confusion matrix.

### Cluster (PBS)

`jobs/train.pbs` is a template. Set `DATA_PATH`, `OUTPUT_DIR`, and `CONDA_ENV`
(or edit the file) and submit with `qsub`. It exports
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, which is required for deterministic cuBLAS.

## Outputs

A training run writes the model weights, `config.json`, fold indices, training
log, training curves, and a full evaluation suite (metrics, confusion matrices,
ROC/PR data, per-class and per-fold CSVs, predictions with fold IDs) to the
output directory. Model weights are **not** tracked in git; distribute them via
releases or Git LFS.

## Reproducibility

Runs are seeded and record an environment snapshot and a run ID. Results can
still vary across hardware, driver, and library versions; pin your environment
for exact reproduction.

## Versioning

This project follows [Semantic Versioning](https://semver.org/). Versions below
1.0.0 may introduce breaking changes in any minor release.

## License

Released under the [MIT License](LICENSE). Replace the copyright holder in
`LICENSE` to match your institution's policy if needed (the default is
"The SPARSH Authors").

A license only grants rights to people who actually receive the code. Keeping
this repository **private** shares nothing publicly, regardless of the license.
Confirm your institution's IP and publication policy before making it public.

## Acknowledgements

**TODO:** funding, institutional, and collaborator acknowledgements.

## Citation

**TODO:** add a `CITATION.cff` or preferred citation once available.
