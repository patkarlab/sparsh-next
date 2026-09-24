# SPARSH-next

Development line of SPARSH (**S**ubtype **P**rediction via **A**daptive **R**ecognition from **S**parse **H**ematological data): a feed-forward classifier trained on Illumina methylation arrays and applied to low-coverage Oxford Nanopore (ONT) methylation calls.

The first commit of this repository is SPARSH v0.1.0 (patkarlab/sparsh, commit 02dbb97) unchanged. Every later commit is a readable diff against it.

> **Research use only.** This software is not a medical device, has not been cleared or approved by any regulatory authority, and must not be used for clinical decision-making. Model outputs require confirmation by validated orthogonal methods.

## What changed from v0.1.0

| Area | v0.1.0 | SPARSH-next |
| --- | --- | --- |
| Class imbalance | Balanced sampler plus inverse-frequency focal weights. `--upsample_minority` adds pre-masked copies, which masks rare classes twice and sets every weight to 1.0 | Balanced sampler only (`--imbalance sampler`); the v0.1.0 variants stay selectable for comparison |
| Sparsity in training | IID masking; observed CpGs keep their array beta; one coverage per epoch, from 3% to 20% | Read-level simulation: Poisson reads per CpG, value = methylated fraction of the reads; coverage drawn per sample, 2–50% (`--train_sim`, `--coverage_mode`) |
| Missing values | Filled with 0.5 at load time (0 in binary mode) | Kept as NaN until the model's input encoding (`midpoint` = 0.5, or `scaled`) |
| Model selection | Early stopping, LR schedule and checkpoint chosen on the fold that is reported | An inner validation split inside each training fold makes these choices; the outer fold is scored once |
| Reported CV | Dense arrays only | Dense arrays plus simulated ONT reads and masking at 3–30% coverage, on identical inputs in every run |
| Probabilities | Uncalibrated (the README said calibrated) | Temperature scaling fitted on the inner validation split |
| Deployed model | A final model trained with a different schedule | The fold ensemble that CV evaluated; `--final_model` optionally adds one model trained with the same recipe |
| Related samples | Not handled | `--groups_file` or `--group_col` for grouped CV; `scripts/check_data.py` finds duplicate clusters |
| Labels | Merges hard-coded; class exclusion by substring | `configs/label_map.json`, saved with each model and applied to ONT truth files; exact or prefix exclusion |
| ONT evaluation | Samples with unmatched labels dropped silently; macro metrics only | Out-of-scheme samples reported and counted as errors; accuracy, balanced accuracy, top-2, callable share and accuracy at the threshold, by coverage bin |
| Removed | | Optuna tuning, binary mode and feature selection (not used) |

## Quick start

Full server setup, step by step: [docs/SETUP.md](docs/SETUP.md). Which runs to do and how to read them: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

```bash
bash tests/smoke_test.sh                  # about 1 minute on CPU; must print SMOKE TEST PASSED
bash jobs/check_settings.sh               # paths, environment, disk; must print SETTINGS OK
bash jobs/probe_gpus.sh                   # GPU model and epoch time on each GPU queue (random numbers only)
qsub jobs/check_data.pbs                  # class list, duplicates, study confounding
bash jobs/submit_experiments.sh           # five comparison runs, one GPU job each
python scripts/compare_runs.py ~/sparsh_next_runs/{legacy,default,scaled,mask_sim,schedule}
qsub -v RUN_NAME=default,ONT_DIR=/path/to/ont_csvs,TRUTH=/path/to/truth.csv jobs/predict.pbs
```

## Inputs

- **Training pickle**: a pandas DataFrame with one row per sample; CpG columns named `cg...` holding beta values in [0, 1] (missing values allowed); `ANNOTATION`; optionally `Sample_ID` (otherwise the index is used) and `Source_Dataset`.
- **ONT sample**: a CSV with one data row. The first column is the sample name; every other column is a CpG with the methylated fraction of reads, between 0 and 1.
- **Ground truth**: a CSV with columns `sample,true_label`. Raw subtype names are fine; the model's label map is applied.

## Outputs of a training run

| File | Content |
| --- | --- |
| `config.json` | Everything needed to reuse the run: model, training settings, data, inference settings, per-fold records |
| `cv_metrics_by_condition.csv` | Accuracy, balanced accuracy, macro F1, top-2, callable share and accuracy at the threshold, ECE and NLL for each evaluation condition |
| `cv_recall_by_class.csv` | Recall per class and condition |
| `cv_predictions_<condition>.csv` | Out-of-fold predictions with probabilities |
| `cv_confusion_dense.csv`, `cv_confusion_reads_0.20.csv` | Confusion matrices |
| `folds.csv`, `folds_summary.csv` | Fold membership; best epoch, temperature and run time per fold |
| `fold_models/` | Weights of the fold models, the ensemble used by `predict.py` (about 1.5 GB each at full size) |
| `training_history.csv`, `training_curves.png` | Per-epoch losses |
| `class_mapping.json`, `label_map.json`, `selected_cpgs.json` | Classes, label merges, CpG order |

## Layout

```
configs/label_map.json   label merges (edit this, not the code)
data/                    training-data loader and the single ONT loader
models/                  SparseNN and the sparsity simulation
training/                nested cross-validation
evaluation/              metrics
scripts/                 train.py, predict.py, check_data.py, compare_runs.py, gpu_benchmark.py
jobs/                    PBS jobs; settings.sh holds all paths, check_settings.sh checks them
tests/                   synthetic data and the smoke test
docs/                    SETUP.md, EXPERIMENTS.md
```

## Licence

MIT; see [LICENSE](LICENSE).
