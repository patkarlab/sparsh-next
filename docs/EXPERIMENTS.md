# Choosing the best recipe

Five runs, one GPU job each, decide the recipe on your data (`bash jobs/submit_experiments.sh`). All five are scored on the same samples with identical simulated ONT inputs, so their tables can be compared line by line.

| Recipe | Differs from `default` by | Question it answers |
| --- | --- | --- |
| `default` | (balanced sampler only; read-level simulation; coverage 2–50% drawn per sample; midpoint encoding) | The new recipe |
| `legacy` | v0.1.0 recipe: pre-masked upsampling to the largest class, IID masking, one coverage per epoch from 3% to 20% | How much the fixes gain under the same fair evaluation |
| `scaled` | Inverted-dropout input encoding | Does coverage-invariant input scaling help at your coverage? |
| `mask_sim` | Training keeps array betas at observed CpGs instead of simulated reads | Value of simulating reads |
| `schedule` | v0.1.0 coverage schedule (3% to 20%, one level per epoch) instead of 2–50% per sample | Which coverage range to train on |

`sampler_alpha` (the v0.1.0 recipe without `--upsample_minority`) is not in the set because it collapsed in every test below; to see it on your data: `qsub -v RECIPE=default,RUN_NAME=sampler_alpha,EXTRA_ARGS="--imbalance sampler_alpha" jobs/train.pbs`.

Resources per run, within the 48 GB and 12 hour request: peak memory about 21 GB for `default` and 28 GB for `legacy` (measured on synthetic data with 29 classes of 6 to 370 samples and one tenth of the CpGs, then scaled up; conservative); run time roughly 1–3 hours on one A40 (an estimate; `bash jobs/probe_gpus.sh` measures the epoch time on each GPU queue, and the training log prints the elapsed time every 25 epochs). The `a40` queue runs at most 2 jobs per user at a time; list more queues in `TRAIN_QUEUES` to run the five together (docs/SETUP.md step 6).

## Reading the results

SPARSH-next classifies nanopore samples only, so `compare_runs.py` shows the `reads_*` rows: simulated runs in which a fraction of the model's CpGs has one or a few reads. The `dense` row (array profiles) and the `mask_*` rows (array beta values at the covered CpGs) describe inputs a nanopore run never produces; `--all_conditions` shows them.

1. **Find your ONT coverage.** `python scripts/ont_coverage.py --run ~/sparsh_next_runs/default /path/to/ont_folder` gives, per folder, the percentage of model CpGs with at least one read (median, range, samples per band), checks that the files load, and shows whether the values look like read calls (mostly exactly 0 or 1). Use the `reads_*` rows that span that range.
2. **Main metric: balanced accuracy** on those rows, so that rare subtypes count as much as common ones. Check `cv_recall_by_class.csv` for the rarest classes.
3. **Operating point:** `callable_share_0.90` (fraction of samples called at confidence 0.90 or more) and `accuracy_callable_0.90` (accuracy of those calls). These depend on calibration, which can differ by coverage. The table *callable share at 98% accuracy of calls* does not: it calls samples from the most confident down, stopping where the calls would fall below 98% correct, and shows the confidence cut-off at which that happens. A recipe that calls more samples there knows better when it is right; a cut-off far from 0.90 means its confidences are off at that coverage.
4. **Noise:** differences of 1–2 points between single runs can be chance. Rerun the top two recipes with another seed and compare again, for example `qsub -v RECIPE=default,SEED=43,RUN_NAME=default_seed43 jobs/train.pbs`.
5. **Decision:** take the recipe with the highest mean balanced accuracy over the `reads_*` rows in your coverage range, provided it does not call fewer samples at 98% accuracy of calls. Then run `jobs/predict.pbs` with that run on the ONT cohort.

Calibration changes confidence, so a threshold of 0.90 does not mean the same as it did for the v0.1.0 model. Re-derive the operating point from the CV `reads_*` rows and from the ONT evaluation.

## What the synthetic tests showed

Two synthetic cohorts (10 classes of 6–300 samples, 4,000 CpGs, noisy betas with partially methylated class signatures), 3-fold nested CV, 80 epochs, small network, same code as these recipes. Mean of the two cohorts, simulated ONT reads:

| Recipe | Balanced accuracy at 5% | 10% | 20% | 30% | Called at ≥0.90, 20% | Accuracy of calls, 20% |
| --- | --- | --- | --- | --- | --- | --- |
| `legacy` | 0.35 | 0.49 | 0.65 | 0.68 | 71% | 97.7% |
| `default` | 0.56 | 0.75 | 0.94 | 0.97 | 91% | 99.5% |
| `scaled` | 0.60 | 0.81 | 0.95 | 0.98 | 82% | 99.9% |
| `mask_sim` | 0.43 | 0.68 | 0.84 | 0.96 | 84% | 99.6% |
| `schedule` | 0.51 | 0.77 | 0.96 | 0.99 | 94% | 99.9% |
| `sampler_alpha` | 0.09 | 0.12 | 0.11 | 0.13 | 0% | – |

- The fixes matter most: `legacy` trails `default` by 20–30 points of balanced accuracy at every coverage. On dense arrays `legacy` reached 0.89 accuracy but only 0.52–0.54 balanced accuracy, because rare-subtype samples were pulled into large classes.
- Simulating reads beats masking array values (`mask_sim` is 7–13 points lower at 5–20%).
- `scaled` and `schedule` were within about 5 points of `default`, in both directions depending on coverage and metric. `scaled` had the highest balanced accuracy at 5–10% coverage but called fewer samples at 0.90 than `default` or `schedule`. These two are what your own runs should settle; if both beat `default`, try `RECIPE=scaled_schedule`.
- Inverse-frequency class weights on top of the balanced sampler (`sampler_alpha`) made the network call almost everything a rare subtype, and early stopping kept epoch 1.

Synthetic data show mechanisms, not the size of the effects on your cohort; the five runs decide.

## Second round: matched to the nanopore files

`scripts/ont_coverage.py` on the lab's 299 nanopore samples showed two differences from what the first five runs assumed:

- **Coverage** is far higher: median 38% of the model's CpGs (range 3.8–87%). Two thirds of the samples lie above 30%, the highest coverage the first round scored, and 83 lie above 50%, the top of its training range.
- **Values** are always exactly 0 or 1. The `reads` simulation reports the methylated fraction (0, 1/3, 1/2, ...) wherever a CpG has two or more reads, which is common at these depths.

Three runs address this: `bash jobs/submit_experiments.sh wide scaled_wide scaled_wide_log`.

| Recipe | Encoding | Training coverage | Question it answers |
| --- | --- | --- | --- |
| `wide` | midpoint | 2–95%, uniform | Baseline at the real coverage and value type |
| `scaled_wide` | scaled | 2–95%, uniform | Encoding, at the real coverage |
| `scaled_wide_log` | scaled | 2–95%, log-uniform (more weight on low coverage) | How to spread training over coverage |

All three train on `binary` values: one 0/1 call per covered CpG, the majority of its reads, and a random call when the reads split evenly (which covers either tie rule of the nanopore pipeline). Early stopping and temperature use inner validation at 5–90% coverage. The outer folds are scored at 5, 10, 20, 30, 50, 70 and 90% coverage, both as `binary` and as `oneread` (the call of a single read per CpG, the right simulation if the pipeline keeps one read per CpG).

Compare them with the cohort's coverage mix:

```bash
python scripts/compare_runs.py ~/sparsh_next_runs/{wide,scaled_wide,scaled_wide_log} \
    --ont_coverage ~/sparsh_next_runs/data_checks/ont_coverage.csv
```

The row *your ONT samples* places every sample's coverage between the two nearest evaluated coverages, interpolates the metric there and averages: the value expected on this cohort, computed from coverage alone.

## Scoring on real nanopore samples

Keep a labelled nanopore cohort for one blinded scoring of the chosen model. Every choice (recipe, threshold, preprocessing) is made on cross-validation and on label-free properties of the nanopore files, such as coverage and value type. Run `jobs/predict.pbs` without `TRUTH`; whoever holds the labels scores `predictions.csv`. Comparing several models on the cohort turns it into a selection set and makes its accuracy optimistic; if that is needed, hold part of the cohort back untouched.

## After choosing a recipe

- **Coverage range:** set `--cov_min` and `--cov_max` (random mode) to span your ONT coverage distribution with some margin, for example its 5th percentile halved to its 95th percentile doubled.
- **Deployment model:** `predict.py` uses the fold ensemble by default, the models cross-validation evaluated. `RECIPE=final` also trains one model on all samples (minus an inner split); cross-validation cannot score it, so choose between the two before any blinded scoring.
- **Loss settings:** label smoothing (`--label_smoothing 0`) and focal gamma (`--focal_gamma 0` or `1`) make probabilities less confident; temperature scaling corrects part of that. Worth one comparison each, passed through `EXTRA_ARGS`, for example `qsub -v RECIPE=default,RUN_NAME=default_ls0,EXTRA_ARGS="--label_smoothing 0" jobs/train.pbs`.
- **Width:** `--hidden_dims 2048 1024 512` against the default `1024 512 256`, if GPU memory allows (`EXTRA_ARGS="--hidden_dims 2048 1024 512"`).
- **Known limitation:** an observed value of exactly 0.5 (one of two reads methylated) is encoded like a missing CpG, about 1–3% of observed CpGs at 10–30% coverage with the `reads` simulation. Files with 0/1 values (`binary`, `oneread`) never contain 0.5.
