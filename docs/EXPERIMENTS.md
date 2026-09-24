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

Resources per run, within the 48 GB and 12 hour request: peak memory about 21 GB for `default` and 28 GB for `legacy` (measured on synthetic data with 29 classes of 6 to 370 samples and one tenth of the CpGs, then scaled up; conservative); run time roughly 1–3 hours on one A40 (an estimate; the training log prints the elapsed time every 25 epochs).

## Reading the results

1. **Find your ONT coverage.** `coverage_pct` in `predict.py` output (or the old `coverage_report.csv`) is the percentage of model CpGs with at least one read. Use the `reads_*` rows closest to that range, typically `reads_0.10` to `reads_0.30`.
2. **Main metric: balanced accuracy** on those rows, so that rare subtypes count as much as common ones. Check `cv_recall_by_class.csv` for the rarest classes.
3. **Operating point:** `callable_share_0.90` (fraction of samples called at confidence 0.90 or more) and `accuracy_callable_0.90` (accuracy of those calls). A recipe that calls more samples at the same accuracy is better.
4. **The `dense` row** is performance on array data. It is a sanity check, not the deployment condition.
5. **Noise:** differences of 1–2 points between single runs can be chance. Rerun the top two recipes with another seed and compare again, for example `qsub -v RECIPE=default,SEED=43,RUN_NAME=default_seed43 jobs/train.pbs`.
6. **Decision:** take the recipe with the highest mean balanced accuracy over the `reads_*` rows in your coverage range, provided `accuracy_callable_0.90` is not lower. Then run `jobs/predict.pbs` with that run on the ONT cohort.

Calibration changes confidence, so a threshold of 0.90 does not mean the same as it did for the v0.1.0 model. Re-derive the operating point from `accuracy_callable` and `callable_share` in the CV `reads_*` rows and in the ONT evaluation.

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

## After choosing a recipe

- **Coverage range:** set `--cov_min` and `--cov_max` (random mode) to span your ONT coverage distribution with some margin, for example its 5th percentile halved to its 95th percentile doubled.
- **Deployment model:** `predict.py` uses the fold ensemble by default. `RECIPE=final` also trains one model on all samples (minus an inner split); compare both on the ONT cohort with `--use ensemble` and `--use final`.
- **Loss settings:** label smoothing (`--label_smoothing 0`) and focal gamma (`--focal_gamma 0` or `1`) make probabilities less confident; temperature scaling corrects part of that. Worth one comparison each, passed through `EXTRA_ARGS`, for example `qsub -v RECIPE=default,RUN_NAME=default_ls0,EXTRA_ARGS="--label_smoothing 0" jobs/train.pbs`.
- **Width:** `--hidden_dims 2048 1024 512` against the default `1024 512 256`, if GPU memory allows (`EXTRA_ARGS="--hidden_dims 2048 1024 512"`).
- **Known limitation:** an observed value of exactly 0.5 (one of two reads methylated) is encoded like a missing CpG, about 1–3% of observed CpGs at 10–30% coverage.
