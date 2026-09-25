# SPARSH-next strategy, locked 25 September 2026 (v0.2.0)

This file fixes the training and reporting strategy that produced the model scored on the lab's nanopore validation cohort on 25 September 2026. Later work changes the class scheme or adds data; the recipe below stays as it is unless a documented comparison on training arrays replaces it.

## The locked recipe

`qsub -q h200 -v RECIPE=locked,RUN_NAME=<name> jobs/train.pbs` reproduces it. It is the fourth-round run `dil` (job 45218), with the class exclusion written into the recipe.

| Part | Setting |
| --- | --- |
| Training data | `AL_24Sep2026.pkl`, exclusion list `exclude_24Sep2026.txt`, groups `groups_no_normals.csv` (jobs/settings.sh) |
| Classes | Classes starting with MPAL, AML_NOS or B-ALL_NOS are left out, and so are AML-MR and AML_MECOM-r: 30 classes. Label merges come from `configs/label_map.json` |
| Input | Simulated nanopore calls from array beta values: one 0/1 call per covered CpG (`binary`), coverage drawn per sample, uniform on 2–95% |
| Encoding | `scaled`: observed calls are centred (x − 0.5), missing CpGs are set to 0, and the result is divided by the sample's observed fraction (the inverted-dropout rule) |
| Dilution | Half of the leukaemia training samples are mixed with a normal marrow of the fit set, at a blast fraction drawn from 20–100% (`--dilution_prob 0.5 --blast_min 0.2`). The inner validation sets are diluted the same way, and normal marrows are never diluted |
| Network | Feed-forward 1024-512-256, dropout 0.45, focal loss (gamma 2) with label smoothing 0.05, balanced sampler (5 per class per batch), AdamW with learning rate 1e-4 and weight decay 1e-4, up to 300 epochs |
| Model selection | Nested 5-fold CV. An inner split (15%) of each training fold sets early stopping, the learning-rate schedule and the temperature. The outer fold is scored once |
| Deployed model | The ensemble of the 5 fold networks, with calibrated probabilities |
| Reporting | Threshold 0.90. When no subtype reaches it, the summed probability of a family, then of a lineage, is used (`configs/class_hierarchy.json`). AML-MR and AML_MECOM-r are reported as "AML, subtype undetermined" |

## Why this recipe

**Cross-validation on training arrays** (binary reads, mean over coverages 5–90%), against the same recipe without dilution (`dil_base`):

| | Without dilution | Locked |
| --- | --- | --- |
| Balanced accuracy, undiluted | 0.878 | 0.900 |
| Balanced accuracy, 50% blasts | 0.569 | 0.835 |
| Balanced accuracy, 30% blasts | 0.153 | 0.718 |
| Leukaemias at 30% blasts called normal marrow (30% coverage) | 64% | 5% |
| Normal_Control_BM recall at 30% coverage | 0.960 | 0.944 |

The decision rule was written down before the runs, and all four conditions were met. There is one cost: at 98% accuracy of calls, the model can call 64% of undiluted samples, against 71% without dilution.

**Validation cohort** (user's scoring, 188 samples with ground truth, reported here and not used for any code change):

| Measure | Result |
| --- | --- |
| Top-1 correct | 175 of 188 (93.1%) |
| Correct class ranked first or second | 180 of 188 |
| Reported at subtype level | 147 (78%), of which 143 (97.3%) correct |
| Lineage right | 185 of 188 |

The legacy v3.2.1 model's 93.5% (144/154, ASH abstract) was measured on a different set of samples.

## Rules for changes

1. The nanopore samples are the validation cohort. No code, architecture or preprocessing change is based on their results. Any change is judged on the training arrays, by cross-validation, against a rule written down before the runs.
2. A change of the class scheme uses the locked recipe unchanged, so that its effect is the only difference. Classes that exist in both runs are compared by per-class recall. New classes are judged by their own recall and by what they take from their neighbours.
3. Results on the current nanopore cohort are no longer an independent estimate for models chosen after seeing them. Such models need new samples.
4. Differences of 1–2 points between single runs are within seed noise. Decisions that hinge on them need seeds 42, 43 and 44.
5. `compare_runs.py --ont_coverage` weights metrics by the validation cohort's coverage (label-free). Decisions made from 25 September 2026 use the equal-weight "mean over" rows instead.
