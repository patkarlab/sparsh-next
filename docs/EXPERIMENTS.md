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

## Third round: call errors, grouped classes, reported call

The blinded nanopore scoring of 24 September 2026 (summary only: 207 samples with truth, model without AML-MR and AML_MECOM-r) found that real samples confuse classes the way simulated data at much lower coverage do, and that most confident errors were diagnoses with no model class. Three changes follow. The nanopore samples are the validation cohort: from 25 September 2026 no code, architecture or preprocessing change may be based on their results (docs/STRATEGY.md). Because a summary of their scoring motivated this round, an independent estimate of its result needs new samples.

**1. Measure the call error (label-free).** `qsub -v RUN_NAME=scaled_wide,ONT_DIR=/path/to/ont_folder jobs/ont_call_error.pbs`. At CpGs methylated (or unmethylated) in almost every training array, whatever the subtype, a nanopore call that disagrees with the arrays is an error of the nanopore data. The log gives the *matched call error*: the simulated per-read error rate that produces the same disagreement. The recipes below assume about 10%. If the measured rate is clearly different, pass `EXTRA_ARGS="--val_call_error <rate> --call_error_max <twice the rate> --eval_call_errors 0 <rate>"`. If it is near zero, call errors do not explain the lost accuracy (dilution by normal cells and array-to-nanopore differences at subtype-specific CpGs remain), and the error recipes are not expected to help.

**2. Runs.** Each against the second-round choice, `scaled_wide`. All three are scored on the outer folds without call errors (the second-round rows) and at 10% (`binary-err10_*`, `oneread-err10_*`):

| Recipe | Differs from `scaled_wide` by | Question it answers |
| --- | --- | --- |
| `scaled_wide_aml_other` | AML-MR and AML_MECOM-r trained as one class, `AML_other`, reported as "AML, no specific subtype" (`configs/label_map_aml_other.json`) | Group the hard classes instead of dropping them |
| `scaled_wide_aml_other_err` | The same classes, plus per-read call errors drawn per training sample from 0–20%, with inner validation (early stopping, temperature) at 10% | Does training on noisy reads help on noisy reads? (against `scaled_wide_aml_other`) |
| `scaled_wide_err` | Call errors as above, with the class set of the job's settings | The same question for another class set; its baseline is `scaled_wide` with `EXTRA_ARGS="--eval_call_errors 0 0.1"`, and both runs take the same class options (for the dropped scheme, add `--exclude_classes AML-MR AML_MECOM-r` to both) |

`bash jobs/submit_experiments.sh scaled_wide_aml_other scaled_wide_aml_other_err` runs the first pair. The comparison is between runs with the same classes, expected on the cohort's coverage mix (`compare_runs.py --ont_coverage`). The call-error check found no measurable per-read error in the lab's nanopore files (below), so the rows without call errors (`binary_*`) decide and the `-err10` rows are a robustness check. Differences of 1–2 points are within single-run noise; run seeds 42, 43 and 44 of the recipes that decide.

**3. Dropping or grouping.** The two class sets cannot be compared on balanced accuracy, because the class lists differ. Compare instead:
- what the dropped-class model does with the classes it never saw: `qsub -v RUN_NAME=scaled_wide_no_mr_mecom jobs/score_excluded.pbs` gives the share of AML-MR and AML_MECOM-r arrays called at 0.90 or more as some other class (every such call is wrong), and where they go;
- what grouping costs the other classes: `by_class_*.csv` from `scripts/hierarchy_report.py`, class by class, for the classes both models share (recall, confident errors, and how often a class is called `AML_other`).

Suggested rule, to fix before looking: group if the dropped-class model calls at least 20% of those arrays confidently as another class at 30% coverage (near the cohort's median of 38%), and grouping lowers no shared class's recall by more than 5 points at the same coverage.

**4. Reported call.** `scripts/predict.py` now reports each sample at the most specific level that reaches the threshold: subtype, family of related subtypes, or lineage (`configs/class_hierarchy.json`). What this adds, on cross-validation predictions:

```bash
python scripts/hierarchy_report.py ~/sparsh_next_runs/{scaled_wide_no_mr_mecom,scaled_wide_aml_other} \
    --ont_coverage ~/sparsh_next_runs/data_checks/ont_coverage_AL.csv
```

It prints, per condition, the share called at subtype level, the share called at any level and the accuracy of those calls, and writes per-class outcomes and the most confused class pairs to `<run>/hierarchy_report/`. The families in `configs/class_hierarchy.json` are a starting proposal (HOX-related AML; T-ALL other than TAL1). Revise them from biology and from the confused pairs in cross-validation, never from nanopore results, and fix the file before a locked nanopore set is scored. A family is only worth having if its members are confused with each other and the family call means something clinically.

**Results of 24 September 2026.** Call error (job 45102, 254 samples): at the 5,707 CpGs unmethylated in 99% of training arrays, the nanopore calls were methylated 0.1% of the time against a mean array beta of 2.5%; at the 33 methylated ones, 100.0% against 97.8%. The matched per-read error is below zero (about -2%): the nanopore calls are more extreme than the array values, so reads simulated from arrays are, if anything, noisier than real reads. Excluded classes (job 45103, `scaled_wide_no_mr_mecom`): AML-MR arrays were called at 0.90 or more as another class in 0% at 10–30% coverage and 4% at 50–70%; AML_MECOM-r arrays in 4% and 8%. Most of the rest were reported as "AML, subtype undetermined".

## Fourth round: dilution by normal marrow; platform check

Per-read call errors do not explain why real samples behave like simulated data at much lower coverage. All the lab's nanopore samples are bone marrow with more than 20% blasts, while the training arrays come mostly from high-blast diagnostic samples. A subtype's methylation signal scales with the blast fraction, so a sample at 30% blasts, against arrays at about 80%, carries roughly (0.3/0.8)^2, about 1/7, of the information per covered CpG: the same as going from 38% to about 5% coverage. `models/dilution.py` mixes leukaemia array profiles with normal-marrow arrays before the reads are simulated. Scored samples are diluted with normal marrows of their own outer fold, which the network has not seen, as a patient's own normal cells would be.

**1. Platform check (label-free).** `qsub -v RUN_NAME=scaled_wide,ONT_DIR=/path/to/ont_folder jobs/platform_check.pbs` compares, CpG by CpG, the mean nanopore call with the array beta, on CpGs whose mean beta is nearly the same in every class, normal marrow included (so neither case mix nor blast percentage can explain a difference). The mapping table shows whether the two scales agree; `discordant.csv` lists CpGs where they differ by more than 0.3.

**2. Runs.** With the class options you have settled on (here the dropped scheme):

```bash
qsub -q h200 -v RECIPE=scaled_wide_dil_base,RUN_NAME=dil_base,EXTRA_ARGS="--exclude_classes AML-MR AML_MECOM-r" jobs/train.pbs
qsub -q h100 -v RECIPE=scaled_wide_dil,RUN_NAME=dil,EXTRA_ARGS="--exclude_classes AML-MR AML_MECOM-r" jobs/train.pbs
```

| Recipe | Differs from `scaled_wide` by | Question it answers |
| --- | --- | --- |
| `scaled_wide_dil_base` | Scored on binary reads only, undiluted and with every leukaemia sample of the outer fold diluted to 50% and 30% blasts (`binary-blast50_*`, `binary-blast30_*`) | Baseline: how much does dilution cost the current model? |
| `scaled_wide_dil` | The same scoring; half the leukaemia training samples diluted to 20–100% blasts, and the inner validation sets likewise | Does training on diluted profiles recover it, and at what cost to undiluted samples and to normal marrow? |

Normal marrows are never diluted, so a model trained on diluted leukaemias may call more normal marrows leukaemia: check `Normal_Control_BM` in `cv_recall_by_class.csv`. On synthetic data the gain at 30–50% blasts was large and the cost to the stand-in partner class was clear, so this row matters.

**3. Rule, fixed before looking** (binary rows, the equal-weight mean over coverages): adopt dilution if balanced accuracy improves by at least 2 points at both 50% and 30% blasts, falls by no more than 1 point undiluted, and Normal_Control_BM recall at `binary_0.30` falls by no more than 5 points. The 50% and 30% rows count equally; nothing is weighted by properties of the validation cohort.

**Result (25 September 2026, `dil_base` against `dil`, 30 classes).** Balanced accuracy, mean over coverages: undiluted 0.878 → 0.900, 50% blasts 0.569 → 0.835, 30% blasts 0.153 → 0.718; Normal_Control_BM recall at `binary_0.30` 0.960 → 0.944. All four conditions are met and dilution is adopted. Without it, 64% of leukaemias at 30% blasts were called normal marrow (4.6% with it). Cost outside the rule: the callable share at 98% accuracy of calls, undiluted, fell from 0.71 to 0.64. The recipe is locked as `locked` (docs/STRATEGY.md).

## Fifth round: class scheme (locked recipe)

The class scheme is revised from biology and public genotypes, on the training arrays only; the recipe stays `locked`. Evidence: project notes `npm1-idh-cluster.md` and `lamprey-marlin-sparsh-comparison.md`. The lists are in `RELABEL_DIR`, one row per sample with its reason.

- **AML_NPM1_IDH** (20 samples; Lamprey's NPM1_IDH, MARLIN's HOX Grp 3). NPM1-mutated AML with an IDH1 or IDH2 mutation by sequencing (Beat AML 2.0, TCGA-LAML, TARGET) that lies in the island next to AML_IDH in the t-SNE of `AL_24Sep2026`. Of the Beat AML and TCGA NPM1-mutated cases in the island, 17 of 18 are IDH-mutant; of their 90 IDH-wild-type NPM1 cases, 1 lies in it.
  - The other 13 NPM1- and IDH-mutated cases lie outside the island, 9 of them with a DNMT3A mutation, and keep the HOX label.
  - Island members whose NPM1 or IDH status is unknown are left out (below).
  - AML_IDH stays a separate class (IDH-mutated without NPM1). Its main group sits with AML-MR and MECOM-r, and merging it with the new class would drop the NPM1 information from the report.
- **AML_NUP98-NSD1** (50): HOX-labelled samples with a NUP98::NSD1 fusion in GEO (GSE190931), all paediatric. They form most of a subcluster that also holds KMT2A::ELL cases, which keep the KMT2A-r label.
- **AML_ETV6-MNX1** (8):
  - 4 with the fusion in GEO (GSE190931);
  - 4 TARGET cases with t(7;12)(q36;p13) and trisomy 19 in the TARGET karyotype.

  The 8 form one tight group. AML_ETV6-r keeps 9 samples with other ETV6 rearrangements.
- **T-ALL_TAL1-like**: T-ALL_TAL1 renamed (Lamprey's name), because the class is a methylation group, mostly CIMP-low, rather than a genotype.
- **ETV6-rearranged MPAL-T/M** belongs to T-ALL_HOXA9_ETP, as in Lamprey's ETP class. The training set has no such arrays: its only MPAL arrays are 3 TARGET cases with BCL11B activation, which are in AL_BCL11B-r. Nothing moves now; the rule applies to arrays added later.
- **Data clean-up**, the same in both runs (exclusion list and groups file). Left out (61):
  - 11 GSE124617 re-deposits of TCGA samples that are in the training set under their TCGA barcode;
  - 25 post-treatment samples of an IDH-inhibitor study (GSE153347);
  - 22 island members whose NPM1 or IDH status is unknown;
  - 2 IDH1-mutated, NPM1-negative island members with KMT2A-PTD;
  - 1 sample with an ETV6::MNX1 fusion call that lies far from the other 8.

  Samples are grouped by patient, so that a patient never sits on both sides of a split: 170 patients with 340 samples. They are mostly TARGET diagnosis/relapse pairs, plus repeat arrays and Beat AML and TCGA patients with more than one sample.

```bash
qsub -v RELABEL_DIR=$HOME/sparsh_next_runs/data_checks/relabel_25Sep jobs/prepare_relabel.pbs
# then, with the files it writes:
CHK=$HOME/sparsh_next_runs/data_checks
qsub -q h200 -v RECIPE=locked,RUN_NAME=locked_clean,EXCLUDE_IDS=$CHK/exclude_25Sep2026.txt,GROUPS_FILE=$CHK/groups_25Sep2026.csv jobs/train.pbs
qsub -q h100 -v RECIPE=locked,RUN_NAME=locked_relabel,DATA_PATH=/home/patkarlab/AL_Methylation_Classifier/data/AL_25Sep2026_relabel.pkl,EXCLUDE_IDS=$CHK/exclude_25Sep2026.txt,GROUPS_FILE=$CHK/groups_25Sep2026.csv jobs/train.pbs
# when both have finished:
python scripts/compare_class_schemes.py ~/sparsh_next_runs/locked_clean ~/sparsh_next_runs/locked_relabel \
    --rename T-ALL_TAL1=T-ALL_TAL1-like --output $CHK/round5_rule.csv
```

`locked_clean` has the old labels and the clean-up; `locked_relabel` has both. `scripts/compare_class_schemes.py` applies the rule below, from the cross-validation predictions (top-1, as in `cv_recall_by_class.csv`).

**Rule, fixed before the runs.** Two refinements were added on 25 September 2026, before either run was started: shared classes are compared like for like, and small classes may lose one sample.

- Keep a new class if its recall at `binary_0.30` is at least 0.70.
- No shared class may lose more than 5 points of recall at `binary_0.30`, or one sample where that is more (classes under 20 samples).
  - A shared class is compared on the samples whose label is the same in both runs, after the TAL1-like rename. So a class that gives members to a new class (HOX, AML_ETV6-r) is judged on the members it keeps.
- The mean recall over the shared classes at `binary_0.30` (equal class weight, the same samples) may fall by no more than 1 point.

If a shared class fails, the column `errors_second` shows which classes took its samples; the new class that took them is merged back first.

**Amendment, adopted on 25 September 2026 at 21:49 IST, after the first result.** A failing shared class counts against a new class only when it lost samples to that new class. Other failing shared classes are reported but block no new class. `scripts/compare_class_schemes.py` prints this attribution for each new class.

**Result.**
- **First comparison**, `locked_clean` against `locked_relabel` (25 September):
  - All three new classes reached the recall they needed: AML_NPM1_IDH 20 of 20, AML_ETV6-MNX1 7 of 8, AML_NUP98-NSD1 40 of 50.
  - The mean over shared classes rose from 0.891 to 0.901.
  - AML_IDH lost 4 of 54 samples, 3 of them to AML_NPM1_IDH. All three are NPM1-negative IDH cases by TCGA sequencing, so AML_NPM1_IDH is merged back: its samples return to their original labels.
  - T-ALL_NKX2 lost 3 of 16 to other T-ALL classes. Under the amendment this does not count.
- **Confirmation run** `locked_relabel_b`, compared with `locked_clean` (26 September). It has AML_NUP98-NSD1, AML_ETV6-MNX1 and the rename, and no AML_NPM1_IDH.
  - AML_NUP98-NSD1 recalled 39 of 50 and AML_ETV6-MNX1 7 of 8.
  - The mean over shared classes went from 0.891 to 0.893.
  - AML_ETV6-r (9 samples) fell from 5 to 3 correct, and both lost samples went to AML_ETV6-MNX1. That is two samples where one is allowed, so AML_ETV6-MNX1 is merged back.
  - AML_IDH (3 samples, to HOX) and T-ALL_NKX2 (2, to T-ALL_TAL1-like) also failed, without losing samples to a new class.
- **Scheme after the fifth round:**
  - AML_NUP98-NSD1 added;
  - T-ALL_TAL1 renamed T-ALL_TAL1-like;
  - AML_NPM1_IDH and AML_ETV6-MNX1 not kept.

```bash
# confirmation run; relabel_25Sep_b holds the NUP98-NSD1 and ETV6-MNX1 lists and the fifth-round drop and patient lists
qsub -v RELABEL_DIR=$CHK/relabel_25Sep_b,TAG=25Sep2026b jobs/prepare_relabel.pbs
qsub -q h200 -v RECIPE=locked,RUN_NAME=locked_relabel_b,DATA_PATH=/home/patkarlab/AL_Methylation_Classifier/data/AL_25Sep2026b_relabel.pkl,EXCLUDE_IDS=$CHK/exclude_25Sep2026b.txt,GROUPS_FILE=$CHK/groups_25Sep2026b.csv jobs/train.pbs
python scripts/compare_class_schemes.py ~/sparsh_next_runs/locked_clean ~/sparsh_next_runs/locked_relabel_b \
    --rename T-ALL_TAL1=T-ALL_TAL1-like --output $CHK/round5b_rule.csv
```

### T-ALL label audit

The T-ALL labels are checked on the training arrays with `scripts/label_audit.py`, which follows Lamprey's label cleaning. It uses two signals:

- agreement with the 20 nearest neighbours in a PCA of the 50,000 most variable CpGs;
- the cross-validation probability each sample received for its own label in the run `dil`.

Flags are for review; nothing is relabelled automatically. Run it on the original pickle (the default `DATA_PATH`), so that the labels match those of `dil`:

```bash
qsub -v CLASSES="T-ALL AL_BCL11B",RUN_NAME=dil,ANNOTATIONS=$HOME/sparsh_next_runs/data_checks/relabel_25Sep/tall_annotations.csv,AUDIT_NAME=label_audit_tall jobs/label_audit.pbs
```

`tall_annotations.csv` gives each sample's source series and, where GEO has it, its CIMP status. For GSE69954 and GSE272021 (192 of the 347 samples), no per-sample genetic subtype is public, so their labels may come from clustering; the project note `lamprey-marlin-sparsh-comparison.md` has the details.

## Sixth round: label clean-up (locked recipe, fifth-round scheme)

Training labels are checked against independent genetics and against the methylation itself, on the training arrays only. The evidence is in the project note `lamprey-marlin-sparsh-comparison.md`.

- **Methylation** is `scripts/label_audit.py`, within the lineage and across all samples, with CV from `dil`.
- **Independent genetics:**
  - RNA-seq for GSE272021 (GSE272023);
  - GEO fusion calls (GSE190931);
  - TARGET clinical data;
  - GEO karyotype subtypes (GSE49031);
  - Beat AML and TCGA sequencing.

**Rule, approved on 25 September 2026 before any clean-up run.**

- **Labels with independent genetics:** keep a label only when both the genetics and the methylation support it.
  - Methylation is taken to contradict a label under Lamprey's rule: at most 1 of the 20 nearest neighbours share the label, and the own-label CV probability is at most 0.05.
  - Relabel only when the genetics and the methylation agree on another class; otherwise leave the sample out.
  - A missing lesion counts against a label only where the test would have found it: IDH hotspots on the exome, and fusions on RNA-seq together with the karyotype. It does not count where the test is blind: UBTF tandem duplications (never tested) and CEBPA on the exome.
- **Labels from clustering alone** (T-ALL without RNA: GSE69954, GSE147667 and 20 GSE272021 samples): leave the sample out when the own-label CV probability is at most 0.05 and most neighbours carry another label. Never relabel them.
- **The two lab arrays in the list** (26CGH0740, 25RSEQ247) are dropped, as decided by the user.
- **Beat AML and TCGA against their sequencing** (approved on 25 September at 22:20 IST):
  - 20 samples are left out:
    - 12 AML_KMT2A-r without a KMT2A fusion;
    - 3 AML_IDH without an IDH mutation;
    - 3 AML_mutated CEBPA with one CEBPA mutation outside the bZIP;
    - 1 HOX sample with KMT2A::ELL;
    - 1 AML_FET-ETS sample without the fusion.
  - 3 TCGA HOX samples with NUP98::NSD1 go to AML_NUP98-NSD1, because cross-validation in `locked_relabel` called them that (0.82 to 0.96 at `binary_0.90`).

The lists are in `relabel_26Sep`:
- `relabel_cleanup.csv`: 12 relabels;
- `drop_cleanup.csv`: the 82 samples left out, with reasons;
- `drop_26Sep2026.txt`: the fifth-round drop list plus the clean-up (143);
- `relabel_nup98_nsd1.csv` from the fifth round, and the patient list.

The fifth-round lists for AML_NPM1_IDH and AML_ETV6-MNX1 are not in this folder, because both classes were merged back.

**Base run.** No finished run has the scheme left after the fifth round, so the clean-up is compared with `locked_nsd1`. That run has the same scheme (AML_NUP98-NSD1 and the rename) and the fifth-round drop list, but no clean-up. Its lists are in `relabel_26Sep_nsd1`: `relabel_nup98_nsd1.csv`, `drop_26Sep2026nsd1.txt` (the fifth-round drop list) and `patients_26Sep2026nsd1.csv`. The two runs can train at the same time.

```bash
R6=$HOME/sparsh_next_runs/data_checks/relabel_26Sep
RN=$HOME/sparsh_next_runs/data_checks/relabel_26Sep_nsd1
CHK=$HOME/sparsh_next_runs/data_checks
qsub -v RELABEL_DIR=$RN,TAG=26Sep2026nsd1 jobs/prepare_relabel.pbs
qsub -v RELABEL_DIR=$R6,TAG=26Sep2026 jobs/prepare_relabel.pbs
# then, with the files they write:
qsub -q h200 -v RECIPE=locked,RUN_NAME=locked_nsd1,DATA_PATH=/home/patkarlab/AL_Methylation_Classifier/data/AL_26Sep2026nsd1_relabel.pkl,EXCLUDE_IDS=$CHK/exclude_26Sep2026nsd1.txt,GROUPS_FILE=$CHK/groups_26Sep2026nsd1.csv jobs/train.pbs
qsub -q h200 -v RECIPE=locked,RUN_NAME=locked_cleanup,DATA_PATH=/home/patkarlab/AL_Methylation_Classifier/data/AL_26Sep2026_relabel.pkl,EXCLUDE_IDS=$CHK/exclude_26Sep2026.txt,GROUPS_FILE=$CHK/groups_26Sep2026.csv jobs/train.pbs
# when both have finished:
python scripts/compare_class_schemes.py ~/sparsh_next_runs/locked_nsd1 ~/sparsh_next_runs/locked_cleanup \
    --allowed_samples 2 --output $CHK/round6_rule.csv
```

**Rule for keeping the clean-up, fixed on 26 September 2026 at 06:28 IST, before either run.**
- The comparison uses the samples present in both runs with the same label. No shared class may lose more than 5 points of recall at `binary_0.30`, or two samples where that is more.
- The mean recall over the shared classes may fall by no more than 1 point.

The samples left out or relabelled are not part of this comparison.

The allowance is two samples, where the fifth round allowed one, because of run-to-run variation. In both fifth-round comparisons, classes whose labels did not change moved by up to 2 or 3 samples between runs; T-ALL_NKX2, for example, lost 3 and then 2. A round without new classes gives the amendment nothing to attribute such losses to. With a one-sample allowance, the clean-up would then be rejected on variation alone. The fifth-round verdicts stand as judged under the one-sample allowance.

## Scoring on real nanopore samples

Keep a labelled nanopore cohort for one blinded scoring of the chosen model. Every choice (recipe, threshold, preprocessing) is made on cross-validation and on label-free properties of the nanopore files, such as coverage and value type. Run `jobs/predict.pbs` without `TRUTH`; whoever holds the labels scores `predictions.csv`. Comparing several models on the cohort turns it into a selection set and makes its accuracy optimistic; if that is needed, hold part of the cohort back untouched.

## After choosing a recipe

- **Coverage range:** set `--cov_min` and `--cov_max` (random mode) to span your ONT coverage distribution with some margin, for example its 5th percentile halved to its 95th percentile doubled.
- **Deployment model:** `predict.py` uses the fold ensemble by default, the models cross-validation evaluated. `RECIPE=final` also trains one model on all samples (minus an inner split); cross-validation cannot score it, so choose between the two before any blinded scoring.
- **Loss settings:** label smoothing (`--label_smoothing 0`) and focal gamma (`--focal_gamma 0` or `1`) make probabilities less confident; temperature scaling corrects part of that. Worth one comparison each, passed through `EXTRA_ARGS`, for example `qsub -v RECIPE=default,RUN_NAME=default_ls0,EXTRA_ARGS="--label_smoothing 0" jobs/train.pbs`.
- **Width:** `--hidden_dims 2048 1024 512` against the default `1024 512 256`, if GPU memory allows (`EXTRA_ARGS="--hidden_dims 2048 1024 512"`).
- **Known limitation:** an observed value of exactly 0.5 (one of two reads methylated) is encoded like a missing CpG, about 1–3% of observed CpGs at 10–30% coverage with the `reads` simulation. Files with 0/1 values (`binary`, `oneread`) never contain 0.5.
