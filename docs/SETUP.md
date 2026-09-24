# Setting up SPARSH-next on the lab server

SPARSH-next lives in its own folder, its own GitHub repository and its own conda environment, and writes only to its own runs folder. Nothing below changes:

- the existing `sparsh` folder on the server and its git history
- the `patkarlab/sparsh` repository on GitHub
- the `meth_sim` conda environment
- the training data (read only) and any existing output folders

Run every command in a VS Code terminal connected to the server (Remote-SSH; Terminal > New Terminal). Commands are shown in grey boxes; lines starting with `#` are comments.

## 1. Create an empty repository on GitHub

On github.com, signed in as patkarlab: **New repository** > name `sparsh-next` > **Private** > leave "Add a README", ".gitignore" and "license" unticked > **Create repository**. Keep the page open; it shows the repository address.

## 2. Put the code on the server

Make a projects folder if you do not have one, outside the existing `sparsh` folder:

```bash
mkdir -p ~/projects
```

Drag `sparsh-next.zip` from Finder onto the `projects` folder in the VS Code Explorer (or upload it with Cyberduck). Then:

```bash
cd ~/projects
unzip sparsh-next.zip
cd sparsh-next
git log --oneline
```

You should see the commit list, with "Baseline: SPARSH v0.1.0" at the bottom.

## 3. Push to the new GitHub repository

First check that the server can reach GitHub:

```bash
git ls-remote https://github.com/patkarlab/sparsh.git > /dev/null && echo "GitHub reachable"
```

If it prints `GitHub reachable`:

```bash
git remote add origin https://github.com/patkarlab/sparsh-next.git
git push -u origin main
```

If git asks for a password, paste a GitHub personal access token rather than your account password (github.com > Settings > Developer settings > Personal access tokens; give it access to `sparsh-next`). If you already push to `patkarlab/sparsh` from this server with an SSH key, use `git@github.com:patkarlab/sparsh-next.git` as the address instead. Pushing from the VS Code Source Control panel also works and offers to sign in to GitHub in the browser.

If the server cannot reach GitHub, run the same unzip and push commands in the Mac Terminal instead. The server copy from step 2 is still the one you work with.

Refresh the GitHub page; the files and commits should now be there.

## 4. Create a separate environment

Clone the environment that already works with your GPUs. Cloning copies it; `meth_sim` itself is not touched, and nothing new needs installing:

```bash
conda create --name sparsh_next --clone meth_sim
conda activate sparsh_next
python -c "import torch, sklearn, pandas, matplotlib; print('torch', torch.__version__, '| GPU visible:', torch.cuda.is_available())"
```

`GPU visible: False` is normal on a login node without a GPU. If cloning fails, you can use `meth_sim` directly: set `CONDA_ENV:=meth_sim` in `jobs/settings.sh` and do not install anything into it; SPARSH-next needs no extra packages.

## 5. Check the installation

```bash
bash tests/smoke_test.sh
```

This trains and evaluates on synthetic data on the CPU for about a minute, writing only to a temporary folder. The last line must be `SMOKE TEST PASSED`.

## 6. Point the jobs at your data

Open `jobs/settings.sh` in VS Code and edit the values after `:=`:

- `DATA_PATH`: the training pickle you use now (it is only read).
- `RUNS_DIR`: where runs are written. Each full run saves about 8 GB of model weights, so choose a disk with room for about 40 GB.
- `EXCLUDE_IDS`: the sample exclusion list you use now, if any.

Compare the `#PBS` lines at the top of `jobs/train.pbs` (queue, GPU, memory, walltime) with a PBS script that already works for you and copy yours over if they differ. If the environment does not activate inside a job, replace the activation block in `jobs/common.sh` with the lines from that working script.

## 7. Check the data

```bash
qsub jobs/check_data.pbs
qstat -u $USER
```

When it finishes, open `RUNS_DIR/data_checks/` (default `~/sparsh_next_runs/data_checks/`):

- `class_counts.csv`: confirm these are the classes you expect after the label merges and exclusions.
- `source_by_class.csv`: classes that come almost entirely from one Source_Dataset are listed in the job output.
- `missing_by_source.csv`: probes missing from a whole dataset (a platform fingerprint).
- `duplicates.csv`: sample pairs that are near-identical. If they are the same patient or sample, set `GROUPS_FILE:=$RUNS_DIR/data_checks/groups.csv` in `jobs/settings.sh` so they always fall in the same fold.

## 8. Train the comparison runs

```bash
bash jobs/submit_experiments.sh
qstat -u $USER
```

This submits five GPU jobs (legacy, default, scaled, mask_sim, schedule), each writing to its own folder in `RUNS_DIR`. A run never overwrites an existing folder. Each job's log is `sn_<recipe>.o<jobid>` in the `sparsh-next` folder, and `training_log.txt` in the run folder.

## 9. Compare the runs

```bash
python scripts/compare_runs.py ~/sparsh_next_runs/legacy ~/sparsh_next_runs/default \
    ~/sparsh_next_runs/scaled ~/sparsh_next_runs/mask_sim ~/sparsh_next_runs/schedule \
    --output ~/sparsh_next_runs/comparison.csv
```

How to read the tables and choose a recipe: [EXPERIMENTS.md](EXPERIMENTS.md).

## 10. Predict and evaluate ONT samples

```bash
qsub -v RUN_NAME=default,ONT_DIR=/path/to/ont_csvs,TRUTH=/path/to/truth.csv jobs/predict.pbs
```

Results go to `RUNS_DIR/default/ont_eval/`: `predictions.csv`, `summary_metrics.json`, `evaluation_by_coverage.csv`, `evaluation_by_class.csv`, `out_of_scheme_samples.csv`, `confusion_matrix.csv/png`, and `input_errors.csv` for files that could not be read. The truth file needs the columns `sample,true_label`; raw subtype names are mapped with the label map. Use `EVAL_NAME=...` to write a second evaluation into a different folder.

## 11. Saving your own changes

In the VS Code Source Control panel: write a message, **Commit**, then **Sync Changes** (push). Or in the terminal:

```bash
git add -A
git commit -m "Describe the change"
git push
```

Run outputs, model weights and data are excluded by `.gitignore`, so only code is pushed.

## Removing SPARSH-next

Nothing else depends on it. Delete `~/projects/sparsh-next` and `RUNS_DIR`, run `conda env remove --name sparsh_next`, and delete the `sparsh-next` repository on GitHub (Settings > Danger Zone).
