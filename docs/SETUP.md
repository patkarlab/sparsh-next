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

Files from Claude arrive in `~/inbox/from_claude`. Copy the zip from there into a projects folder, outside the existing `sparsh` folder, and unpack it:

```bash
mkdir -p ~/projects
cp ~/inbox/from_claude/sparsh-next.zip ~/projects/
cd ~/projects
unzip sparsh-next.zip
cd sparsh-next
git log --oneline
```

You should see the commit list, with "Baseline: SPARSH v0.1.0" at the bottom. Later changes from Claude arrive in the same inbox as `.patch` files; see [Applying updates from Claude](#applying-updates-from-claude).

## 3. Push to the new GitHub repository

The repository belongs to the patkarlab GitHub account. The GitHub login already saved on the server belongs to another account, so this repository pushes over SSH with its own key, registered on the patkarlab account and used only through the host name `github-patkarlab`. Other repositories on the server keep their login.

Make the key and print it:

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
ssh-keygen -t ed25519 -N "" -C "patkarlab@ln1 sparsh-next" -f ~/.ssh/id_ed25519_patkarlab
cat ~/.ssh/id_ed25519_patkarlab.pub
```

In a private browser window, sign in to github.com as patkarlab and open https://github.com/settings/ssh/new. Title `ln1 sparsh-next`, key type Authentication Key, paste the printed line, **Add SSH key**. Then, once only:

```bash
cat >> ~/.ssh/config <<'EOF'

Host github-patkarlab
    HostName ssh.github.com
    Port 443
    User git
    IdentityFile ~/.ssh/id_ed25519_patkarlab
    IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config
ssh -T github-patkarlab
```

At the first connection, accept GitHub's host key if its fingerprint is one of those GitHub publishes (docs.github.com, "GitHub's SSH key fingerprints"). The last command must print `Hi patkarlab! You've successfully authenticated`. Port 443 is used because port 22 is often blocked on institutional networks. Then:

```bash
git remote add origin github-patkarlab:patkarlab/sparsh-next.git
git push -u origin main
```

Refresh the GitHub page; the files and commits should now be there. Pushing with **Sync** in VS Code uses the same key.

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

## 6. Check the job settings

Queues, resources and environment in `jobs/settings.sh` and the `#PBS` lines follow your working job, `~/train_focal_fast.pbs`:

| Setting | Value |
| --- | --- |
| Training runs | GPU queue(s) in `TRAIN_QUEUES` (default `a40`), 8 CPUs, 1 GPU, 48 GB, 12 hours per run |
| Data checks and ONT prediction | `short` queue, CPU only, 8 CPUs, 48 GB, 2 hours |
| Environment | `module load cuda/12.3`, then `conda activate sparsh_next` from `~/miniconda3` |
| Runs (`RUNS_DIR`) | `~/sparsh_next_runs`; about 8 GB of model weights per run |
| Classes left out (`EXCLUDE_PREFIXES`) | MPAL, AML_NOS, B-ALL_NOS, as in the v0.1.0 job template (`train_focal_fast.pbs` leaves out only MPAL) |

Open `jobs/settings.sh` in VS Code and set the training data: `DATA_PATH` to the full path of the training pickle, and `EXCLUDE_IDS` to its exclusion list (one Sample_ID per line), or leave `EXCLUDE_IDS` empty. For example:

```bash
: "${DATA_PATH:=/home/patkarlab/data/new_training_set.pkl}"
: "${EXCLUDE_IDS=}"
```

Then check:

```bash
bash jobs/check_settings.sh
```

The last line must be `SETTINGS OK`. The check also shows the free space and your quota for `RUNS_DIR`, and the training queues.

### Which GPU queues to use

The `a40` queue runs at most 2 GPU jobs per user at a time, so the five comparison runs would run two at a time. To see what each GPU queue offers (GPU model, whether the environment works on it, and the time of one training epoch), run a short test with random numbers on each; it reads no data and saves nothing:

```bash
bash jobs/probe_gpus.sh
```

When the jobs have finished (`qstat -u $USER`), print the results with the `grep` line the script shows. List the queues to use in `TRAIN_QUEUES`, for example `: "${TRAIN_QUEUES:=a40 a40 A40b h100 h200}"`; `jobs/submit_experiments.sh` sends the five runs to them in turn.

## 7. Check the data

```bash
qsub jobs/check_data.pbs
qstat -u $USER
```

The job log, `sparsh_next_checks.o<job number>` in the `sparsh-next` folder, shows how many samples the exclusion list removed and which classes were dropped. Then open `RUNS_DIR/data_checks/` (default `~/sparsh_next_runs/data_checks/`):

- `class_counts.csv`: confirm these are the classes you expect after the label merges and exclusions.
- `source_by_class.csv`: classes that come almost entirely from one Source_Dataset are listed in the job output.
- `missing_by_source.csv`: probes missing from a whole dataset (a platform fingerprint).
- `duplicates.csv`: sample pairs that are near-identical. If they are the same patient or sample, set `GROUPS_FILE:=$RUNS_DIR/data_checks/groups.csv` in `jobs/settings.sh` so they always fall in the same fold.

## 8. Train the comparison runs

```bash
bash jobs/submit_experiments.sh
qstat -u $USER
```

This submits five GPU jobs (legacy, default, scaled, mask_sim, schedule) to the queues in `TRAIN_QUEUES`, each writing to its own folder in `RUNS_DIR`. A run never overwrites an existing folder. Each job's log is `sn_<recipe>.o<jobid>` in the `sparsh-next` folder, and `training_log.txt` in the run folder.

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

## Applying updates from Claude

Updates arrive in `~/inbox/from_claude` as `.patch` files. Each one becomes a new commit with its own message:

```bash
cd ~/projects/sparsh-next
git am ~/inbox/from_claude/<patch file>
git log --oneline | head -3
```

If git stops with "Please tell me who you are", nothing has been applied yet. Set your name and email for this repository only (the `patkarlab` account may be shared) and run the same `git am` again:

```bash
git config user.name "Your Name"
git config user.email "you@example.org"
git am ~/inbox/from_claude/<patch file>
```

If git stops with "does not match index", a file the patch changes has uncommitted edits of yours. Run `git am --abort`, then commit those edits (or discard them with `git checkout -- <file>`), and apply again. Push afterwards (`git push`) so GitHub has the update.

## Removing SPARSH-next

Nothing else depends on it. Delete `~/projects/sparsh-next` and `RUNS_DIR`, run `conda env remove --name sparsh_next`, and delete the `sparsh-next` repository on GitHub (Settings > Danger Zone). If nothing else uses them, also remove the `github-patkarlab` block from `~/.ssh/config`, the two `~/.ssh/id_ed25519_patkarlab` files, and the key under Settings > SSH and GPG keys on the patkarlab account.
