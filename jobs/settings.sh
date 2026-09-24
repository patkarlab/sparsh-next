# SPARSH-next job settings. Every job in jobs/ reads this file; check it with
#   bash jobs/check_settings.sh
# A value passed with "qsub -v NAME=value" takes precedence over the value here.
# Queues, resources and environment follow the lab's working job (~/train_focal_fast.pbs).

# Training data, only read. DATA_PATH: full path of the training pickle. EXCLUDE_IDS: full path of its
# sample exclusion list (one Sample_ID per line), or leave it empty for none.
: "${DATA_PATH:=}"
: "${EXCLUDE_IDS=}"

: "${RUNS_DIR:=$HOME/sparsh_next_runs}"      # each run gets its own new folder here (about 8 GB of weights per run)
: "${CONDA_ENV:=sparsh_next}"                # clone of meth_sim (docs/SETUP.md step 4)
: "${CONDA_BASE:=$HOME/miniconda3}"
: "${CUDA_MODULE=cuda/12.3}"                 # module loaded before activation, as in train_focal_fast.pbs; "" = none

# GPU queues for jobs/submit_experiments.sh, used in turn for the five runs (a40 allows 2 per user at a
# time; further runs wait). Example after jobs/probe_gpus.sh: "a40 a40 A40b h100 h200"
: "${TRAIN_QUEUES:=a40}"

# Classes left out: MPAL, AML_NOS and B-ALL_NOS, as in the v0.1.0 job template.
# train_focal_fast.pbs leaves out only MPAL; to match it, change this to "MPAL".
: "${EXCLUDE_PREFIXES:=MPAL AML_NOS B-ALL_NOS}"
: "${GROUPS_FILE:=}"                         # optional: CSV Sample_ID,group (e.g. $RUNS_DIR/data_checks/groups.csv)
