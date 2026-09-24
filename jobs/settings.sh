# SPARSH-next job settings. Every job in jobs/ reads this file.
# Values are pre-filled from the lab's working job (~/train_focal_fast.pbs); check them with
#   bash jobs/check_settings.sh
# A value passed with "qsub -v NAME=value" takes precedence over the value here.
# The existing sparsh folder is only read (training pickle and exclusion list); nothing is written there.

# Folder that holds data/imputed_nonbin_greater5.pkl (the folder train_focal_fast.pbs is submitted from)
: "${SPARSH_DIR:=$HOME/projects/sparsh}"
: "${DATA_PATH:=$SPARSH_DIR/data/imputed_nonbin_greater5.pkl}"   # training pickle (read only)
: "${EXCLUDE_IDS=$SPARSH_DIR/data/ids_to_junk_fs.txt}"           # sample exclusion list; set to "" for none

: "${RUNS_DIR:=$HOME/sparsh_next_runs}"      # each run gets its own new folder here (about 8 GB of weights per run)
: "${CONDA_ENV:=sparsh_next}"                # clone of meth_sim (docs/SETUP.md step 4)
: "${CONDA_BASE:=$HOME/miniconda3}"
: "${CUDA_MODULE=cuda/12.3}"                 # module loaded before activation, as in train_focal_fast.pbs; "" = none

# Classes left out: MPAL, AML_NOS and B-ALL_NOS, as in the v0.1.0 job template.
# train_focal_fast.pbs leaves out only MPAL; to match it, change this to "MPAL".
: "${EXCLUDE_PREFIXES:=MPAL AML_NOS B-ALL_NOS}"
: "${GROUPS_FILE:=}"                         # optional: $RUNS_DIR/data_checks/groups.csv after review (SETUP.md step 7)
