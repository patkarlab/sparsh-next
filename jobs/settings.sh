# SPARSH-next job settings. Edit the values after ":=" once; every job in jobs/ reads this file.
# A value passed with "qsub -v NAME=value" takes precedence over the value here.
# Nothing here points at the existing sparsh repository, its outputs or its environment.

: "${DATA_PATH:=/path/to/methylation.pkl}"          # the training pickle you use now (read only, never modified)
: "${RUNS_DIR:=$HOME/sparsh_next_runs}"             # each run gets its own new folder here; about 8 GB of
                                                    # model weights per run, so use a large disk if $HOME is small
: "${CONDA_ENV:=sparsh_next}"                       # the cloned environment from docs/SETUP.md step 4
: "${EXCLUDE_IDS:=}"                                # optional: the sample exclusion (junk) list you use now
: "${EXCLUDE_PREFIXES:=MPAL AML_NOS B-ALL_NOS}"     # classes left out, as in the v0.1.0 job
: "${GROUPS_FILE:=}"                                # optional: $RUNS_DIR/data_checks/groups.csv after review
