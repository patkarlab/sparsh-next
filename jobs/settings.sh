# SPARSH-next job settings. Edit these once; every job in jobs/ reads this file.
# Nothing here points at the existing sparsh repository, its outputs or its environment.

DATA_PATH=/path/to/methylation.pkl         # the training pickle you use now (read only, never modified)
RUNS_DIR=$HOME/sparsh_next_runs            # each run gets its own new folder inside this
CONDA_ENV=sparsh_next                      # the cloned environment from docs/SETUP.md step 5
EXCLUDE_IDS=""                             # optional: the sample exclusion (junk) list you use now
EXCLUDE_PREFIXES="MPAL AML_NOS B-ALL_NOS"  # classes left out, as in the v0.1.0 job
GROUPS_FILE=""                             # optional: $RUNS_DIR/data_checks/groups.csv once duplicates are reviewed
