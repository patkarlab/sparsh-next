#!/bin/bash
# Submit the five comparison runs, one GPU job each. From the sparsh-next folder:
#   bash jobs/submit_experiments.sh
# The settings are checked first (jobs/check_settings.sh); nothing is submitted if a check fails.
# The current jobs/settings.sh is copied into RUNS_DIR and every job reads that copy,
# so editing settings.sh while jobs wait in the queue cannot change them.
set -e
cd "$(dirname "$0")/.."
bash jobs/check_settings.sh || { echo "Nothing submitted."; exit 1; }
source jobs/settings.sh
snapshot="$RUNS_DIR/settings_submitted_$(date +%Y%m%d_%H%M%S).sh"
cp jobs/settings.sh "$snapshot"
for recipe in legacy default scaled mask_sim schedule; do
  if [ -e "$RUNS_DIR/$recipe" ]; then
    echo "Skipping $recipe: $RUNS_DIR/$recipe already exists (runs never overwrite each other)"
    continue
  fi
  qsub -N "sn_${recipe}" -v RECIPE="${recipe}",SETTINGS_FILE="${snapshot}" jobs/train.pbs
done
echo "Settings used: $snapshot"
echo "Check progress with: qstat -u \$USER"
