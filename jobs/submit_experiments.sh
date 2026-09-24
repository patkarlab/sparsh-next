#!/bin/bash
# Submit comparison runs, one GPU job each. From the sparsh-next folder:
#   bash jobs/submit_experiments.sh                                      # the five first-round runs
#   bash jobs/submit_experiments.sh wide scaled_wide scaled_wide_log     # any recipes from jobs/train.pbs
# The settings are checked first (jobs/check_settings.sh); nothing is submitted if a check fails.
# Runs go to the queues in TRAIN_QUEUES (jobs/settings.sh) in turn.
# The current jobs/settings.sh is copied into RUNS_DIR and every job reads that copy,
# so editing settings.sh while jobs wait in the queue cannot change them.
set -e
cd "$(dirname "$0")/.."
bash jobs/check_settings.sh || { echo "Nothing submitted."; exit 1; }
source jobs/settings.sh
snapshot="$RUNS_DIR/settings_submitted_$(date +%Y%m%d_%H%M%S).sh"
cp jobs/settings.sh "$snapshot"
read -r -a queues <<< "${TRAIN_QUEUES:-a40}"
i=0
recipes=("$@")
[ ${#recipes[@]} -gt 0 ] || recipes=(legacy default scaled mask_sim schedule)
for recipe in "${recipes[@]}"; do
  if [ -e "$RUNS_DIR/$recipe" ]; then
    echo "Skipping $recipe: $RUNS_DIR/$recipe already exists (runs never overwrite each other)"
    continue
  fi
  queue="${queues[$((i % ${#queues[@]}))]}"
  i=$((i + 1))
  echo "$recipe -> queue $queue"
  qsub -q "$queue" -N "sn_${recipe}" -v RECIPE="${recipe}",SETTINGS_FILE="${snapshot}" jobs/train.pbs
done
echo "Settings used: $snapshot"
echo "Check progress with: qstat -u \$USER"
