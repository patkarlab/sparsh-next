#!/bin/bash
# Submit the five comparison runs, one GPU job each. Run from the sparsh-next folder:
#   bash jobs/submit_experiments.sh
set -e
for recipe in legacy default scaled mask_sim schedule; do
  qsub -N "sn_${recipe}" -v RECIPE="${recipe}" jobs/train.pbs
done
echo "Submitted. Check progress with: qstat -u \$USER"
