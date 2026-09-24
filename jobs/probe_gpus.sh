#!/bin/bash
# Submit jobs/gpu_benchmark.pbs once to each GPU queue: GPU model, PyTorch support and the time of
# one full-size training epoch, with random numbers (no data read, nothing trained or saved).
# From the sparsh-next folder:
#   bash jobs/probe_gpus.sh                 # queues a40 A40b h100 h200 sch
#   bash jobs/probe_gpus.sh a40 h200        # chosen queues
# Results, once the jobs have finished (qstat -u $USER):
#   grep -h -E "^(Queue|Node|GPU|RESULT|Warm|Peak)" ~/sparsh_next_runs/gpu_benchmark/*.log
cd "$(dirname "$0")/.." || exit 1
source jobs/settings.sh
out="$RUNS_DIR/gpu_benchmark"
mkdir -p "$out"
queues=("$@")
[ ${#queues[@]} -gt 0 ] || queues=(a40 A40b h100 h200 sch)
for q in "${queues[@]}"; do
  if job=$(qsub -q "$q" -N "sn_bench_$q" -o "$out/$q.log" jobs/gpu_benchmark.pbs 2>&1); then
    echo "$q: submitted ($job)"
  else
    echo "$q: not accepted: $job"
  fi
done
echo "When the jobs have finished:"
echo "  grep -h -E \"^(Queue|Node|GPU|RESULT|Warm|Peak)\" $out/*.log"
