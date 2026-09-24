#!/bin/bash
# End-to-end check on synthetic data: CPU only, about one minute, writes only to a
# temporary folder that is deleted afterwards. Run from the sparsh-next folder:
#   bash tests/smoke_test.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

step() {
    local name=$1; shift
    if "$@" > "$TMP/$name.log" 2>&1; then
        echo "ok    $name"
    else
        echo "FAIL  $name"; tail -30 "$TMP/$name.log"; exit 1
    fi
}

step make_data    python tests/make_synthetic.py --output_dir "$TMP/data"
step check_data   python scripts/check_data.py --data_path "$TMP/data/train.pkl" --output_dir "$TMP/checks" \
                      --exclude_prefixes MPAL
printf '# samples to leave out\nGSM100001\n\nGSM100002,replicate\n' > "$TMP/exclude.txt"
step exclusion    python scripts/check_data.py --data_path "$TMP/data/train.pkl" --output_dir "$TMP/checks_excl" \
                      --exclude_prefixes MPAL --exclude_ids "$TMP/exclude.txt"
grep -q "Excluding 2 listed samples" "$TMP/exclusion.log" || { echo "FAIL  exclusion list not applied"; exit 1; }
step train_new    python scripts/train.py --data_path "$TMP/data/train.pkl" --output_dir "$TMP/run_new" \
                      --exclude_prefixes MPAL --groups_file "$TMP/checks/groups.csv" \
                      --n_folds 2 --epochs 3 --hidden_dims 32 16 --device cpu --final_model
step train_legacy python scripts/train.py --data_path "$TMP/data/train.pkl" --output_dir "$TMP/run_legacy" \
                      --exclude_prefixes MPAL --n_folds 2 --epochs 3 --hidden_dims 32 16 --device cpu \
                      --imbalance legacy_upsample --train_sim mask --val_sim mask --coverage_mode schedule
step predict      python scripts/predict.py --model_dir "$TMP/run_new" --ont_dir "$TMP/data/ont" \
                      --ground_truth "$TMP/data/ground_truth.csv" --output_dir "$TMP/pred" --device cpu
step compare      python scripts/compare_runs.py "$TMP/run_legacy" "$TMP/run_new" --output "$TMP/comparison.csv"
step ont_coverage python scripts/ont_coverage.py --run "$TMP/run_new" "$TMP/data/ont" --output "$TMP/ont_coverage.csv"
step duplicates   python - "$TMP" <<'PY'
import sys
import numpy as np
from data.ont import read_ont_csv
f = f"{sys.argv[1]}/repeated_probes.csv"
open(f, "w").write("sample,cg1,cg2,cg1,cg3,cg2\nS1,1,,0,0.5,1\n")
cpgs = ["cg1", "cg2", "cg3", "cg4"]
x, info = read_ont_csv(f, cpgs)                        # default: mean of the copies with a value
assert np.allclose(x[:3], [0.5, 1.0, 0.5]) and np.isnan(x[3]), x
assert (info["n_repeated_probes"], info["n_repeated_multi_observed"], info["n_repeated_disagree"]) == (2, 1, 1), info
assert (info["n_observed"], info["n_observed_first"], info["max_copies"]) == (3, 2, 2), info
x, _ = read_ont_csv(f, cpgs, duplicates="first")
assert x[0] == 1.0 and np.isnan(x[1]), x
try:
    read_ont_csv(f, cpgs, duplicates="error")
    raise SystemExit("the error rule did not refuse repeated probes")
except ValueError:
    pass
print("repeated probes ok")
PY
step benchmark    python scripts/gpu_benchmark.py --device cpu --n_cpgs 2000 --hidden_dims 16 --epochs 1 \
                      --largest_class 20 --n_inner 5 --n_test 5

for f in run_new/cv_metrics_by_condition.csv run_new/fold_models/fold1.pt run_new/model.pt \
         run_new/config.json pred/predictions.csv pred/summary_metrics.json checks/groups.csv; do
    [ -s "$TMP/$f" ] || { echo "FAIL  expected output missing: $f"; exit 1; }
done
step contents python - "$TMP" <<'PY'
import json, sys
import pandas as pd
tmp = sys.argv[1]
s = json.load(open(f"{tmp}/pred/summary_metrics.json"))
assert s["n_with_truth"] == 8, s["n_with_truth"]
assert s["n_out_of_scheme"] == 1, "the out-of-scheme sample must be reported, not dropped"
g = pd.read_csv(f"{tmp}/checks/groups.csv")
assert len(g) == 2, "the planted duplicate pair must be found"
m = pd.read_csv(f"{tmp}/run_new/cv_metrics_by_condition.csv")
assert {"dense", "reads_0.20", "mask_0.20"} <= set(m["condition"])
c = pd.read_csv(f"{tmp}/comparison.csv", index_col=0)
assert "callable_share_at_accuracy_0.98" in set(c["metric"]), "calibration-free table missing"
assert not any(str(i).startswith(("dense", "mask_")) for i in c.index), "only reads_* rows by default"
o = pd.read_csv(f"{tmp}/ont_coverage.csv")
assert len(o) == 8 and o["error"].isna().all(), o
assert o["coverage_pct"].between(10, 40).all() and (o["share_0_or_1"] > 0.8).all(), o
print("contents ok")
PY
echo "SMOKE TEST PASSED"
