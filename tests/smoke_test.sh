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
step compare      python scripts/compare_runs.py "$TMP/run_legacy" "$TMP/run_new"

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
print("contents ok")
PY
echo "SMOKE TEST PASSED"
