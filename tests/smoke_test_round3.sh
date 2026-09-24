#!/bin/bash
# Checks for the third-round additions (per-read call errors, AML_other grouping, reported-call fallback,
# excluded-class check, call-error estimate) on synthetic data: CPU only, about 30 seconds, writes only to a
# temporary folder that is deleted afterwards. Run from the sparsh-next folder, after tests/smoke_test.sh:
#   bash tests/smoke_test_round3.sh
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
step train_wide   python scripts/train.py --data_path "$TMP/data/train.pkl" --output_dir "$TMP/run_wide" \
                      --exclude_prefixes MPAL --n_folds 2 --epochs 3 --hidden_dims 32 16 --device cpu \
                      --train_sim binary --val_sim binary --eval_sims binary oneread --cov_max 0.95 \
                      --coverage_dist uniform --input_encoding scaled --val_coverages 0.1 0.5 0.9 \
                      --eval_coverages 0.1 0.5 0.9
step ont_coverage python scripts/ont_coverage.py --run "$TMP/run_wide" "$TMP/data/ont" --output "$TMP/ont_coverage.csv"
# Third round: per-read call errors, AML_other grouping, family fallback, excluded classes, call-error estimate
cat > "$TMP/label_map_other.json" <<'EOF'
{"merge": {"AML_mutated NPM1": "AML_mutated_NPM1_Nup98_DEKnup214", "AML_NUP98-r": "AML_mutated_NPM1_Nup98_DEKnup214",
           "B-ALL_PAX5 alt": "B-ALL_PAX5_BCR-ABL1_B-ALL_BCR-ABL1_like",
           "B-ALL_BCR-ABL1 like": "B-ALL_PAX5_BCR-ABL1_B-ALL_BCR-ABL1_like",
           "B-ALL_BCR-ABL1": "B-ALL_PAX5_BCR-ABL1_B-ALL_BCR-ABL1_like", "AML_KMT2A-r": "AML_other"}}
EOF
cat > "$TMP/hierarchy.json" <<'EOF'
{"lineage_by_prefix": [["AML", "AML"], ["B-ALL", "B-ALL"], ["T-ALL", "T-ALL"]],
 "families": {"B-ALL, PAX5 group": ["B-ALL_PAX5_BCR-ABL1_B-ALL_BCR-ABL1_like", "B-ALL_PAX5 P80R"],
              "family absent from this model": ["X1", "X2"]},
 "no_subtype_classes": ["AML_other"]}
EOF
step train_err    python scripts/train.py --data_path "$TMP/data/train.pkl" --output_dir "$TMP/run_err" \
                      --exclude_prefixes MPAL --n_folds 2 --epochs 3 --hidden_dims 32 16 --device cpu \
                      --train_sim binary --val_sim binary --eval_sims binary oneread --cov_max 0.95 \
                      --coverage_dist uniform --input_encoding scaled --val_coverages 0.1 0.5 0.9 \
                      --eval_coverages 0.1 0.5 0.9 --call_error_max 0.2 --val_call_error 0.1 --eval_call_errors 0 0.1 \
                      --label_map "$TMP/label_map_other.json" --exclude_classes B-ALL_TCF3-PBX1
step compare_err  python scripts/compare_runs.py "$TMP/run_wide" "$TMP/run_err" \
                      --ont_coverage "$TMP/ont_coverage.csv" --output "$TMP/comparison_err.csv"
step hierarchy    python scripts/hierarchy_report.py "$TMP/run_err" "$TMP/run_wide" --hierarchy "$TMP/hierarchy.json" \
                      --ont_coverage "$TMP/ont_coverage.csv"
step excluded     python scripts/score_excluded.py --run "$TMP/run_err" --hierarchy "$TMP/hierarchy.json" \
                      --coverages 0.1 0.5
step call_error   python scripts/ont_call_error.py "$TMP/data/ont" --run "$TMP/run_err" --output "$TMP/call_error.csv" \
                      --high 0.7 --low 0.3 --min_share 0.9 --n_check 40
step predict_h    python scripts/predict.py --model_dir "$TMP/run_err" --ont_dir "$TMP/data/ont" \
                      --ground_truth "$TMP/data/ground_truth.csv" --output_dir "$TMP/pred_err" --device cpu \
                      --hierarchy "$TMP/hierarchy.json"
step call_errors  python - <<'PY'
import numpy as np
import torch
from models.corruption import corrupt_numpy, corrupt_rows, corrupt_torch
X = np.random.default_rng(0).random((3, 20000)).astype(np.float32)
a = corrupt_rows(X, ["a", "b", "c"], 0.3, "binary", 12345, salt=2)
b = corrupt_rows(X, ["a", "b", "c"], 0.3, "binary", 12345, salt=2, call_error=0.0)
assert np.array_equal(a, b, equal_nan=True), "a zero call error must leave the inputs of earlier runs unchanged"
ones, zeros = np.ones(300000, np.float32), np.zeros(300000, np.float32)
for sim in ("reads", "binary", "oneread"):
    hi = np.nanmean(corrupt_numpy(ones, 0.3, sim, np.random.default_rng(1), 0.1))
    lo = np.nanmean(corrupt_numpy(zeros, 0.3, sim, np.random.default_rng(2), 0.1))
    assert 0.88 < hi < 0.92 and 0.08 < lo < 0.12, (sim, hi, lo)
t = corrupt_torch(torch.ones(2, 300000), torch.tensor([[0.5], [0.5]]), "oneread", torch.tensor([[0.0], [0.2]]))
m = [float(torch.nanmean(r)) for r in t]
assert m[0] == 1.0 and 0.78 < m[1] < 0.82, m
try:
    corrupt_numpy(ones, 0.3, "mask", np.random.default_rng(1), 0.1)
    raise SystemExit("call errors on the mask simulation were not refused")
except ValueError:
    pass
print("call errors ok")
PY
step hierarchy_none python scripts/hierarchy_report.py "$TMP/run_wide" --hierarchy none
step hierarchy_unit python - <<'PY'
import json
import tempfile
import numpy as np
from evaluation.hierarchy import hierarchical_calls, load_hierarchy, score_calls
cfg = {"lineage_by_prefix": [["T-ALL", "T-ALL"], ["AML", "AML"]],
       "families": {"T-ALL, not TAL1": ["T-ALL_TLX1", "T-ALL_TLX3", "T-ALL_NKX2"]},
       "no_subtype_classes": ["AML_other"]}
f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
json.dump(cfg, f)
f.close()
classes = ["AML_KMT2A-r", "AML_other", "T-ALL_TAL1", "T-ALL_TLX1", "T-ALL_TLX3"]   # no T-ALL_NKX2 in this model
h = load_hierarchy(f.name, classes)
probs = np.array([[0.00, 0.00, 0.05, 0.50, 0.45],     # family call
                  [0.02, 0.95, 0.01, 0.01, 0.01],     # background class: AML, no specific subtype
                  [0.50, 0.45, 0.02, 0.02, 0.01],     # lineage call: AML
                  [0.30, 0.10, 0.30, 0.20, 0.10],     # nothing reaches 0.90
                  [0.96, 0.01, 0.01, 0.01, 0.01]])    # subtype
calls = hierarchical_calls(probs, classes, h, 0.90)
assert list(calls["reported_level"]) == ["family", "no_subtype", "lineage", "none", "subtype"], calls
truth = ["T-ALL_NKX2", "AML_other", "AML NOS", "T-ALL_TAL1", "AML_KMT2A-r"]
pred = [classes[i] for i in probs.argmax(1)]
s = score_calls(truth, pred, calls, h)
assert list(s["reported_correct"].fillna(-1)) == [1.0, 1.0, 1.0, -1.0, 1.0], s
print("hierarchy ok")
PY
step contents_err python - "$TMP" <<'PY'
import json
import os
import sys
import numpy as np
import pandas as pd
tmp = sys.argv[1]
m = pd.read_csv(f"{tmp}/run_err/cv_metrics_by_condition.csv")
want = {"dense"} | {f"{s}{e}_{c}" for s in ("binary", "oneread") for e in ("", "-err10") for c in ("0.10", "0.50", "0.90")}
assert set(m["condition"]) == want, set(m["condition"]) ^ want
classes = set(json.load(open(f"{tmp}/run_err/class_mapping.json")).values())
assert "AML_other" in classes and "B-ALL_TCF3-PBX1" not in classes and "AML_KMT2A-r" not in classes, classes
t = json.load(open(f"{tmp}/run_err/config.json"))["training"]
assert t["call_error_max"] == 0.2 and t["val_call_error"] == 0.1 and t["eval_call_errors"] == [0.0, 0.1], t
c = pd.read_csv(f"{tmp}/comparison_err.csv", index_col=0)
for row in ("your ONT samples, binary-err10_*", "mean over oneread-err10_*", "binary-err10_0.50", "binary_0.50"):
    assert row in c.index, f"row {row} missing from the comparison"
h = pd.read_csv(f"{tmp}/run_err/hierarchy_report/hierarchy_by_condition.csv", index_col=0)
levels = ["share_subtype", "share_no_subtype", "share_family", "share_lineage", "share_none"]
assert np.allclose(h[levels].sum(axis=1), 1.0), h[levels]
assert (h["share_reported_any_level"] >= h["share_subtype"] - 1e-9).all()
assert all(str(i).startswith(("binary", "oneread")) for i in h.index), list(h.index)
bc = sorted(f for f in os.listdir(f"{tmp}/run_err/hierarchy_report") if f.startswith("by_class_"))
assert bc == ["by_class_binary_0.10.csv"], bc
e = pd.read_csv(f"{tmp}/run_err/excluded_check/summary.csv")
assert list(e["class"]) == ["B-ALL_TCF3-PBX1"] * 2 and (e["n"] == 7).all(), e
assert len(pd.read_csv(f"{tmp}/run_err/excluded_check/per_sample.csv")) == 14
ce = pd.read_csv(f"{tmp}/call_error.csv")
assert len(ce) == 8 and ce["matched_e_mean"].notna().all(), ce
p = pd.read_csv(f"{tmp}/pred_err/predictions.csv")
assert {"reported_call", "reported_level", "family", "lineage", "lineage_prob"} <= set(p.columns)
assert set(p["reported_level"]) <= {"subtype", "no_subtype", "family", "lineage", "none"}
s = json.load(open(f"{tmp}/pred_err/summary_metrics.json"))
assert s["reported_call"]["all_with_truth"]["n"] == 8, s["reported_call"]
print("third-round contents ok")
PY
echo "ROUND 3 CHECKS PASSED"
