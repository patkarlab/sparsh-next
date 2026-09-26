#!/bin/bash
# Checks for the fifth-round additions (relabelled training pickle with class renames, merged groups, NPM1/IDH
# check with island decisions, locked recipe arguments, label audit, comparison of two class schemes) on synthetic
# data: CPU only, one to two minutes, writes only to a temporary folder that is deleted afterwards. Run from the
# sparsh-next folder:
#   bash tests/smoke_test_round5.sh
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
step lists        python - "$TMP" <<'PY'
import sys
import numpy as np
import pandas as pd
tmp = sys.argv[1]
df = pd.read_pickle(f"{tmp}/data/train.pkl")
ids = (df["Sample_ID"] if "Sample_ID" in df.columns else pd.Series(df.index)).astype(str).to_numpy()
lab = df["ANNOTATION"].astype(str).to_numpy()
npm1 = ids[lab == "AML_mutated NPM1"]
assert len(npm1) >= 40, len(npm1)
g = pd.DataFrame({"train_id": npm1[:36], "cohort": ["A"] * 18 + ["B"] * 18,
                  "group": (["IDH_mut"] * 6 + ["IDH_wt"] * 12) * 2, "IDH": "", "DNMT3A": "", "TET2": ""})
g.loc[g["group"] == "IDH_mut", "IDH"] = "IDH1 p.R132H"
g.loc[[7, 8], "TET2"] = "p.Q1034*"
g.to_csv(f"{tmp}/genotypes.csv", index=False)
isl = pd.DataFrame({"Sample_ID": [npm1[0], npm1[1], npm1[6], npm1[36], npm1[37], npm1[38], "NOT_A_SAMPLE"],
                    "evidence": ["genotype_IDH_mut", "genotype_IDH_mut", "genotype_IDH_wt", "unknown", "unknown",
                                 "known_fusion:NPM1-MLF1", "unknown"]})
isl.to_csv(f"{tmp}/island.csv", index=False)
kmt = ids[lab == "AML_KMT2A-r"][:5]
pd.DataFrame({"Sample_ID": kmt, "new_label": "AML_KMT2A_test", "reason": "test"}).to_csv(f"{tmp}/relabel_b.csv", index=False)
pd.DataFrame({"Sample_ID": [kmt[0]], "new_label": "OTHER"}).to_csv(f"{tmp}/relabel_clash.csv", index=False)
pd.DataFrame({"Sample_ID": ["X_missing"], "new_label": "Y"}).to_csv(f"{tmp}/relabel_unknown.csv", index=False)
pd.DataFrame({"Sample_ID": ids[:4], "group": ["d1", "d1", "d2", "d2"]}).to_csv(f"{tmp}/groups.csv", index=False)
pd.DataFrame({"Sample_ID": [ids[1], ids[2], ids[10], ids[11]], "patient": ["p1", "p1", "p2", "p2"]}).to_csv(
    f"{tmp}/patients.csv", index=False)
print("lists written")
PY
step idh_check    python scripts/npm1_idh_check.py --data_path "$TMP/data/train.pkl" --genotypes "$TMP/genotypes.csv" \
                      --npm1_class "AML_mutated NPM1" --island "$TMP/island.csv" --new_label AML_HOX_IDH \
                      --out_dir "$TMP/idh" --repeats 2
step pickle       python scripts/make_training_pickle.py --data_path "$TMP/data/train.pkl" --output "$TMP/new.pkl" \
                      --relabel "$TMP/idh/relabel_AML_HOX_IDH.csv" "$TMP/relabel_b.csv" --rename "T-ALL=T-ALL_renamed"
step rename_bad   bash -c "python scripts/make_training_pickle.py --data_path '$TMP/data/train.pkl' --output '$TMP/z.pkl' \
                      --rename NO_SUCH_CLASS=X 2>&1 | grep -q 'no sample has the label'"
step pickle_again bash -c "python scripts/make_training_pickle.py --data_path '$TMP/data/train.pkl' --output '$TMP/new.pkl' \
                      2>&1 | grep -q 'exists'"
step clash        bash -c "python scripts/make_training_pickle.py --data_path '$TMP/data/train.pkl' --output '$TMP/x.pkl' \
                      --relabel '$TMP/relabel_b.csv' '$TMP/relabel_clash.csv' 2>&1 | grep -q 'different new labels'"
step unknown_id   bash -c "python scripts/make_training_pickle.py --data_path '$TMP/data/train.pkl' --output '$TMP/y.pkl' \
                      --relabel '$TMP/relabel_unknown.csv' 2>&1 | grep -q 'not in the pickle'"
step groups       python scripts/merge_groups.py --groups "$TMP/groups.csv" --patients "$TMP/patients.csv" \
                      --output "$TMP/groups_merged.csv"
step train_locked python scripts/train.py --data_path "$TMP/new.pkl" --output_dir "$TMP/run_locked" \
                      --exclude_prefixes MPAL --n_folds 2 --epochs 3 --hidden_dims 32 16 --device cpu \
                      --train_sim binary --val_sim binary --eval_sims binary oneread --cov_min 0.02 --cov_max 0.95 \
                      --coverage_dist uniform --input_encoding scaled --val_coverages 0.1 0.5 0.9 \
                      --eval_coverages 0.1 0.5 0.9 --dilution_prob 0.5 --blast_min 0.2 --val_dilution \
                      --eval_blasts 1 0.5 0.3 --eval_sims binary --normal_class T-ALL_renamed \
                      --exclude_classes B-ALL_TCF3-PBX1 --groups_file "$TMP/groups_merged.csv" \
                      --min_samples 2
step train_orig   python scripts/train.py --data_path "$TMP/data/train.pkl" --output_dir "$TMP/run_orig" \
                      --exclude_prefixes MPAL --n_folds 2 --epochs 3 --hidden_dims 32 16 --device cpu \
                      --train_sim binary --val_sim binary --cov_min 0.02 --cov_max 0.95 \
                      --coverage_dist uniform --input_encoding scaled --val_coverages 0.1 0.5 0.9 \
                      --eval_coverages 0.1 0.5 0.9 --dilution_prob 0.5 --blast_min 0.2 --val_dilution \
                      --eval_blasts 1 0.5 0.3 --eval_sims binary --normal_class T-ALL \
                      --exclude_classes B-ALL_TCF3-PBX1 --groups_file "$TMP/groups_merged.csv" \
                      --min_samples 2
step schemes      python scripts/compare_class_schemes.py "$TMP/run_orig" "$TMP/run_locked" \
                      --rename T-ALL=T-ALL_renamed --conditions binary_0.50 binary_0.90 --output "$TMP/schemes.csv"
step schemes_bad  bash -c "python scripts/compare_class_schemes.py '$TMP/run_orig' '$TMP/run_locked' \
                      --rename NO_SUCH_CLASS=X --conditions binary_0.50 2>&1 | grep -q 'not a class'"
step schemes_rule python - "$TMP" <<'PY'
# The rule on hand-made predictions: attribution of failing shared classes (amendment of 25 September 2026),
# --allowed_samples, and a round without new classes.
import subprocess
import sys
from pathlib import Path
import pandas as pd
tmp = Path(sys.argv[1]) / "rule"
def write(name, rows):
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["sample_id", "true_label", "prediction"]).to_csv(d / "cv_predictions_binary_0.30.csv", index=False)
old = [(f"s{i:02d}", c, c) for c, r in (("A", range(1, 11)), ("B", range(11, 21)), ("C", range(21, 31)),
                                        ("D", range(31, 41))) for i in r]
new = []
for s, t, _ in old:
    i = int(s[1:])
    t2 = "D_new" if 31 <= i <= 35 else t
    p = "D_new" if i in (1, 2) else ("C" if i in (11, 12) else t2)   # A loses 2 to D_new, B loses 2 to C
    new.append((s, t2, p))
noc = [(s, t, "C" if s in ("s11", "s12") else p) for s, t, p in old]  # no new class; B loses 2 to C
write("a", old); write("b", new); write("c", noc)
def run(first, second, *extra):
    r = subprocess.run([sys.executable, "scripts/compare_class_schemes.py", str(tmp / first), str(tmp / second),
                        "--max_mean_drop", "0.5", *extra], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout
out = run("a", "b")
assert "D_new: recall 1.000, merge it back: A lost 2 to it and fails" in out, out
assert "block no new class): B" in out, out
out = run("a", "b", "--allowed_samples", "2")
assert "D_new: recall 1.000, keep" in out and "max(0.05, 2 samples): yes" in out, out
assert "Failing shared classes" not in out, out
out = run("a", "c")
assert "No new classes: the change fails the rule" in out and "block the change): B" in out, out
out = run("a", "c", "--allowed_samples", "2")
assert "No new classes: keep the change" in out, out
print("rule checks ok")
PY
step audit        python scripts/label_audit.py --data_path "$TMP/new.pkl" --classes AML B-ALL_High \
                      --run "$TMP/run_locked" --n_cpgs 2000 --n_pcs 10 --k 5 --out_dir "$TMP/audit"
step audit_norun  python scripts/label_audit.py --data_path "$TMP/data/train.pkl" --n_cpgs 1000 --n_pcs 5 --k 5 \
                      --out_dir "$TMP/audit_norun"
step contents     python - "$TMP" <<'PY'
import json
import os
import sys
import pandas as pd
tmp = sys.argv[1]
old = pd.read_pickle(f"{tmp}/data/train.pkl")
new = pd.read_pickle(f"{tmp}/new.pkl")
assert new.shape[0] == old.shape[0] and "ANNOTATION_ORIGINAL" in new.columns
assert (new["ANNOTATION_ORIGINAL"].astype(str).to_numpy() == old["ANNOTATION"].astype(str).to_numpy()).all()
dec = pd.read_csv(f"{tmp}/idh/island_decisions.csv", dtype=str)
move = dict(zip(dec["Sample_ID"], dec["decision"]))
isl = pd.read_csv(f"{tmp}/island.csv", dtype=str)
e = dict(zip(isl["Sample_ID"], isl["evidence"]))
for s, ev in e.items():
    if s == "NOT_A_SAMPLE":
        assert s not in move
    elif ev == "genotype_IDH_mut":
        assert move[s].startswith("move"), (s, move[s])
    elif ev in ("genotype_IDH_wt", "known_fusion:NPM1-MLF1"):
        assert move[s] == "keep label", (s, move[s])
rel = pd.read_csv(f"{tmp}/idh/relabel_AML_HOX_IDH.csv", dtype=str)
ch = pd.read_csv(f"{tmp}/new.pkl.changes.csv", dtype=str)
assert set(ch.loc[ch["file"] != "--rename", "Sample_ID"]) == \
    set(rel["Sample_ID"]) | set(pd.read_csv(f"{tmp}/relabel_b.csv", dtype=str)["Sample_ID"])
assert (ch.loc[ch["file"] == "--rename", "new_label"] == "T-ALL_renamed").all()
ids = (new["Sample_ID"] if "Sample_ID" in new.columns else pd.Series(new.index, index=new.index)).astype(str)
lab = dict(zip(ids, new["ANNOTATION"].astype(str)))
assert all(lab[s] == "AML_HOX_IDH" for s in rel["Sample_ID"])
g = pd.read_csv(f"{tmp}/groups_merged.csv", dtype=str)
assert g["group"].nunique() == 2 and len(g) == 6, g          # {0,1,2,3} joined through p1, and {10,11}
cfg = json.load(open(f"{tmp}/run_locked/config.json"))
assert "AML_HOX_IDH" in json.dumps(cfg["data"].get("class_counts", {})), "new class missing from the model"
assert cfg["data"]["groups_file"].endswith("groups_merged.csv")
for f in ("summary.txt", "cv_auc.csv", "scores.csv", "differential_cpgs.csv"):
    assert os.path.exists(f"{tmp}/idh/{f}"), f
assert (lab_all := new["ANNOTATION"].astype(str)).str.startswith("T-ALL_renamed").sum() > 0 and not (lab_all == "T-ALL").any()
au = pd.read_csv(f"{tmp}/audit/audit.csv")
assert {"same_label_of_5", "neighbour_majority", "cv_own_prob_mean", "cv_own_prob_binary_0.90", "flag_lamprey",
        "flag_review"} <= set(au.columns), au.columns
assert au["cv_own_prob_mean"].notna().sum() > 0 and au["label"].str.startswith(("AML", "B-ALL_High")).all()
assert au["same_label_of_5"].between(0, 5).all() and au["same_label_fraction"].between(0, 1).all()
size = au["label"].map(au["label"].value_counts())
small = size <= 5                                         # fewer possible same-label neighbours than k
assert small.any(), "no small class in the audit"
assert ((au.loc[small, "same_label_fraction"] * (size[small] - 1)).round() == au.loc[small, "same_label_of_5"]).all()
assert os.path.exists(f"{tmp}/audit_norun/flagged.csv")
sc = pd.read_csv(f"{tmp}/schemes.csv")
assert set(sc["condition"]) == {"binary_0.50", "binary_0.90"}, sc["condition"].unique()
s5 = sc[sc["condition"] == "binary_0.50"].set_index("class")
assert set(s5.index[s5["kind"] == "new"]) == {"AML_HOX_IDH", "AML_KMT2A_test"}, s5
assert s5.loc["T-ALL_renamed", "kind"] == "shared"
po = pd.read_csv(f"{tmp}/run_orig/cv_predictions_binary_0.50.csv")
pl = pd.read_csv(f"{tmp}/run_locked/cv_predictions_binary_0.50.csv")
assert s5.loc["AML_KMT2A-r", "n"] == (po["true_label"] == "AML_KMT2A-r").sum() - 5      # 5 moved to the new class
assert s5.loc["T-ALL_renamed", "n"] == (po["true_label"] == "T-ALL").sum()
t = pl[pl["true_label"] == "AML_KMT2A_test"]
assert abs(s5.loc["AML_KMT2A_test", "recall_second"] - (t["prediction"] == "AML_KMT2A_test").mean()) < 1e-3
assert (s5.loc[s5["kind"] == "shared", "allowed_drop"] >= 0.05 - 1e-9).all()
print("fifth-round contents ok")
PY
step train_pbs_recipe bash -c "grep -q 'locked)' jobs/train.pbs && grep -q 'exclude_classes AML-MR AML_MECOM-r' jobs/train.pbs"
# prepare_relabel.pbs outside PBS, with a stand-in conda: default TAG, refusal to overwrite, a second TAG
mkdir -p "$TMP/pbs/rel" "$TMP/pbs/conda/etc/profile.d"
printf 'conda() { export CONDA_DEFAULT_ENV="$2"; }\n' > "$TMP/pbs/conda/etc/profile.d/conda.sh"
printf 'DATA_PATH=%s\nRUNS_DIR=%s\nEXCLUDE_IDS=\nGROUPS_FILE=\nCONDA_ENV=test\nCONDA_BASE=%s\n' \
    "$TMP/data/train.pkl" "$TMP/pbs/runs" "$TMP/pbs/conda" > "$TMP/pbs/settings.sh"
cp "$TMP/relabel_b.csv" "$TMP/pbs/rel/relabel_b.csv"
for tag in 25Sep2026 26Sep2026; do
    cut -d, -f1 "$TMP/relabel_b.csv" | sed -n '2,3p' > "$TMP/pbs/rel/drop_$tag.txt"
    cp "$TMP/patients.csv" "$TMP/pbs/rel/patients_$tag.csv"
done
export PBS_ENV="SETTINGS_FILE=$TMP/pbs/settings.sh RELABEL_DIR=$TMP/pbs/rel NEW_DATA_PATH= RENAMES=T-ALL=T-ALL_x"
step relabel_pbs  bash -c "env $PBS_ENV bash jobs/prepare_relabel.pbs && \
                      test -f '$TMP/pbs/runs/data_checks/exclude_25Sep2026.txt' && test -f '$TMP/data/AL_25Sep2026_relabel.pkl'"
step relabel_pbs_again bash -c "env $PBS_ENV bash jobs/prepare_relabel.pbs 2>&1 | grep -q 'exists; choose a new TAG'"
step relabel_pbs_tag bash -c "env $PBS_ENV TAG=26Sep2026 bash jobs/prepare_relabel.pbs && \
                      test -f '$TMP/pbs/runs/data_checks/groups_26Sep2026.csv' && test -f '$TMP/data/AL_26Sep2026_relabel.pkl'"
echo "ROUND 5 CHECKS PASSED"
