#!/bin/bash
# Checks for the fifth-round additions (relabelled training pickle, merged groups, NPM1/IDH check with island
# decisions, locked recipe arguments) on synthetic data: CPU only, about a minute, writes only to a temporary
# folder that is deleted afterwards. Run from the sparsh-next folder:
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
                      --relabel "$TMP/idh/relabel_AML_HOX_IDH.csv" "$TMP/relabel_b.csv"
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
                      --eval_blasts 1 0.5 0.3 --eval_sims binary --normal_class T-ALL \
                      --exclude_classes B-ALL_TCF3-PBX1 --groups_file "$TMP/groups_merged.csv" \
                      --min_samples 2
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
assert set(ch["Sample_ID"]) == set(rel["Sample_ID"]) | set(pd.read_csv(f"{tmp}/relabel_b.csv", dtype=str)["Sample_ID"])
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
print("fifth-round contents ok")
PY
step train_pbs_recipe bash -c "grep -q 'locked)' jobs/train.pbs && grep -q 'exclude_classes AML-MR AML_MECOM-r' jobs/train.pbs"
echo "ROUND 5 CHECKS PASSED"
