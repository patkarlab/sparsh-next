#!/usr/bin/env python3
"""
Label audit of the training arrays: which samples disagree with their label?

    python scripts/label_audit.py --data_path /path/AL_24Sep2026.pkl --classes T-ALL AL_BCL11B \
        --run ~/sparsh_next_runs/dil --annotations tall_annotations.csv --out_dir label_audit_tall

Training arrays only. Two independent signals, as in the label cleaning of Lamprey
(Achterberg et al., medRxiv 2026):

1. Neighbourhood. PCA (default 100 components) of the most variable CpGs
   (default 50,000) among the audited samples; for each sample, how many of its
   k nearest neighbours (default 20) share its label, and which label most of
   them carry.
2. Cross-validation. With --run (a finished train.py run on the same samples),
   the out-of-fold probability each sample received for its own label: at
   --condition (default binary_0.90) and averaged over the undiluted binary_*
   conditions. Samples absent from the run, or whose label differs from the one
   the run used, get no value.

A sample is flagged "lamprey" when at most 1 of its k neighbours shares its label
and its mean CV probability for that label is at most 0.05 (Lamprey's rule), and
"review" when at most a quarter of the neighbours that could share its label do
(k, or the class size minus one for a class smaller than k + 1) or its CV
probability at --condition is below 0.5. Flags are for review, not automatic
relabelling. Labels are those of the training pickle after --label_map.

--annotations adds columns (for example source series and CIMP status) by Sample_ID.

Outputs in --out_dir: audit.csv (one row per audited sample), flagged.csv,
summary.txt (per-class counts and label against neighbour-majority label).
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data.dataset import load_label_map, load_training_data  # noqa: E402

LOG = []


def say(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


def cv_own_probabilities(run: Path, sample_ids, labels, condition: str):
    """Own-label CV probability at `condition` and mean over undiluted binary_* conditions."""
    files = sorted(run.glob("cv_predictions_*.csv"))
    undiluted = [f for f in files if re.fullmatch(r"cv_predictions_binary_\d\.\d+\.csv", f.name)]
    if not undiluted:
        say(f"No undiluted binary_* predictions in {run}; CV signal skipped")
        return None
    own = {}
    for f in undiluted:
        cond = f.name[len("cv_predictions_"):-len(".csv")]
        p = pd.read_csv(f)
        p["sample_id"] = p["sample_id"].astype(str).str.strip()
        p = p.set_index("sample_id")
        vals, pred = [], []
        for s, lab in zip(sample_ids, labels):
            col = f"prob_{lab}"
            if s in p.index and col in p.columns and str(p.at[s, "true_label"]) == lab:
                vals.append(float(p.at[s, col]))
                pred.append(str(p.at[s, "prediction"]))
            else:
                vals.append(np.nan)
                pred.append("")
        own[cond] = (np.array(vals), np.array(pred, dtype=object))
    mean = np.nanmean(np.vstack([v for v, _ in own.values()]), axis=0)
    at, pred_at = own.get(condition, (np.full(len(sample_ids), np.nan), np.full(len(sample_ids), "", dtype=object)))
    say(f"CV probabilities from {run}: {len(own)} undiluted binary conditions; "
        f"{int(np.isfinite(mean).sum())} of {len(sample_ids)} samples matched with the same label")
    return {"cv_own_prob_mean": mean, f"cv_own_prob_{condition}": at, f"cv_prediction_{condition}": pred_at}


def main():
    ap = argparse.ArgumentParser(description="Label audit of training arrays (neighbourhood and CV)")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--label_map", default=str(REPO / "configs" / "label_map.json"))
    ap.add_argument("--exclude_ids", default=None)
    ap.add_argument("--classes", nargs="*", default=[], help="Label prefixes to audit (default: all)")
    ap.add_argument("--run", default=None, help="Finished train.py run for the CV signal")
    ap.add_argument("--condition", default="binary_0.90")
    ap.add_argument("--n_cpgs", type=int, default=50000)
    ap.add_argument("--n_pcs", type=int, default=100)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--annotations", nargs="*", default=[], help="CSV files with a Sample_ID column")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    bundle = load_training_data(args.data_path, load_label_map(args.label_map), args.exclude_ids)
    labels = np.asarray(bundle["labels"], dtype=object)
    ids = np.asarray(bundle["sample_ids"], dtype=object)
    sel = np.ones(len(labels), bool) if not args.classes else \
        np.array([any(str(l).startswith(p) for p in args.classes) for l in labels])
    if sel.sum() < args.k + 2:
        sys.exit(f"Only {int(sel.sum())} samples match {args.classes}")
    X = bundle["X"][sel]
    labels, ids, raw = labels[sel], ids[sel], np.asarray(bundle["labels_raw"], dtype=object)[sel]
    del bundle
    say(f"Audited: {len(ids)} samples in {len(set(labels))} classes")
    say(pd.Series(labels).value_counts().to_string())

    with np.errstate(invalid="ignore"):
        var = np.nanvar(X, axis=0)
    var[~np.isfinite(var)] = -1
    top = np.argsort(-var)[:min(args.n_cpgs, X.shape[1])]
    Z = X[:, top]
    mu = np.nanmean(Z, axis=0)
    Z = np.where(np.isnan(Z), mu, Z) - mu
    n_pcs = min(args.n_pcs, len(ids) - 1, Z.shape[1])
    pcs = PCA(n_components=n_pcs, svd_solver="randomized", random_state=args.seed).fit_transform(Z)
    nn = NearestNeighbors(n_neighbors=args.k + 1).fit(pcs)
    _, idx = nn.kneighbors(pcs)
    neigh = labels[idx[:, 1:]]
    same = (neigh == labels[:, None]).sum(axis=1)
    class_size = pd.Series(labels).map(pd.Series(labels).value_counts()).to_numpy()
    same_frac = same / np.maximum(np.minimum(args.k, class_size - 1), 1)   # a class of 6 has at most 5 same-label neighbours
    maj, maj_n = [], []
    for row in neigh:
        vc = pd.Series(row).value_counts()
        maj.append(vc.index[0])
        maj_n.append(int(vc.iloc[0]))
    say(f"\nNeighbourhood: {len(top)} most variable CpGs, {n_pcs} principal components, k = {args.k}")

    audit = pd.DataFrame({"Sample_ID": ids, "label": labels, "raw_label": raw,
                          f"same_label_of_{args.k}": same, "same_label_fraction": same_frac,
                          "neighbour_majority": maj, "neighbour_majority_n": maj_n})
    cv = cv_own_probabilities(Path(args.run).expanduser(), ids, labels, args.condition) if args.run else None
    if cv:
        for key, val in cv.items():
            audit[key] = val
    for path in args.annotations:
        a = pd.read_csv(path, dtype=str)
        if "Sample_ID" not in a.columns:
            sys.exit(f"{path} has no Sample_ID column")
        audit = audit.merge(a.drop_duplicates("Sample_ID"), on="Sample_ID", how="left")

    low = audit[f"same_label_of_{args.k}"] <= 1
    few = audit["same_label_fraction"] <= 0.25
    if cv:
        audit["flag_lamprey"] = low & (audit["cv_own_prob_mean"] <= 0.05)
        audit["flag_review"] = few | (audit[f"cv_own_prob_{args.condition}"] < 0.5)
    else:
        audit["flag_lamprey"] = low
        audit["flag_review"] = few
    audit.to_csv(out / "audit.csv", index=False, float_format="%.4f")
    flagged = audit[audit["flag_review"] | audit["flag_lamprey"]].sort_values(["label", f"same_label_of_{args.k}"])
    flagged.to_csv(out / "flagged.csv", index=False, float_format="%.4f")

    summ = audit.groupby("label").agg(n=("Sample_ID", "size"),
                                      median_same=(f"same_label_of_{args.k}", "median"),
                                      review=("flag_review", "sum"), lamprey=("flag_lamprey", "sum"))
    if cv:
        summ["median_cv_own_prob"] = audit.groupby("label")[f"cv_own_prob_{args.condition}"].median()
    say("\nPer class:")
    say(summ.to_string(float_format=lambda v: f"{v:.2f}"))
    say(f"\nFlagged for review: {int(audit['flag_review'].sum())}; by Lamprey's rule: {int(audit['flag_lamprey'].sum())}")
    if len(flagged):
        say("\nFlagged samples, label (rows) against neighbour-majority label (columns):")
        say(pd.crosstab(flagged["label"], flagged["neighbour_majority"]).to_string())
    (out / "summary.txt").write_text("\n".join(LOG) + "\n")
    say(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
