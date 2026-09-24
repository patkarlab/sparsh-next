#!/usr/bin/env python3
"""
Synthetic data for the smoke test: a training pickle in the SPARSH format, ONT-like
CSVs simulated from held-out samples, and a ground-truth file that uses raw
(pre-merge) labels and one out-of-scheme label, so every code path is exercised.

    python tests/make_synthetic.py --output_dir /tmp/sparsh_synth
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from models.corruption import corrupt_numpy  # noqa: E402

CLASS_SIZES = {
    "B-ALL_High hyperdiploid": 120, "AML_KMT2A-r": 90, "AML_mutated NPM1": 60, "AML_NUP98-r": 15,
    "B-ALL_ETV6-RUNX1": 70, "T-ALL": 50, "B-ALL_BCR-ABL1": 20, "B-ALL_BCR-ABL1 like": 18,
    "B-ALL_PAX5 alt": 12, "B-ALL_PAX5 P80R": 8, "B-ALL_TCF3-PBX1": 7, "MPAL_BCR-ABL1": 6,
}
MERGED = {"AML_mutated NPM1": "HOX", "AML_NUP98-r": "HOX",
          "B-ALL_BCR-ABL1": "PH", "B-ALL_BCR-ABL1 like": "PH", "B-ALL_PAX5 alt": "PH"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_cpg", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.output_dir)
    (out / "ont").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    p = args.n_cpg
    state = rng.choice([0, 1, 2], size=p, p=[0.30, 0.55, 0.15])
    base = np.where(state == 0, 0.06, np.where(state == 1, 0.88, 0.50))
    signatures = {}
    rows, labels, sources = [], [], []
    for ci, (cls, n) in enumerate(CLASS_SIZES.items()):
        key = MERGED.get(cls, cls)
        if key not in signatures:
            idx = rng.choice(p, size=200, replace=False)
            tgt = base[idx].copy()
            tgt[:100] = np.where(base[idx[:100]] > 0.5, 0.25, 0.75)
            tgt[100:] = np.where(base[idx[100:]] > 0.5, 0.55, 0.45)
            signatures[key] = (idx, tgt)
        idx, tgt = signatures[key]
        for _ in range(n):
            mu = base.copy()
            mu[idx] = tgt
            a = np.clip(mu * 15, 0.3, None)
            b = np.clip((1 - mu) * 15, 0.3, None)
            rows.append(rng.beta(a, b).astype(np.float32))
            labels.append(cls)
            sources.append("GSE_A" if ci % 3 else "GSE_B")
    X = np.vstack(rows)
    X[rng.random(X.shape) < 0.002] = np.nan
    # one technical replicate of sample 0, to be found by scripts/check_data.py
    rep = np.clip(X[0] + rng.normal(0, 0.01, p), 0, 1).astype(np.float32)
    X = np.vstack([X, rep])
    labels.append(labels[0])
    sources.append("GSE_C")

    cpgs = [f"cg{i:08d}" for i in range(p)]
    df = pd.DataFrame(X, columns=cpgs)
    df.insert(0, "Sample_ID", [f"GSM{100000 + i}" for i in range(len(df))])
    df["ANNOTATION"] = labels
    df["Source_Dataset"] = sources
    df.to_pickle(out / "train.pkl")

    # ONT-like samples: new draws from a few classes, read-level simulation at 15-30% coverage
    truth = []
    picks = ["B-ALL_ETV6-RUNX1", "T-ALL", "AML_mutated NPM1", "AML_KMT2A-r", "B-ALL_High hyperdiploid",
             "B-ALL_BCR-ABL1", "B-ALL_TCF3-PBX1", "B-ALL_High hyperdiploid"]
    for k, cls in enumerate(picks):
        idx, tgt = signatures[MERGED.get(cls, cls)]
        mu = base.copy()
        mu[idx] = tgt
        beta = rng.beta(np.clip(mu * 15, 0.3, None), np.clip((1 - mu) * 15, 0.3, None))
        x = corrupt_numpy(beta.astype(np.float32), float(rng.uniform(0.15, 0.30)), "reads", rng)
        sid = f"ONT{k + 1:02d}"
        pd.DataFrame([x], columns=cpgs, index=[sid]).to_csv(out / "ont" / f"{sid}.csv")
        truth.append((sid, cls))
    truth[-1] = (truth[-1][0], "B-ALL_Low hyperdiploid")        # out-of-scheme label
    pd.DataFrame(truth, columns=["sample", "true_label"]).to_csv(out / "ground_truth.csv", index=False)
    print(f"Wrote {out}/train.pkl ({len(df)} samples, {p} CpGs), {len(picks)} ONT CSVs and ground_truth.csv")


if __name__ == "__main__":
    main()
