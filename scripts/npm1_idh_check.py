#!/usr/bin/env python3
"""
Does NPM1-mutated AML with an IDH1/2 co-mutation have its own methylation profile
in the training arrays?

    python npm1_idh_check.py --data_path /path/to/training.pkl \
        --genotypes npm1_genotypes.csv --out_dir npm1_idh_out

Training arrays only. npm1_genotypes.csv lists NPM1-mutated cases with public
sequencing (Beat AML 2.0 and TCGA-LAML), one row per case: train_id (the
Sample_ID used in the training pickle), cohort, group (IDH_mut or IDH_wt), and
the IDH, DNMT3A and TET2 variants.

1. Differential methylation, IDH_mut against IDH_wt (Welch t-test per CpG,
   Benjamini-Hochberg FDR, difference in mean beta).
2. How well the two groups separate:
   - repeated stratified 5-fold cross-validation;
   - leave-one-cohort-out: train on Beat AML, test on TCGA, and the reverse,
     which guards against a batch effect between the two cohorts.
   CpGs are chosen inside each training part only. Two scores are used: a
   logistic regression on the 1,000 CpGs with the largest |t|, and the mean
   beta of the 200 most hypermethylated CpGs.
3. TET2: IDH-wild-type cases with a TET2 mutation are scored separately,
   because TET2 loss can mimic the IDH hypermethylation.
4. A signature from all genotyped cases (the most hypermethylated CpGs) is
   applied to every other sample of the NPM1 class, and of any class whose name
   contains IDH, to estimate how many untyped IDH-like cases each class holds.
   The cut-off gives 95% specificity on the genotyped IDH_wt cases (in-sample).

5. With --island (a CSV of candidate samples, Sample_ID and evidence), writes
   --relabel_out: the candidates to move to --new_label. A candidate moves if
   its evidence is genotype_IDH_mut or IDH_label, or if its evidence is unknown
   and its score reaches the cut-off. It never moves if its evidence is
   genotype_IDH_wt or starts with known_fusion. island_decisions.csv records
   every candidate with its score and the decision.

Outputs in --out_dir: summary.txt, differential_cpgs.csv (top 5,000 by |t|),
cv_auc.csv, scores.csv and, with --island, island_decisions.csv.

Sample IDs come from the Sample_ID column, or from the index, as in data/dataset.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

LOG = []


def say(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


def welch_t(a: np.ndarray, b: np.ndarray):
    """Per-column Welch t (a minus b), ignoring NaN, and its two-sided p-value."""
    na, nb = np.sum(~np.isnan(a), 0), np.sum(~np.isnan(b), 0)
    ma, mb = np.nanmean(a, 0), np.nanmean(b, 0)
    va, vb = np.nanvar(a, 0, ddof=1), np.nanvar(b, 0, ddof=1)
    se2 = va / na + vb / nb
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ma - mb) / np.sqrt(se2)
        df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    p = 2 * stats.t.sf(np.abs(t), df)
    bad = ~np.isfinite(t) | (na < 3) | (nb < 3)
    t[bad], p[bad] = 0.0, 1.0
    return t, p, ma - mb


def bh_fdr(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def fill_nan(train: np.ndarray, *others):
    """Replace NaN by the training-part column mean (0.5 where a column is all NaN)."""
    mu = np.nanmean(train, 0)
    mu = np.where(np.isnan(mu), 0.5, mu)
    return [np.where(np.isnan(x), mu, x) for x in (train, *others)]


def fold_scores(X_tr, y_tr, X_te, n_lr=1000, n_hyper=200, seed=0):
    """Out-of-sample scores for the test part: logistic regression and hypermethylation mean."""
    t, _, _ = welch_t(X_tr[y_tr == 1], X_tr[y_tr == 0])
    top = np.argsort(-np.abs(t))[:n_lr]
    hyper = np.argsort(-t)[:n_hyper]
    a, b = fill_nan(X_tr[:, top], X_te[:, top])
    mu, sd = a.mean(0), a.std(0) + 1e-6
    lr = LogisticRegression(C=0.1, max_iter=5000, random_state=seed)
    lr.fit((a - mu) / sd, y_tr)
    p_lr = lr.predict_proba((b - mu) / sd)[:, 1]
    s_hyper = np.nanmean(X_te[:, hyper], 1)
    return p_lr, s_hyper


def auc(y, s):
    return roc_auc_score(y, s) if len(set(y)) == 2 else float("nan")


def main():
    ap = argparse.ArgumentParser(description="NPM1 with and without IDH1/2: methylation check on training arrays")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--genotypes", required=True)
    ap.add_argument("--npm1_class", default="AML_HOXdys_NPM1_NUP98_DEK-NUP214_UBTF")
    ap.add_argument("--out_dir", default="npm1_idh_out")
    ap.add_argument("--n_signature", type=int, default=200, help="Hypermethylated CpGs in the final signature")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--island", default=None, help="CSV Sample_ID,evidence of candidates for --new_label")
    ap.add_argument("--new_label", default="AML_HOX_IDH")
    ap.add_argument("--relabel_out", default=None, help="Default: <out_dir>/relabel_<new_label>.csv")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    say(f"Loading {args.data_path}")
    df = pd.read_pickle(args.data_path)
    if "Sample_ID" in df.columns:
        ids = df["Sample_ID"].astype(str).str.strip().to_numpy()
    else:
        ids = df.index.astype(str).str.strip().to_numpy()
    labels = df["ANNOTATION"].astype(str).to_numpy()
    cpgs = np.array([c for c in df.columns if str(c).startswith("cg")])
    say(f"{len(df)} samples, {len(cpgs)} CpGs")

    geno = pd.read_csv(args.genotypes, dtype=str).fillna("")
    row_of = {s: i for i, s in enumerate(ids)}
    geno["row"] = geno["train_id"].map(row_of)
    missing = geno[geno["row"].isna()]
    if len(missing):
        say(f"Not in the training data ({len(missing)}): {', '.join(missing['train_id'])}")
    geno = geno.dropna(subset=["row"]).copy()
    geno["row"] = geno["row"].astype(int)
    geno["label"] = labels[geno["row"]]
    say("\nGenotyped NPM1 cases in the training data:")
    say(geno.groupby(["cohort", "group"]).size().to_string())
    say("\nTheir current training labels:")
    say(geno.groupby(["group", "label"]).size().to_string())

    idh_classes = sorted({l for l in labels if "IDH" in l.upper()})
    say(f"\nClasses whose name contains IDH: {idh_classes or 'none'}")
    island = None
    if args.island:
        island = pd.read_csv(args.island, dtype=str).fillna("")
        if not {"Sample_ID", "evidence"} <= set(island.columns):
            sys.exit(f"{args.island} must have columns Sample_ID and evidence")
        absent = sorted(set(island["Sample_ID"]) - set(row_of))
        if absent:
            say(f"Island candidates not in the training data ({len(absent)}): {', '.join(absent[:10])}")
        island = island[island["Sample_ID"].isin(row_of)].copy()
    island_rows = [] if island is None else [row_of[s] for s in island["Sample_ID"]]
    other_rows = np.flatnonzero(np.isin(labels, [args.npm1_class] + idh_classes))
    other_rows = np.union1d(other_rows, np.array(island_rows, dtype=int))
    other_rows = np.setdiff1d(other_rows, geno["row"].to_numpy())
    rows = np.concatenate([geno["row"].to_numpy(), other_rows])
    X_all = df.iloc[rows, df.columns.get_indexer(cpgs)].to_numpy(dtype=np.float32)
    del df
    n_g = len(geno)
    X = X_all[:n_g]
    y = (geno["group"] == "IDH_mut").to_numpy().astype(int)
    cohort = geno["cohort"].to_numpy()
    say(f"\nIDH_mut {int(y.sum())}, IDH_wt {int((1 - y).sum())}")

    # 1. Differential methylation
    t, p, d = welch_t(X[y == 1], X[y == 0])
    q = bh_fdr(p)
    for thr in (0.1, 0.2, 0.3):
        up, down = int(((q < 0.05) & (d >= thr)).sum()), int(((q < 0.05) & (d <= -thr)).sum())
        say(f"CpGs with FDR < 0.05 and |delta beta| >= {thr}: {up} higher in IDH_mut, {down} lower")
    top = np.argsort(-np.abs(t))[:5000]
    pd.DataFrame({"cpg": cpgs[top], "t": t[top], "delta_beta": d[top], "p": p[top], "fdr": q[top],
                  "mean_idh_mut": np.nanmean(X[y == 1][:, top], 0),
                  "mean_idh_wt": np.nanmean(X[y == 0][:, top], 0)}).to_csv(
        out / "differential_cpgs.csv", index=False, float_format="%.4g")

    # 2. Separation: repeated CV and leave-one-cohort-out
    cv_rows = []
    oof_lr, oof_hyper = [], []
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats, random_state=args.seed)
    for k, (tr, te) in enumerate(rskf.split(X, y)):
        p_lr, s_h = fold_scores(X[tr], y[tr], X[te], seed=args.seed + k)
        oof_lr.append((te, p_lr))
        oof_hyper.append((te, s_h))
    for name, parts in (("logistic_1000", oof_lr), ("hyper_mean_200", oof_hyper)):
        per_repeat = []
        for r in range(args.repeats):
            s = np.full(len(y), np.nan)
            for te, v in parts[r * 5:(r + 1) * 5]:
                s[te] = v
            per_repeat.append(auc(y, s))
        cv_rows.append({"test": "5-fold CV", "score": name, "auc": np.mean(per_repeat), "sd": np.std(per_repeat)})
    for train_c in sorted(set(cohort)):
        tr, te = cohort == train_c, cohort != train_c
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        p_lr, s_h = fold_scores(X[tr], y[tr], X[te], seed=args.seed)
        test_c = "+".join(sorted(set(cohort[te])))
        cv_rows.append({"test": f"train {train_c}, test {test_c}", "score": "logistic_1000", "auc": auc(y[te], p_lr), "sd": np.nan})
        cv_rows.append({"test": f"train {train_c}, test {test_c}", "score": "hyper_mean_200", "auc": auc(y[te], s_h), "sd": np.nan})
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(out / "cv_auc.csv", index=False, float_format="%.3f")
    say("\nSeparation of IDH_mut from IDH_wt (AUC; CpGs chosen inside each training part):")
    say(cv.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # 3 and 4. Final signature from all genotyped cases, applied to the rest
    sig = np.argsort(-t)[:args.n_signature]
    say(f"\nSignature: {len(sig)} CpGs, delta beta {d[sig].min():.2f}-{d[sig].max():.2f}, "
        f"largest FDR {q[sig].max():.2g}")
    score_all = np.nanmean(X_all[:, sig], 1)
    s_g = score_all[:n_g]
    cut = float(np.quantile(s_g[y == 0], 0.95))
    sens = float(np.mean(s_g[y == 1] >= cut))
    say(f"Cut-off {cut:.3f} (95% specificity on genotyped IDH_wt, in-sample): sensitivity {sens:.2f} in-sample")
    tet2 = (geno["TET2"] != "").to_numpy()
    for lab, m in (("IDH_mut", y == 1), ("IDH_wt, TET2-mutant", (y == 0) & tet2), ("IDH_wt, TET2-wild-type", (y == 0) & ~tet2)):
        if m.any():
            say(f"  {lab}: n={int(m.sum())}, median score {np.median(s_g[m]):.3f}, above cut-off {int((s_g[m] >= cut).sum())}")
    rest_labels = labels[other_rows]
    s_rest = score_all[n_g:]
    say("\nUntyped samples scored with the signature:")
    for lab in [args.npm1_class] + idh_classes:
        m = rest_labels == lab
        if m.any():
            say(f"  {lab}: n={int(m.sum())}, above cut-off {int((s_rest[m] >= cut).sum())} "
                f"({100 * np.mean(s_rest[m] >= cut):.0f}%), median score {np.median(s_rest[m]):.3f}")

    scores = pd.DataFrame({"Sample_ID": ids[rows], "ANNOTATION": labels[rows], "score": score_all,
                           "idh_like": score_all >= cut})
    scores["genotype_group"] = list(geno["group"]) + [""] * len(other_rows)
    scores["cohort"] = list(geno["cohort"]) + [""] * len(other_rows)
    scores["IDH"] = list(geno["IDH"]) + [""] * len(other_rows)
    scores["TET2"] = list(geno["TET2"]) + [""] * len(other_rows)
    scores["DNMT3A"] = list(geno["DNMT3A"]) + [""] * len(other_rows)
    scores.to_csv(out / "scores.csv", index=False, float_format="%.4f")

    if island is not None:
        sc = dict(zip(scores["Sample_ID"], scores["score"]))
        island["label"] = [labels[row_of[s]] for s in island["Sample_ID"]]
        island["score"] = island["Sample_ID"].map(sc)
        ev = island["evidence"].str.strip()
        move = ev.isin(["genotype_IDH_mut", "IDH_label"]) | ((ev == "unknown") & (island["score"] >= cut))
        move &= ~(ev.eq("genotype_IDH_wt") | ev.str.startswith("known_fusion"))
        island["decision"] = np.where(move, "move to " + args.new_label, "keep label")
        island.to_csv(out / "island_decisions.csv", index=False, float_format="%.4f")
        rel = island.loc[move, ["Sample_ID"]].assign(new_label=args.new_label,
                                                     reason="island next to AML_IDH; evidence " + ev[move])
        rel_path = Path(args.relabel_out) if args.relabel_out else out / f"relabel_{args.new_label}.csv"
        rel.to_csv(rel_path, index=False)
        say(f"\nIsland candidates ({len(island)}):")
        say(island.groupby(["label", "evidence", "decision"]).size().to_string())
        say(f"{int(move.sum())} samples to move to {args.new_label}: {rel_path}")
    (out / "summary.txt").write_text("\n".join(LOG) + "\n")
    say(f"\nWritten to {out}")


if __name__ == "__main__":
    sys.exit(main())
