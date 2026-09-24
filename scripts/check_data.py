#!/usr/bin/env python3
"""
SPARSH-next data checks, run before training.

Writes to --output_dir:
  class_counts.csv         raw label -> model class, sample counts, kept or dropped
  source_by_class.csv      Source_Dataset x class table (study confounding)
  missing_by_source.csv    missing values per Source_Dataset, incl. probes absent from a whole dataset
  nearest_neighbour.csv    each sample's most correlated other sample (inspect the distribution)
  duplicates.csv           pairs with correlation >= --dup_threshold (likely the same sample or patient)
  groups.csv               Sample_ID,group for duplicate clusters; pass to train.py --groups_file
  sex_chromosome_probes.txt  only with --probe_annotation

Example:
    python scripts/check_data.py --data_path /path/to/methylation.pkl --output_dir runs/data_checks \
        --exclude_prefixes MPAL AML_NOS B-ALL_NOS
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data.dataset import filter_classes, load_label_map, load_training_data  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="SPARSH-next data checks")
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--label_map", default=str(REPO / "configs" / "label_map.json"))
    p.add_argument("--exclude_ids", "--junk_path", dest="exclude_ids", default=None)
    p.add_argument("--min_samples", type=int, default=5)
    p.add_argument("--exclude_classes", nargs="*", default=[])
    p.add_argument("--exclude_prefixes", nargs="*", default=[])
    p.add_argument("--top_variable", type=int, default=10000, help="CpGs used for the correlation check")
    p.add_argument("--dup_threshold", type=float, default=0.98)
    p.add_argument("--probe_annotation", default=None,
                   help="Optional CSV with probe ID and chromosome columns (e.g. from the Illumina manifest)")
    return p.parse_args()


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    label_map = load_label_map(None if args.label_map.lower() == "none" else args.label_map)
    b = load_training_data(args.data_path, label_map, args.exclude_ids)
    X, labels, raw, ids, sources = b["X"], b["labels"], b["labels_raw"], b["sample_ids"], b["sources"]

    # ---------------------------------------------------------- classes
    keep = filter_classes(labels, args.min_samples, args.exclude_classes, args.exclude_prefixes)
    cc = (pd.DataFrame({"raw_label": raw, "model_class": labels, "kept": keep})
          .groupby(["model_class", "raw_label", "kept"]).size().rename("n").reset_index()
          .sort_values(["kept", "model_class", "raw_label"], ascending=[False, True, True]))
    cc.to_csv(out / "class_counts.csv", index=False)
    kept_counts = pd.Series(labels[keep]).value_counts()
    print(f"\n{len(kept_counts)} classes, {int(keep.sum())} samples kept "
          f"(smallest class: {kept_counts.min()}, largest: {kept_counts.max()})")

    # ---------------------------------------------------------- source confounding
    if sources is not None:
        tab = pd.crosstab(pd.Series(labels[keep], name="class"), pd.Series(sources[keep], name="Source_Dataset"))
        tab.to_csv(out / "source_by_class.csv")
        share = tab.max(axis=1) / tab.sum(axis=1)
        single = share[share >= 0.9].sort_values(ascending=False)
        print(f"\nClasses with >=90% of samples from a single Source_Dataset: {len(single)} of {len(tab)}")
        for cls, s in single.items():
            print(f"  {cls}: {100 * s:.0f}% from {tab.loc[cls].idxmax()} (n={int(tab.loc[cls].sum())})")

        rows = []
        for src in np.unique(sources):
            m = sources == src
            miss = np.isnan(X[m])
            rows.append({"Source_Dataset": src, "n_samples": int(m.sum()),
                         "missing_fraction": float(miss.mean()),
                         "probes_missing_in_all_samples": int(miss.all(axis=0).sum()),
                         "probes_missing_in_half_or_more": int((miss.mean(axis=0) >= 0.5).sum())})
        ms = pd.DataFrame(rows).sort_values("probes_missing_in_all_samples", ascending=False)
        ms.to_csv(out / "missing_by_source.csv", index=False)
        flagged = ms[ms["probes_missing_in_all_samples"] > 0]
        print(f"\nDatasets with probes missing in every sample (a platform fingerprint if the class mix differs): "
              f"{len(flagged)}")
        if len(flagged):
            print(flagged.to_string(index=False))
    else:
        print("\nNo Source_Dataset column: study confounding not checked")

    # ---------------------------------------------------------- duplicates
    Xk = X[keep]
    ids_k, labels_k = ids[keep], labels[keep]
    src_k = sources[keep] if sources is not None else np.array([""] * len(ids_k), dtype=object)
    var = np.nanvar(Xk, axis=0)
    var[np.isnan(var)] = -1.0
    top = np.argsort(var)[::-1][:min(args.top_variable, Xk.shape[1])]
    Z = Xk[:, top].astype(np.float64)
    col_mean = np.nanmean(Z, axis=0)
    Z = np.where(np.isnan(Z), col_mean, Z)
    Z -= Z.mean(axis=1, keepdims=True)
    Z /= np.maximum(Z.std(axis=1, keepdims=True), 1e-12)
    R = (Z @ Z.T) / Z.shape[1]
    np.fill_diagonal(R, -np.inf)
    nn_idx = R.argmax(axis=1)
    nn = pd.DataFrame({"sample_id": ids_k, "label": labels_k, "source": src_k,
                       "nearest_sample": ids_k[nn_idx], "nearest_label": labels_k[nn_idx],
                       "nearest_source": src_k[nn_idx], "r": R[np.arange(len(ids_k)), nn_idx]})
    nn.sort_values("r", ascending=False).to_csv(out / "nearest_neighbour.csv", index=False, float_format="%.4f")
    qs = np.quantile(nn["r"], [0.5, 0.9, 0.99])
    print(f"\nNearest-neighbour correlation on the {len(top)} most variable CpGs: "
          f"median {qs[0]:.3f}, 90th pct {qs[1]:.3f}, 99th pct {qs[2]:.3f}, max {nn['r'].max():.3f}")

    ii, jj = np.where(np.triu(R >= args.dup_threshold, k=1))
    pairs = pd.DataFrame({"sample_a": ids_k[ii], "sample_b": ids_k[jj], "r": R[ii, jj],
                          "label_a": labels_k[ii], "label_b": labels_k[jj],
                          "source_a": src_k[ii], "source_b": src_k[jj]})
    pairs.sort_values("r", ascending=False).to_csv(out / "duplicates.csv", index=False, float_format="%.4f")
    uf = UnionFind(len(ids_k))
    for a, c in zip(ii, jj):
        uf.union(a, c)
    roots = np.array([uf.find(i) for i in range(len(ids_k))])
    in_cluster = np.isin(roots, roots[ii]) | np.isin(roots, roots[jj]) if len(ii) else np.zeros(len(ids_k), bool)
    groups = pd.DataFrame({"Sample_ID": ids_k[in_cluster], "group": [f"dup{r}" for r in roots[in_cluster]]})
    groups.to_csv(out / "groups.csv", index=False)
    n_diff = int((pairs["label_a"] != pairs["label_b"]).sum()) if len(pairs) else 0
    print(f"Pairs with r >= {args.dup_threshold}: {len(pairs)} ({n_diff} with different labels) in "
          f"{groups['group'].nunique() if len(groups) else 0} clusters -> groups.csv")
    if len(pairs):
        print("Check the top of duplicates.csv; if these are the same patient or sample, train with "
              f"--groups_file {out / 'groups.csv'}")

    # ---------------------------------------------------------- sex chromosomes (optional)
    if args.probe_annotation:
        ann = pd.read_csv(args.probe_annotation, dtype=str, low_memory=False)
        pcol = next((c for c in ann.columns if c.lower() in ("probe", "ilmnid", "name", "probe_id")), None)
        ccol = next((c for c in ann.columns if c.lower() in ("chr", "chromosome", "chr_hg38", "chr_hg19")), None)
        if pcol is None or ccol is None:
            sys.exit("probe_annotation needs a probe column (probe/IlmnID/Name) and a chromosome column (chr/CHR)")
        chrom = dict(zip(ann[pcol].str.strip(), ann[ccol].astype(str).str.replace("chr", "", regex=False)))
        on_x = sum(1 for c in b["cpg_ids"] if chrom.get(c) == "X")
        on_y = sum(1 for c in b["cpg_ids"] if chrom.get(c) == "Y")
        unknown = sum(1 for c in b["cpg_ids"] if c not in chrom)
        text = f"CpGs on chrX: {on_x}\nCpGs on chrY: {on_y}\nCpGs not in the annotation: {unknown}\n"
        (out / "sex_chromosome_probes.txt").write_text(text)
        print("\n" + text)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
