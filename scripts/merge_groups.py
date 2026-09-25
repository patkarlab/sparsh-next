#!/usr/bin/env python3
"""
Merge sample groupings for cross-validation into one groups file.

    python scripts/merge_groups.py --groups ~/sparsh_next_runs/data_checks/groups_no_normals.csv \
        --patients patients_25Sep2026.csv --output ~/sparsh_next_runs/data_checks/groups_25Sep2026.csv

- --groups: files in the format of scripts/check_data.py (Sample_ID,group), for
  example clusters of near-identical arrays.
- --patients: files Sample_ID,patient, for example a diagnosis and a relapse
  sample of one patient, or a lab's repeat array of the same sample.

Samples joined by any file end up in one group, and groups that share a sample
are merged. train.py --groups_file then keeps each group within one fold, so
a patient never sits on both sides of a split. Samples in no file form their
own group and are not written.
"""

import argparse
import sys

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Merge groups and patient IDs into one groups file")
    ap.add_argument("--groups", nargs="*", default=[], help="CSV Sample_ID,group")
    ap.add_argument("--patients", nargs="*", default=[], help="CSV Sample_ID,patient")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    n_links = 0
    for path, col in [(p, "group") for p in args.groups] + [(p, "patient") for p in args.patients]:
        df = pd.read_csv(path, dtype=str)
        if not {"Sample_ID", col} <= set(df.columns):
            sys.exit(f"{path} must have columns Sample_ID and {col}")
        df = df.dropna(subset=["Sample_ID", col])
        for sid, key in zip(df["Sample_ID"].str.strip(), df[col].str.strip()):
            if sid and key:
                union("s:" + sid, f"k:{path}:{key}")
                n_links += 1
    samples = [k for k in parent if k.startswith("s:")]
    roots = pd.Series({s[2:]: find(s) for s in samples})
    sizes = roots.value_counts()
    roots = roots[roots.map(sizes) > 1]
    names = {r: f"g{i:04d}" for i, r in enumerate(sorted(roots.unique()))}
    out = pd.DataFrame({"Sample_ID": roots.index, "group": roots.map(names).to_numpy()}).sort_values(["group", "Sample_ID"])
    out.to_csv(args.output, index=False)
    print(f"{n_links} links from {len(args.groups) + len(args.patients)} file(s): {out['group'].nunique()} groups "
          f"of 2 or more samples, {len(out)} samples -> {args.output}")


if __name__ == "__main__":
    main()
