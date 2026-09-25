#!/usr/bin/env python3
"""
Write a new training pickle with per-sample label changes, leaving the input untouched.

    python scripts/make_training_pickle.py \
        --data_path /path/AL_24Sep2026.pkl --output /path/AL_25Sep2026.pkl \
        --relabel relabel_hox_idh.csv relabel_nup98_nsd1.csv

Each --relabel CSV has the columns Sample_ID and new_label, and optionally reason.
Only the ANNOTATION column changes. The original label is kept in a new column,
ANNOTATION_ORIGINAL, which training ignores because only cg... columns are read.
Sample IDs come from the Sample_ID column, or from the index when there is no
such column, as in data/dataset.py.

The script refuses to run when:
- a Sample_ID in a relabel file is not in the pickle;
- one sample is given two different new labels;
- the output file already exists.

Samples to leave out are better listed in an exclusion file (EXCLUDE_IDS in
jobs/settings.sh) than dropped here, so that one pickle serves every run.
--drop_ids exists for a pickle that must not contain them at all.

Also written: <output>.changes.csv, one row per changed or dropped sample with
its old label, new label and reason; the counts are printed.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

LABEL = "ANNOTATION"
SAMPLE = "Sample_ID"


def read_relabels(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(p, dtype=str).fillna("")
        missing = {SAMPLE, "new_label"} - set(df.columns)
        if missing:
            sys.exit(f"{p}: missing column(s) {sorted(missing)}")
        df[SAMPLE] = df[SAMPLE].str.strip()
        df["new_label"] = df["new_label"].str.strip()
        if "reason" not in df.columns:
            df["reason"] = ""
        df["file"] = Path(p).name
        frames.append(df[[SAMPLE, "new_label", "reason", "file"]])
    rel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=[SAMPLE, "new_label", "reason", "file"])
    rel = rel[(rel[SAMPLE] != "") & (rel["new_label"] != "")]
    clash = rel.groupby(SAMPLE)["new_label"].nunique()
    clash = clash[clash > 1]
    if len(clash):
        sys.exit(f"Samples given different new labels: {list(clash.index[:10])}")
    return rel.drop_duplicates(SAMPLE)


def main():
    ap = argparse.ArgumentParser(description="New training pickle with per-sample label changes")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--relabel", nargs="*", default=[], help="CSV files Sample_ID,new_label[,reason]")
    ap.add_argument("--drop_ids", default=None, help="Optional file of Sample_IDs to remove (one per line, or a CSV "
                                                      "with a Sample_ID column)")
    args = ap.parse_args()

    out = Path(args.output)
    if out.exists():
        sys.exit(f"{out} exists; choose a new name (the input pickle is never overwritten)")
    rel = read_relabels(args.relabel)
    drops = set()
    if args.drop_ids:
        p = Path(args.drop_ids)
        if p.suffix == ".csv":
            drops = set(pd.read_csv(p, dtype=str)[SAMPLE].str.strip())
        else:
            drops = {line.strip() for line in p.read_text().splitlines() if line.strip()}

    print(f"Loading {args.data_path}", flush=True)
    df = pd.read_pickle(args.data_path)
    if LABEL not in df.columns:
        sys.exit(f"No {LABEL} column in {args.data_path}")
    ids = (df[SAMPLE] if SAMPLE in df.columns else pd.Series(df.index, index=df.index)).astype(str).str.strip()
    if ids.duplicated().any():
        sys.exit(f"Duplicate sample IDs in the input, e.g. {list(ids[ids.duplicated()].unique()[:5])}")
    where = pd.Series(range(len(df)), index=ids.to_numpy())

    unknown = sorted(set(rel[SAMPLE]) - set(where.index))
    if unknown:
        sys.exit(f"{len(unknown)} relabelled Sample_ID(s) are not in the pickle, e.g. {unknown[:10]}")
    unknown_drops = sorted(drops - set(where.index))
    if unknown_drops:
        print(f"Note: {len(unknown_drops)} Sample_ID(s) to drop are not in the pickle, e.g. {unknown_drops[:5]}")

    old = df[LABEL].astype(str).to_numpy()
    if "ANNOTATION_ORIGINAL" not in df.columns:
        df["ANNOTATION_ORIGINAL"] = df[LABEL]
    rows = where.loc[rel[SAMPLE]].to_numpy()
    label_pos = df.columns.get_loc(LABEL)
    for r, new in zip(rows, rel["new_label"]):
        df.iat[r, label_pos] = new
    changes = pd.DataFrame({SAMPLE: rel[SAMPLE].to_numpy(), "old_label": old[rows],
                            "new_label": rel["new_label"].to_numpy(), "reason": rel["reason"].to_numpy(),
                            "file": rel["file"].to_numpy()})
    changes = changes[changes["old_label"] != changes["new_label"]]

    if drops:
        keep = ~ids.isin(drops).to_numpy()
        dropped = pd.DataFrame({SAMPLE: ids[~keep].to_numpy(), "old_label": old[~keep], "new_label": "(dropped)",
                                "reason": f"listed in {Path(args.drop_ids).name}", "file": Path(args.drop_ids).name})
        changes = pd.concat([changes, dropped], ignore_index=True)
        df = df.loc[keep]

    print("\nChanges (old label -> new label):")
    if len(changes):
        print(changes.groupby(["old_label", "new_label"]).size().to_string())
    else:
        print("  none")
    print(f"\nClass counts after the change ({len(df)} samples):")
    print(df[LABEL].value_counts().to_string())

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out)
    changes.to_csv(str(out) + ".changes.csv", index=False)
    print(f"\nWritten {out} and {out}.changes.csv")


if __name__ == "__main__":
    main()
