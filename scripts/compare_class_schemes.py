#!/usr/bin/env python3
"""
Compare two runs whose class schemes differ, and apply the fifth-round rule (docs/EXPERIMENTS.md).

    python scripts/compare_class_schemes.py ~/sparsh_next_runs/locked_clean ~/sparsh_next_runs/locked_relabel \
        --rename T-ALL_TAL1=T-ALL_TAL1-like --output ~/sparsh_next_runs/data_checks/round5_rule.csv

The first run has the old labels, the second the new ones. --rename maps class names of
the first run to those of the second when a class was only renamed. Only the
cross-validation predictions on training arrays are read (cv_predictions_<condition>.csv),
top-1, as in cv_recall_by_class.csv.

- New classes (in the second run only): recall over all their samples.
- Shared classes: recall on the samples whose label is the same in both runs, so a class
  that loses members to a new class is compared on the members it keeps.

Rule, per --conditions (default binary_0.30): a new class passes if its recall is at least
--min_new_recall (0.70); each shared class may lose at most --max_class_drop (5 points) or
one sample, whichever is larger; the mean recall over the shared classes (equal weight) may
fall by at most --max_mean_drop (1 point). For a shared class that fails, the table of where
its lost samples went shows which new class took them.

Output: one row per class and condition (--output), and a printed summary.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def read_predictions(run: Path, condition: str) -> pd.DataFrame:
    f = run / f"cv_predictions_{condition}.csv"
    if not f.exists():
        sys.exit(f"{f} not found")
    p = pd.read_csv(f, usecols=["sample_id", "true_label", "prediction"], dtype=str)
    p["sample_id"] = p["sample_id"].str.strip()
    if p["sample_id"].duplicated().any():
        sys.exit(f"{f}: duplicate sample IDs")
    return p.set_index("sample_id")


def compare(a: pd.DataFrame, b: pd.DataFrame, args, condition: str):
    only_a, only_b = a.index.difference(b.index), b.index.difference(a.index)
    if len(only_a) or len(only_b):
        print(f"  Note: {len(only_a)} samples only in the first run, {len(only_b)} only in the second "
              "(different exclusion lists?); they are left out of the shared-class comparison.")
    m = a.join(b, how="inner", lsuffix="_a", rsuffix="_b")
    unchanged = m[m["true_label_a"] == m["true_label_b"]]
    moved = m[m["true_label_a"] != m["true_label_b"]]
    classes_a = set(a["true_label"])
    new_classes = sorted(set(b["true_label"]) - classes_a)

    rows = []
    for c in new_classes:
        s = b[b["true_label"] == c]
        recall = float((s["prediction"] == c).mean())
        errors = s.loc[s["prediction"] != c, "prediction"].value_counts()
        rows.append({"condition": condition, "class": c, "kind": "new", "n": len(s), "recall_first": np.nan,
                     "recall_second": recall, "change": np.nan, "allowed_drop": np.nan,
                     "passes": recall >= args.min_new_recall,
                     "errors_second": "; ".join(f"{k} {v}" for k, v in errors.head(3).items())})
    for c, s in unchanged.groupby("true_label_b"):
        ra = float((s["prediction_a"] == c).mean())
        rb = float((s["prediction_b"] == c).mean())
        allowed = max(args.max_class_drop, 1.0 / len(s))
        lost = s[(s["prediction_a"] == c) & (s["prediction_b"] != c)]["prediction_b"].value_counts()
        rows.append({"condition": condition, "class": c, "kind": "shared", "n": len(s), "recall_first": ra,
                     "recall_second": rb, "change": rb - ra, "allowed_drop": allowed,
                     "passes": ra - rb <= allowed + 1e-9,
                     "errors_second": "; ".join(f"{k} {v}" for k, v in lost.head(3).items())})
    table = pd.DataFrame(rows)
    shared = table[table["kind"] == "shared"]
    mean_a, mean_b = shared["recall_first"].mean(), shared["recall_second"].mean()
    verdict = {
        "shared_classes_pass": bool(shared["passes"].all()),
        "mean_shared_first": mean_a, "mean_shared_second": mean_b,
        "mean_shared_pass": bool(mean_a - mean_b <= args.max_mean_drop + 1e-9),
        "moved": moved.groupby(["true_label_a", "true_label_b"]).size(),
    }
    return table, verdict


def main():
    ap = argparse.ArgumentParser(description="Compare runs with different class schemes (fifth-round rule)")
    ap.add_argument("first_run", help="Run with the old labels")
    ap.add_argument("second_run", help="Run with the new labels")
    ap.add_argument("--rename", nargs="*", default=[], help="OLD=NEW class names, first run to second")
    ap.add_argument("--conditions", nargs="*", default=["binary_0.30"])
    ap.add_argument("--min_new_recall", type=float, default=0.70)
    ap.add_argument("--max_class_drop", type=float, default=0.05)
    ap.add_argument("--max_mean_drop", type=float, default=0.01)
    ap.add_argument("--output", default=None, help="CSV, one row per class and condition")
    args = ap.parse_args()

    renames = {}
    for pair in args.rename:
        if "=" not in pair:
            sys.exit(f"--rename expects OLD=NEW, got {pair!r}")
        old, new = (x.strip() for x in pair.split("=", 1))
        renames[old] = new
    first, second = Path(args.first_run).expanduser(), Path(args.second_run).expanduser()
    tables = []
    pd.set_option("display.width", 200)
    for cond in args.conditions:
        a, b = read_predictions(first, cond), read_predictions(second, cond)
        unknown = sorted(set(renames) - set(a["true_label"]))
        if unknown:
            sys.exit(f"--rename: {unknown} not a class of {first.name}")
        a = a.replace({"true_label": renames, "prediction": renames})
        print(f"\n== {cond}: {first.name} (old labels) against {second.name} (new labels)")
        table, v = compare(a, b, args, cond)
        tables.append(table)
        if len(v["moved"]):
            print("Samples whose label changed (first run -> second run):")
            print(v["moved"].to_string())
        show = table.copy()
        for col in ("recall_first", "recall_second", "change", "allowed_drop"):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        print(show.drop(columns="condition").to_string(index=False))
        shared_ok = v["shared_classes_pass"] and v["mean_shared_pass"]
        print(f"\nShared classes, none loses more than max({args.max_class_drop:.2f}, one sample): "
              f"{'yes' if v['shared_classes_pass'] else 'NO'}")
        print(f"Mean recall over shared classes: {v['mean_shared_first']:.4f} -> {v['mean_shared_second']:.4f} "
              f"(may fall by {args.max_mean_drop:.2f}): {'yes' if v['mean_shared_pass'] else 'NO'}")
        for _, r in table[table["kind"] == "new"].iterrows():
            if not r["passes"]:
                what = f"recall below {args.min_new_recall:.2f}: merge it back"
            elif shared_ok:
                what = "keep"
            else:
                what = "recall passes, but the shared-class criteria fail (see errors_second for where samples went)"
            print(f"  {r['class']}: recall {r['recall_second']:.3f}, {what}")
    if args.output:
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(tables, ignore_index=True).to_csv(out, index=False, float_format="%.4f")
        print(f"\nWritten {out}")


if __name__ == "__main__":
    main()
