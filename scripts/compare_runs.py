#!/usr/bin/env python3
"""
Compare SPARSH-next runs side by side.

    python scripts/compare_runs.py runs/legacy runs/next_default runs/next_mask \
        --metrics balanced_accuracy accuracy callable_share_0.90 --output runs/comparison.csv

Rows are evaluation conditions (dense arrays; simulated ONT reads or masking
at each coverage); columns are runs. Every sample's corrupted input depends only
on its Sample_ID, the condition and eval_seed, so runs on the same data are scored
on identical inputs. With the same seed, fold count and grouping they also share
the same folds (checked below).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Compare SPARSH-next runs")
    p.add_argument("runs", nargs="+", help="Run output directories")
    p.add_argument("--metrics", nargs="+", default=["balanced_accuracy", "accuracy", "callable_share_0.90",
                                                    "accuracy_callable_0.90"])
    p.add_argument("--output", default=None, help="Optional CSV with all tables stacked")
    args = p.parse_args()

    tables, settings, keys, splits = {}, [], set(), set()
    names = [Path(r).name for r in args.runs]
    for run, name in zip(args.runs, names):
        run = Path(run)
        if names.count(name) > 1:          # same folder name in different places: use the full path
            name = str(run)
        f = run / "cv_metrics_by_condition.csv"
        if not f.exists():
            sys.exit(f"{f} not found (is this a finished SPARSH-next run?)")
        tables[name] = pd.read_csv(f).set_index("condition")
        cfg = json.loads((run / "config.json").read_text())
        t, m, d = cfg["training"], cfg["model"], cfg["data"]
        settings.append({"run": name, "imbalance": t["imbalance"], "encoding": m["input_encoding"],
                         "train_sim": t["train_sim"], "coverage": t["coverage_mode"], "calibrated": t["calibrate"],
                         "n_samples": d["n_samples"], "n_classes": m["n_classes"],
                         "median_best_epoch": pd.Series([r["best_epoch"] for r in cfg["cv_folds"]]).median()})
        keys.add((d["data_path"], d["n_samples"], d["n_cpgs"], d.get("cpg_list"),
                  json.dumps(d.get("class_counts"), sort_keys=True), d.get("groups_file"), d.get("group_col"),
                  t["eval_seed"], tuple(t["eval_coverages"])))
        splits.add((t["seed"], t["n_folds"]))

    print(pd.DataFrame(settings).to_string(index=False))
    if len(keys) > 1:
        print("\nWARNING: runs differ in data, CpG set, classes, grouping, eval_seed or evaluation coverages; "
              "they are not directly comparable.")
    elif len(splits) > 1:
        print("\nNote: runs use different fold splits (seed, fold count or grouping). Samples and evaluation "
              "inputs are identical, so differences include run-to-run noise; this is how to measure that noise.")

    stacked = []
    for metric in args.metrics:
        cols = {}
        for name, t in tables.items():
            if metric in t.columns:
                cols[name] = t[metric]
        if not cols:
            print(f"\n(metric {metric} not found)")
            continue
        lacking = [name for name in tables if name not in cols]
        if lacking:
            print(f"\n(metric {metric} is missing for: {lacking}; a different --threshold?)")
        table = pd.DataFrame(cols)
        ont = [c for c in table.index if c.startswith("reads_")]
        if ont:
            table.loc["mean over reads_*"] = table.loc[ont].mean()
        print(f"\n{metric}")
        print(table.to_string(float_format=lambda v: f"{v:.3f}"))
        stacked.append(table.assign(metric=metric))
    if args.output and stacked:
        pd.concat(stacked).to_csv(args.output, float_format="%.4f")
        print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
