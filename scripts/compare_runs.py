#!/usr/bin/env python3
"""
Compare SPARSH-next runs side by side.

    python scripts/compare_runs.py runs/legacy runs/next_default runs/next_mask \
        --metrics balanced_accuracy accuracy callable_share_0.90 --output runs/comparison.csv

Rows are evaluation conditions (dense arrays; simulated ONT reads or masking
at each coverage); columns are runs. Runs are comparable because every run
scores the same outer folds on identically corrupted inputs, provided the data,
seed, fold count and eval_seed are the same (checked below).
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

    tables, settings, keys = {}, [], set()
    for run in args.runs:
        run = Path(run)
        f = run / "cv_metrics_by_condition.csv"
        if not f.exists():
            sys.exit(f"{f} not found (is this a finished SPARSH-next run?)")
        tables[run.name] = pd.read_csv(f).set_index("condition")
        cfg = json.loads((run / "config.json").read_text())
        t, m = cfg["training"], cfg["model"]
        settings.append({"run": run.name, "imbalance": t["imbalance"], "encoding": m["input_encoding"],
                         "train_sim": t["train_sim"], "coverage": t["coverage_mode"], "calibrated": t["calibrate"],
                         "n_samples": cfg["data"]["n_samples"], "n_classes": m["n_classes"],
                         "median_best_epoch": pd.Series([r["best_epoch"] for r in cfg["cv_folds"]]).median()})
        keys.add((cfg["data"]["data_path"], cfg["data"]["n_samples"], t["seed"], t["n_folds"], t["eval_seed"],
                  cfg["data"].get("groups_file"), cfg["data"].get("group_col")))

    print(pd.DataFrame(settings).to_string(index=False))
    if len(keys) > 1:
        print("\nWARNING: runs differ in data, seed, folds, grouping or eval_seed; they were not scored on the "
              "same inputs and are not directly comparable.")

    stacked = []
    for metric in args.metrics:
        cols = {}
        for name, t in tables.items():
            if metric in t.columns:
                cols[name] = t[metric]
        if not cols:
            print(f"\n(metric {metric} not found)")
            continue
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
