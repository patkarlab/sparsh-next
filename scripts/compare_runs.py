#!/usr/bin/env python3
"""
Compare SPARSH-next runs side by side.

    python scripts/compare_runs.py ~/sparsh_next_runs/{wide,scaled_wide} \
        --ont_coverage ~/sparsh_next_runs/data_checks/ont_coverage.csv --output comparison.csv

SPARSH-next classifies nanopore samples only, so the tables show the simulated
nanopore conditions, named <simulation>_<f>, where a fraction f of the model's
CpGs is covered: reads_* (the methylated fraction of the reads at each CpG),
binary_* (one 0/1 call per CpG from its reads) and oneread_* (the call of a
single read). Runs scored with per-read call errors (--eval_call_errors) add
rows such as binary-err10_* (10% of read calls wrong). --all_conditions adds the dense array profile and the mask_*
conditions (array beta values at the covered CpGs), which no nanopore run
produces. For each simulation (and call error), a row gives the mean over its coverages, and
with --ont_coverage (the CSV written by scripts/ont_coverage.py) a second row
gives the value expected on those samples: each sample's coverage is placed
between the two nearest evaluated coverages and the metric interpolated there.
Only coverage is used, never labels.

The last two tables come from each run's saved predictions: the share of
samples that can be called, most confident first, while at least
--target_accuracy of the calls are correct (default 0.98), and the confidence
cut-off at which that happens. Unlike callable_share_0.90, this share does not
depend on how well the confidences are calibrated at each coverage, so it
compares how well each model knows when it is right.

Every sample's corrupted input depends only on its Sample_ID, the condition and
eval_seed, so runs on the same data are scored on identical inputs. With the
same seed, fold count and grouping they also share the same folds (checked below).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.conditions import (add_summary_rows, is_nanopore, nanopore_labels,  # noqa: E402
                                   read_ont_coverage, split_condition)


def callable_at_accuracy(confidence: np.ndarray, correct: np.ndarray, target: float):
    """
    Largest share of samples that can be called, most confident first, with at least
    `target` of the calls correct, and the lowest confidence among those calls. Samples
    with equal confidence are called together. Returns (0.0, nan) if no set of calls
    reaches the target.
    """
    order = np.argsort(-confidence, kind="stable")
    conf = confidence[order]
    hits = np.cumsum(correct[order])
    k = np.arange(1, len(conf) + 1)
    block_end = np.r_[conf[1:] < conf[:-1], True]
    ok = block_end & (hits >= target * k - 1e-9)
    if not ok.any():
        return 0.0, float("nan")
    last = int(np.flatnonzero(ok)[-1])
    return (last + 1) / len(conf), float(conf[last])


def trained_coverage(t: dict) -> str:
    if t["coverage_mode"] == "schedule":
        lo, hi = 1.0 - t["mask_start"], 1.0 - t["mask_end"]
        return f"schedule {round(100 * lo, 2):g}-{round(100 * hi, 2):g}%"
    lo, hi = t["cov_min"], t["cov_max"]
    dist = "uniform" if t.get("coverage_dist", "loguniform") == "uniform" else "log"
    return f"{round(100 * lo, 2):g}-{round(100 * hi, 2):g}% {dist}"


def main():
    p = argparse.ArgumentParser(description="Compare SPARSH-next runs")
    p.add_argument("runs", nargs="+", help="Run output directories")
    p.add_argument("--metrics", nargs="+", default=["balanced_accuracy", "accuracy", "callable_share_0.90",
                                                    "accuracy_callable_0.90"])
    p.add_argument("--target_accuracy", type=float, default=0.98,
                   help="Accuracy of calls for the calibration-free callable share (default 0.98)")
    p.add_argument("--all_conditions", action="store_true",
                   help="Also show the dense array profile and the mask_* conditions")
    p.add_argument("--ont_coverage", default=None,
                   help="CSV from scripts/ont_coverage.py; adds the value expected at those samples' coverages")
    p.add_argument("--output", default=None, help="Optional CSV with all tables stacked")
    args = p.parse_args()
    if not 0.0 < args.target_accuracy <= 1.0:
        sys.exit("--target_accuracy must be between 0 and 1, e.g. 0.98")

    ont_cov = None
    if args.ont_coverage:
        try:
            ont_cov = read_ont_coverage(args.ont_coverage)
        except ValueError as e:
            sys.exit(str(e))
        print(f"ONT coverage from {args.ont_coverage}: {len(ont_cov)} samples, median {100 * np.median(ont_cov):.1f}% "
              f"(range {100 * ont_cov.min():.1f}-{100 * ont_cov.max():.1f}%)")

    tables, run_dirs, settings, data_keys, eval_keys, eval_grids, splits = {}, {}, [], set(), set(), set(), set()
    names = [Path(r).name for r in args.runs]
    for run, name in zip(args.runs, names):
        run = Path(run)
        if names.count(name) > 1:          # same folder name in different places: use the full path
            name = str(run)
        f = run / "cv_metrics_by_condition.csv"
        if not f.exists():
            sys.exit(f"{f} not found (is this a finished SPARSH-next run?)")
        table = pd.read_csv(f).set_index("condition")
        if not args.all_conditions:
            table = table[[is_nanopore(c) for c in table.index]]
        tables[name], run_dirs[name] = table, run
        cfg = json.loads((run / "config.json").read_text())
        t, m, d = cfg["training"], cfg["model"], cfg["data"]
        settings.append({"run": name, "imbalance": t["imbalance"], "encoding": m["input_encoding"],
                         "train_sim": t["train_sim"], "trained_coverage": trained_coverage(t),
                         "call_error": f"0-{t['call_error_max']:g}" if t.get("call_error_max") else "0",
                         "calibrated": t["calibrate"], "n_samples": d["n_samples"], "n_classes": m["n_classes"],
                         "median_best_epoch": pd.Series([r["best_epoch"] for r in cfg["cv_folds"]]).median()})
        data_keys.add((d["data_path"], d["n_samples"], d["n_cpgs"], d.get("cpg_list"),
                       json.dumps(d.get("class_counts"), sort_keys=True), d.get("groups_file"), d.get("group_col")))
        eval_keys.add(t["eval_seed"])
        eval_grids.add((tuple(t.get("eval_sims", ["reads", "mask"])), tuple(t["eval_coverages"]),
                        tuple(t.get("eval_call_errors", [0.0]))))
        splits.add((t["seed"], t["n_folds"]))

    print(pd.DataFrame(settings).to_string(index=False))
    if len(data_keys) > 1 or len(eval_keys) > 1:
        print("\nWARNING: runs differ in data, CpG set, classes, grouping or eval_seed; "
              "they are not directly comparable.")
    elif len(splits) > 1:
        print("\nNote: runs use different fold splits (seed, fold count or grouping). Samples and evaluation "
              "inputs are identical, so differences include run-to-run noise; this is how to measure that noise.")
    if len(eval_grids) > 1:
        print("\nNote: runs were scored on different simulations, coverages or call errors; a blank cell means "
              "that run was not scored on that row.")
    if not args.all_conditions:
        print("\nSimulated nanopore conditions only; --all_conditions adds dense and mask_*.")
    if ont_cov is not None:
        all_conditions = [c for t in tables.values() for c in t.index]
        for label in nanopore_labels(all_conditions):
            covs = sorted({split_condition(c)[1] for c in all_conditions
                           if is_nanopore(c) and split_condition(c)[0] == label})
            if covs:
                out = int(((ont_cov < covs[0]) | (ont_cov > covs[-1])).sum())
                if out:
                    verb = "lies" if out == 1 else "lie"
                    print(f"Note: {out} of {len(ont_cov)} ONT samples {verb} outside the evaluated {label}_* "
                          f"coverages ({100 * covs[0]:g}-{100 * covs[-1]:g}%) and take the value at the nearest end.")

    stacked = []
    for metric in args.metrics:
        cols = {name: t[metric] for name, t in tables.items() if metric in t.columns}
        if not cols:
            print(f"\n(metric {metric} not found)")
            continue
        lacking = [name for name in tables if name not in cols]
        if lacking:
            print(f"\n(metric {metric} is missing for: {lacking}; a different --threshold?)")
        table = add_summary_rows(pd.DataFrame(cols), ont_cov)
        print(f"\n{metric}")
        print(table.to_string(float_format=lambda v: f"{v:.3f}"))
        stacked.append(table.assign(metric=metric))

    # Calibration-free operating point, from the saved out-of-fold predictions
    target = args.target_accuracy
    shares, cutoffs, missing = {}, {}, []
    for name, table in tables.items():
        s, c = {}, {}
        for cond in table.index:
            f = run_dirs[name] / f"cv_predictions_{cond}.csv"
            if not f.exists():
                missing.append(f"{name}/{f.name}")
                continue
            pred = pd.read_csv(f, usecols=["confidence", "correct"])
            s[cond], c[cond] = callable_at_accuracy(pred["confidence"].to_numpy(dtype=float),
                                                    pred["correct"].to_numpy(dtype=float), target)
        shares[name], cutoffs[name] = pd.Series(s, dtype=float), pd.Series(c, dtype=float)
    if missing:
        print(f"\n(saved predictions missing, left blank: {missing[:5]})")
    label = f"{100 * target:g}%"
    share_table = add_summary_rows(pd.DataFrame(shares), ont_cov)
    cutoff_table = pd.DataFrame(cutoffs)
    print(f"\ncallable share at {label} accuracy of calls (most confident first; does not depend on calibration)")
    print(share_table.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nconfidence cut-off that gives it (compare with 0.90)")
    print(cutoff_table.to_string(float_format=lambda v: f"{v:.3f}"))
    stacked.append(share_table.assign(metric=f"callable_share_at_accuracy_{target:.2f}"))
    stacked.append(cutoff_table.assign(metric=f"cutoff_at_accuracy_{target:.2f}"))

    if args.output and stacked:
        pd.concat(stacked).to_csv(args.output, float_format="%.4f")
        print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
