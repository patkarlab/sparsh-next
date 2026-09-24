#!/usr/bin/env python3
"""
How much of a model's CpG set each nanopore sample covers, checked before prediction.

    python scripts/ont_coverage.py --run ~/sparsh_next_runs/default /path/to/ont_folder [more folders or CSV files]

Every CSV directly inside each folder (the files scripts/predict.py --ont_dir reads)
goes through the same loader as prediction (data/ont.py). For each folder it prints:
- files read, and files the loader rejects, with the reason;
- CpG columns per file, and how many of them are model CpGs;
- coverage, the percentage of the model's CpGs with a value: median, range and the
  number of samples per band, next to the coverage range the run was trained on;
- the average number of reads per CpG that this coverage implies;
- the share of observed values that are exactly 0 or 1. A CpG covered by a single
  read is always 0 or 1, which is what the read simulation used in training produces
  at low coverage; a low share means the files hold something else (for example
  smoothed values or modification probabilities).
Nothing is written unless --output is given (one line per file).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data.ont import read_ont_csv  # noqa: E402

BAND_EDGES = [0, 3, 5, 10, 20, 30, 50, 101]
BAND_NAMES = ["<3%", "3-5%", "5-10%", "10-20%", "20-30%", "30-50%", ">=50%"]


def trained_range(config: dict):
    t = config["training"]
    if t["coverage_mode"] == "schedule":
        return 1.0 - t["mask_start"], 1.0 - t["mask_end"]
    return t["cov_min"], t["cov_max"]


def inputs_to_files(items):
    """(label, files, csv files in subfolders) per argument; None for a path that does not exist."""
    groups = []
    for item in items:
        path = Path(item).expanduser()
        if path.is_dir():
            files = sorted(path.glob("*.csv"))
            nested = 0 if files else sum(1 for _ in path.rglob("*.csv"))
            groups.append((str(path), files, nested))
        elif path.is_file():
            groups.append((str(path), [path], 0))
        else:
            groups.append((str(path), None, 0))
    return groups


def main():
    p = argparse.ArgumentParser(description="Coverage of nanopore samples on a model's CpGs")
    p.add_argument("inputs", nargs="+", help="Folders of per-sample CSV files, or CSV files")
    p.add_argument("--run", required=True, help="A finished run folder; its CpG list and training range are used")
    p.add_argument("--output", default=None, help="Optional CSV with one line per file")
    args = p.parse_args()

    run = Path(args.run).expanduser()
    if not (run / "selected_cpgs.json").exists() or not (run / "config.json").exists():
        sys.exit(f"{run} is not a finished SPARSH-next run (selected_cpgs.json or config.json missing)")
    cpg_ids = json.loads((run / "selected_cpgs.json").read_text())
    lo, hi = trained_range(json.loads((run / "config.json").read_text()))
    print(f"Model CpGs: {len(cpg_ids)} (from {run}); trained on coverage {100 * lo:g}-{100 * hi:g}%")

    records = []
    for label, files, nested in inputs_to_files(args.inputs):
        print(f"\n== {label}")
        if files is None:
            print("   not found")
            continue
        if not files:
            where = f"; {nested} in subfolders, which predict.py does not read" if nested else ""
            print(f"   no CSV files directly in this folder{where}")
            continue
        rows, errors = [], []
        for f in files:
            try:
                x, info = read_ont_csv(f, cpg_ids)
            except Exception as e:  # report every unreadable file and carry on
                errors.append((f.name, str(e)))
                records.append({"input": label, "file": f.name, "error": str(e)})
                continue
            v = x[np.isfinite(x)]
            row = {"input": label, "file": f.name, "sample": f.stem.strip().upper(),
                   "n_columns": info["n_columns"], "n_model_cpgs": info["n_matched"],
                   "n_observed": info["n_observed"], "coverage_pct": 100.0 * info["coverage"],
                   "share_0_or_1": float(np.isin(v, (0.0, 1.0)).mean()) if len(v) else float("nan"),
                   "mean_value": float(v.mean()) if len(v) else float("nan"), "error": ""}
            rows.append(row)
            records.append(row)

        print(f"   {len(files)} CSV file{'s' if len(files) != 1 else ''}: {len(rows)} read, "
              f"{len(errors)} rejected by the loader")
        for name, message in errors[:3]:
            print(f"     {message if message.startswith(name) else f'{name}: {message}'}")
        if not rows:
            continue
        df = pd.DataFrame(rows)
        cov = df["coverage_pct"]
        median_depth = -math.log(max(1e-9, 1.0 - cov.median() / 100.0))
        print(f"   CpG columns per file: median {df['n_columns'].median():.0f}; "
              f"model CpGs among them: median {df['n_model_cpgs'].median():.0f} of {len(cpg_ids)}")
        print(f"   Coverage of the model CpGs: median {cov.median():.1f}% (range {cov.min():.1f}-{cov.max():.1f}%), "
              f"about {median_depth:.2f} reads per CpG at the median")
        bands = pd.cut(cov, bins=BAND_EDGES, right=False, labels=BAND_NAMES).value_counts(sort=False)
        print("   Samples per band: " + "  ".join(f"{b} {int(n)}" for b, n in bands.items()))
        below, above = int((cov < 100 * lo).sum()), int((cov > 100 * hi).sum())
        print(f"   Outside the trained range ({100 * lo:g}-{100 * hi:g}%): {below} below, {above} above")
        print(f"   Observed values that are exactly 0 or 1: median {100 * df['share_0_or_1'].median():.0f}% per sample; "
              f"mean observed value: median {df['mean_value'].median():.2f}")

    if args.output and records:
        pd.DataFrame(records).to_csv(args.output, index=False, float_format="%.4f")
        print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
