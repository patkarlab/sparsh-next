#!/usr/bin/env python3
"""
Array beta values against nanopore calls, CpG by CpG, without labels.

    python scripts/platform_check.py /path/to/ont_folder --run ~/sparsh_next_runs/scaled_wide \
        --output_dir ~/sparsh_next_runs/data_checks/platform_ont_samples_AL

The model is trained on array beta values turned into simulated reads, which
assumes that a CpG's array beta equals the chance that a nanopore read there is
methylated. This script checks that assumption on the real nanopore files.

The mean nanopore call at a CpG across samples depends on the cohort's mix of
subtypes and on its blast percentages, which differ from the training arrays.
So the comparison is made on CpGs whose array means hardly depend on class:
the mean beta of every class, normal marrow included, lies within
--max_class_range of the others. At those CpGs neither the case mix nor
dilution by normal cells can move the nanopore mean by more than that range, so
a larger difference between the platforms is a platform effect.

Printed and written to --output_dir:
  mapping.csv     array beta (reference = median of the class means) in bins of 0.05
                  against the mean nanopore call: how the two scales relate
  discordant.csv  class-stable CpGs whose nanopore mean differs from the array
                  reference by more than --discordance, with the numbers
  per_cpg.csv     every model CpG: array reference, class range, nanopore mean and
                  the number of nanopore samples with a call
Nanopore calls that are majority votes of several reads are slightly more
extreme than a single read's chance of being methylated; at the cohort's
coverage most covered CpGs have one read. Only nanopore values are read, never
their labels; array labels are used only to find class-stable CpGs.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data.dataset import load_training_data  # noqa: E402
from data.ont import DUPLICATE_RULES, read_ont_csv  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Array beta against nanopore calls per CpG (label-free)")
    p.add_argument("inputs", nargs="+", help="Folders of per-sample nanopore CSV files, or CSV files")
    p.add_argument("--run", required=True, help="A finished run folder (CpG list, training data, exclusions)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--data_path", default=None, help="Training pickle (default: the run's)")
    p.add_argument("--max_class_range", type=float, default=0.1,
                   help="Class-stable CpGs: class means (normal marrow included) within this range")
    p.add_argument("--min_class_size", type=int, default=5, help="Classes smaller than this are ignored")
    p.add_argument("--min_ont_samples", type=int, default=40,
                   help="CpGs called in fewer nanopore samples are left out of the comparison")
    p.add_argument("--discordance", type=float, default=0.3,
                   help="Difference between nanopore mean and array reference that counts as discordant")
    p.add_argument("--duplicate_probes", choices=DUPLICATE_RULES, default="mean")
    args = p.parse_args()

    run = Path(args.run).expanduser()
    config = json.loads((run / "config.json").read_text())
    cpg_ids = json.loads((run / "selected_cpgs.json").read_text())
    label_map = json.loads((run / "label_map.json").read_text()).get("merge", {})
    d = config["data"]
    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- arrays: class means
    bundle = load_training_data(args.data_path or d["data_path"], label_map, d.get("exclude_ids"),
                                str(run / "selected_cpgs.json"))
    X, labels = bundle["X"], bundle["labels"]
    names, counts = np.unique(labels, return_counts=True)
    used = [c for c, n in zip(names, counts) if n >= args.min_class_size]
    means = np.full((len(used), X.shape[1]), np.nan, dtype=np.float32)
    for k, c in enumerate(used):
        rows = X[labels == c]
        with np.errstate(invalid="ignore"):
            means[k] = np.nanmean(rows, axis=0) if np.isnan(rows).any() else rows.mean(axis=0)
    with np.errstate(invalid="ignore"):
        reference = np.nanmedian(means, axis=0)
        class_range = np.nanmax(means, axis=0) - np.nanmin(means, axis=0)
    stable = class_range <= args.max_class_range
    del X
    print(f"Arrays: {len(labels)} samples, {len(used)} classes with at least {args.min_class_size}; "
          f"{int(stable.sum())} of {len(cpg_ids)} CpGs are class-stable (class means within "
          f"{args.max_class_range:g})")

    # ---------------------------------------------------------------- nanopore: mean call per CpG
    total = np.zeros(len(cpg_ids), dtype=np.float64)
    n_called = np.zeros(len(cpg_ids), dtype=np.int64)
    n_files, errors = 0, []
    for item in args.inputs:
        path = Path(item).expanduser()
        files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
        for f in files:
            try:
                x, _ = read_ont_csv(f, cpg_ids, duplicates=args.duplicate_probes)
            except Exception as e:  # report and carry on
                errors.append((f.name, str(e)))
                continue
            obs = np.isfinite(x)
            total[obs] += x[obs]
            n_called[obs] += 1
            n_files += 1
    if not n_files:
        sys.exit("No readable nanopore files")
    with np.errstate(invalid="ignore", divide="ignore"):
        ont_mean = np.where(n_called > 0, total / np.maximum(n_called, 1), np.nan)
    enough = n_called >= args.min_ont_samples
    print(f"Nanopore: {n_files} samples read ({len(errors)} rejected); CpGs called in at least "
          f"{args.min_ont_samples} samples: {int(enough.sum())}")

    per_cpg = pd.DataFrame({"cpg": cpg_ids, "array_reference": reference, "class_range": class_range,
                            "class_stable": stable, "ont_mean": ont_mean, "ont_samples": n_called})
    per_cpg["difference"] = per_cpg["ont_mean"] - per_cpg["array_reference"]
    per_cpg.to_csv(out / "per_cpg.csv", index=False, float_format="%.4f")

    comp = per_cpg[stable & enough & np.isfinite(reference)].copy()
    if not len(comp):
        sys.exit("No class-stable CpG with enough nanopore calls; raise --max_class_range or lower --min_ont_samples")
    edges = np.linspace(0.0, 1.0, 21)
    comp["bin"] = pd.cut(comp["array_reference"], edges, include_lowest=True)
    mapping = comp.groupby("bin", observed=True).agg(
        n_cpgs=("cpg", "size"), array_mean=("array_reference", "mean"), ont_mean=("ont_mean", "mean"),
        ont_q25=("ont_mean", lambda s: s.quantile(0.25)), ont_q75=("ont_mean", lambda s: s.quantile(0.75)))
    mapping.to_csv(out / "mapping.csv", float_format="%.4f")
    disc = comp[comp["difference"].abs() > args.discordance].drop(columns="bin")
    disc.sort_values("difference", key=np.abs, ascending=False).to_csv(out / "discordant.csv", index=False,
                                                                        float_format="%.4f")

    r = float(np.corrcoef(comp["array_reference"], comp["ont_mean"])[0, 1])
    mid = comp[(comp["array_reference"] > 0.2) & (comp["array_reference"] < 0.8)]
    print(f"\nClass-stable CpGs compared: {len(comp)}; Pearson r {r:.3f}; mean difference (nanopore - array) "
          f"{comp['difference'].mean():+.3f}; median absolute difference {comp['difference'].abs().median():.3f}")
    if len(mid):
        print(f"  with array reference 0.2-0.8: {len(mid)} CpGs, mean difference {mid['difference'].mean():+.3f}, "
              f"median absolute difference {mid['difference'].abs().median():.3f}")
    print(f"Discordant (|difference| > {args.discordance:g}): {len(disc)} "
          f"({100 * len(disc) / len(comp):.2f}% of those compared); "
          f"nanopore higher {int((disc['difference'] > 0).sum())}, lower {int((disc['difference'] < 0).sum())}")
    print("\nArray reference (bin) against mean nanopore call:")
    print(mapping.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nOn the diagonal, the simulation's assumption holds. Nanopore means further from 0.5 than the array "
          "values mean the arrays compress beta values, so reads simulated from them are noisier than real ones. "
          "A systematic shift, or many discordant CpGs, points to probes to check or drop before the next model "
          "is fixed.")
    if errors:
        print(f"{len(errors)} files could not be read, e.g. {errors[0][0]}: {errors[0][1]}")
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
