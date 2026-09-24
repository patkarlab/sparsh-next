#!/usr/bin/env python3
"""
Per-read call error of nanopore samples, estimated without labels.

    python scripts/ont_call_error.py --run ~/sparsh_next_runs/scaled_wide /path/to/ont_folder \
        --output ~/sparsh_next_runs/data_checks/ont_call_error.csv

Some CpGs are methylated in almost every training array whatever the subtype
(beta >= --high in at least --min_share of the arrays), and some are
unmethylated in almost all (beta <= --low). At these CpGs the blast percentage
and the diagnosis do not matter, so a nanopore call that disagrees with the
arrays is an error of the nanopore data: base-calling and modification calling,
the pipeline's 0/1 thresholding, or a platform difference at that CpG.

With a per-read error rate e (models/corruption.py), a read at a CpG with array
beta b is methylated with probability b(1 - 2e) + e. The mean call at the
observed constitutive CpGs therefore gives e, separately on the methylated side
(e_high) and the unmethylated side (e_low). A negative value means the
nanopore calls are more extreme than the array beta values (arrays compress
beta towards the middle).

The same estimator is applied to training arrays run through the read
simulation without call errors and with 10%, at a grid of coverages spanning
the samples', so its bias is known at every coverage (majority calls at CpGs
with several reads hide part of the error as coverage rises). "Matched call
error" is the simulated rate that gives the value seen in a real sample,
interpolated at that sample's coverage: the number to use for --val_call_error.

Only the nanopore values are read, never their labels. Like coverage, this is a
property of the inputs. It does not measure dilution by normal cells, which
leaves constitutive CpGs unchanged.
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
from models.corruption import corrupt_rows  # noqa: E402

CHECK_ERROR = 0.10


def estimate(x: np.ndarray, high: np.ndarray, low: np.ndarray, mean_beta: np.ndarray) -> dict:
    """Error estimates from one sample's values (NaN = no data) at the constitutive CpGs."""
    obs = np.isfinite(x)
    oh, ol = high & obs, low & obs
    out = {"n_high": int(oh.sum()), "n_low": int(ol.sum())}
    if oh.any():
        f, b = float(x[oh].mean()), float(mean_beta[oh].mean())
        out.update(mean_call_high=f, array_beta_high=b, e_high=(b - f) / (2.0 * b - 1.0))
    else:
        out.update(mean_call_high=np.nan, array_beta_high=np.nan, e_high=np.nan)
    if ol.any():
        f, b = float(x[ol].mean()), float(mean_beta[ol].mean())
        out.update(mean_call_low=f, array_beta_low=b, e_low=(f - b) / (1.0 - 2.0 * b))
    else:
        out.update(mean_call_low=np.nan, array_beta_low=np.nan, e_low=np.nan)
    out["e_mean"] = float(np.nanmean([out["e_high"], out["e_low"]])) if (oh.any() or ol.any()) else np.nan
    return out


def iqr_text(s: pd.Series) -> str:
    s = s.dropna()
    if not len(s):
        return "n/a"
    return f"{100 * s.median():.1f}% (IQR {100 * s.quantile(0.25):.1f}-{100 * s.quantile(0.75):.1f}%)"


def main():
    p = argparse.ArgumentParser(description="Label-free per-read call error of nanopore samples")
    p.add_argument("inputs", nargs="+", help="Folders of per-sample CSV files, or CSV files")
    p.add_argument("--run", required=True, help="A finished run folder (CpG list, training data, exclusions)")
    p.add_argument("--data_path", default=None, help="Training pickle (default: the run's)")
    p.add_argument("--high", type=float, default=0.9)
    p.add_argument("--low", type=float, default=0.1)
    p.add_argument("--min_share", type=float, default=0.99,
                   help="Share of training arrays that must be at or beyond --high / --low")
    p.add_argument("--n_check", type=int, default=150, help="Training arrays for the estimator check")
    p.add_argument("--duplicate_probes", choices=DUPLICATE_RULES, default="mean")
    p.add_argument("--output", default=None, help="Optional CSV, one line per file")
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()

    run = Path(args.run).expanduser()
    config = json.loads((run / "config.json").read_text())
    cpg_ids = json.loads((run / "selected_cpgs.json").read_text())
    d = config["data"]
    bundle = load_training_data(args.data_path or d["data_path"], {}, d.get("exclude_ids"),
                                str(run / "selected_cpgs.json"))
    X = bundle["X"]
    n_obs = np.isfinite(X).sum(axis=0)
    enough = n_obs >= 0.9 * len(X)
    with np.errstate(invalid="ignore", divide="ignore"):
        share_high = (X >= args.high).sum(axis=0) / np.maximum(n_obs, 1)
        share_low = (X <= args.low).sum(axis=0) / np.maximum(n_obs, 1)
        mean_beta = np.nansum(X, axis=0) / np.maximum(n_obs, 1)
    high = enough & (share_high >= args.min_share)
    low = enough & (share_low >= args.min_share)
    print(f"Training arrays: {len(X)}; of {len(cpg_ids)} model CpGs, {int(high.sum())} are >= {args.high} and "
          f"{int(low.sum())} are <= {args.low} in at least {100 * args.min_share:g}% of arrays "
          f"(mean array beta {mean_beta[high].mean():.3f} and {mean_beta[low].mean():.3f})")
    if high.sum() < 1000 or low.sum() < 1000:
        print("Warning: fewer than 1,000 constitutive CpGs on one side; estimates will be noisy "
              "(lower --min_share or widen --high/--low)")

    records = []
    for item in args.inputs:
        path = Path(item).expanduser()
        files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
        for f in files:
            try:
                x, info = read_ont_csv(f, cpg_ids, duplicates=args.duplicate_probes)
            except Exception as e:  # report and carry on
                records.append({"input": str(path), "file": f.name, "error": str(e)})
                continue
            v = x[np.isfinite(x)]
            records.append({"input": str(path), "file": f.name, "sample": f.stem.strip().upper(),
                            "coverage_pct": 100.0 * info["coverage"],
                            "share_0_or_1": float(np.isin(v, (0.0, 1.0)).mean()) if len(v) else np.nan,
                            **estimate(x, high, low, mean_beta), "error": ""})
    df = pd.DataFrame(records)
    ok = df[df["error"] == ""] if "error" in df else df
    if not len(ok):
        sys.exit("No readable nanopore files")

    # estimator check on simulated nanopore data from training arrays, on a grid of coverages
    binary_like = ok["share_0_or_1"].median() > 0.95
    sim = "binary" if binary_like else "reads"
    covs = ok["coverage_pct"].to_numpy(dtype=float) / 100.0
    grid = np.unique(np.round(np.quantile(covs, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]), 3))
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(X), size=min(args.n_check, len(X)), replace=False)
    sides = ("e_high", "e_low", "e_mean")
    base = {side: [] for side in sides}
    slope = {side: [] for side in sides}
    print(f"\nEstimator check: {len(pick)} training arrays simulated as '{sim}' at {len(grid)} coverages; "
          f"median estimate without call errors and with {100 * CHECK_ERROR:g}%")
    for cov in grid:
        est = {}
        for err in (0.0, CHECK_ERROR):
            Xs = corrupt_rows(X[pick], bundle["sample_ids"][pick], float(cov), sim, args.seed, salt=7,
                              call_error=err)
            est[err] = pd.DataFrame([estimate(r, high, low, mean_beta) for r in Xs]).median()
        for side in sides:
            b0, b1 = est[0.0][side], est[CHECK_ERROR][side]
            base[side].append(b0)
            slope[side].append((b1 - b0) / CHECK_ERROR)
        print(f"  {100 * cov:5.1f}% coverage: e_mean {100 * est[0.0]['e_mean']:.1f}% and "
              f"{100 * est[CHECK_ERROR]['e_mean']:.1f}%")
    for side in sides:
        b = np.interp(covs, grid, np.array(base[side], dtype=float))
        k = np.interp(covs, grid, np.array(slope[side], dtype=float))
        ok = ok.assign(**{f"matched_{side}": (ok[side].to_numpy(dtype=float) - b) / k})
    df = df.merge(ok[["file", "input"] + [f"matched_{s}" for s in ("e_high", "e_low", "e_mean")]],
                  on=["file", "input"], how="left")

    for label, g in ok.groupby("input", sort=False):
        print(f"\n== {label}: {len(g)} samples, coverage median {g['coverage_pct'].median():.1f}%; "
              f"constitutive CpGs observed per sample: median {g['n_high'].median():.0f} methylated, "
              f"{g['n_low'].median():.0f} unmethylated")
        print(f"   mean call at methylated CpGs {g['mean_call_high'].median():.3f} (array beta "
              f"{g['array_beta_high'].median():.3f}); at unmethylated CpGs {g['mean_call_low'].median():.3f} "
              f"(array beta {g['array_beta_low'].median():.3f})")
        print(f"   estimated error: methylated side {iqr_text(g['e_high'])}, unmethylated side {iqr_text(g['e_low'])}")
        print(f"   matched call error (simulated rate giving the same estimate): methylated side "
              f"{iqr_text(g['matched_e_high'])}, unmethylated side {iqr_text(g['matched_e_low'])}, "
              f"both {iqr_text(g['matched_e_mean'])}")
        if len(g) >= 10:
            rho = g[["coverage_pct", "matched_e_mean"]].corr(method="spearman").iloc[0, 1]
            print(f"   Spearman correlation of the matched error with coverage: {rho:.2f}")
    print("\nA per-read error should not depend on coverage; a strong correlation points to something else, "
          "such as the pipeline's handling of CpGs with several reads.")
    errors = df[df["error"] != ""] if "error" in df else df.iloc[0:0]
    if len(errors):
        print(f"{len(errors)} files could not be read, e.g. {errors['file'].iloc[0]}: {errors['error'].iloc[0]}")
    if args.output:
        df.to_csv(args.output, index=False, float_format="%.5f")
        print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
