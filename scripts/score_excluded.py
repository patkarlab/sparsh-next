#!/usr/bin/env python3
"""
How a model calls training arrays of classes it was NOT trained on.

    python scripts/score_excluded.py --run ~/sparsh_next_runs/scaled_wide_no_mr_mecom

A class left out of training (for example AML-MR and AML_MECOM-r) does not leave
the clinic: such patients are still sequenced, and the model must call them
something. This script takes the arrays of the left-out classes from the
training pickle, simulates nanopore reads from them exactly as cross-validation
scores its outer folds (same simulation, seed and clipping; per-read call
errors optional), and predicts them with the run's networks. None of these
samples was used to train, stop or calibrate any of those networks, so every
network can be used. Only training arrays and their labels are read.

Default classes: the run's --exclude_classes. Others: --classes (exact names
after the run's label map) or --prefixes. Classes the model was trained on are
refused, because those samples are in-sample for some networks.

Output folder (default <run>/excluded_check):
  per_sample.csv   one row per sample and coverage: prediction, confidence, called at
                   the threshold, reported call and level (family and lineage fallback)
  summary.csv      per class and coverage: share called at the threshold (every such
                   call is wrong, since the class is not in the model), share reported
                   at each level, the most common calls
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data.dataset import load_training_data  # noqa: E402
from evaluation.hierarchy import default_hierarchy_path, hierarchical_calls, load_hierarchy  # noqa: E402
from models.corruption import READ_SIMS, corrupt_rows  # noqa: E402
from models.ensemble import ensemble_probs, load_models, load_run  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Predict training arrays of classes the model was not trained on")
    p.add_argument("--run", required=True, help="Finished run folder")
    p.add_argument("--classes", nargs="*", default=None,
                   help="Exact class names after the run's label map (default: the run's --exclude_classes)")
    p.add_argument("--prefixes", nargs="*", default=[], help="Also every class starting with these")
    p.add_argument("--data_path", default=None, help="Training pickle (default: the run's)")
    p.add_argument("--sim", choices=READ_SIMS, default=None, help="Default: the run's first nanopore --eval_sims")
    p.add_argument("--coverages", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    p.add_argument("--call_error", type=float, default=0.0, help="Per-read call error rate of the simulation")
    p.add_argument("--hierarchy", default=None, help="Default configs/class_hierarchy.json; 'none' = no families")
    p.add_argument("--use", choices=["auto", "ensemble", "final"], default="auto")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    run = Path(args.run).expanduser()
    config, classes, cpg_ids, label_map = load_run(run)
    t, d = config["training"], config["data"]
    wanted = list(args.classes) if args.classes is not None else list(d.get("exclude_classes") or [])
    if not wanted and not args.prefixes:
        sys.exit("The run excluded no classes by name; give --classes or --prefixes")
    sim = args.sim or next((s for s in t.get("eval_sims", ["reads"]) if s in READ_SIMS), "reads")
    if not 0.0 <= args.call_error < 0.5:
        sys.exit("--call_error must be between 0 and 0.5")
    threshold = config["inference"].get("threshold", 0.90)
    clip = config["inference"].get("clip_observed")

    bundle = load_training_data(args.data_path or d["data_path"], label_map, d.get("exclude_ids"),
                                str(run / "selected_cpgs.json"))
    labels = bundle["labels"]
    chosen = np.isin(labels, wanted) | np.array([any(str(l).startswith(pf) for pf in args.prefixes) for l in labels])
    found = sorted(set(labels[chosen]))
    absent = [c for c in wanted if c not in set(labels)]
    if absent:
        print(f"Not in the data after the label map: {absent}")
    trained = [c for c in found if c in classes]
    if trained:
        sys.exit(f"The model was trained on {trained}; their samples are in-sample for some networks. "
                 "Use the run's cv_predictions_*.csv for those classes instead.")
    if not chosen.any():
        sys.exit("No samples of the requested classes")
    X, ids, true = bundle["X"][chosen], bundle["sample_ids"][chosen], labels[chosen]
    print(f"{len(ids)} samples: " + ", ".join(f"{c} {int((true == c).sum())}" for c in found))

    device = args.device if (not args.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"
    use, models = load_models(run, config, args.use, device)
    hpath = default_hierarchy_path() if args.hierarchy is None else (
        None if args.hierarchy.lower() == "none" else Path(args.hierarchy))
    h = load_hierarchy(str(hpath) if hpath else None, classes)
    print(f"Model: {run} ({use}, {len(models)} network(s)); simulation {sim}, call error {args.call_error:g}, "
          f"threshold {threshold:.2f}")

    rows = []
    for cov in args.coverages:
        Xc = corrupt_rows(X, ids, cov, sim, t["eval_seed"], salt=2, call_error=args.call_error)
        if clip is not None:
            Xc = np.where(np.isnan(Xc), Xc, np.clip(Xc, clip[0], clip[1])).astype(np.float32)
        probs = ensemble_probs(models, Xc, device)
        order = np.argsort(-probs, axis=1)
        calls = hierarchical_calls(probs, classes, h, threshold)
        for i in range(len(ids)):
            rows.append({"sample_id": ids[i], "true_label": true[i], "coverage": cov,
                         "prediction": classes[order[i, 0]], "confidence": probs[i, order[i, 0]],
                         "top2_class": classes[order[i, 1]] if len(classes) > 1 else "",
                         "called": probs[i, order[i, 0]] >= threshold,
                         "reported_call": calls.at[i, "reported_call"],
                         "reported_level": calls.at[i, "reported_level"],
                         "lineage": calls.at[i, "lineage"], "lineage_prob": calls.at[i, "lineage_prob"]})
    per = pd.DataFrame(rows)

    summary = []
    for (cls, cov), g in per.groupby(["true_label", "coverage"]):
        called = g[g["called"]]
        top_calls = called["prediction"].value_counts().head(3)
        summary.append({
            "class": cls, "coverage": cov, "n": len(g),
            "called_at_threshold": g["called"].mean(),
            "median_confidence": g["confidence"].median(),
            **{f"reported_{lev}": (g["reported_level"] == lev).mean()
               for lev in ("subtype", "no_subtype", "family", "lineage", "none")},
            "called_as": "; ".join(f"{c} {n}" for c, n in top_calls.items()),
            "lineage_calls": "; ".join(f"{c} {n}" for c, n in
                                       g.loc[g["reported_level"] == "lineage", "lineage"].value_counts().items()),
        })
    summary = pd.DataFrame(summary)

    out = Path(args.output_dir) if args.output_dir else run / "excluded_check"
    out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "per_sample.csv", index=False, float_format="%.4f")
    summary.to_csv(out / "summary.csv", index=False, float_format="%.3f")
    shown = summary[["class", "coverage", "n", "called_at_threshold", "median_confidence", "reported_family",
                     "reported_lineage", "called_as"]]
    print("\nCalled at the threshold means a confident call of a class the sample does not belong to:")
    print(shown.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
