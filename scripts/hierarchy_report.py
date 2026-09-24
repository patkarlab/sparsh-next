#!/usr/bin/env python3
"""
What the family fallback adds, from each run's saved out-of-fold predictions (training data only).

    python scripts/hierarchy_report.py ~/sparsh_next_runs/{scaled_wide_no_mr_mecom,scaled_wide_aml_other} \
        --ont_coverage ~/sparsh_next_runs/data_checks/ont_coverage_AL.csv

For every simulated nanopore condition, each sample is reported at the most
specific level that reaches the threshold (subtype, family or lineage; see
evaluation/hierarchy.py and configs/class_hierarchy.json), and the tables give,
per run:
  share_subtype           called at subtype level (with share_no_subtype, the callable
                          share at the threshold)
  share_reported_any_level called at any level
  accuracy_reported       share of those calls that are correct at the reported level
  share_family, share_lineage, share_no_subtype
with the mean over coverages and, with --ont_coverage, the value expected on
those samples' coverages (coverage only, never labels).

Written into each run folder, under hierarchy_report/:
  hierarchy_by_condition.csv       the numbers above for every condition
  by_class_<condition>.csv         per true class: recall, confident errors, where the
                                   calls went (subtype / no specific subtype / family /
                                   lineage / not callable), accuracy at the reported level
  confusion_pairs_<condition>.csv  class pairs ranked by how often they are confused, with
                                   whether they share a family or lineage: the evidence
                                   for refining the families
The per-class and pair tables use --detail_condition. Default: the binary
condition nearest 5% coverage, because the nanopore scoring of 24 September
2026 found that real samples confuse classes the way simulated data at much
lower coverage do. Runs with different class sets (for example with AML-MR and
AML_MECOM-r dropped or grouped) are compared class by class in by_class_*.csv.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.conditions import (add_summary_rows, is_nanopore, parse_condition,  # noqa: E402
                                   read_ont_coverage)
from evaluation.hierarchy import (LEVELS, default_hierarchy_path, hierarchical_calls, load_hierarchy,  # noqa: E402
                                  score_calls, summarize_calls)
from models.ensemble import load_run  # noqa: E402

SHOWN = ["share_subtype", "share_reported_any_level", "accuracy_reported", "share_family", "share_lineage",
         "share_no_subtype"]


def read_predictions(path: Path, classes):
    df = pd.read_csv(path)
    missing = [c for c in classes if f"prob_{c}" not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: no probability columns for {missing[:3]}")
    return df, df[[f"prob_{c}" for c in classes]].to_numpy(dtype=float)


def by_class(df: pd.DataFrame, calls: pd.DataFrame, scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    d = pd.concat([df[["true_label", "prediction", "confidence"]].reset_index(drop=True), calls, scored], axis=1)
    d["correct"] = d["true_label"] == d["prediction"]
    d["confident_error"] = (d["confidence"] >= threshold) & ~d["correct"]
    g = d.groupby("true_label")
    table = pd.DataFrame({"n": g.size(), "recall": g["correct"].mean(),
                          "confident_error": g["confident_error"].mean()})
    for lev in LEVELS:
        table[f"reported_{lev}"] = g["reported_level"].apply(lambda s, lev=lev: (s == lev).mean())
    table["accuracy_reported"] = g["reported_correct"].mean()
    wrong = d[~d["correct"]]
    common = {t: w["prediction"].value_counts().index[0] for t, w in wrong.groupby("true_label")}
    table["most_common_wrong_call"] = [common.get(t, "") for t in table.index]
    return table


def confusion_pairs(df: pd.DataFrame, h, top: int = 25) -> pd.DataFrame:
    counts = df.groupby("true_label").size()
    cm = pd.crosstab(df["true_label"], df["prediction"])
    rows = []
    classes = sorted(counts.index)
    for i, a in enumerate(classes):
        for b in classes[i + 1:]:
            ab = int(cm.at[a, b]) if (a in cm.index and b in cm.columns) else 0
            ba = int(cm.at[b, a]) if (b in cm.index and a in cm.columns) else 0
            if ab + ba == 0:
                continue
            rows.append({"class_a": a, "class_b": b, "a_called_b": ab, "b_called_a": ba,
                         "rate": (ab + ba) / (counts[a] + counts[b]),
                         "same_family": bool(h.family_of.get(a)) and h.family_of.get(a) == h.family_of.get(b),
                         "same_lineage": h.lineage(a) == h.lineage(b)})
    out = pd.DataFrame(rows)
    return out.sort_values("rate", ascending=False).head(top) if len(out) else out


def pick_detail(conditions, wanted: str) -> str:
    if wanted != "auto":
        if wanted not in conditions:
            sys.exit(f"--detail_condition {wanted} is not one of {conditions}")
        return wanted
    binary = [c for c in conditions if parse_condition(c)[0] == "binary" and parse_condition(c)[1] == 0.0]
    pool = binary or conditions
    return min(pool, key=lambda c: abs(parse_condition(c)[2] - 0.05))


def main():
    p = argparse.ArgumentParser(description="Family and lineage fallback on cross-validation predictions")
    p.add_argument("runs", nargs="+", help="Finished run folders")
    p.add_argument("--hierarchy", default=None,
                   help="Default configs/class_hierarchy.json; 'none' = lineage by class-name prefix only")
    p.add_argument("--threshold", type=float, default=None, help="Default: each run's stored threshold")
    p.add_argument("--ont_coverage", default=None, help="CSV from scripts/ont_coverage.py (coverage only)")
    p.add_argument("--detail_condition", default="auto", help="Condition for the per-class and pair tables")
    args = p.parse_args()

    if args.hierarchy is None:
        hierarchy_path = default_hierarchy_path()
    else:
        hierarchy_path = None if args.hierarchy.lower() == "none" else Path(args.hierarchy)
    ont_cov = None
    if args.ont_coverage:
        try:
            ont_cov = read_ont_coverage(args.ont_coverage)
        except ValueError as e:
            sys.exit(str(e))

    per_metric = {m: {} for m in SHOWN}
    for run in args.runs:
        run = Path(run).expanduser()
        name = run.name
        config, classes, _, _ = load_run(run)
        threshold = args.threshold if args.threshold is not None else config["inference"].get("threshold", 0.90)
        h = load_hierarchy(str(hierarchy_path) if hierarchy_path else None, classes)
        out = run / "hierarchy_report"
        out.mkdir(exist_ok=True)
        saved = {f.name[len("cv_predictions_"):-len(".csv")] for f in run.glob("cv_predictions_*.csv")}
        order = pd.read_csv(run / "cv_metrics_by_condition.csv")["condition"].tolist()
        conditions = [c for c in order if c in saved and is_nanopore(c)]
        if not conditions:
            sys.exit(f"{run}: no saved nanopore predictions (cv_predictions_<condition>.csv)")
        print(f"\n== {name}: {len(classes)} classes, threshold {threshold:.2f}")
        print(h.describe())

        rows = {}
        detail = pick_detail(conditions, args.detail_condition)
        for cond in conditions:
            df, probs = read_predictions(run / f"cv_predictions_{cond}.csv", classes)
            calls = hierarchical_calls(probs, classes, h, threshold)
            scored = score_calls(df["true_label"], df["prediction"], calls, h)
            rows[cond] = summarize_calls(calls, scored)
            if cond == detail:
                by_class(df, calls, scored, threshold).to_csv(out / f"by_class_{cond}.csv", float_format="%.3f")
                pairs = confusion_pairs(df, h)
                pairs.to_csv(out / f"confusion_pairs_{cond}.csv", index=False, float_format="%.3f")
        table = pd.DataFrame(rows).T
        table.index.name = "condition"
        table.to_csv(out / "hierarchy_by_condition.csv", float_format="%.4f")
        for m in SHOWN:
            per_metric[m][name] = table[m]
        print(f"Written to {out} (per-class and pair tables at {detail})")
        if len(pairs):
            print(f"Most confused pairs at {detail} (rate = confusions / samples of both classes):")
            print(pairs.head(10).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    for m in SHOWN:
        table = add_summary_rows(pd.DataFrame(per_metric[m]), ont_cov)
        print(f"\n{m}")
        print(table.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
