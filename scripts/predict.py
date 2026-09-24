#!/usr/bin/env python3
"""
SPARSH-next prediction on ONT samples, with optional evaluation.

Examples
    # a directory of per-sample CSVs, fold ensemble, with ground truth
    python scripts/predict.py --model_dir runs/next_default --ont_dir /path/to/ont_csvs \
        --output_dir runs/next_default/ont_eval --ground_truth truth.csv

    # one or a few samples
    python scripts/predict.py --model_dir runs/next_default --samples S01.csv --output_dir tmp/S01

Input CSVs: wide format, one row, one column per CpG, values = methylated
fraction in [0, 1] (see data/ont.py). The model directory supplies everything
else: CpG list, classes, label map, input handling and temperatures.

Ground truth: CSV with columns sample,true_label. The training label map is
applied to it, so raw subtype names are scored against the merged classes.
Samples whose true label is not a model class are reported and counted as
errors in the overall accuracy, never dropped silently.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data.dataset import apply_label_map  # noqa: E402
from data.ont import DUPLICATE_RULES, read_ont_csv  # noqa: E402
from evaluation.metrics import balanced_accuracy, confusion_frame, expected_calibration_error  # noqa: E402
from models.sparse_nn import load_model, predict_logits, softmax_np  # noqa: E402

COVERAGE_BINS = [0.0, 0.05, 0.10, 0.20, 0.30, 1.01]


def parse_args():
    p = argparse.ArgumentParser(description="SPARSH-next prediction and evaluation on ONT samples")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--ont_dir", default=None, help="Directory of per-sample CSV files")
    p.add_argument("--samples", nargs="*", default=[], help="Individual CSV files")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--ground_truth", default=None, help="CSV with columns sample,true_label")
    p.add_argument("--no_label_map", action="store_true", help="Truth labels are already model class names")
    p.add_argument("--use", choices=["auto", "ensemble", "final"], default="auto",
                   help="auto = fold ensemble if saved, otherwise the final model")
    p.add_argument("--threshold", type=float, default=None, help="Default: the value stored with the model")
    p.add_argument("--min_coverage", type=float, default=0.01,
                   help="Samples with fewer observed CpGs (fraction) are marked not callable")
    p.add_argument("--duplicate_probes", choices=DUPLICATE_RULES, default="mean",
                   help="Probe IDs repeated in a file: mean of the copies with a value (default), "
                        "first column only, or refuse the file (see data/ont.py)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_models(model_dir: Path, config: dict, use: str, device: str):
    inf = config["inference"]
    folds, final = inf.get("fold_models"), inf.get("final_model")
    if use == "auto":
        use = "ensemble" if folds else "final"
    entries = folds if use == "ensemble" else ([final] if final else None)
    if not entries:
        sys.exit(f"No {use} weights recorded in {model_dir}/config.json")
    models = []
    for entry in entries:
        path = model_dir / entry["file"]
        if not path.exists():
            sys.exit(f"Missing weights file {path}")
        models.append((load_model(path, config["model"], device), float(entry.get("temperature", 1.0))))
    return use, models


def coverage_bin(c: float) -> str:
    for lo, hi in zip(COVERAGE_BINS[:-1], COVERAGE_BINS[1:]):
        if lo <= c < hi:
            return f"{int(lo * 100)}-{int(min(hi, 1.0) * 100)}%"
    return "?"


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        sys.exit(f"Output directory {out} is not empty. Choose a new --output_dir or pass --overwrite.")
    out.mkdir(parents=True, exist_ok=True)
    device = args.device if (not args.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"

    config = json.loads((model_dir / "config.json").read_text())
    idx_to_class = {int(k): v for k, v in json.loads((model_dir / "class_mapping.json").read_text()).items()}
    classes = [idx_to_class[i] for i in range(len(idx_to_class))]
    cpg_ids = json.loads((model_dir / "selected_cpgs.json").read_text())
    label_map = json.loads((model_dir / "label_map.json").read_text()).get("merge", {})
    threshold = args.threshold if args.threshold is not None else config["inference"].get("threshold", 0.90)
    clip = config["inference"].get("clip_observed")
    use, models = load_models(model_dir, config, args.use, device)
    print(f"Model: {model_dir} ({use}, {len(models)} network(s), {len(cpg_ids)} CpGs, {len(classes)} classes)")

    files = sorted(Path(args.ont_dir).glob("*.csv")) if args.ont_dir else []
    files += [Path(s) for s in args.samples]
    unique, seen = [], set()
    for f in files:                      # the same file given twice is read once
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(f)
    files = unique
    if not files:
        sys.exit("No input CSV files (use --ont_dir and/or --samples)")
    stems = pd.Series([f.stem.strip().upper() for f in files])
    if stems.duplicated().any():
        sys.exit(f"Different files map to the same sample name (case is ignored): "
                 f"{sorted(set(stems[stems.duplicated()]))}")

    # ------------------------------------------------------------ load samples
    rows, X_ok, errors = [], [], []
    for f in files:
        sample = f.stem.strip().upper()
        try:
            x, info = read_ont_csv(f, cpg_ids, duplicates=args.duplicate_probes)
        except (ValueError, OSError) as e:
            errors.append({"sample": sample, "file": str(f), "error": str(e)})
            print(f"  ERROR {f.name}: {e}")
            continue
        if clip is not None:
            x = np.where(np.isnan(x), np.nan, np.clip(x, clip[0], clip[1])).astype(np.float32)
        rows.append({"sample": sample, "file": f.name, "n_columns": info["n_columns"],
                     "n_matched": info["n_matched"], "n_observed": info["n_observed"],
                     "coverage_pct": 100.0 * info["coverage"], "n_repeated_probes": info["n_repeated_probes"]})
        X_ok.append(x)
    if errors:
        pd.DataFrame(errors).to_csv(out / "input_errors.csv", index=False)
    if not rows:
        sys.exit("No readable samples")
    X = np.vstack(X_ok)
    meta = pd.DataFrame(rows)
    with_rep = meta["n_repeated_probes"] > 0
    if with_rep.any():
        print(f"Repeated probe IDs in {int(with_rep.sum())} of {len(meta)} files (median "
              f"{meta.loc[with_rep, 'n_repeated_probes'].median():.0f} per file); copies combined by "
              f"--duplicate_probes {args.duplicate_probes}")

    # ------------------------------------------------------------ predict
    probs = np.mean([softmax_np(predict_logits(m, X, device), t) for m, t in models], axis=0)
    order = np.argsort(-probs, axis=1)
    meta["prediction"] = [classes[i] for i in order[:, 0]]
    meta["confidence"] = probs[np.arange(len(probs)), order[:, 0]]
    for r in range(1, min(3, len(classes))):
        meta[f"top{r + 1}_class"] = [classes[i] for i in order[:, r]]
        meta[f"top{r + 1}_prob"] = probs[np.arange(len(probs)), order[:, r]]
    if "top2_class" not in meta:
        meta["top2_class"] = ""
    enough = meta["coverage_pct"] >= 100.0 * args.min_coverage
    meta["callable"] = enough & (meta["confidence"] >= threshold)
    meta["note"] = np.where(enough, "", f"coverage below {100 * args.min_coverage:.1f}%")
    prob_df = pd.DataFrame(probs, columns=[f"prob_{c}" for c in classes])
    pd.concat([meta, prob_df], axis=1).to_csv(out / "predictions.csv", index=False, float_format="%.5f")
    print(f"Predicted {len(meta)} samples ({len(errors)} unreadable); coverage median "
          f"{meta['coverage_pct'].median():.1f}% (range {meta['coverage_pct'].min():.1f}-"
          f"{meta['coverage_pct'].max():.1f}%); callable at >= {threshold:.2f}: {int(meta['callable'].sum())}")

    if not args.ground_truth:
        return

    # ------------------------------------------------------------ evaluate
    gt = pd.read_csv(args.ground_truth, dtype=str)
    if not {"sample", "true_label"} <= set(gt.columns):
        sys.exit("Ground truth needs columns: sample,true_label")
    blank = gt["sample"].isna() | gt["true_label"].isna() | (gt["true_label"].fillna("").str.strip() == "")
    if blank.any():
        print(f"Ground truth: {int(blank.sum())} rows without a sample or label are ignored: "
              f"{gt.loc[blank, 'sample'].fillna('?').tolist()[:10]}")
        gt = gt[~blank].copy()
    gt["sample"] = gt["sample"].str.strip().str.upper()
    gt["true_label_raw"] = gt["true_label"].str.strip()
    gt["true_label"] = gt["true_label_raw"] if args.no_label_map else apply_label_map(gt["true_label_raw"], label_map, log=False)
    if gt["sample"].duplicated().any():
        sys.exit(f"Duplicated samples in ground truth: {gt.loc[gt['sample'].duplicated(), 'sample'].tolist()[:5]}")
    ev = meta.merge(gt[["sample", "true_label_raw", "true_label"]], on="sample", how="inner")
    no_truth = sorted(set(meta["sample"]) - set(gt["sample"]))
    no_file = sorted(set(gt["sample"]) - set(meta["sample"]))
    ev["in_scheme"] = ev["true_label"].isin(classes)
    ev["correct"] = ev["prediction"] == ev["true_label"]
    ev["top2_correct"] = ev["correct"] | (ev["top2_class"] == ev["true_label"])
    ev["coverage_bin"] = [coverage_bin(c / 100.0) for c in ev["coverage_pct"]]
    ev.to_csv(out / "evaluation_per_sample.csv", index=False, float_format="%.5f")

    ins = ev[ev["in_scheme"]]
    y_true = ins["true_label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    y_pred = ins["prediction"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    summary = {
        "model_dir": str(model_dir), "models_used": use, "threshold": threshold,
        "duplicate_probes": args.duplicate_probes,
        "n_predicted": int(len(meta)), "n_unreadable": len(errors),
        "n_with_truth": int(len(ev)), "n_without_truth": len(no_truth), "truth_without_file": no_file,
        "n_out_of_scheme": int((~ev["in_scheme"]).sum()),
        "out_of_scheme_labels": sorted(ev.loc[~ev["in_scheme"], "true_label"].unique().tolist()),
        "all_with_truth": {
            "accuracy_out_of_scheme_counted_wrong": float(ev["correct"].mean()),
            "callable_share": float(ev["callable"].mean()),
            "accuracy_callable": float(ev.loc[ev["callable"], "correct"].mean()) if ev["callable"].any() else None,
        },
        "in_scheme": {
            "n": int(len(ins)),
            "accuracy": float(ins["correct"].mean()) if len(ins) else None,
            "balanced_accuracy": balanced_accuracy(y_true, y_pred) if len(ins) else None,
            "top2_accuracy": float(ins["top2_correct"].mean()) if len(ins) else None,
            "callable_share": float(ins["callable"].mean()) if len(ins) else None,
            "accuracy_callable": float(ins.loc[ins["callable"], "correct"].mean()) if ins["callable"].any() else None,
            "ece": expected_calibration_error(ins["confidence"].to_numpy(), ins["correct"].to_numpy()) if len(ins) else None,
        },
    }
    by_cov = ev.groupby("coverage_bin").agg(
        n=("sample", "size"), accuracy=("correct", "mean"), top2_accuracy=("top2_correct", "mean"),
        callable_share=("callable", "mean"))
    by_cov["accuracy_callable"] = ev[ev["callable"]].groupby("coverage_bin")["correct"].mean()
    by_cov.to_csv(out / "evaluation_by_coverage.csv", float_format="%.3f")
    by_class = ins.groupby("true_label").agg(n=("sample", "size"), recall=("correct", "mean"),
                                             callable_share=("callable", "mean"))
    by_class.to_csv(out / "evaluation_by_class.csv", float_format="%.3f")
    ev.loc[~ev["in_scheme"], ["sample", "true_label_raw", "true_label", "prediction", "confidence", "callable",
                               "coverage_pct"]].to_csv(out / "out_of_scheme_samples.csv", index=False)

    labels = sorted(set(ev["true_label"]) | set(ev["prediction"]))
    cm = confusion_frame(ev["true_label"], ev["prediction"], labels)
    cm.to_csv(out / "confusion_matrix.csv")
    try:  # the figure is optional; a plotting problem must never stop the evaluation
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        size = max(8, 0.45 * len(labels))
        fig, ax = plt.subplots(figsize=(size, size))
        ax.imshow(cm.to_numpy(), cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = cm.iat[i, j]
                if v:
                    ax.text(j, i, str(v), ha="center", va="center", fontsize=6)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True (after label map)")
        plt.tight_layout()
        plt.savefig(out / "confusion_matrix.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"(confusion_matrix.png not drawn: {e})")

    with open(out / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("\nBy coverage:\n" + by_cov.to_string(float_format=lambda v: f"{v:.3f}"))
    if no_truth:
        print(f"\n{len(no_truth)} predicted samples have no ground truth (listed in summary_metrics.json count only)")
    if summary["n_out_of_scheme"]:
        print(f"{summary['n_out_of_scheme']} samples have a true label outside the model's classes: "
              "see out_of_scheme_samples.csv (counted as errors in the overall accuracy)")


if __name__ == "__main__":
    main()
