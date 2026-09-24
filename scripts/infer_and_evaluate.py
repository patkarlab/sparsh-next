#!/usr/bin/env python3
"""
SPARSH Inference & Evaluation Pipeline
"""

import argparse
import sys
from pathlib import Path
import json

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, f1_score, precision_score, recall_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.sparse_nn import SparseNN


def load_predictor(model_dir, device):
    with open(Path(model_dir) / "config.json") as f:
        config = json.load(f)

    with open(Path(model_dir) / "class_mapping.json") as f:
        idx_to_class = {int(k): v for k, v in json.load(f).items()}

    with open(Path(model_dir) / "selected_cpgs.json") as f:
        selected_cpgs = json.load(f)

    # Normalize CpG names at load time
    selected_cpgs_norm = [str(c).strip() for c in selected_cpgs]
    cpg_to_idx = {cpg: i for i, cpg in enumerate(selected_cpgs_norm)}

    model = SparseNN(
        input_dim=config["input_dim"],
        hidden_dims=config["hidden_dims"],
        n_classes=config["n_classes"],
        dropout=0.0,
        activation=config.get("activation", "gelu"),
    )
    model.load_state_dict(
        torch.load(Path(model_dir) / "model.pt", map_location=device)
    )
    model.to(device)
    model.eval()

    return model, idx_to_class, cpg_to_idx, len(selected_cpgs_norm), config


def load_sample(csv_path, cpg_to_idx, n_features, binary_features, fill_value):

    df = pd.read_csv(csv_path, index_col=0)
    row = df.iloc[0]

    # Initialize with the mode-appropriate fill value
    x = np.full(n_features, fill_value, dtype=np.float32)

    n_matched = 0
    for probe, val in row.items():
        probe_norm = str(probe).strip()
        if probe_norm not in cpg_to_idx or pd.isna(val):
            continue

        idx = cpg_to_idx[probe_norm]
        fval = float(val)

        if binary_features:

            x[idx] = 0.0 if fval < 0.5 else 1.0
        else:

            if fval <= 0.0:
                x[idx] = 0.05
            elif fval >= 1.0:
                x[idx] = 0.95
            else:
                x[idx] = fval

        n_matched += 1

    return x, n_matched


def main():
    parser = argparse.ArgumentParser(description="SPARSH Inference & Evaluation")
    parser.add_argument("--model_dir",   type=str, required=True)
    parser.add_argument("--ont_dir",     type=str, required=True)
    parser.add_argument("--output_dir",  type=str, required=True)
    parser.add_argument(
        "--ground_truth", type=str, default=None,
        help="CSV with columns: sample,true_label",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--binary_features", action="store_true", default=False,
        help=(
            "ONT samples provide binary {0,1} methylation calls. "
            "Missing CpGs are filled with 0.0 (not 0.5). "
            "Exact 0/1 values are preserved without clipping. "
            "Must match the --binary_features flag used during training."
        )
    )
    parser.add_argument(
        "--binary_threshold", type=float, default=0.5,
        help="Threshold for binarizing ONT values (default 0.5). "
             "Values < threshold -> 0.0, >= threshold -> 1.0."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load model and config
    model, idx_to_class, cpg_to_idx, n_features, saved_config = load_predictor(
        args.model_dir, device
    )
    class_to_idx = {v: k for k, v in idx_to_class.items()}


    trained_binary = saved_config.get("data", {}).get("binary_features", False)
    if trained_binary != args.binary_features:
        print(
            f"\nWARNING: Model was trained with binary_features={trained_binary} "
            f"but inference is running with binary_features={args.binary_features}. "
            f"Input distributions will mismatch. Results may be unreliable.\n"
        )


    fill_value = 0.0 if args.binary_features else 0.5
    print(f"Inference mode: {'BINARY' if args.binary_features else 'CONTINUOUS'}")
    print(f"Missing CpG fill value: {fill_value}")

    # Load ONT samples
    ont_dir   = Path(args.ont_dir)
    csv_files = sorted(ont_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No ONT CSV files found in {ont_dir}")

    sample_names = [f.stem.strip().upper() for f in csv_files]

    X = np.full((len(csv_files), n_features), fill_value, dtype=np.float32)
    coverages = []

    for i, f in enumerate(csv_files):
        X[i], n_matched = load_sample(
            f, cpg_to_idx, n_features,
            binary_features=args.binary_features,
            fill_value=fill_value,
        )
        coverage_pct = n_matched / n_features * 100
        coverages.append(coverage_pct)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Loaded {i+1}/{len(csv_files)}: {f.stem} "
                  f"coverage={coverage_pct:.1f}% ({n_matched}/{n_features} CpGs)")

    print(f"\nCoverage summary: mean={np.mean(coverages):.1f}%, "
          f"min={np.min(coverages):.1f}%, max={np.max(coverages):.1f}%")

    coverage_df = pd.DataFrame({
        "sample":       sample_names,
        "cpgs_matched": [int(c * n_features / 100) for c in coverages],
        "coverage_pct": coverages,
    })
    coverage_df.to_csv(output_dir / "coverage_report.csv", index=False)

    # Predict
    with torch.no_grad():
        logits = model(torch.tensor(X).to(device))
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        preds  = np.argmax(probs, axis=1)

    pred_classes = [idx_to_class[i] for i in preds]

    # Save predictions with top-3
    n_classes    = len(idx_to_class)
    top3_indices = np.argsort(probs, axis=1)[:, ::-1][:, :3]

    pred_rows = []
    for i in range(len(csv_files)):
        row = {
            "sample":       sample_names[i],
            "prediction":   pred_classes[i],
            "confidence":   probs[i, preds[i]],
            "coverage_pct": coverages[i],
        }
        for rank in range(3):
            cls_idx = top3_indices[i, rank]
            row[f"top{rank+1}_class"] = idx_to_class[cls_idx]
            row[f"top{rank+1}_prob"]  = probs[i, cls_idx]
        pred_rows.append(row)

    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(output_dir / "predictions.csv", index=False)
    print(f"\nSaved predictions: {output_dir / 'predictions.csv'}")

    # Evaluation (only if ground truth provided)
    if args.ground_truth:
        gt_df = pd.read_csv(args.ground_truth, dtype=str)
        gt_df["sample"]     = gt_df["sample"].str.strip().str.upper()
        gt_df["true_label"] = gt_df["true_label"].str.strip()
        gt_map = dict(zip(gt_df["sample"], gt_df["true_label"]))

        y_true, y_pred, y_probs_list = [], [], []
        for i, s in enumerate(sample_names):
            if s in gt_map and gt_map[s] in class_to_idx:
                y_true.append(gt_map[s])
                y_pred.append(pred_classes[i])
                y_probs_list.append(probs[i])

        if not y_true:
            print("No matching ground truth labels found — skipping evaluation")
            return

        y_probs_eval = np.array(y_probs_list)
        y_true_idx   = [class_to_idx[y] for y in y_true]

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        pd.DataFrame(report).T.to_csv(output_dir / "classification_report.csv")

        labels = sorted(set(y_true) | set(y_pred))
        cm     = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df  = pd.DataFrame(cm, index=labels, columns=labels)
        cm_df.to_csv(output_dir / "confusion_matrix.csv")

        fig_size = max(10, len(labels))
        plt.figure(figsize=(fig_size, fig_size - 2))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
        plt.close()

        macro_f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)
        macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        macro_recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)

        summary = {
            "n_samples":        len(y_true),
            "inference_mode":   "binary" if args.binary_features else "continuous",
            "fill_value_used":  fill_value,
            "macro_f1":         macro_f1,
            "macro_precision":  macro_precision,
            "macro_recall":     macro_recall,
        }

        y_true_bin = np.zeros_like(y_probs_eval)
        for i, idx in enumerate(y_true_idx):
            y_true_bin[i, idx] = 1

        try:
            roc_auc = roc_auc_score(
                y_true_bin, y_probs_eval, average="macro", multi_class="ovr"
            )
            summary["macro_roc_auc"] = roc_auc
        except Exception as e:
            print(f"  ROC AUC failed: {e}")

        with open(output_dir / "summary_metrics.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n  Macro F1:        {macro_f1:.4f}")
        print(f"  Macro Precision: {macro_precision:.4f}")
        print(f"  Macro Recall:    {macro_recall:.4f}")
        if "macro_roc_auc" in summary:
            print(f"  Macro ROC AUC:   {summary['macro_roc_auc']:.4f}")

        plt.figure(figsize=(12, 8))
        for i, cls in idx_to_class.items():
            if i < y_true_bin.shape[1]:
                precision, recall, _ = precision_recall_curve(
                    y_true_bin[:, i], y_probs_eval[:, i]
                )
                plt.plot(recall, precision, label=cls, alpha=0.7)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves (per class)")
        plt.legend(fontsize="x-small", ncol=2)
        plt.tight_layout()
        plt.savefig(output_dir / "pr_curves.png", dpi=150)
        plt.close()

    print(f"\nInference & evaluation complete -> {output_dir}")


if __name__ == "__main__":
    main()