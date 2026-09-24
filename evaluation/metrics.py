"""
Classification Metrics for Model Evaluation.

Comprehensive evaluation utilities including:
- Accuracy, F1, Precision, Recall
- Balanced accuracy
- Per-class sensitivity & specificity
- ROC & PR curves with AUC
- Full CSV export utilities
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
)


# ==========================================================
# Internal Utilities
# ==========================================================

def _ensure_output_path(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _validate_probs(y_probs: np.ndarray):
    if y_probs.ndim != 2:
        raise ValueError("y_probs must be 2D (n_samples, n_classes)")
    if np.any(np.isnan(y_probs)):
        raise ValueError("y_probs contains NaN values")


def _compute_uncertainty_metrics(y_probs: np.ndarray) -> Dict[str, np.ndarray]:
    eps = 1e-12
    n_samples, n_classes = y_probs.shape

    max_probability = np.max(y_probs, axis=1)
    entropy = -np.sum(y_probs * np.log(y_probs + eps), axis=1)

    if n_classes > 1:
        normalized_entropy = entropy / np.log(n_classes)
        sorted_probs = np.sort(y_probs, axis=1)
        top2_margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    else:
        normalized_entropy = np.zeros(n_samples)
        top2_margin = np.zeros(n_samples)

    return {
        "max_probability": max_probability,
        "predictive_entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "top2_margin": top2_margin,
    }


# ==========================================================
# Core Metrics
# ==========================================================

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:

    from sklearn.metrics import balanced_accuracy_score

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict[str, Dict[str, float]]:

    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]

    if len(class_names) != n_classes:
        raise ValueError("class_names length must match number of classes")

    per_class = {}

    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        per_class[cls] = {
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
            "support": int(tp + fn),
        }

    return per_class


# ==========================================================
# Confusion Matrix
# ==========================================================

def get_confusion_matrix_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx_to_class: Dict[int, str],
) -> Dict[str, Any]:

    cm = confusion_matrix(y_true, y_pred)
    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    return {
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "normalized_matrix": cm_normalized.tolist(),
    }


# ==========================================================
# ROC & PR
# ==========================================================

def get_roc_curve_data(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    idx_to_class: Dict[int, str],
) -> Dict[str, Any]:

    from sklearn.preprocessing import label_binarize

    _validate_probs(y_probs)

    n_classes = len(idx_to_class)
    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]

    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    roc_data = {}

    for i, class_name in enumerate(class_names):

        try:
            fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], y_probs[:, i])
            auc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
        except ValueError:
            fpr, tpr, thresholds = [0.0], [0.0], [0.0]
            auc = 0.0

        roc_data[class_name] = {
            "fpr": list(fpr),
            "tpr": list(tpr),
            "thresholds": list(thresholds),
            "auc": float(auc),
        }

    try:
        macro_auc = roc_auc_score(
            y_true_bin, y_probs, average="macro", multi_class="ovr"
        )
    except ValueError:
        macro_auc = 0.0

    roc_data["macro_average"] = {"auc": float(macro_auc)}

    return roc_data


# ==========================================================
# CSV EXPORTS
# ==========================================================

def export_confusion_matrix_csv(
    y_true, y_pred, idx_to_class, output_path
):
    output_path = _ensure_output_path(output_path)

    cm = confusion_matrix(y_true, y_pred)
    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]

    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    df.index.name = "True_Label"
    df.to_csv(output_path)


def export_summary_metrics_csv(
    y_true,
    y_pred,
    y_probs,
    idx_to_class,
    output_path,
):
    from sklearn.preprocessing import label_binarize

    output_path = _ensure_output_path(output_path)
    _validate_probs(y_probs)

    metrics = compute_classification_metrics(y_true, y_pred)

    n_classes = len(idx_to_class)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    try:
        macro_auc = roc_auc_score(
            y_true_bin, y_probs, average="macro", multi_class="ovr"
        )
        weighted_auc = roc_auc_score(
            y_true_bin, y_probs, average="weighted", multi_class="ovr"
        )
        micro_auc = roc_auc_score(y_true_bin, y_probs, average="micro")
    except ValueError:
        macro_auc = weighted_auc = micro_auc = 0.0

    metrics["macro_auc"] = macro_auc
    metrics["weighted_auc"] = weighted_auc
    metrics["micro_auc"] = micro_auc

    pd.DataFrame([metrics]).to_csv(
        output_path, index=False, float_format="%.4f"
    )


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx_to_class: Dict[int, str],
) -> str:
    """
    Generate formatted sklearn classification report.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        idx_to_class: Mapping from integer index to class name.
        
    Returns:
        Formatted string report.
    """
    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    return classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )


def get_pr_curve_data(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    idx_to_class: Dict[int, str],
) -> Dict[str, Any]:
    """
    Generate precision-recall curve data for each class.
    
    Args:
        y_true: True labels (n_samples,).
        y_probs: Predicted probabilities (n_samples, n_classes).
        idx_to_class: Mapping from integer index to class name.
        
    Returns:
        Dictionary mapping class names to PR curve data.
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize
    
    _validate_probs(y_probs)
    
    n_classes = len(idx_to_class)
    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    
    pr_data = {}
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(
            y_true_bin[:, i], y_probs[:, i]
        )
        ap = average_precision_score(y_true_bin[:, i], y_probs[:, i])
        
        pr_data[class_name] = {
            "precision": list(precision),
            "recall": list(recall),
            "ap": float(ap),
        }
    
    return pr_data


def export_confusion_matrix_normalized_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx_to_class: Dict[int, str],
    output_path,
) -> None:
    """
    Export row-normalized confusion matrix to CSV.
    """
    output_path = _ensure_output_path(output_path)
    
    cm = confusion_matrix(y_true, y_pred)
    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )
    
    df = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)
    df.index.name = "True_Label"
    df.to_csv(output_path, float_format="%.4f")


def export_classification_report_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx_to_class: Dict[int, str],
    output_path,
) -> None:
    """
    Export sklearn classification report to CSV.
    """
    output_path = _ensure_output_path(output_path)
    
    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    df = pd.DataFrame(report_dict).T
    df.index.name = "class"
    df.to_csv(output_path, float_format="%.4f")


def export_per_class_metrics_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx_to_class: Dict[int, str],
    output_path,
) -> None:
    """
    Export per-class sensitivity, specificity, precision, F1 to CSV.
    """
    output_path = _ensure_output_path(output_path)
    
    per_class = compute_per_class_metrics(y_true, y_pred, 
                                          [idx_to_class[i] for i in sorted(idx_to_class.keys())])
    
    rows = []
    for class_name, metrics in per_class.items():
        rows.append({
            "class": class_name,
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "precision": metrics["precision"],
            "f1": metrics["f1"],
            "support": metrics["support"],
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, float_format="%.4f")


def export_roc_curve_csv(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    idx_to_class: Dict[int, str],
    output_path,
) -> None:
    """
    Export ROC curve data to CSV.
    """
    output_path = _ensure_output_path(output_path)
    
    roc_data = get_roc_curve_data(y_true, y_probs, idx_to_class)
    
    rows = []
    for class_name, data in roc_data.items():
        if class_name == "macro_average":
            continue
        
        auc = data["auc"]
        for fpr, tpr, thresh in zip(data["fpr"], data["tpr"], data["thresholds"]):
            rows.append({
                "class": class_name,
                "fpr": fpr,
                "tpr": tpr,
                "threshold": thresh,
                "auc": auc,
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, float_format="%.6f")


def export_roc_auc_summary_csv(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    idx_to_class: Dict[int, str],
    output_path,
) -> None:
    """
    Export per-class AUC scores to CSV.
    """
    output_path = _ensure_output_path(output_path)
    
    roc_data = get_roc_curve_data(y_true, y_probs, idx_to_class)
    
    rows = []
    for class_name, data in roc_data.items():
        if "auc" in data:
            rows.append({
                "class": class_name,
                "auc": data["auc"],
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, float_format="%.4f")


def export_pr_curve_csv(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    idx_to_class: Dict[int, str],
    output_path,
) -> None:
    """
    Export precision-recall curve data to CSV.
    """
    output_path = _ensure_output_path(output_path)
    
    pr_data = get_pr_curve_data(y_true, y_probs, idx_to_class)
    
    rows = []
    for class_name, data in pr_data.items():
        ap = data["ap"]
        for prec, rec in zip(data["precision"], data["recall"]):
            rows.append({
                "class": class_name,
                "precision": prec,
                "recall": rec,
                "average_precision": ap,
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, float_format="%.6f")


def export_predictions_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    idx_to_class: Dict[int, str],
    output_path,
    sample_ids: Optional[List[str]] = None,
) -> None:
    """
    Export raw predictions with probabilities to CSV.
    """
    output_path = _ensure_output_path(output_path)
    
    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    
    uncertainty = _compute_uncertainty_metrics(y_probs)
    predicted_probability = y_probs[np.arange(len(y_pred)), y_pred]

    data = {
        "true_label_idx": y_true,
        "pred_label_idx": y_pred,
        "true_label": [idx_to_class[i] for i in y_true],
        "pred_label": [idx_to_class[i] for i in y_pred],
        "correct": (y_true == y_pred).astype(int),
        "predicted_probability": predicted_probability,
        "max_probability": uncertainty["max_probability"],
        "predictive_entropy": uncertainty["predictive_entropy"],
        "normalized_entropy": uncertainty["normalized_entropy"],
        "top2_margin": uncertainty["top2_margin"],
    }
    
    if sample_ids is not None:
        data["sample_id"] = sample_ids
    
    for i, class_name in enumerate(class_names):
        data[f"prob_{class_name}"] = y_probs[:, i]
    
    df = pd.DataFrame(data)
    
    cols = list(df.columns)
    if "sample_id" in cols:
        cols.remove("sample_id")
        cols = ["sample_id"] + cols
    df = df[cols]

    df.to_csv(output_path, index=False, float_format="%.6f")


def export_predictions_with_folds_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    idx_to_class: Dict[int, str],
    output_path,
    fold_ids: Optional[np.ndarray] = None,
    sample_ids: Optional[List[str]] = None,
) -> None:
    """
    Export predictions CSV with fold IDs.
    """
    output_path = _ensure_output_path(output_path)
    
    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]

    uncertainty = _compute_uncertainty_metrics(y_probs)
    predicted_probability = y_probs[np.arange(len(y_pred)), y_pred]

    data = {
        "true_label_idx": y_true,
        "pred_label_idx": y_pred,
        "true_label": [idx_to_class[i] for i in y_true],
        "pred_label": [idx_to_class[i] for i in y_pred],
        "correct": (y_true == y_pred).astype(int),
        "predicted_probability": predicted_probability,
        "max_probability": uncertainty["max_probability"],
        "predictive_entropy": uncertainty["predictive_entropy"],
        "normalized_entropy": uncertainty["normalized_entropy"],
        "top2_margin": uncertainty["top2_margin"],
    }

    if sample_ids is not None:
        data["sample_id"] = sample_ids

    if fold_ids is not None:
        data["fold_id"] = fold_ids

    for i, class_name in enumerate(class_names):
        data[f"prob_{class_name}"] = y_probs[:, i]

    df = pd.DataFrame(data)

    cols = list(df.columns)
    priority = []
    for col in ["sample_id", "fold_id"]:
        if col in cols:
            cols.remove(col)
            priority.append(col)
    df = df[priority + cols]

    df.to_csv(output_path, index=False, float_format="%.6f")


def export_per_fold_metrics_csv(
    fold_metrics: List[Dict[str, Any]],
    output_path,
) -> None:
    """
    Export per-fold metrics to CSV.
    """
    output_path = _ensure_output_path(output_path)
    
    df = pd.DataFrame(fold_metrics)
    df.to_csv(output_path, index=False, float_format="%.4f")


