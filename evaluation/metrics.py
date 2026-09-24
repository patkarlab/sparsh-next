"""
SPARSH-next metrics.

All summaries are computed from class probabilities. Class-level averages use
only the classes present in the truth set (a class that is merely predicted
cannot have a recall), and every confusion matrix is built with an explicit
label list, so rows and columns always match their names.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Weighted mean |accuracy - confidence| over equal-width confidence bins."""
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(confidence) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean recall over the classes present in y_true."""
    classes = np.unique(y_true)
    if len(classes) == 0:
        return float("nan")
    return float(np.mean([np.mean(y_pred[y_true == c] == c) for c in classes]))


def summarize_probs(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.90) -> Dict[str, float]:
    """Accuracy, balanced accuracy, macro F1, top-2, confidence-threshold and calibration metrics."""
    y_true = np.asarray(y_true)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = pred == y_true
    order = np.argsort(-probs, axis=1)
    top2 = (order[:, :2] == y_true[:, None]).any(axis=1)
    p_true = probs[np.arange(len(y_true)), y_true]
    callable_ = conf >= threshold
    classes = np.unique(y_true)
    return {
        "n": int(len(y_true)),
        "accuracy": float(correct.mean()),
        "balanced_accuracy": balanced_accuracy(y_true, pred),
        "macro_f1": float(f1_score(y_true, pred, labels=classes, average="macro", zero_division=0)),
        "top2_accuracy": float(top2.mean()),
        "mean_confidence": float(conf.mean()),
        f"callable_share_{threshold:.2f}": float(callable_.mean()),
        f"accuracy_callable_{threshold:.2f}": float(correct[callable_].mean()) if callable_.any() else float("nan"),
        "ece": expected_calibration_error(conf, correct),
        "nll": float(-np.mean(np.log(np.clip(p_true, 1e-12, 1.0)))),
    }


def recall_by_class(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> np.ndarray:
    """Recall per class index (NaN for classes absent from y_true)."""
    pred = probs.argmax(axis=1)
    out = np.full(n_classes, np.nan)
    for c in range(n_classes):
        m = y_true == c
        if m.any():
            out[c] = float(np.mean(pred[m] == c))
    return out


def predictions_frame(
    sample_ids: Sequence[str],
    probs: np.ndarray,
    idx_to_class: Dict[int, str],
    true_labels: Optional[Sequence[str]] = None,
    extra: Optional[Dict[str, Sequence]] = None,
    include_probabilities: bool = True,
) -> pd.DataFrame:
    """One row per sample: prediction, confidence, top-3, optional truth and per-class probabilities."""
    names = [idx_to_class[i] for i in range(len(idx_to_class))]
    order = np.argsort(-probs, axis=1)
    data: Dict[str, Sequence] = {"sample_id": list(sample_ids)}
    for key, values in (extra or {}).items():
        data[key] = list(values)
    if true_labels is not None:
        data["true_label"] = list(true_labels)
    data["prediction"] = [names[i] for i in order[:, 0]]
    data["confidence"] = probs[np.arange(len(probs)), order[:, 0]]
    if true_labels is not None:
        data["correct"] = [int(t == p) for t, p in zip(true_labels, data["prediction"])]
    for rank in range(1, min(3, probs.shape[1])):
        data[f"top{rank + 1}_class"] = [names[i] for i in order[:, rank]]
        data[f"top{rank + 1}_prob"] = probs[np.arange(len(probs)), order[:, rank]]
    df = pd.DataFrame(data)
    if include_probabilities:
        df = pd.concat([df, pd.DataFrame(probs, columns=[f"prob_{n}" for n in names])], axis=1)
    return df


def confusion_frame(true_names: Sequence[str], pred_names: Sequence[str], labels: List[str]) -> pd.DataFrame:
    """Counts with rows = truth, columns = prediction, in the given label order."""
    index = {name: i for i, name in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(true_names, pred_names):
        m[index[t], index[p]] += 1
    df = pd.DataFrame(m, index=labels, columns=labels)
    df.index.name = "true_label"
    return df
