"""
SPARSH-next: loading a finished run's networks and averaging their probabilities.

Used by scripts/predict.py and scripts/score_excluded.py, so both score with
exactly the networks and temperatures that training saved.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .sparse_nn import SparseNN, load_model, predict_logits, softmax_np


def load_run(model_dir: Path) -> Tuple[dict, List[str], List[str], Dict[str, str]]:
    """(config, classes in index order, CpG list, label map) of a finished run."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    idx_to_class = {int(k): v for k, v in json.loads((model_dir / "class_mapping.json").read_text()).items()}
    classes = [idx_to_class[i] for i in range(len(idx_to_class))]
    cpg_ids = json.loads((model_dir / "selected_cpgs.json").read_text())
    label_map = json.loads((model_dir / "label_map.json").read_text()).get("merge", {})
    return config, classes, cpg_ids, label_map


def load_models(model_dir: Path, config: dict, use: str, device: str) -> Tuple[str, List[Tuple[SparseNN, float]]]:
    """The fold ensemble or the final model, each with its temperature. use: auto, ensemble or final."""
    inf = config["inference"]
    folds, final = inf.get("fold_models"), inf.get("final_model")
    if use == "auto":
        use = "ensemble" if folds else "final"
    entries = folds if use == "ensemble" else ([final] if final else None)
    if not entries:
        sys.exit(f"No {use} weights recorded in {model_dir}/config.json")
    models = []
    for entry in entries:
        path = Path(model_dir) / entry["file"]
        if not path.exists():
            sys.exit(f"Missing weights file {path}")
        models.append((load_model(path, config["model"], device), float(entry.get("temperature", 1.0))))
    return use, models


def ensemble_probs(models: List[Tuple[SparseNN, float]], X: np.ndarray, device: str) -> np.ndarray:
    """Mean of the calibrated class probabilities of the networks."""
    return np.mean([softmax_np(predict_logits(m, X, device), t) for m, t in models], axis=0)
