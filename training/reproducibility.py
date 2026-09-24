"""
SPARSH Reproducibility and Environment Utilities.

This module provides:
1. Full deterministic mode (seeds, CUDA, cuDNN)
2. Environment metadata capture
3. Config snapshot persistence
4. Run ID generation and tracking
"""

import os
import sys
import json
import hashlib
import platform
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_deterministic_mode(seed: int = 42) -> None:
    """
    Set full deterministic mode for reproducible training.

    Configures all random number generators and backend flags
    to ensure bitwise reproducibility across runs.

    Args:
        seed: Random seed for all generators.
    """
    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cuDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Enable deterministic algorithms globally
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Older PyTorch versions
        torch.use_deterministic_algorithms(True)

    logger.info(f"Deterministic mode set with seed={seed}")


def generate_run_id(seed: int = 42, timestamp: Optional[str] = None) -> str:
    """
    Generate a unique run ID from seed and timestamp.

    Args:
        seed: Random seed used for this run.
        timestamp: Optional timestamp string. Generated if not provided.

    Returns:
        Short unique run ID string.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw = f"{seed}_{timestamp}"
    short_hash = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"run_{timestamp}_{short_hash}"


def get_environment_metadata() -> Dict[str, Any]:
    """
    Capture full environment metadata for reproducibility.

    Returns:
        Dictionary with platform, Python, PyTorch, CUDA, and package info.
    """
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cudnn_version": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A",
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
        "numpy_version": np.__version__,
    }

    if torch.cuda.is_available():
        metadata["pytorch"]["cuda_version"] = torch.version.cuda
        metadata["pytorch"]["gpu_name"] = torch.cuda.get_device_name(0)
        metadata["pytorch"]["gpu_count"] = torch.cuda.device_count()

    # Capture key package versions
    try:
        import sklearn
        metadata["sklearn_version"] = sklearn.__version__
    except ImportError:
        pass

    try:
        import optuna
        metadata["optuna_version"] = optuna.__version__
    except ImportError:
        pass

    try:
        import pandas
        metadata["pandas_version"] = pandas.__version__
    except ImportError:
        pass

    return metadata


def save_config_snapshot(
    config: Dict[str, Any],
    output_dir: Path,
    run_id: str,
) -> None:
    """
    Save a complete config snapshot including environment metadata.

    Args:
        config: Training configuration dictionary.
        output_dir: Directory to save the snapshot.
        run_id: Unique run identifier.
    """
    snapshot = {
        "run_id": run_id,
        "config": config,
        "environment": get_environment_metadata(),
    }

    output_path = output_dir / "config_snapshot.json"
    with open(output_path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    logger.info(f"Config snapshot saved: {output_path}")


def save_split_indices(
    train_indices: np.ndarray,
    holdout_indices: np.ndarray,
    output_dir: Path,
    seed: int,
) -> None:
    """
    Save train/holdout split indices for exact reproduction.

    Args:
        train_indices: Indices of training samples.
        holdout_indices: Indices of holdout test samples.
        output_dir: Directory to save indices.
        seed: Seed used for the split.
    """
    split_data = {
        "seed": seed,
        "train_indices": train_indices.tolist(),
        "holdout_indices": holdout_indices.tolist(),
        "n_train": len(train_indices),
        "n_holdout": len(holdout_indices),
    }

    output_path = output_dir / "split_indices.json"
    with open(output_path, "w") as f:
        json.dump(split_data, f)

    logger.info(f"Split indices saved: {output_path} (train={len(train_indices)}, holdout={len(holdout_indices)})")


def save_fold_indices(
    fold_splits: list,
    output_dir: Path,
    seed: int,
) -> None:
    """
    Save CV fold indices for exact reproduction.

    Args:
        fold_splits: List of (train_idx, val_idx) tuples per fold.
        output_dir: Directory to save indices.
        seed: Seed used for the splits.
    """
    fold_data = {
        "seed": seed,
        "n_folds": len(fold_splits),
        "folds": [
            {
                "fold": i + 1,
                "train_indices": train_idx.tolist(),
                "val_indices": val_idx.tolist(),
                "n_train": len(train_idx),
                "n_val": len(val_idx),
            }
            for i, (train_idx, val_idx) in enumerate(fold_splits)
        ],
    }

    output_path = output_dir / "fold_indices.json"
    with open(output_path, "w") as f:
        json.dump(fold_data, f)

    logger.info(f"Fold indices saved: {output_path}")
