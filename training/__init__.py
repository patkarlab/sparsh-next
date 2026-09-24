"""
SPARSH Training module with cross-validation, tuning, and reproducibility.
"""

from .trainer import (
    train_epoch,
    evaluate,
    cross_validate,
    train_final_model,
)
from .tuning import (
    tune_hyperparameters,
    get_tuned_config,
)
from .training_curves import TrainingCurveTracker
from .reproducibility import (
    set_deterministic_mode,
    generate_run_id,
    get_environment_metadata,
    save_config_snapshot,
    save_split_indices,
    save_fold_indices,
)

__all__ = [
    "train_epoch",
    "evaluate",
    "cross_validate",
    "train_final_model",
    "tune_hyperparameters",
    "get_tuned_config",
    "TrainingCurveTracker",
    "set_deterministic_mode",
    "generate_run_id",
    "get_environment_metadata",
    "save_config_snapshot",
    "save_split_indices",
    "save_fold_indices",
]
