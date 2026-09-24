"""SPARSH-next training module: nested cross-validation, final model, reproducibility."""

from .reproducibility import generate_run_id, get_environment_metadata, set_deterministic_mode
from .trainer import (
    IMBALANCE_MODES,
    BalancedBatchSampler,
    FocalLoss,
    TrainConfig,
    cross_validate,
    evaluation_conditions,
    fit_temperature,
    inner_split,
    split_indices,
    train_final_model,
    train_one_model,
)
from .training_curves import save_history

__all__ = [
    "IMBALANCE_MODES",
    "BalancedBatchSampler",
    "FocalLoss",
    "TrainConfig",
    "cross_validate",
    "evaluation_conditions",
    "fit_temperature",
    "generate_run_id",
    "get_environment_metadata",
    "inner_split",
    "save_history",
    "set_deterministic_mode",
    "split_indices",
    "train_final_model",
    "train_one_model",
]
