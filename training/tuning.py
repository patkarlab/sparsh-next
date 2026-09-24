# # tuning.py file


"""
SPARSH Hyperparameter Tuning with Optuna (PHASED).
"""

import numpy as np
import torch
import optuna
from optuna.trial import Trial
from typing import Dict, Any, Optional, Tuple
import logging
import json
import traceback
from pathlib import Path

from training.trainer import cross_validate

logger = logging.getLogger(__name__)


def tune_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 40,
    n_folds: int = 3,
    epochs: int = 150,
    device: str = "cuda",
    seed: int = 42,
    output_dir: Optional[str] = None,
    samples_per_class_per_batch: int = 5,
    n_features: int = 0,
    cpg_ids: Optional[list] = None,
    early_stopping_patience: int = 20,
    lr_scheduler_patience: int = 10,
    lr_scheduler_factor: float = 0.5,
    phase: str = "phase1",                 # "phase1" or "phase2"
    fixed_params: Optional[Dict[str, Any]] = None,  # used in phase2
) -> Dict[str, Any]:

    assert phase in {"phase1", "phase2"}, "phase must be 'phase1' or 'phase2'"

    n_input_features = X.shape[1]
    n_classes = len(np.unique(y))
    trial_logs = []

    device = device if torch.cuda.is_available() else "cpu"

    def objective(trial: Trial) -> float:

        # ============================================================
        # ARCHITECTURE
        # ============================================================
        if phase == "phase1":
            # 🔒 FIXED architecture (strong baseline)
            hidden_dim_1 = 1024
            hidden_dim_2 = 512
            hidden_dim_3 = 256
        else:
            # 🔓 Phase-2: tune architecture ONLY
            hidden_dim_1 = trial.suggest_categorical(
                "hidden_dim_1", [512, 1024, 2048]
            )
            hidden_dim_2 = trial.suggest_categorical(
                "hidden_dim_2", [128, 256, 512]
            )
            hidden_dim_3 = trial.suggest_categorical(
                "hidden_dim_3", [64, 128, 256]
            )

        # ============================================================
        # TRAINING DYNAMICS
        # ============================================================
        if phase == "phase2":
            # 🔒 FIXED from Phase-1 best
            assert fixed_params is not None, "fixed_params required for phase2"

            learning_rate = fixed_params["learning_rate"]
            weight_decay = fixed_params["weight_decay"]
            dropout = fixed_params["dropout"]
            label_smoothing = fixed_params["label_smoothing"]
            mask_start = fixed_params["mask_start"]
            mask_end = fixed_params["mask_end"]
            focal_gamma = fixed_params["focal_gamma"]
            gaussian_noise_std = fixed_params["gaussian_noise_std"]

        else:
            # 🔓 Phase-1: tune optimisation behaviour
            # dropout = trial.suggest_float("dropout", 0.45, 0.65)
            dropout = trial.suggest_float("dropout", 0.40, 0.50)

            learning_rate = trial.suggest_float(
                # "learning_rate", 5e-5, 5e-4, log=True
                "learning_rate", 3e-5, 2e-4, log=True
            )

            weight_decay = trial.suggest_float(
                "weight_decay", 1e-4, 4e-3, log=True
            )

            mask_start = trial.suggest_float("mask_start", 0.96, 0.98)
            mask_end = trial.suggest_float("mask_end", 0.80, 0.85)

            label_smoothing = trial.suggest_float(
                "label_smoothing", 0.0, 0.05, step=0.05
            )

            focal_gamma = trial.suggest_categorical(
                "focal_gamma", [1.5, 2.0, 2.5]
            )

            gaussian_noise_std = trial.suggest_categorical(
                "gaussian_noise_std", [0.0, 0.01]
            )

        if mask_end > mask_start:
            mask_end = mask_start

        input_dim = n_features if n_features > 0 else n_input_features

        model_config = {
            "input_dim": input_dim,
            "hidden_dims": [hidden_dim_1, hidden_dim_2, hidden_dim_3],
            "n_classes": n_classes,
            "dropout": dropout,
            "activation": "gelu",
        }

        try:
            results = cross_validate(
                X=X,
                y=y,
                model_config=model_config,
                n_folds=n_folds,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                mask_start=mask_start,
                mask_end=mask_end,
                early_stopping_patience=early_stopping_patience,
                lr_scheduler_patience=lr_scheduler_patience,
                lr_scheduler_factor=lr_scheduler_factor,
                device=device,
                seed=seed,
                samples_per_class_per_batch=samples_per_class_per_batch,
                focal_gamma=focal_gamma,
                gaussian_noise_std=gaussian_noise_std,
                n_features=n_features,
                cpg_ids=cpg_ids,
                label_smoothing=label_smoothing,
            )
        except Exception as e:
            logger.error(f"Trial {trial.number} FAILED: {e}")
            logger.error(traceback.format_exc())
            raise optuna.exceptions.TrialPruned(str(e))

        score = results["mean_macro_f1"]
        acc = results["mean_accuracy"]

        logger.info(
            f"[{phase.upper()}] Trial {trial.number}: macro_f1={score:.4f}, acc={acc:.4f}"
        )

        return score

    # ============================================================
    # OPTUNA STUDY
    # ============================================================
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        #pruner=optuna.pruners.MedianPruner(),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20, interval_steps=10,),
    )

    logger.info(f"Starting Optuna {phase.upper()} with {n_trials} trials")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_value = study.best_value

    logger.info(f"[{phase.upper()}] Best macro-F1 = {best_value:.4f}")
    logger.info(f"[{phase.upper()}] Best params = {best_params}")

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"tuning_results_{phase}.json", "w") as f:
            json.dump(
                {
                    "phase": phase,
                    "best_value": best_value,
                    "best_params": best_params,
                    "n_trials": n_trials,
                },
                f,
                indent=2,
            )

    return best_params


# ============================================================
# BUILD FINAL CONFIG FROM BEST PARAMS
# ============================================================
def get_tuned_config(
    best_params: Dict[str, Any],
    n_features: int,
    n_classes: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    hidden_dims = [
        best_params.get("hidden_dim_1", 1024),
        best_params.get("hidden_dim_2", 512),
        best_params.get("hidden_dim_3", 256),
    ]

    model_config = {
        "input_dim": n_features,
        "hidden_dims": hidden_dims,
        "n_classes": n_classes,
        "dropout": best_params.get("dropout", 0.55),
        "activation": "gelu",
    }

    training_config = {
        "learning_rate": best_params.get("learning_rate", 1e-4),
        "weight_decay": best_params.get("weight_decay", 1e-4),
        "mask_start": best_params.get("mask_start", 0.97),
        "mask_end": best_params.get("mask_end", 0.80),
        "label_smoothing": best_params.get("label_smoothing", 0.05),
        "focal_gamma": best_params.get("focal_gamma", 2.0),
        "gaussian_noise_std": best_params.get("gaussian_noise_std", 0.0),
    }

    return model_config, training_config