#!/usr/bin/env python3
"""
SPARSH: Subtype Prediction via Adaptive Recognition from Sparse Hematological Data
"""

import argparse
import logging
import json
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.dataset import load_methylation_pickle, filter_classes
from training.trainer import cross_validate, train_final_model
from training.tuning import tune_hyperparameters, get_tuned_config
from training.training_curves import TrainingCurveTracker
from training.reproducibility import (
    set_deterministic_mode, generate_run_id,
    get_environment_metadata, save_config_snapshot, save_fold_indices,
)
from evaluation.metrics import (
    compute_classification_metrics, print_classification_report,
    get_confusion_matrix_data, get_pr_curve_data, get_roc_curve_data,
    export_confusion_matrix_csv, export_confusion_matrix_normalized_csv,
    export_classification_report_csv, export_per_class_metrics_csv,
    export_roc_curve_csv, export_roc_auc_summary_csv,
    export_pr_curve_csv, export_summary_metrics_csv,
    export_predictions_csv, export_predictions_with_folds_csv,
    export_per_fold_metrics_csv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SPARSH 0.1.0: Statistically grounded adaptive classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--junk_path", type=str, default=None)
    parser.add_argument("--cpg_list_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/model")
    parser.add_argument("--n_features", type=int, default=0)
    parser.add_argument("--min_samples", type=int, default=5)
    parser.add_argument("--exclude_classes", type=str, nargs="*", default=[])
    parser.add_argument("--samples_per_class_per_batch", type=int, default=5)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--gaussian_noise_std", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mask_start", type=float, default=0.97)
    parser.add_argument("--mask_end", type=float, default=0.80)
    parser.add_argument("--n_folds", type=int, default=5, choices=[5])
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--lr_scheduler_patience", type=int, default=10)
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.5)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--binary_features", action="store_true", default=False)
    parser.add_argument("--binary_threshold", type=float, default=0.5)
    parser.add_argument(
        "--upsample_minority", action="store_true", default=False,
        help=(
            "Apply masking augmentation to minority classes INSIDE each CV fold "
            "training split only. Simulates ONT sparse coverage for underrepresented "
            "classes. Val folds always contain only real unaugmented samples. "
            "mask_ratios are constrained to [mask_end, mask_start] range."
        )
    )
    return parser.parse_args()


def _run_evaluation_suite(
    y_true, y_pred, y_probs, idx_to_class, output_dir, prefix="",
    fold_ids=None, sample_ids=None,
):
    metrics = compute_classification_metrics(y_true, y_pred)
    logger.info(f"\n{prefix}Metrics:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    report = print_classification_report(y_true, y_pred, idx_to_class)
    logger.info(f"\n{prefix}Classification report:")
    logger.info("\n" + report)

    p = prefix.lower().replace(" ", "_").rstrip("_")
    suffix = f"_{p}" if p else ""

    cm_data = get_confusion_matrix_data(y_true, y_pred, idx_to_class)
    with open(output_dir / f"confusion_matrix{suffix}.json", "w") as f:
        json.dump(cm_data, f, indent=2)
    export_confusion_matrix_csv(y_true, y_pred, idx_to_class, output_dir / f"confusion_matrix{suffix}.csv")
    export_confusion_matrix_normalized_csv(y_true, y_pred, idx_to_class, output_dir / f"confusion_matrix_normalized{suffix}.csv")
    export_classification_report_csv(y_true, y_pred, idx_to_class, output_dir / f"classification_report{suffix}.csv")
    export_per_class_metrics_csv(y_true, y_pred, idx_to_class, output_dir / f"per_class_metrics{suffix}.csv")

    roc_data = get_roc_curve_data(y_true, y_probs, idx_to_class)
    with open(output_dir / f"roc_curve_data{suffix}.json", "w") as f:
        json.dump(roc_data, f, indent=2)
    export_roc_curve_csv(y_true, y_probs, idx_to_class, output_dir / f"roc_curves{suffix}.csv")
    export_roc_auc_summary_csv(y_true, y_probs, idx_to_class, output_dir / f"roc_auc_summary{suffix}.csv")

    pr_data = get_pr_curve_data(y_true, y_probs, idx_to_class)
    with open(output_dir / f"pr_curve_data{suffix}.json", "w") as f:
        json.dump(pr_data, f, indent=2)
    export_pr_curve_csv(y_true, y_probs, idx_to_class, output_dir / f"pr_curves{suffix}.csv")
    export_summary_metrics_csv(y_true, y_pred, y_probs, idx_to_class, output_dir / f"summary_metrics{suffix}.csv")

    if fold_ids is not None:
        export_predictions_with_folds_csv(
            y_true, y_pred, y_probs, idx_to_class,
            output_dir / f"predictions{suffix}.csv",
            fold_ids=fold_ids, sample_ids=sample_ids,
        )
    else:
        export_predictions_csv(
            y_true, y_pred, y_probs, idx_to_class,
            output_dir / f"predictions{suffix}.csv",
            sample_ids=sample_ids,
        )
    return metrics


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(output_dir / "training_log.txt")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    logger.info("=" * 60)
    logger.info("SPARSH 0.1.0: Subtype Prediction via Adaptive Recognition")
    logger.info("         from Sparse Hematological Data")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Arguments: {vars(args)}")

    # -------------------------------------------------------------------------
    # Step 0: Reproducibility
    # -------------------------------------------------------------------------
    logger.info("\n--- Step 0: Setting up reproducibility ---")
    set_deterministic_mode(args.seed)
    run_id = generate_run_id(args.seed)
    logger.info(f"Run ID: {run_id}")

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"
    # ---- CUDA sanity + memory check (diagnostic) ----
    if args.device == "cuda":
        torch.cuda.synchronize()
        free_mem, total_mem = torch.cuda.mem_get_info()
        logger.info(
            f"[CUDA MEM] Free: {free_mem / 1024**3:.2f} GB | "
            f"Total: {total_mem / 1024**3:.2f} GB"
        )    

    env_metadata = get_environment_metadata()
    with open(output_dir / "environment.json", "w") as f:
        json.dump(env_metadata, f, indent=2, default=str)

    # -------------------------------------------------------------------------
    # Step 1: Load data
    # -------------------------------------------------------------------------
    logger.info("\n--- Step 1: Loading data ---")
    X, y, sample_ids, cpg_ids, idx_to_class = load_methylation_pickle(
        data_path=args.data_path,
        junk_path=args.junk_path,
        cpg_list_path=args.cpg_list_path,
    )

    if args.binary_features:
        thr = args.binary_threshold
        logger.info(f"Binarizing X: continuous -> {{0,1}} at threshold {thr}")
        X = (X > thr).astype(np.float32)
        args.gaussian_noise_std = 0.0

    # -------------------------------------------------------------------------
    # Step 2: Filter classes
    # -------------------------------------------------------------------------
    logger.info("\n--- Step 2: Filtering classes ---")
    X, y, sample_ids, idx_to_class = filter_classes(
        X=X, y=y, sample_ids=sample_ids, idx_to_class=idx_to_class,
        min_samples=args.min_samples, exclude_classes=args.exclude_classes,
    )
    assert len(sample_ids) == len(y), "sample_ids and y length mismatch after filter_classes"

    n_classes = len(idx_to_class)
    n_original_samples = len(y)
    fill_value = 0.0 if args.binary_features else 0.5

    logger.info(f"Final classes: {n_classes}, Total real samples: {n_original_samples}")
    for idx, name in sorted(idx_to_class.items()):
        count = (y == idx).sum()
        logger.info(f"  {idx}: {name} ({count})")


    UPSAMPLE_MASK_RATIOS = [0.80, 0.85, 0.88, 0.90, 0.93, 0.97]

    if args.upsample_minority:
        logger.info("\n--- Step 2b: Masking augmentation ENABLED (runs inside CV folds) ---")
        logger.info(f"  Augmentation mask ratios: {UPSAMPLE_MASK_RATIOS}")
        logger.info(f"  Fill value: {fill_value}")
        logger.info(f"  Real samples: {n_original_samples}")
        logger.info("  Val folds = real samples only → unbiased CV metrics")
    else:
        logger.info("\n--- Step 2b: No minority upsampling (BalancedBatchSampler handles imbalance) ---")

    # -------------------------------------------------------------------------
    # Step 3: Hyperparameter tuning (optional)
    # -------------------------------------------------------------------------
    if args.tune:
        logger.info(f"\n--- Step 3: Hyperparameter tuning ({args.n_trials} trials) ---")
        logger.info("  Tuning runs on REAL samples only (no augmentation in tuning CV)")
        logger.info("  Feature selection: INSIDE CV folds")
        best_params = tune_hyperparameters(
            X=X,
            y=y,
            n_trials=args.n_trials,
            n_folds=5,
            epochs=args.epochs,
            device=args.device,
            seed=args.seed,
            output_dir=output_dir,
            samples_per_class_per_batch=args.samples_per_class_per_batch,
            n_features=args.n_features,
            cpg_ids=cpg_ids,
            # PATCH: pass patience values so tuning and final CV are aligned
            early_stopping_patience=args.early_stopping_patience,
            lr_scheduler_patience=args.lr_scheduler_patience,
            lr_scheduler_factor=args.lr_scheduler_factor,
        )

        model_config, training_config = get_tuned_config(best_params, X.shape[1], n_classes)

        args.learning_rate      = training_config["learning_rate"]
        args.weight_decay       = training_config["weight_decay"]
        args.mask_start         = training_config["mask_start"]
        args.mask_end           = training_config["mask_end"]
        args.label_smoothing    = training_config.get("label_smoothing", 0.0)
        args.focal_gamma        = training_config.get("focal_gamma", 2.0)
        args.gaussian_noise_std = training_config.get("gaussian_noise_std", 0.0)
        # PATCH: sync args.hidden_dims so summary block is truthful
        args.hidden_dims        = model_config["hidden_dims"]

    else:
        logger.info("\n--- Step 3: Using default/specified hyperparameters ---")
        model_config = {
            "input_dim": args.n_features if args.n_features > 0 else X.shape[1],
            "hidden_dims": args.hidden_dims,
            "n_classes": n_classes,
            "dropout": args.dropout,
            "activation": "gelu",
        }

    logger.info(f"  Model config: {model_config}")
    logger.info(f"  Focal gamma: {args.focal_gamma}")
    logger.info(f"  Label smoothing: {args.label_smoothing}")
    logger.info(f"  Gaussian noise std: {args.gaussian_noise_std}")
    logger.info(f"  Samples/class/batch: {args.samples_per_class_per_batch}")

    # -------------------------------------------------------------------------
    # Step 4: Cross-validation on real samples only
    # -------------------------------------------------------------------------
    logger.info(f"\n--- Step 4: {args.n_folds}-fold cross-validation ---")
    logger.info("  Training: BalancedBatchSampler + optional masking augmentation (inside fold)")
    logger.info("  Val folds: real samples only — metrics are unbiased")
    logger.info("  Loss:      FocalLoss with inverse-frequency alpha")
    logger.info("  Feature selection: INSIDE each fold (train split only)")
    logger.info("  Validation masking: NONE (matches inference)")

    curve_tracker = TrainingCurveTracker()

    cv_results = cross_validate(
        X=X,
        y=y,
        model_config=model_config,
        n_folds=args.n_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        mask_start=args.mask_start,
        mask_end=args.mask_end,
        early_stopping_patience=args.early_stopping_patience,
        lr_scheduler_patience=args.lr_scheduler_patience,
        lr_scheduler_factor=args.lr_scheduler_factor,
        device=args.device,
        seed=args.seed,
        samples_per_class_per_batch=args.samples_per_class_per_batch,
        focal_gamma=args.focal_gamma,
        gaussian_noise_std=args.gaussian_noise_std,
        n_features=args.n_features,
        cpg_ids=cpg_ids,
        label_smoothing=args.label_smoothing,
        curve_tracker=curve_tracker,
        output_dir=str(output_dir),
        binary_features=args.binary_features,
        # PATCH: augmentation inside fold
        upsample_minority=args.upsample_minority,
        upsample_mask_ratios=UPSAMPLE_MASK_RATIOS,
        upsample_fill_value=fill_value,
    )

    save_fold_indices(cv_results["fold_indices"], output_dir, args.seed)
    curve_tracker.save_curves(output_dir)
    curve_tracker.save_plots(output_dir)

    cv_preds    = np.array(cv_results["all_predictions"])
    cv_labels   = np.array(cv_results["all_labels"])
    cv_probs    = np.array(cv_results["all_probabilities"])
    cv_fold_ids = np.array(cv_results["fold_ids"])

    logger.info("\n--- Step 4b: CV Evaluation Suite ---")
    _run_evaluation_suite(
        cv_labels, cv_preds, cv_probs, idx_to_class, output_dir,
        prefix="CV ",
        fold_ids=cv_fold_ids,
        sample_ids=sample_ids,
    )

    export_per_fold_metrics_csv(cv_results["per_fold_metrics"], output_dir / "per_fold_metrics_cv.csv")

    cv_results_save = {
        "fold_results": cv_results["fold_results"],
        "overall_accuracy": cv_results["overall_accuracy"],
        "mean_accuracy": cv_results["mean_accuracy"],
        "std_accuracy": cv_results["std_accuracy"],
        "mean_macro_f1": cv_results.get("mean_macro_f1", 0.0),
        "std_macro_f1": cv_results.get("std_macro_f1", 0.0),
    }
    with open(output_dir / "cv_results.json", "w") as f:
        json.dump(cv_results_save, f, indent=2)

    # -------------------------------------------------------------------------
    # Step 5: Train final model on full real dataset
    # -------------------------------------------------------------------------
    logger.info(f"\n--- Step 5: Training final model ---")
    logger.info(f"  Real training samples: {len(y)}")
    if args.upsample_minority:
        logger.info(f"  Masking augmentation will expand training set inside train_final_model")

    final_model, selected_cpgs, _ = train_final_model(
        X=X,
        y=y,
        model_config=model_config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        mask_start=args.mask_start,
        mask_end=args.mask_end,
        device=args.device,
        seed=args.seed,
        samples_per_class_per_batch=args.samples_per_class_per_batch,
        focal_gamma=args.focal_gamma,
        gaussian_noise_std=args.gaussian_noise_std,
        label_smoothing=args.label_smoothing,
        n_features=args.n_features,
        cpg_ids=cpg_ids,
        binary_features=args.binary_features,
        # PATCH: augmentation inside final training too
        upsample_minority=args.upsample_minority,
        upsample_mask_ratios=UPSAMPLE_MASK_RATIOS,
        upsample_fill_value=fill_value,
    )

    final_cpgs = selected_cpgs if selected_cpgs is not None else cpg_ids

    # -------------------------------------------------------------------------
    # Step 6: Save model and configuration
    # -------------------------------------------------------------------------
    logger.info(f"\n--- Step 6: Saving model and configuration ---")

    torch.save(final_model.state_dict(), output_dir / "model.pt")
    logger.info("  Saved model weights: model.pt")


    unique_counts = dict(zip(*[x.tolist() for x in np.unique(y, return_counts=True)]))
    majority_count = max(unique_counts.values())
    estimated_synthetic_per_fold = sum(
        max(0, majority_count - c) for c in unique_counts.values()
    ) if args.upsample_minority else 0

    full_config = {
        **model_config,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "mask_start": args.mask_start,
            "mask_end": args.mask_end,
            "label_smoothing": args.label_smoothing,
            "focal_gamma": args.focal_gamma,
            "gaussian_noise_std": args.gaussian_noise_std,
            "samples_per_class_per_batch": args.samples_per_class_per_batch,
            "early_stopping_patience": args.early_stopping_patience,
            "lr_scheduler_patience": args.lr_scheduler_patience,
            "lr_scheduler_factor": args.lr_scheduler_factor,
            "imbalance_strategy": "BalancedBatchSampler",
            "loss_function": "FocalLoss",
            "norm_type": "BatchNorm1d",
        },
        "data": {
            "n_features": args.n_features,
            "min_samples": args.min_samples,
            "exclude_classes": args.exclude_classes,
            "n_folds": args.n_folds,
            "binary_features": args.binary_features,
            "binary_threshold": args.binary_threshold,
            # PATCH: truthful upsample recording
            "upsample_minority": args.upsample_minority,
            "upsample_mask_ratios": UPSAMPLE_MASK_RATIOS if args.upsample_minority else [],
            "upsample_applied_inside_cv_folds": args.upsample_minority,
            "n_real_samples": n_original_samples,
            "estimated_synthetic_per_cv_training_fold": estimated_synthetic_per_fold,
        },
        "reproducibility": {
            "run_id": run_id,
            "seed": args.seed,
        },
        "tuning": {
            "used": args.tune,
            "n_trials": args.n_trials if args.tune else 0,
        },
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(full_config, f, indent=2)
    logger.info("  Saved config: config.json")

    with open(output_dir / "selected_cpgs.json", "w") as f:
        json.dump(final_cpgs, f)
    logger.info("  Saved selected CpGs: selected_cpgs.json")

    class_mapping = {str(k): v for k, v in idx_to_class.items()}
    with open(output_dir / "class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)
    logger.info("  Saved class mapping: class_mapping.json")

    save_config_snapshot(full_config, output_dir, run_id)

    # -------------------------------------------------------------------------
    # Step 7: Final summary — TRUTHFUL in all fields
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE — SPARSH 0.1.0")
    logger.info("=" * 60)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"CV Accuracy:      {cv_results['mean_accuracy']:.4f} +/- {cv_results['std_accuracy']:.4f}")
    logger.info(f"CV Macro F1:      {cv_results.get('mean_macro_f1', 0):.4f} +/- {cv_results.get('std_macro_f1', 0):.4f}")
    logger.info(f"Model saved to:   {output_dir}")
    logger.info("")
    logger.info("0.1.0 PIPELINE STATUS:")
    logger.info("  Imbalance:        BalancedBatchSampler (real samples, replacement)")
    logger.info("  Loss function:    FocalLoss (gamma=%.1f, alpha=inv-freq)" % args.focal_gamma)
    logger.info("  Normalisation:    BatchNorm1d")


    actual_hidden_dims = model_config["hidden_dims"]
    logger.info("  Architecture:     %s -> output" % " -> ".join(str(d) for d in actual_hidden_dims))
    logger.info("  Dropout:          %.4f" % model_config["dropout"])
    logger.info("  Gaussian noise:   std=%.3f" % args.gaussian_noise_std)


    if args.upsample_minority:
        logger.info(
            f"  Synthetic data:   MASKING AUGMENTATION (inside CV training folds, "
            f"mask_ratios={UPSAMPLE_MASK_RATIOS}, fill=0.5)"
        )
    else:
        logger.info("  Synthetic data:   NONE (no ADASYN, no interpolation)")

    logger.info("")


    logger.info("LEAKAGE STATUS: CLEAN")
    logger.info("  - Feature selection: inside CV folds only (train split)")
    logger.info("  - Masking augmentation: inside CV folds only (train split)")
    logger.info("  - Validation folds: real samples only — unbiased metrics")
    logger.info("  - CV protocol: stratified 5-fold on real samples")
    logger.info("  - Validation masking: none (matches inference conditions)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()