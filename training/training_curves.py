

"""
SPARSH Training Curve Tracking and Visualization.

Tracks per-epoch metrics per fold and provides:
- Aggregated mean/std curves
- JSON + CSV export
- Automatic plot generation
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TrainingCurveTracker:
    def __init__(self):
        self.fold_curves: Dict[int, Dict] = {}
        self.current_fold: Optional[int] = None

    # --------------------------------------------------
    # Fold Management
    # --------------------------------------------------

    def start_fold(self, fold: int) -> None:
        self.current_fold = fold
        self.fold_curves[fold] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "mask_ratio": [],
            "learning_rate": [],
            "per_class_f1": [],
        }

    # --------------------------------------------------
    # Record Epoch
    # --------------------------------------------------

    def record_epoch(
        self,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        mask_ratio: float,
        learning_rate: float = 0.0,
        per_class_f1: Optional[Dict[str, float]] = None,
    ) -> None:

        if self.current_fold is None:
            return

        curves = self.fold_curves[self.current_fold]

        curves["train_loss"].append(train_loss)
        curves["val_loss"].append(val_loss)
        curves["train_acc"].append(train_acc)
        curves["val_acc"].append(val_acc)
        curves["mask_ratio"].append(mask_ratio)
        curves["learning_rate"].append(learning_rate)

        if per_class_f1 is not None:
            curves["per_class_f1"].append(per_class_f1)
        else:
            curves["per_class_f1"].append({})

    # --------------------------------------------------
    # Aggregation
    # --------------------------------------------------

    def _min_epochs(self) -> int:
        return min(len(c["train_loss"]) for c in self.fold_curves.values())

    def get_aggregated_curves(self) -> Dict[str, Dict[str, list]]:
        if not self.fold_curves:
            return {}

        min_epochs = self._min_epochs()
        aggregated = {}

        metrics = ["train_loss", "val_loss", "train_acc", "val_acc"]

        for metric in metrics:
            values = np.array([
                curves[metric][:min_epochs]
                for curves in self.fold_curves.values()
            ])
            aggregated[metric] = {
                "mean": values.mean(axis=0).tolist(),
                "std": values.std(axis=0).tolist(),
            }

        return aggregated

    def get_aggregated_per_class_f1(self) -> Dict[str, Dict[str, list]]:
        if not self.fold_curves:
            return {}

        min_epochs = self._min_epochs()

        # Collect class names
        class_names = set()
        for curves in self.fold_curves.values():
            for epoch_dict in curves["per_class_f1"][:min_epochs]:
                class_names.update(epoch_dict.keys())

        aggregated = {}

        for class_name in class_names:
            values = []
            for curves in self.fold_curves.values():
                class_values = [
                    epoch_dict.get(class_name, 0.0)
                    for epoch_dict in curves["per_class_f1"][:min_epochs]
                ]
                values.append(class_values)

            arr = np.array(values)
            aggregated[class_name] = {
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0).tolist(),
            }

        return aggregated

    # --------------------------------------------------
    # Saving
    # --------------------------------------------------

    def save_curves(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON
        with open(output_dir / "training_curves.json", "w") as f:
            json.dump(self.fold_curves, f, indent=2)

        # Save CSVs
        self._save_per_fold_csv(output_dir)
        self._save_aggregated_csv(output_dir)

    def _save_per_fold_csv(self, output_dir: Path) -> None:
        try:
            import pandas as pd
        except ImportError:
            return

        rows = []

        for fold, curves in self.fold_curves.items():
            n_epochs = len(curves["train_loss"])

            for epoch in range(n_epochs):
                row = {
                    "fold": fold + 1,
                    "epoch": epoch + 1,
                    "train_loss": curves["train_loss"][epoch],
                    "val_loss": curves["val_loss"][epoch],
                    "train_acc": curves["train_acc"][epoch],
                    "val_acc": curves["val_acc"][epoch],
                    "mask_ratio": curves["mask_ratio"][epoch],
                    "learning_rate": curves["learning_rate"][epoch],
                }
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_dir / "training_curves_per_fold.csv", index=False)

    def _save_aggregated_csv(self, output_dir: Path) -> None:
        try:
            import pandas as pd
        except ImportError:
            return

        aggregated = self.get_aggregated_curves()
        if not aggregated:
            return

        n_epochs = len(aggregated["train_loss"]["mean"])
        data = {"epoch": list(range(1, n_epochs + 1))}

        for metric in aggregated:
            data[f"{metric}_mean"] = aggregated[metric]["mean"]
            data[f"{metric}_std"] = aggregated[metric]["std"]

        df = pd.DataFrame(data)
        df.to_csv(output_dir / "training_curves_summary.csv", index=False)

    # --------------------------------------------------
    # Plotting
    # --------------------------------------------------

    def save_plots(self, output_dir: Path) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        aggregated = self.get_aggregated_curves()

        # -------- Loss Plot --------
        fig, ax = plt.subplots(figsize=(8, 5))

        for fold, curves in self.fold_curves.items():
            epochs = range(1, len(curves["train_loss"]) + 1)
            ax.plot(epochs, curves["train_loss"], alpha=0.2)
            ax.plot(epochs, curves["val_loss"], alpha=0.2)

        if aggregated:
            epochs = range(1, len(aggregated["train_loss"]["mean"]) + 1)
            ax.plot(epochs, aggregated["train_loss"]["mean"], linewidth=2, label="Train Mean")
            ax.plot(epochs, aggregated["val_loss"]["mean"], linewidth=2, label="Val Mean")

        ax.set_title("Loss Curves")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "training_loss_curves.png", dpi=150)
        plt.close()

        # -------- Accuracy Plot --------
        fig, ax = plt.subplots(figsize=(8, 5))

        for fold, curves in self.fold_curves.items():
            epochs = range(1, len(curves["train_acc"]) + 1)
            ax.plot(epochs, curves["train_acc"], alpha=0.2)
            ax.plot(epochs, curves["val_acc"], alpha=0.2)

        if aggregated:
            epochs = range(1, len(aggregated["train_acc"]["mean"]) + 1)
            ax.plot(epochs, aggregated["train_acc"]["mean"], linewidth=2, label="Train Mean")
            ax.plot(epochs, aggregated["val_acc"]["mean"], linewidth=2, label="Val Mean")

        ax.set_title("Accuracy Curves")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "training_accuracy_curves.png", dpi=150)
        plt.close()

        # -------- Mask Ratio --------
        fig, ax = plt.subplots(figsize=(8, 4))
        for fold, curves in self.fold_curves.items():
            epochs = range(1, len(curves["mask_ratio"]) + 1)
            ax.plot(epochs, curves["mask_ratio"], alpha=0.4)

        ax.set_title("Mask Ratio Schedule")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mask Ratio")
        plt.tight_layout()
        plt.savefig(output_dir / "mask_ratio_schedule.png", dpi=150)
        plt.close()

        # -------- Per-Class F1 --------
        aggregated_f1 = self.get_aggregated_per_class_f1()
        if aggregated_f1:
            fig, ax = plt.subplots(figsize=(8, 5))

            for class_name, stats in aggregated_f1.items():
                mean = np.array(stats["mean"])
                epochs = range(1, len(mean) + 1)
                ax.plot(epochs, mean, label=class_name)

            ax.set_title("Per-Class F1 (Mean)")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("F1 Score")
            ax.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "per_class_f1_curves.png", dpi=150)
            plt.close()
