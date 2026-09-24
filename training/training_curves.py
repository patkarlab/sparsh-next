"""
SPARSH-next: training history export (CSV and one figure).

Each row of the history is one epoch of one model (fold1 ... foldK, final).
The inner validation loss is the model-selection signal; the outer folds are
never evaluated during training.
"""

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def save_history(history: List[Dict], output_dir: Path) -> None:
    if not history:
        return
    output_dir = Path(output_dir)
    df = pd.DataFrame(history)
    df.to_csv(output_dir / "training_history.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipped training_curves.png")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for tag, part in df.groupby("model"):
        axes[0].plot(part["epoch"], part["train_loss"], label=tag)
        axes[1].plot(part["epoch"], part["inner_val_nll"], label=tag)
        axes[2].plot(part["epoch"], part["inner_val_acc"], label=tag)
    axes[0].set_title("Training loss (focal)")
    axes[1].set_title("Inner validation NLL (model selection)")
    axes[2].set_title("Inner validation accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)
