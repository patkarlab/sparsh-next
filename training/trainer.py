# trainer.py file



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

from models.sparse_nn import SparseNN, apply_progressive_mask, get_mask_ratio
from data.dataset import (
    calculate_feature_importance,
    select_top_features,
)
from training.training_curves import TrainingCurveTracker

logger = logging.getLogger(__name__)


# =============================================================================
# Focal Loss
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Emphasises hard, misclassified examples by down-weighting easy ones.
    Per-class alpha provides implicit class balancing without synthetic data.

    Args:
        gamma: Focusing parameter. Higher values focus more on hard examples
            (default: 2.0).
        alpha: Per-class weight tensor of shape (n_classes,). Typically set
            to inverse class frequency (normalised). If None, no per-class
            weighting is applied.
        label_smoothing: Label smoothing factor applied to the CE component
            (default: 0.0). Toggleable via config.
        reduction: 'mean' or 'sum' (default: 'mean').
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        # Register alpha as a buffer so it moves with .to(device)
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Raw model output of shape (batch, n_classes).
            targets: Ground-truth class indices of shape (batch,).

        Returns:
            Scalar loss value.
        """
        # Standard cross-entropy (with optional label smoothing) per sample
        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        # p_t: probability assigned to the TRUE class (hard label, not smoothed)
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal weight: down-weights easy examples
        focal_weight = (1.0 - p_t) ** self.gamma

        loss = focal_weight * ce_loss

        # Per-class alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def _build_focal_loss(
    y: np.ndarray,
    n_classes: int,
    gamma: float,
    label_smoothing: float,
    device: str,
) -> FocalLoss:
    """
    Construct FocalLoss with inverse-frequency alpha from label array.

    Alpha is computed as inverse class frequency normalised so that the
    mean alpha equals 1.0, preserving the overall loss scale.

    Args:
        y: Integer label array (training split only).
        n_classes: Total number of classes.
        gamma: Focal loss gamma.
        label_smoothing: Label smoothing factor.
        device: Target device string.

    Returns:
        FocalLoss instance with alpha buffer on the correct device.
    """
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    # Avoid division by zero for classes absent in this fold
    counts = np.where(counts == 0, 1.0, counts)
    inv_freq = 1.0 / counts
    # Normalise: mean alpha = 1.0
    inv_freq = inv_freq / inv_freq.mean()
    alpha = torch.FloatTensor(inv_freq).to(device)
    return FocalLoss(
        gamma=gamma,
        alpha=alpha,
        label_smoothing=label_smoothing,
        reduction="mean",
    )


# =============================================================================
# Balanced Batch Sampler
# =============================================================================

class BalancedBatchSampler:
    """
    Balanced batch sampler:
    - Equal samples_per_class per class per batch
    - Majority / medium classes: sampled WITHOUT replacement per epoch
    - Minority classes: sampled WITH replacement
    """

    def __init__(
        self,
        labels: np.ndarray,
        samples_per_class: int = 3,
        seed: int = 42,
    ):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.classes = np.unique(self.labels)
        self.samples_per_class = samples_per_class
        self.seed = seed

        self.class_indices: Dict[int, np.ndarray] = {
            int(c): np.where(self.labels == c)[0] for c in self.classes
        }

        self.max_class_count = max(len(v) for v in self.class_indices.values())
        self.n_batches = int(
            np.ceil(self.max_class_count / self.samples_per_class)
        )

        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)

        # prepare pools for non-minority classes
        pools = {}
        ptrs = {}

        for cls, idx in self.class_indices.items():
            if len(idx) >= self.samples_per_class:
                pool = idx.copy()
                rng.shuffle(pool)
                pools[cls] = pool
                ptrs[cls] = 0

        for _ in range(self.n_batches):
            batch = []

            for cls, idx in self.class_indices.items():
                if len(idx) >= self.samples_per_class:
                    # consume without replacement
                    need = self.samples_per_class
                    while need > 0:
                        pool = pools[cls]
                        ptr = ptrs[cls]
                        available = len(pool) - ptr

                        if available == 0:
                            pool = idx.copy()
                            rng.shuffle(pool)
                            pools[cls] = pool
                            ptr = 0
                            available = len(pool)

                        take = min(need, available)
                        batch.extend(pool[ptr:ptr + take].tolist())
                        ptrs[cls] = ptr + take
                        need -= take
                else:
                    # minority class: oversample
                    sampled = rng.choice(
                        idx, size=self.samples_per_class, replace=True
                    )
                    batch.extend(sampled.tolist())

            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(
    model: SparseNN,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    mask_ratio: float = 0.0,
    gaussian_noise_std: float = 0.0,
    binary_features=False, 
) -> Tuple[float, float]:

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        if gaussian_noise_std > 0.0 and not binary_features:
            noise = torch.randn_like(X_batch) * gaussian_noise_std
            X_batch = (X_batch + noise).clamp(0.0, 1.0)

        # 2. Apply progressive masking (anti-curriculum)
        if mask_ratio > 0:
            fill = 0.0 if binary_features else 0.5
            X_batch = apply_progressive_mask(X_batch, mask_ratio, fill_value=fill)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


def evaluate(
    model: SparseNN,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    mask_ratio: float = 0.0,
    return_probs: bool = False,
) -> Union[Tuple[float, float, np.ndarray, np.ndarray], Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]]:

    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Apply masking if specified (use 0.0 for inference-consistent evaluation)
            if mask_ratio > 0:
                X_batch = apply_progressive_mask(X_batch, mask_ratio)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

            if return_probs:
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    accuracy = (all_preds == all_labels).mean()

    if return_probs:
        all_probs = np.concatenate(all_probs)
        return total_loss / len(all_labels), accuracy, all_preds, all_labels, all_probs

    return total_loss / len(all_labels), accuracy, all_preds, all_labels


# =============================================================================
# Cross-Validation (Leak-Free)
# =============================================================================

def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model_config: Dict[str, Any],
    n_folds: int = 5,
    epochs: int = 150,
    batch_size: int = 64,
    learning_rate: float = 0.0001,
    weight_decay: float = 1e-4,
    mask_start: float = 0.97,
    mask_end: float = 0.80,
    early_stopping_patience: int = 20,
    lr_scheduler_patience: int = 10,
    lr_scheduler_factor: float = 0.5,
    device: str = "cuda",
    seed: int = 42,
    samples_per_class_per_batch: int = 2,
    focal_gamma: float = 2.0,
    gaussian_noise_std: float = 0.0,
    n_features: int = 0,
    cpg_ids: Optional[List[str]] = None,
    label_smoothing: float = 0.0,
    curve_tracker: Optional[TrainingCurveTracker] = None,
    output_dir: Optional[str] = None,
    binary_features=False,
    upsample_minority: bool = False,
    upsample_mask_ratios: Optional[List[float]] = None,
    upsample_fill_value: float = 0.5,
) -> Dict[str, Any]:

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if upsample_mask_ratios is None:
        upsample_mask_ratios = [0.80, 0.85, 0.88, 0.90, 0.93, 0.97]

    n_classes = len(np.unique(y))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_results = []
    all_preds = np.zeros_like(y)
    all_probs = np.zeros((len(y), n_classes), dtype=np.float32)
    fold_ids = np.full(len(y), -1, dtype=np.int32)
    fold_indices_list = []
    per_fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold + 1}/{n_folds}")
        logger.info(f"{'='*60}")

        fold_indices_list.append((train_idx, val_idx))
        fold_ids[val_idx] = fold

        # === LEAK-FREE: All preprocessing on train split only ===
        X_train_fold = X[train_idx].copy()
        y_train_fold = y[train_idx].copy()
        X_val_fold = X[val_idx].copy()
        y_val_fold = y[val_idx].copy()

        # PATCH: apply masking augmentation to training fold only
        if upsample_minority:
            from data.dataset import upsample_with_masking_augmentation

            unique_before, counts_before = np.unique(y_train_fold, return_counts=True)
            logger.info(
                f"  [Fold {fold+1}] Augmenting training fold: "
                f"before min={counts_before.min()}, max={counts_before.max()}, "
                f"total={len(y_train_fold)}"
            )

            X_train_fold, y_train_fold = upsample_with_masking_augmentation(
                X=X_train_fold,
                y=y_train_fold,
                target_count=0,
                mask_ratios=upsample_mask_ratios,
                fill_value=upsample_fill_value,
                random_state=seed + fold,
            )

            unique_after, counts_after = np.unique(y_train_fold, return_counts=True)
            n_synthetic = len(y_train_fold) - len(y[train_idx])
            logger.info(
                f"  [Fold {fold+1}] After augmentation: "
                f"min={counts_after.min()}, max={counts_after.max()}, "
                f"total={len(y_train_fold)}, synthetic_added={n_synthetic}"
            )
            logger.info(
                f"  [Fold {fold+1}] Val fold: {len(y_val_fold)} REAL samples only"
            )

        # Log class distribution
        unique_train, counts_train = np.unique(y_train_fold, return_counts=True)
        logger.info(f"  Train split: {len(y_train_fold)} samples")
        for cls, cnt in zip(unique_train, counts_train):
            logger.info(f"    Class {cls}: {cnt}")

        unique_val, counts_val = np.unique(y_val_fold, return_counts=True)
        logger.info(f"  Val split: {len(val_idx)} samples (no masking at eval)")
        for cls, cnt in zip(unique_val, counts_val):
            logger.info(f"    Class {cls}: {cnt}")

        # --- Internal feature selection (train only) ---
        fold_model_config = model_config.copy()
        if n_features > 0 and cpg_ids is not None:
            logger.info(f"  Selecting top {n_features} features (train data only)...")
            importance = calculate_feature_importance(
                X_train_fold, y_train_fold, random_state=seed + fold
            )
            X_train_fold, _, selected_indices = select_top_features(
                X_train_fold, importance, cpg_ids, n_features=n_features
            )
            # Apply same selection to validation (no leakage: indices from train)
            X_val_fold = X_val_fold[:, selected_indices]
            fold_model_config["input_dim"] = X_train_fold.shape[1]
            logger.info(f"  Features selected: {X_train_fold.shape[1]}")

        # --- Build Focal Loss with inverse-frequency alpha (from train fold) ---
        criterion = _build_focal_loss(
            y_train_fold, n_classes, focal_gamma, label_smoothing, device
        )
        logger.info(
            f"  FocalLoss: gamma={focal_gamma}, label_smoothing={label_smoothing}"
        )

        # --- Create datasets ---
        X_train_t = torch.FloatTensor(X_train_fold)
        y_train_t = torch.LongTensor(y_train_fold)
        X_val_t = torch.FloatTensor(X_val_fold)
        y_val_t = torch.LongTensor(y_val_fold)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)

        # --- BalancedBatchSampler for training (seed varies per fold) ---
        balanced_sampler = BalancedBatchSampler(
            labels=y_train_fold,
            samples_per_class=samples_per_class_per_batch,
            seed=seed + fold,
        )
        effective_batch = n_classes * samples_per_class_per_batch
        logger.info(
            f"  BalancedBatchSampler: {samples_per_class_per_batch} samples/class, "
            f"effective batch ≈ {effective_batch}, batches/epoch = {len(balanced_sampler)}"
        )

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=balanced_sampler,
        )
        # Validation loader: standard, no balanced sampling, no masking
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        # --- Create model ---
        model = SparseNN(**fold_model_config).to(device)

        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
        )

        best_val_loss = float("inf")
        best_val_acc = 0.0
        patience_counter = 0
        best_state = None

        if curve_tracker is not None:
            curve_tracker.start_fold(fold)

        for epoch in range(epochs):
            balanced_sampler.set_epoch(epoch)
            # Calculate current mask ratio (anti-curriculum: high → lower)
            mask_ratio = get_mask_ratio(epoch, epochs, mask_start, mask_end)

            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device,
                mask_ratio, gaussian_noise_std,
                binary_features=binary_features,
            )

            # Validate WITHOUT masking (matches inference conditions)
            val_loss, val_acc, _, _ = evaluate(
                model, val_loader, criterion, device, mask_ratio=0.0
            )

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)

            # Record training curves
            if curve_tracker is not None:
                curve_tracker.record_epoch(
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_acc=train_acc,
                    val_acc=val_acc,
                    mask_ratio=mask_ratio,
                    learning_rate=current_lr,
                )

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                logger.info(f"  Early stopping at epoch {epoch + 1}")
                break

            # Periodic logging
            if (epoch + 1) % 25 == 0:
                logger.info(
                    f"  Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
                    f"mask={mask_ratio:.3f}, lr={current_lr:.6f}"
                )

        # Load best model and get final predictions WITHOUT masking
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        _, _, fold_preds, _, fold_probs = evaluate(
            model, val_loader, criterion, device, mask_ratio=0.0, return_probs=True
        )

        all_preds[val_idx] = fold_preds
        all_probs[val_idx] = fold_probs

        # Compute per-fold metrics
        fold_macro_f1 = f1_score(y_val_fold, fold_preds, average="macro", zero_division=0)
        fold_weighted_f1 = f1_score(y_val_fold, fold_preds, average="weighted", zero_division=0)

        fold_result = {
            "fold": fold + 1,
            "val_loss": best_val_loss,
            "val_acc": best_val_acc,
            "macro_f1": float(fold_macro_f1),
            "weighted_f1": float(fold_weighted_f1),
            "n_train": len(y_train_fold),
            "n_val": len(y_val_fold),
            "stopped_epoch": epoch + 1,
        }
        fold_results.append(fold_result)
        per_fold_metrics.append(fold_result)

        logger.info(
            f"  Fold {fold + 1} - val_acc: {best_val_acc:.4f}, "
            f"macro_f1: {fold_macro_f1:.4f}, weighted_f1: {fold_weighted_f1:.4f}"
        )

    # Compute overall metrics
    overall_acc = (all_preds == y).mean()
    mean_acc = np.mean([f["val_acc"] for f in fold_results])
    std_acc = np.std([f["val_acc"] for f in fold_results])
    mean_f1 = np.mean([f["macro_f1"] for f in fold_results])
    std_f1 = np.std([f["macro_f1"] for f in fold_results])

    results = {
        "fold_results": fold_results,
        "overall_accuracy": overall_acc,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_macro_f1": mean_f1,
        "std_macro_f1": std_f1,
        "all_predictions": all_preds.tolist(),
        "all_labels": y.tolist(),
        "all_probabilities": all_probs.tolist(),
        "fold_indices": fold_indices_list,
        "fold_ids": fold_ids.tolist(),
        "per_fold_metrics": per_fold_metrics,
    }

    logger.info(f"\nCross-validation complete:")
    logger.info(f"  Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
    logger.info(f"  Macro F1: {mean_f1:.4f} +/- {std_f1:.4f}")

    return results


# =============================================================================
# Final Model Training
# =============================================================================

def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    model_config: Dict[str, Any],
    epochs: int = 150,
    batch_size: int = 64,
    learning_rate: float = 0.0001,
    weight_decay: float = 1e-4,
    mask_start: float = 0.97,
    mask_end: float = 0.80,
    device: str = "cuda",
    seed: int = 42,
    samples_per_class_per_batch: int = 2,
    focal_gamma: float = 2.0,
    gaussian_noise_std: float = 0.0,
    label_smoothing: float = 0.0,
    n_features: int = 0,
    cpg_ids: Optional[List[str]] = None,
    binary_features: bool = False,
    upsample_minority: bool = False,
    upsample_mask_ratios: Optional[List[float]] = None,
    upsample_fill_value: float = 0.5,
) -> Tuple[SparseNN, Optional[List[str]], Optional[np.ndarray]]:

    if upsample_mask_ratios is None:
        upsample_mask_ratios = [0.80, 0.85, 0.88, 0.90, 0.93, 0.97]

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    X_train = X.copy()
    y_train = y.copy()
    selected_cpgs_out = None
    selected_indices_out = None

    # Apply feature selection
    final_model_config = model_config.copy()
    if n_features > 0 and cpg_ids is not None:
        logger.info(f"Final model: Selecting top {n_features} features...")
        importance = calculate_feature_importance(
            X_train, y_train, random_state=seed
        )
        X_train, selected_cpgs_out, selected_indices_out = select_top_features(
            X_train, importance, cpg_ids, n_features=n_features
        )
        final_model_config["input_dim"] = X_train.shape[1]

    # PATCH: apply masking augmentation to full training set
    if upsample_minority:
        from data.dataset import upsample_with_masking_augmentation

        unique_before, counts_before = np.unique(y_train, return_counts=True)
        logger.info(
            f"Final model: augmenting training data. "
            f"Before: min={counts_before.min()}, max={counts_before.max()}, total={len(y_train)}"
        )

        X_train, y_train = upsample_with_masking_augmentation(
            X=X_train,
            y=y_train,
            target_count=0,
            mask_ratios=upsample_mask_ratios,
            fill_value=upsample_fill_value,
            random_state=seed,
        )

        unique_after, counts_after = np.unique(y_train, return_counts=True)
        n_synthetic = len(y_train) - len(y)
        logger.info(
            f"Final model: after augmentation. "
            f"min={counts_after.min()}, max={counts_after.max()}, "
            f"total={len(y_train)}, synthetic_added={n_synthetic}"
        )

    n_classes = len(np.unique(y_train))

    # Build Focal Loss with inverse-frequency alpha from full training set
    criterion = _build_focal_loss(
        y_train, n_classes, focal_gamma, label_smoothing, device
    )

    # Create dataset
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y_train)
    dataset = TensorDataset(X_tensor, y_tensor)

    # BalancedBatchSampler for training
    balanced_sampler = BalancedBatchSampler(
        labels=y_train,
        samples_per_class=samples_per_class_per_batch,
        seed=seed,
    )
    train_loader = DataLoader(dataset, batch_sampler=balanced_sampler)

    # Create model
    model = SparseNN(**final_model_config).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    logger.info(f"Training final model for {epochs} epochs...")

    for epoch in range(epochs):
        balanced_sampler.set_epoch(epoch)
        mask_ratio = get_mask_ratio(epoch, epochs, mask_start, mask_end)
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            mask_ratio, gaussian_noise_std,
            binary_features=binary_features,
        )
        scheduler.step()

        if (epoch + 1) % 25 == 0:
            logger.info(f"  Epoch {epoch + 1}: loss={train_loss:.4f}, acc={train_acc:.4f}")

    logger.info("Final model training complete")

    return model, selected_cpgs_out, selected_indices_out