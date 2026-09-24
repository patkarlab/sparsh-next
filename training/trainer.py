"""
SPARSH-next training and nested cross-validation.

Protocol for each outer fold
----------------------------
1. Outer split: StratifiedKFold, or StratifiedGroupKFold when groups are given
   (for example duplicate clusters or patients), so related samples never sit
   on both sides of a fold.
2. Inner split: a stratified, group-aware fraction of the outer TRAINING part
   (inner_val_frac, default 0.15) is held out. It alone drives the learning-rate
   schedule, early stopping, checkpoint choice and temperature calibration.
3. The outer fold is never used for any decision. It is scored once, under
   fixed conditions: dense arrays and simulated ONT sparsity (mask and reads)
   at several coverages, with corruption seeded per sample so every run is
   judged on identical inputs.

Class imbalance (cfg.imbalance)
-------------------------------
sampler         : BalancedBatchSampler, no class weights in the loss (default).
sampler_alpha   : sampler + inverse-frequency focal-loss weights
                  (v0.1.0 without --upsample_minority; corrects imbalance twice).
legacy_upsample : pre-masked copies up to the largest class, weights computed
                  after upsampling, sampler (v0.1.0 with --upsample_minority).
"""

import logging
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from data.dataset import legacy_masked_upsample
from models.corruption import READ_SIMS, corrupt_rows, corrupt_torch, sample_observed_fraction
from models.sparse_nn import SparseNN, predict_logits

logger = logging.getLogger(__name__)

IMBALANCE_MODES = ("sampler", "sampler_alpha", "legacy_upsample")


@dataclass
class TrainConfig:
    # optimisation
    epochs: int = 300
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 30
    lr_scheduler_patience: int = 10
    lr_scheduler_factor: float = 0.5
    samples_per_class_per_batch: int = 5
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05
    # class imbalance
    imbalance: str = "sampler"
    legacy_upsample_ratios: Tuple[float, ...] = (0.80, 0.85, 0.88, 0.90, 0.93, 0.97)
    # sparsity simulated during training
    train_sim: str = "reads"
    coverage_mode: str = "random"
    coverage_dist: str = "loguniform"
    cov_min: float = 0.02
    cov_max: float = 0.5
    mask_start: float = 0.97
    mask_end: float = 0.80
    # validation and evaluation
    n_folds: int = 5
    inner_val_frac: float = 0.15
    val_sim: str = "reads"
    val_coverages: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.3)
    eval_coverages: Tuple[float, ...] = (0.03, 0.05, 0.1, 0.2, 0.3)
    eval_sims: Tuple[str, ...] = ("reads", "mask")
    eval_seed: int = 12345
    calibrate: bool = True
    # clip applied to read-level (ONT-like: READ_SIMS) inputs in CV exactly as predict.py applies it to ONT files;
    # set to (0.05, 0.95) for mask-trained recipes, None otherwise (scripts/train.py decides)
    clip_observed: Optional[Tuple[float, float]] = None
    # hardware and reproducibility
    seed: int = 42
    device: str = "cuda"
    data_on_gpu: str = "auto"
    eval_batch_size: int = 256

    def to_dict(self) -> Dict:
        d = asdict(self)
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


# =============================================================================
# Loss and sampler
# =============================================================================

class FocalLoss(nn.Module):
    """FL = alpha_t * (1 - p_t)^gamma * CE, with optional label smoothing in the CE term."""

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none", label_smoothing=self.label_smoothing)
        with torch.no_grad():
            p_t = F.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = (1.0 - p_t) ** self.gamma * ce
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean()


def inverse_frequency_alpha(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    return torch.tensor(inv / inv.mean(), dtype=torch.float32)


class BalancedBatchSampler:
    """
    Every batch holds samples_per_class draws from every class.

    Classes with at least samples_per_class members are drawn without
    replacement from a shuffled pool that is refilled when exhausted; smaller
    classes are drawn with replacement. An epoch has
    ceil(largest class / samples_per_class) batches, so every class is drawn
    about as often as the largest class has members.
    """

    def __init__(self, labels: np.ndarray, samples_per_class: int = 5, seed: int = 42):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.samples_per_class = samples_per_class
        self.seed = seed
        self.class_indices = {int(c): np.where(self.labels == c)[0] for c in np.unique(self.labels)}
        self.n_batches = int(np.ceil(max(len(v) for v in self.class_indices.values()) / samples_per_class))
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        pools, ptrs = {}, {}
        for cls, idx in self.class_indices.items():
            if len(idx) >= self.samples_per_class:
                pools[cls] = rng.permutation(idx)
                ptrs[cls] = 0
        for _ in range(self.n_batches):
            batch = []
            for cls, idx in self.class_indices.items():
                if cls in pools:
                    need = self.samples_per_class
                    while need > 0:
                        if ptrs[cls] >= len(pools[cls]):
                            pools[cls] = rng.permutation(idx)
                            ptrs[cls] = 0
                        take = min(need, len(pools[cls]) - ptrs[cls])
                        batch.extend(pools[cls][ptrs[cls]:ptrs[cls] + take].tolist())
                        ptrs[cls] += take
                        need -= take
                else:
                    batch.extend(rng.choice(idx, size=self.samples_per_class, replace=True).tolist())
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.n_batches


# =============================================================================
# Splits
# =============================================================================

def split_indices(y: np.ndarray, groups: Optional[np.ndarray], n_splits: int, seed: int):
    """Stratified (and group-aware when groups is given) K-fold index pairs."""
    dummy = np.zeros(len(y))
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(dummy, y))
    # StratifiedGroupKFold(shuffle=True) did not shuffle correctly before scikit-learn 1.8 and lost
    # stratification. Randomise the group order ourselves (seeded) and use the deterministic algorithm,
    # which gives stratified, reproducible folds on every scikit-learn version.
    _, inverse = np.unique(np.asarray(groups, dtype=str), return_inverse=True)
    relabelled = np.random.RandomState(seed).permutation(inverse.max() + 1)[inverse]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)
    return list(splitter.split(dummy, y, relabelled))


def inner_split(y: np.ndarray, groups: Optional[np.ndarray], frac: float, seed: int):
    """Hold out about frac of the rows (stratified, group-aware); returns (fit, inner_val) positions."""
    n_splits = max(2, int(round(1.0 / frac)))
    with warnings.catch_warnings():
        # classes smaller than n_splits are expected here; they simply land in the fit part more often
        warnings.filterwarnings("ignore", message="The least populated class")
        fit, ival = split_indices(y, groups, n_splits, seed)[0]
    return fit, ival


# =============================================================================
# Helpers
# =============================================================================

def resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        return "cpu"
    return device


def _storage_device(n_bytes: int, device: str, mode: str) -> str:
    """Keep the training matrix on the GPU when it fits comfortably (saves a copy per batch)."""
    if not device.startswith("cuda") or mode == "never":
        return "cpu"
    if mode == "always":
        return device
    free, _ = torch.cuda.mem_get_info()
    return device if n_bytes < 0.5 * free else "cpu"


T_MIN, T_MAX = 0.25, 4.0


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """
    Single temperature T minimising the NLL of softmax(logits / T), kept within [0.25, 4].
    When every inner-validation sample is already correct the NLL keeps falling as T -> 0,
    so no temperature is fitted (T = 1).
    """
    if np.all(logits.argmax(axis=1) == y):
        logger.warning("  inner validation is perfectly classified; temperature left at 1.0")
        return 1.0
    lt = torch.tensor(logits, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lt / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.detach().exp().item())
    if not np.isfinite(t) or t <= T_MIN or t >= T_MAX:
        logger.warning(f"  fitted temperature {t:.3f} is outside [{T_MIN}, {T_MAX}]; clipped")
    return float(np.clip(t if np.isfinite(t) else 1.0, T_MIN, T_MAX))


def clip_observed(X: np.ndarray, clip: Optional[Sequence[float]]) -> np.ndarray:
    """Clip observed values (NaN stays NaN), as predict.py does for mask-trained models."""
    if clip is None:
        return X
    return np.where(np.isnan(X), X, np.clip(X, clip[0], clip[1])).astype(np.float32)


def make_inner_sets(X: np.ndarray, sample_ids: np.ndarray, cfg: "TrainConfig") -> List[np.ndarray]:
    """Fixed corrupted copies of the inner validation samples, one per validation coverage."""
    sets = []
    for c in cfg.val_coverages:
        Xc = corrupt_rows(X, sample_ids, c, cfg.val_sim, cfg.seed, salt=1)
        sets.append(clip_observed(Xc, cfg.clip_observed) if cfg.val_sim in READ_SIMS else Xc)
    return sets


def _nll_acc(logits: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    lt = torch.from_numpy(logits)
    yt = torch.from_numpy(y).long()
    return float(F.cross_entropy(lt, yt).item()), float((logits.argmax(1) == y).mean())


# =============================================================================
# Training one model
# =============================================================================

def train_one_model(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    inner_sets: Sequence[np.ndarray],
    y_inner: np.ndarray,
    n_classes: int,
    model_config: Dict,
    cfg: TrainConfig,
    seed: int,
    tag: str,
    history: Optional[List[Dict]] = None,
) -> Dict:
    """
    Train one network on X_fit. Model selection uses only the inner validation
    sets (pre-corrupted copies of the inner split). Returns the model at its
    best inner-validation epoch and a fitted temperature.
    """
    device = resolve_device(cfg.device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if cfg.imbalance == "legacy_upsample":
        X_fit, y_fit = legacy_masked_upsample(X_fit, y_fit, cfg.legacy_upsample_ratios, seed)
    alpha = inverse_frequency_alpha(y_fit, n_classes) if cfg.imbalance in ("sampler_alpha", "legacy_upsample") else None
    criterion = FocalLoss(cfg.focal_gamma, alpha, cfg.label_smoothing).to(device)
    sampler = BalancedBatchSampler(y_fit, cfg.samples_per_class_per_batch, seed)

    store = _storage_device(X_fit.nbytes, device, cfg.data_on_gpu)
    X_store = torch.from_numpy(np.ascontiguousarray(X_fit, dtype=np.float32)).to(store)
    y_store = torch.from_numpy(y_fit.astype(np.int64)).to(store)

    model = SparseNN(**model_config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_scheduler_factor, patience=cfg.lr_scheduler_patience
    )

    best_nll, best_epoch, best_state, wait = float("inf"), -1, None, 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        sampler.set_epoch(epoch)
        loss_sum, correct, seen, cov_sum = 0.0, 0, 0, 0.0
        for batch in sampler:
            idx = torch.as_tensor(batch, device=store)
            xb = X_store.index_select(0, idx).to(device, non_blocking=True)
            yb = y_store.index_select(0, idx).to(device, non_blocking=True)
            frac = sample_observed_fraction(
                len(batch), cfg.coverage_mode, epoch, cfg.epochs,
                cfg.cov_min, cfg.cov_max, cfg.mask_start, cfg.mask_end, device, cfg.coverage_dist,
            )
            xb = corrupt_torch(xb, frac, cfg.train_sim)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(batch)
            correct += int((logits.argmax(1) == yb).sum().item())
            seen += len(batch)
            cov_sum += float(frac.sum().item())

        per_set = [_nll_acc(predict_logits(model, Xs, device, cfg.eval_batch_size), y_inner) for Xs in inner_sets]
        val_nll = float(np.mean([p[0] for p in per_set]))
        val_acc = float(np.mean([p[1] for p in per_set]))
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step(val_nll)

        if history is not None:
            history.append({
                "model": tag, "epoch": epoch + 1, "train_loss": loss_sum / seen, "train_acc": correct / seen,
                "inner_val_nll": val_nll, "inner_val_acc": val_acc, "mean_train_coverage": cov_sum / seen,
                "learning_rate": lr_now,
            })

        if val_nll < best_nll - 1e-6:
            best_nll, best_epoch, wait = val_nll, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if (epoch + 1) % 25 == 0 or epoch == 0:
            logger.info(f"  [{tag}] epoch {epoch + 1}: train_loss={loss_sum / seen:.4f} "
                        f"inner_val_nll={val_nll:.4f} inner_val_acc={val_acc:.3f} lr={lr_now:.2e} "
                        f"({time.time() - t0:.0f}s)")
        if wait >= cfg.early_stopping_patience:
            logger.info(f"  [{tag}] early stop at epoch {epoch + 1}; best epoch {best_epoch}")
            break
        if not np.isfinite(val_nll):
            raise RuntimeError(f"[{tag}] inner validation loss is not finite at epoch {epoch + 1}")

    if best_state is None:
        raise RuntimeError(f"[{tag}] no finite inner-validation loss was ever recorded")
    model.load_state_dict(best_state)
    model.eval()

    temperature = 1.0
    if cfg.calibrate:
        pooled = np.concatenate([predict_logits(model, Xs, device, cfg.eval_batch_size) for Xs in inner_sets])
        temperature = fit_temperature(pooled, np.tile(y_inner, len(inner_sets)))
    logger.info(f"  [{tag}] best epoch {best_epoch} (inner val NLL {best_nll:.4f}); temperature {temperature:.3f}")

    del X_store, y_store
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {"model": model, "best_epoch": best_epoch, "epochs_run": epoch + 1,
            "best_inner_val_nll": best_nll, "temperature": temperature}


# =============================================================================
# Evaluation conditions and nested cross-validation
# =============================================================================

def evaluation_conditions(eval_coverages: Sequence[float], eval_sims: Sequence[str] = ("reads", "mask")
                          ) -> List[Tuple[str, Optional[str], Optional[float]]]:
    """(name, simulation, observed fraction); 'dense' means the array profile as measured."""
    conds = [("dense", None, None)]
    for sim in eval_sims:
        for c in eval_coverages:
            conds.append((f"{sim}_{c:.2f}", sim, float(c)))
    names = [c[0] for c in conds]
    if len(set(names)) != len(names):
        raise ValueError(f"--eval_coverages give duplicate condition names at 2 decimals: {names}")
    return conds


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: np.ndarray,
    groups: Optional[np.ndarray],
    n_classes: int,
    model_config: Dict,
    cfg: TrainConfig,
    fold_model_dir: Optional[Path] = None,
    history: Optional[List[Dict]] = None,
) -> Dict:
    """Nested CV. Returns out-of-fold logits per evaluation condition and per-fold records."""
    device = resolve_device(cfg.device)
    conditions = evaluation_conditions(cfg.eval_coverages, cfg.eval_sims)
    folds = split_indices(y, groups, cfg.n_folds, cfg.seed)
    n = len(y)
    logits = {name: np.full((n, n_classes), np.nan, dtype=np.float32) for name, _, _ in conditions}
    fold_of = np.full(n, -1, dtype=np.int64)
    temperature_of = np.full(n, np.nan, dtype=np.float64)
    records = []

    for k, (tr, te) in enumerate(folds):
        t0 = time.time()
        g_tr = groups[tr] if groups is not None else None
        fit_rel, ival_rel = inner_split(y[tr], g_tr, cfg.inner_val_frac, cfg.seed + 1000 + k)
        fit_idx, ival_idx = tr[fit_rel], tr[ival_rel]
        logger.info(f"Fold {k + 1}/{len(folds)}: fit {len(fit_idx)}, inner val {len(ival_idx)}, outer test {len(te)}")
        missing = sorted(set(np.unique(y[te])) - set(np.unique(y[fit_idx])))
        if missing:
            logger.warning(f"  classes in the outer fold but absent from training: {missing}")

        inner_sets = make_inner_sets(X[ival_idx], sample_ids[ival_idx], cfg)
        result = train_one_model(X[fit_idx], y[fit_idx], inner_sets, y[ival_idx], n_classes,
                                 model_config, cfg, seed=cfg.seed + k, tag=f"fold{k + 1}", history=history)
        del inner_sets
        model = result["model"]

        for name, sim, frac in conditions:
            Xc = corrupt_rows(X[te], sample_ids[te], frac, sim, cfg.eval_seed, salt=2)
            if sim in READ_SIMS:
                Xc = clip_observed(Xc, cfg.clip_observed)
            logits[name][te] = predict_logits(model, Xc, device, cfg.eval_batch_size)
            del Xc
        fold_of[te] = k
        temperature_of[te] = result["temperature"]

        if fold_model_dir is not None:
            fold_model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), fold_model_dir / f"fold{k + 1}.pt")
        records.append({
            "fold": k + 1, "n_fit": int(len(fit_idx)), "n_inner_val": int(len(ival_idx)), "n_test": int(len(te)),
            "best_epoch": result["best_epoch"], "epochs_run": result["epochs_run"],
            "best_inner_val_nll": result["best_inner_val_nll"], "temperature": result["temperature"],
            "minutes": round((time.time() - t0) / 60.0, 1),
        })
        del model, result
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return {"conditions": conditions, "logits": logits, "fold_of": fold_of,
            "temperature_of": temperature_of, "records": records, "folds": folds}


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: np.ndarray,
    groups: Optional[np.ndarray],
    n_classes: int,
    model_config: Dict,
    cfg: TrainConfig,
    history: Optional[List[Dict]] = None,
) -> Dict:
    """
    Same recipe on all samples except an inner split (about inner_val_frac, 1/7 by
    default) held out for early stopping and temperature calibration.
    """
    fit_rel, ival_rel = inner_split(y, groups, cfg.inner_val_frac, cfg.seed + 999)
    inner_sets = make_inner_sets(X[ival_rel], sample_ids[ival_rel], cfg)
    logger.info(f"Final model: fit {len(fit_rel)}, inner val {len(ival_rel)}")
    return train_one_model(X[fit_rel], y[fit_rel], inner_sets, y[ival_rel], n_classes,
                           model_config, cfg, seed=cfg.seed + 999, tag="final", history=history)
