"""
SPARSH-next: dilution of leukaemia profiles by normal marrow.

A marrow sample with a blast fraction b is, at every CpG, a mixture of the
blasts' methylation and that of the normal cells:
    beta_mix = b * beta_leukaemia + (1 - b) * beta_normal
The training arrays come mostly from high-blast diagnostic samples, while a
nanopore sample may have as few as 20% blasts. Mixing an array profile with a
normal-marrow array before the reads are simulated reproduces that dilution.
The label does not change, and normal marrows are never mixed. Where the
partner has no value, the leukaemia value is kept.

Which normal marrows are used. A patient's own normal cells are never part of
the training data, so the samples that are scored are diluted with normal
marrows the network has not seen, neither as a normal-marrow sample nor as a
dilution partner:
Training   : each leukaemia sample in a batch is diluted with probability
             dilution_prob, at a blast fraction drawn uniformly from
             [blast_min, 1], with a random normal marrow of the fit set.
Validation : the same, drawn once per inner-validation sample (seeded by its
             Sample_ID), with a normal marrow of the inner-validation split, so
             early stopping and temperature see the same mixture.
Evaluation : every leukaemia sample of the outer fold at a fixed blast fraction
             (conditions such as binary-blast30_0.30), with a normal marrow of
             the same outer fold. Only the input is built from it; its own label
             and prediction are untouched.
The partner of a sample is the pool member with the smallest hash of (sample,
partner, blast fraction, seed), so it depends only on the sample and on which
normal marrows are in the pool: runs with the same folds are scored on
identical inputs, and runs with different folds keep the same partner whenever
it lands in the same fold. If a pool has no normal marrow, the training-part
normals are used instead, with a warning.
With dilution_prob = 0, no random numbers are drawn and training is unchanged.
"""

import zlib
from typing import Optional

import numpy as np
import torch


def mix(x, partner, blast):
    """blast * x + (1 - blast) * partner, keeping x where the partner is missing (numpy or torch)."""
    if isinstance(x, torch.Tensor):
        return torch.where(torch.isnan(partner), x, blast * x + (1.0 - blast) * partner)
    return np.where(np.isnan(partner), x, blast * x + (1.0 - blast) * partner).astype(np.float32)


def dilute_batch(xb: torch.Tensor, yb: torch.Tensor, X_store: torch.Tensor, normal_rows: torch.Tensor,
                 normal_idx: int, prob: float, blast_min: float) -> torch.Tensor:
    """Training-time dilution of a batch on the device (see module docstring)."""
    if prob <= 0 or normal_rows.numel() == 0:
        return xb
    device = xb.device
    chosen = (torch.rand(len(yb), device=device) < prob) & (yb != normal_idx)
    k = int(chosen.sum().item())
    if k == 0:
        return xb
    pick = normal_rows[torch.randint(len(normal_rows), (k,), device=normal_rows.device)]
    partner = X_store.index_select(0, pick).to(device, non_blocking=True)
    blast = blast_min + torch.rand((k, 1), device=device) * (1.0 - blast_min)
    out = xb.clone()
    out[chosen] = mix(xb[chosen], partner, blast)
    return out


def _rng(base_seed: int, sample_id: str, tag: str, blast: float = 0.0) -> np.random.Generator:
    return np.random.default_rng([int(base_seed) & 0xFFFFFFFF, zlib.crc32(str(sample_id).encode()),
                                  zlib.crc32(tag.encode()), int(round(float(blast) * 1_000_000))])


def pick_partner(sample_id: str, partner_ids, tag: str, base_seed: int) -> int:
    """Index of the partner with the smallest hash of (sample, partner, tag, seed): stable as pools change."""
    keys = [zlib.crc32(f"{int(base_seed)}|{tag}|{sample_id}|{pid}".encode()) for pid in partner_ids]
    return int(np.argmin(keys))


def dilute_fixed(X: np.ndarray, sample_ids, is_normal: np.ndarray, partners: np.ndarray, partner_ids,
                 blast: float, base_seed: int) -> np.ndarray:
    """Every non-normal row at blast fraction `blast`, each with the partner chosen by pick_partner."""
    if blast >= 1.0:
        return X
    if len(partners) == 0:
        raise ValueError("no normal-marrow arrays available as dilution partners")
    out = X.astype(np.float32, copy=True)
    tag = f"eval{float(blast):.6f}"
    for i, sid in enumerate(sample_ids):
        if is_normal[i]:
            continue
        out[i] = mix(X[i], partners[pick_partner(sid, partner_ids, tag, base_seed)], blast)
    return out


def dilute_random(X: np.ndarray, sample_ids, is_normal: np.ndarray, partners: np.ndarray, partner_ids,
                  prob: float, blast_min: float, base_seed: int) -> np.ndarray:
    """Inner-validation dilution: each non-normal row with probability prob, blast ~ U(blast_min, 1)."""
    if prob <= 0:
        return X
    if len(partners) == 0:
        raise ValueError("no normal-marrow arrays available as dilution partners")
    out = X.astype(np.float32, copy=True)
    for i, sid in enumerate(sample_ids):
        if is_normal[i]:
            continue
        rng = _rng(base_seed, sid, "dilution-val")
        if rng.random() >= prob:
            continue
        blast = blast_min + rng.random() * (1.0 - blast_min)
        out[i] = mix(X[i], partners[pick_partner(sid, partner_ids, "val", base_seed)], blast)
    return out


def partner_pool(X: np.ndarray, y: np.ndarray, sample_ids, normal_idx: Optional[int], preferred: np.ndarray,
                 fallback: np.ndarray, what: str, logger=None):
    """Normal-marrow rows (and their IDs) among `preferred` row indices, else among `fallback`."""
    if normal_idx is None:
        return X[:0], []
    rows = preferred[y[preferred] == normal_idx]
    if len(rows) == 0:
        rows = fallback[y[fallback] == normal_idx]
        if logger is not None:
            logger.warning(f"  no normal marrow in the {what}; its dilution partners come from the training part")
    return X[rows], [str(s) for s in sample_ids[rows]]


def normal_index(idx_to_class: dict, name: Optional[str]) -> Optional[int]:
    """Class index of the normal-marrow class, or None if the model has no such class."""
    for i, c in idx_to_class.items():
        if c == name:
            return int(i)
    return None
