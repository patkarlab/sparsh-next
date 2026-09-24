"""
SPARSH-next: simulating ONT-like sparsity from array beta values.

Simulations
-----------
mask    : each CpG is kept with probability f (the observed fraction) and keeps
          its array beta value; the rest become missing. This is what v0.1.0 did.
reads   : each CpG receives k ~ Poisson(lambda) reads with lambda = -ln(1 - f),
          so a fraction f of CpGs has at least one read. Each read is methylated
          with probability beta, and the value is the methylated fraction of the
          reads (0, 1/2, 1, ...). Implemented by Poisson splitting: methylated ~
          Poisson(lambda * beta), unmethylated ~ Poisson(lambda * (1 - beta)).
binary  : the same reads, reported as one call per CpG: 1 if most reads are
          methylated, 0 if most are not, and a random 0 or 1 when they are split
          evenly. For ONT files whose values are always 0 or 1 (a threshold on the
          methylated fraction; the random tie covers either tie rule).
oneread : a fraction f of CpGs is covered and each reports the call of a single
          read (1 with probability beta), whatever the depth. For ONT files built
          from one read per CpG.
READ_SIMS are the simulations that stand for ONT data.

Observed fraction per training sample
-------------------------------------
schedule : one fraction for the whole epoch, 1 - mask_ratio, moving linearly
           from 1 - mask_start to 1 - mask_end (v0.1.0 behaviour).
random   : a new fraction for every sample drawn between cov_min and cov_max,
           so every coverage level is trained throughout; log-uniform (more
           weight on low coverage) or uniform (coverage_dist).

Evaluation uses corrupt_numpy with a seed derived from the sample ID, the
simulation and the coverage, so every run is scored on identical inputs.
"""

import math
import zlib
from typing import Optional

import numpy as np
import torch

SIMULATIONS = ("mask", "reads", "binary", "oneread")
READ_SIMS = ("reads", "binary", "oneread")
COVERAGE_MODES = ("random", "schedule")
COVERAGE_DISTS = ("loguniform", "uniform")


def get_mask_ratio(epoch: int, max_epochs: int, start: float = 0.97, end: float = 0.80) -> float:
    """Linear schedule of the masked fraction (v0.1.0)."""
    if max_epochs <= 1:
        return end
    return start + (end - start) * (epoch / (max_epochs - 1))


def sample_observed_fraction(
    n: int,
    mode: str,
    epoch: int = 0,
    epochs: int = 1,
    cov_min: float = 0.02,
    cov_max: float = 0.5,
    mask_start: float = 0.97,
    mask_end: float = 0.80,
    device: str = "cpu",
    dist: str = "loguniform",
) -> torch.Tensor:
    """Observed fraction for each of n samples, shape (n, 1)."""
    if mode == "schedule":
        f = 1.0 - get_mask_ratio(epoch, epochs, mask_start, mask_end)
        return torch.full((n, 1), f, device=device)
    if mode == "random":
        u = torch.rand((n, 1), device=device)
        if dist == "uniform":
            return cov_min + u * (cov_max - cov_min)
        if dist == "loguniform":
            lo, hi = math.log(cov_min), math.log(cov_max)
            return torch.exp(lo + u * (hi - lo))
        raise ValueError(f"coverage distribution must be one of {COVERAGE_DISTS}")
    raise ValueError(f"coverage mode must be one of {COVERAGE_MODES}")


def corrupt_torch(x: torch.Tensor, fraction: torch.Tensor, sim: str) -> torch.Tensor:
    """
    Training-time corruption on the device.
    x: (n, p) beta values with NaN for missing; fraction: (n, 1) observed fraction.
    """
    nan = torch.full_like(x, float("nan"))
    valid = ~torch.isnan(x)
    if sim == "mask":
        keep = torch.rand_like(x) < fraction
        return torch.where(keep & valid, x, nan)
    beta = torch.where(valid, x.clamp(0.0, 1.0), torch.zeros_like(x))
    if sim in ("reads", "binary"):
        lam = -torch.log1p(-fraction.clamp(max=1.0 - 1e-6))
        meth = torch.poisson(lam * beta)
        unmeth = torch.poisson(lam * (1.0 - beta))
        reads = meth + unmeth
        observed = (reads > 0) & valid
        if sim == "reads":
            return torch.where(observed, meth / reads.clamp_min(1.0), nan)
        tie_call = (torch.rand_like(x) < 0.5).to(x.dtype)
        call = torch.where(meth == unmeth, tie_call, (meth > unmeth).to(x.dtype))
        return torch.where(observed, call, nan)
    if sim == "oneread":
        observed = (torch.rand_like(x) < fraction) & valid
        call = (torch.rand_like(x) < beta).to(x.dtype)
        return torch.where(observed, call, nan)
    raise ValueError(f"simulation must be one of {SIMULATIONS}")


def corrupt_numpy(x: np.ndarray, fraction: float, sim: str, rng: np.random.Generator) -> np.ndarray:
    """Evaluation-time corruption of one sample or a matrix, reproducible through rng."""
    valid = np.isfinite(x)
    if sim == "mask":
        keep = rng.random(x.shape) < fraction
        out = np.where(keep & valid, x, np.nan)
    elif sim in ("reads", "binary"):
        lam = -math.log1p(-min(float(fraction), 1.0 - 1e-6))
        beta = np.where(valid, np.clip(x, 0.0, 1.0), 0.0)
        meth = rng.poisson(lam * beta)
        unmeth = rng.poisson(lam * (1.0 - beta))
        reads = meth + unmeth
        if sim == "reads":
            with np.errstate(invalid="ignore", divide="ignore"):
                out = np.where((reads > 0) & valid, meth / np.maximum(reads, 1), np.nan)
        else:
            tie_call = rng.random(x.shape) < 0.5
            call = np.where(meth == unmeth, tie_call, meth > unmeth).astype(np.float32)
            out = np.where((reads > 0) & valid, call, np.nan)
    elif sim == "oneread":
        beta = np.where(valid, np.clip(x, 0.0, 1.0), 0.0)
        observed = (rng.random(x.shape) < fraction) & valid
        call = (rng.random(x.shape) < beta).astype(np.float32)
        out = np.where(observed, call, np.nan)
    else:
        raise ValueError(f"simulation must be one of {SIMULATIONS}")
    return out.astype(np.float32)


def sample_rng(base_seed: int, sample_id: str, sim: str, fraction: float, salt: int = 0) -> np.random.Generator:
    """Seed that depends only on the sample and the condition, not on fold membership."""
    return np.random.default_rng([
        int(base_seed) & 0xFFFFFFFF,
        zlib.crc32(str(sample_id).encode()),
        zlib.crc32(sim.encode()),
        int(round(float(fraction) * 1_000_000)),
        int(salt),
    ])


def corrupt_rows(X: np.ndarray, sample_ids, fraction: Optional[float], sim: Optional[str],
                 base_seed: int, salt: int = 0) -> np.ndarray:
    """Corrupt each row with its own reproducible generator. fraction None = dense (unchanged)."""
    if fraction is None or sim is None or fraction >= 1.0:
        return X.astype(np.float32, copy=True)
    out = np.empty_like(X, dtype=np.float32)
    for i, sid in enumerate(sample_ids):
        out[i] = corrupt_numpy(X[i], fraction, sim, sample_rng(base_seed, sid, sim, fraction, salt))
    return out
