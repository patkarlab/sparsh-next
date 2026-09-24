"""
SPARSH-next: loading and preparing the array training data.

Conventions used throughout this code base
------------------------------------------
- X is a float32 matrix (samples x CpGs) of beta values in [0, 1].
- Missing values are NaN. They are NOT replaced by 0.5 here. How a missing
  CpG is presented to the network is decided by the model's input encoding
  (see models/sparse_nn.py), so masked CpGs, failed array probes and CpGs
  without ONT reads are all handled the same way.
- Class labels pass through configs/label_map.json (raw label -> model class)
  before anything else. The same file is applied to ONT ground-truth files.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LABEL_COLUMN = "ANNOTATION"
SAMPLE_COLUMN = "Sample_ID"
SOURCE_COLUMN = "Source_Dataset"


# =============================================================================
# Label map
# =============================================================================

def load_label_map(path: Optional[str]) -> Dict[str, str]:
    """Read configs/label_map.json. Returns an empty map when path is None."""
    if path is None:
        return {}
    with open(path) as f:
        content = json.load(f)
    mapping = content.get("merge", content) if isinstance(content, dict) else None
    if not isinstance(mapping, dict):
        raise ValueError(f"{path}: expected a JSON object with a 'merge' dictionary")
    for raw, merged in mapping.items():
        if not isinstance(raw, str) or not isinstance(merged, str):
            raise ValueError(f"{path}: every label map entry must be text -> text")
    return dict(mapping)


def apply_label_map(labels: Sequence[str], label_map: Dict[str, str], log: bool = True) -> np.ndarray:
    """Replace raw labels by model classes. Logs how many samples each rule changed."""
    labels = np.asarray([str(v).strip() for v in labels], dtype=object)
    if not label_map:
        return labels
    mapped = labels.copy()
    for raw, merged in label_map.items():
        hit = labels == raw
        n = int(hit.sum())
        mapped[hit] = merged
        if log:
            if n:
                logger.info(f"  label map: {raw!r} -> {merged!r} ({n} samples)")
            else:
                logger.warning(f"  label map: no sample is labelled {raw!r} (check the spelling)")
    return mapped


# =============================================================================
# Loading
# =============================================================================

def load_ids_to_exclude(path: Optional[str]) -> Set[str]:
    """One sample ID per line. A missing file is an error, not a silent no-op."""
    if path is None:
        return set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Exclusion file not found: {p}")
    with open(p) as f:
        ids = {line.strip() for line in f if line.strip()}
    logger.info(f"Loaded {len(ids)} sample IDs to exclude from {p}")
    return ids


def load_training_data(
    data_path: str,
    label_map: Optional[Dict[str, str]] = None,
    exclude_ids_path: Optional[str] = None,
    cpg_list_path: Optional[str] = None,
    group_col: Optional[str] = None,
    groups_file: Optional[str] = None,
) -> Dict[str, object]:
    """
    Load the training pickle.

    Expected: a pandas DataFrame with one row per sample, CpG columns named
    cg..., an ANNOTATION column, optionally Sample_ID (otherwise the index is
    used) and Source_Dataset.

    Returns a dict with keys: X, labels_raw, labels, sample_ids, cpg_ids,
    sources (array or None), groups (array or None).
    """
    logger.info(f"Loading training data from {data_path}")
    df = pd.read_pickle(data_path)
    logger.info(f"  DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")

    if SAMPLE_COLUMN in df.columns:
        sample_ids = df[SAMPLE_COLUMN].astype(str).str.strip().to_numpy(dtype=object)
    else:
        logger.info(f"  No {SAMPLE_COLUMN} column: using the DataFrame index as sample IDs")
        sample_ids = df.index.astype(str).str.strip().to_numpy(dtype=object)
    if len(set(sample_ids)) != len(sample_ids):
        dup = pd.Series(sample_ids)[pd.Series(sample_ids).duplicated()].unique()[:5]
        raise ValueError(f"Duplicate sample IDs in the training data, e.g. {list(dup)}")

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Column {LABEL_COLUMN} not found in {data_path}")
    raw = df[LABEL_COLUMN]
    if raw.isna().any():
        raise ValueError(f"{int(raw.isna().sum())} samples have no {LABEL_COLUMN}; fix or exclude them first")
    labels_raw = raw.astype(str).str.strip().to_numpy(dtype=object)

    sources = df[SOURCE_COLUMN].astype(str).to_numpy(dtype=object) if SOURCE_COLUMN in df.columns else None

    meta = {SAMPLE_COLUMN, LABEL_COLUMN, SOURCE_COLUMN}
    if group_col:
        meta.add(group_col)
    cpg_cols = [c for c in df.columns if c not in meta and str(c).startswith("cg")]
    if len(set(cpg_cols)) != len(cpg_cols):
        raise ValueError("Duplicate CpG column names in the training data")

    if cpg_list_path:
        with open(cpg_list_path) as f:
            reference = [str(c) for c in json.load(f)]
        if len(set(reference)) != len(reference):
            raise ValueError(f"{cpg_list_path} lists some CpGs more than once")
        present = set(cpg_cols)
        missing = [c for c in reference if c not in present]
        if missing:
            raise ValueError(f"{len(missing)} CpGs from {cpg_list_path} are not in the data, e.g. {missing[:5]}")
        cpg_cols = reference
        logger.info(f"  Using the {len(cpg_cols)} CpGs listed in {cpg_list_path}, in that order")

    X = df[cpg_cols].to_numpy(dtype=np.float32, copy=True)  # writable copy (pandas copy-on-write safe)
    with np.errstate(invalid="ignore"):
        lo, hi = np.nanmin(X), np.nanmax(X)
    if np.isinf(lo) or np.isinf(hi):
        X[np.isinf(X)] = np.nan
        lo, hi = np.nanmin(X), np.nanmax(X)
    if lo < -1e-6 or hi > 1 + 1e-6:
        raise ValueError(
            f"Values outside [0, 1] (min {lo:.3f}, max {hi:.3f}); "
            "expected beta values, not M-values or percentages"
        )
    n_nan = int(np.isnan(X).sum())
    logger.info(f"  {X.shape[0]} samples x {X.shape[1]} CpGs; missing values: {n_nan} "
                f"({100.0 * n_nan / X.size:.3f}%), kept as missing (not filled with 0.5)")

    groups = None
    if group_col and groups_file:
        raise ValueError("Use either --group_col or --groups_file, not both")
    if group_col:
        if group_col not in df.columns:
            raise ValueError(f"Group column {group_col!r} not found")
        g = df[group_col].to_numpy(dtype=object)
        blank = pd.isna(g) | (pd.Series(g).astype(str).str.strip() == "").to_numpy()
        groups = np.array([("sample:" + s) if b else ("group:" + str(v).strip())
                           for s, v, b in zip(sample_ids, g, blank)], dtype=object)
        logger.info(f"  Groups from column {group_col!r}: {len(set(groups))} groups for {len(groups)} samples "
                    f"({int(blank.sum())} blank = own group)")
    if groups_file:
        gdf = pd.read_csv(groups_file, dtype=str)
        if not {"Sample_ID", "group"} <= set(gdf.columns):
            raise ValueError(f"{groups_file} must have columns Sample_ID and group")
        gdf = gdf.dropna(subset=["Sample_ID", "group"])
        gdf = gdf[(gdf["Sample_ID"].str.strip() != "") & (gdf["group"].str.strip() != "")]
        if gdf["Sample_ID"].str.strip().duplicated().any():
            raise ValueError(f"{groups_file}: a Sample_ID appears more than once")
        gmap = dict(zip(gdf["Sample_ID"].str.strip(), gdf["group"].str.strip()))
        groups = np.array([("group:" + gmap[s]) if s in gmap else ("sample:" + s) for s in sample_ids], dtype=object)
        logger.info(f"  Groups from {groups_file}: {len(set(groups))} groups for {len(groups)} samples")

    del df
    bundle = {
        "X": X,
        "labels_raw": labels_raw,
        "labels": apply_label_map(labels_raw, label_map or {}),
        "sample_ids": sample_ids,
        "cpg_ids": list(map(str, cpg_cols)),
        "sources": sources,
        "groups": groups,
    }

    exclude_ids = load_ids_to_exclude(exclude_ids_path)
    if exclude_ids:
        keep = np.array([s not in exclude_ids for s in sample_ids])
        unknown = exclude_ids - set(sample_ids)
        if unknown:
            logger.warning(f"  {len(unknown)} IDs in the exclusion file are not in the data")
        logger.info(f"  Excluding {int((~keep).sum())} listed samples")
        bundle = subset(bundle, keep)
    return bundle


def subset(bundle: Dict[str, object], keep: np.ndarray) -> Dict[str, object]:
    """Keep the rows where keep is True in every per-sample array of the bundle."""
    out = dict(bundle)
    for key in ("X", "labels_raw", "labels", "sample_ids", "sources", "groups"):
        if out.get(key) is not None:
            out[key] = out[key][keep]
    return out


# =============================================================================
# Class filtering and label encoding
# =============================================================================

def filter_classes(
    labels: np.ndarray,
    min_samples: int = 5,
    exclude_classes: Optional[List[str]] = None,
    exclude_prefixes: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Return a boolean mask of samples to keep.

    exclude_classes: exact class names (after the label map). An unknown name
    is an error, so a typo cannot silently keep a class.
    exclude_prefixes: remove every class whose name starts with the prefix
    (for example "MPAL"). Each removed class is logged.
    """
    exclude_classes = list(exclude_classes or [])
    exclude_prefixes = list(exclude_prefixes or [])
    names, counts = np.unique(labels, return_counts=True)
    present = set(names)
    unknown = [c for c in exclude_classes if c not in present]
    if unknown:
        raise ValueError(f"--exclude_classes names not found after the label map: {unknown}. "
                         f"Available classes: {sorted(present)}")

    drop = set(exclude_classes)
    for prefix in exclude_prefixes:
        matched = [c for c in names if c.startswith(prefix)]
        if not matched:
            logger.warning(f"  --exclude_prefixes {prefix!r} matched no class")
        drop.update(matched)

    keep_classes = []
    for name, count in zip(names, counts):
        if name in drop:
            logger.info(f"  Dropping {name} ({count} samples): excluded")
        elif count < min_samples:
            logger.info(f"  Dropping {name} ({count} samples): fewer than {min_samples}")
        else:
            keep_classes.append(name)
    return np.isin(labels, keep_classes)


def encode_labels(labels: np.ndarray) -> Tuple[np.ndarray, Dict[int, str]]:
    """Alphabetical class order -> integer labels 0..K-1."""
    classes = sorted(set(labels))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[c] for c in labels], dtype=np.int64)
    return y, {i: c for c, i in class_to_idx.items()}


# =============================================================================
# Legacy minority augmentation (kept only to reproduce v0.1.0 for comparison)
# =============================================================================

def legacy_masked_upsample(
    X: np.ndarray,
    y: np.ndarray,
    mask_ratios: Sequence[float],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    v0.1.0 --upsample_minority: raise every class to the largest class count
    with copies that are pre-masked at a random ratio from mask_ratios.
    Masked CpGs are NaN here (v0.1.0 used 0.5, which the midpoint encoding
    reproduces). Not recommended: see docs/EXPERIMENTS.md.
    """
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    target = counts.max()
    X_parts, y_parts = [X], [y]
    for cls, count in zip(classes, counts):
        if count >= target:
            continue
        src = X[y == cls]
        n_new = int(target - count)
        new = np.empty((n_new, X.shape[1]), dtype=np.float32)
        for i in range(n_new):
            row = src[rng.randint(0, count)].copy()
            ratio = mask_ratios[rng.randint(0, len(mask_ratios))]
            row[rng.random_sample(X.shape[1]) < ratio] = np.nan
            new[i] = row
        X_parts.append(new)
        y_parts.append(np.full(n_new, cls, dtype=y.dtype))
    logger.info(f"  Legacy masked upsampling: {len(y)} -> {sum(len(p) for p in y_parts)} training rows")
    return np.vstack(X_parts), np.concatenate(y_parts)
