# dataset.py file 


"""
SPARSH Data Loading and Preprocessing.
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Set
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading
# =============================================================================

def load_ids_to_exclude(junk_path: str) -> Set[str]:
    """
    Load sample IDs to exclude from training.
    
    Args:
        junk_path: Path to text file with one sample ID per line.
        
    Returns:
        Set of sample IDs to exclude.
    """
    junk_path = Path(junk_path)
    if not junk_path.exists():
        logger.warning(f"Exclusion file not found: {junk_path}")
        return set()
    
    with open(junk_path, "r") as f:
        ids = set(line.strip() for line in f if line.strip())
    
    logger.info(f"Loaded {len(ids)} sample IDs to exclude")
    return ids

def normalize_and_filter_labels(
    df: pd.DataFrame,
    labels_raw: np.ndarray,
    sample_ids: List[str],
):
    """
    Apply class cleanup rules:
    - Remove JMML_Inter
    - Merge MDS_Hypo + MDS_Hyper → MDS
    - Merge CMML_Hypo + CMML_Hyper → CMML
    """

    labels_raw = pd.Series(labels_raw)
    # now I have added this class in the dataset, so I am not removing it
    # ── Remove JMML_Inter ─────────────────────────────
    #remove_mask = labels_raw == "JMML_Inter"    
    #if remove_mask.any():
    #    logger.info(f"Removing {remove_mask.sum()} JMML_Inter samples")
    #    df = df.loc[~remove_mask].reset_index(drop=True)
    #    labels_raw = labels_raw.loc[~remove_mask].reset_index(drop=True)
    #    sample_ids = df["Sample_ID"].astype(str).tolist()

    # ── Merge MDS classes ─────────────────────────────
    labels_raw = labels_raw.replace({
        # "MDS_Hypo": "MDS",
        # "MDS_Hyper": "MDS",
        # "CMML_Hypo": "CMML",
        # "CMML_Hyper": "CMML",
        # "JMML_Inter": "JMML", # this was wrong suggest by nikhil sir keep seprate classes of JMML
        # "JMML_Hyper": "JMML",
        # "JMML_Hypo": "JMML",
        # "MDS_Hypo": "MDS",
        # "MDS_Hyper": "MDS",
        # "MDS-LB_SF3B1_mut": "MDS",
        "MDS_SF3B1_mut": "MDS",
        "AML_DEK-NUP214": "AML_mutated_NPM1_Nup98_DEKnup214",
        "AML_NUP98-r": "AML_mutated_NPM1_Nup98_DEKnup214",
        "AML_mutated NPM1": "AML_mutated_NPM1_Nup98_DEKnup214",
        "B-ALL_PAX5 alt": "B-ALL_PAX5_BCR-ABL1_B-ALL_BCR-ABL1_like",
        "B-ALL_BCR-ABL1 like": "B-ALL_PAX5_BCR-ABL1_B-ALL_BCR-ABL1_like",
        "B-ALL_BCR-ABL1": "B-ALL_PAX5_BCR-ABL1_B-ALL_BCR-ABL1_like",
    })

    # ── Merge CMML classes ────────────────────────────
    labels_raw = labels_raw.replace({
        # "CMML_Hypo": "CMML",
        # "CMML_Hyper": "CMML",
        # "MDS_Hypo": "MDS",
        # "MDS_Hyper": "MDS",
        # "MDS": "MDS_163", THIS WHEN HAVE TO EXCLUTE THIS CLASS OF MDS ONLY
        # "MDS-LB_SF3B1_mut": "MDS",
        "MDS_SF3B1_mut": "MDS",
        "AML_DEK-NUP214": "AML_mutated_NPM1_Nup98_DEKnup214",
        "AML_NUP98-r": "AML_mutated_NPM1_Nup98_DEKnup214",
        "AML_mutated NPM1": "AML_mutated_NPM1_Nup98_DEKnup214",
        "B-ALL_PAX5 alt": "B-ALL_PAX5_BCR-ABL1_B-ALL_BCR-ABL1_like",
        "B-ALL_BCR-ABL1 like": "B-ALL_PAX5_BCR-ABL1_B-ALL_BCR-ABL1_like",
        "B-ALL_BCR-ABL1": "B-ALL_PAX5_BCR-ABL1_B-ALL_BCR-ABL1_like",
        # "JMML_Inter": "JMML", # this was wrong suggest by nikhil sir keep seprate classes of JMML
        # "JMML_Hyper": "JMML",
        # "JMML_Hypo": "JMML",
    })

    return df, labels_raw.values, sample_ids



def load_methylation_pickle(
    data_path: str,
    junk_path: Optional[str] = None,
    cpg_list_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], Dict[int, str]]:
    """
    Load methylation data from pickle file.
    
    Expected format (updated):
    - Column 0          : Sample_ID  (GSM IDs)
    - Columns 1 to -3   : CpG probes (cg...)
    - Second-to-last    : ANNOTATION (class labels)
    - Last              : Source_Dataset (batch info, ignored)
    
    Args:
        data_path: Path to pickle file containing methylation DataFrame.
        junk_path: Optional path to file with sample IDs to exclude.
        cpg_list_path: Optional path to JSON with CpG ordering (for consistency).
        
    Returns:
        Tuple containing:
        - X: Methylation matrix (n_samples, n_features)
        - y: Integer labels (n_samples,)
        - sample_ids: List of sample identifiers
        - cpg_ids: List of CpG probe IDs
        - idx_to_class: Mapping from label index to class name
    """
    data_path = Path(data_path)
    
    logger.info(f"Loading methylation data from {data_path}")
    df = pd.read_pickle(data_path)
    logger.info(f"Loaded DataFrame: {df.shape[0]} samples, {df.shape[1]} columns")

    if "Sample_ID" in df.columns:
        sample_ids = df["Sample_ID"].astype(str).tolist()
    else:
        logger.info("Sample_ID column not found — using index as Sample_ID")
        df = df.reset_index().rename(columns={"index": "Sample_ID"})
        sample_ids = df["Sample_ID"].astype(str).tolist()



    logger.info(f"Sample IDs from 'Sample_ID' column, first 5: {sample_ids[:5]}")


    labels_raw = df["ANNOTATION"].values
    df, labels_raw, sample_ids = normalize_and_filter_labels(df, labels_raw, sample_ids)
    logger.info(f"Label column : ANNOTATION")
    logger.info(f"Batch column : Source_Dataset (ignored in training)")


    meta_cols = ["Sample_ID", "ANNOTATION", "Source_Dataset"]
    cpg_cols  = [c for c in df.columns if c not in meta_cols and str(c).startswith("cg")]
    logger.info(f"Found {len(cpg_cols)} CpG columns")
    

    X = df[cpg_cols].values.astype(np.float32)
    cpg_ids = cpg_cols
    

    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        logger.info(f"Replacing {nan_count} NaN values with 0.5")
        X = np.nan_to_num(X, nan=0.5)
    

    if junk_path:
        junk_ids = load_ids_to_exclude(junk_path)
        keep_mask = [sid not in junk_ids for sid in sample_ids]
        n_excluded = sum(1 for k in keep_mask if not k)
        
        X          = X[keep_mask]
        labels_raw = labels_raw[np.array(keep_mask)]
        sample_ids = [sid for sid, keep in zip(sample_ids, keep_mask) if keep]
        
        logger.info(f"Excluded {n_excluded} junk samples, {len(sample_ids)} remaining")
    

    unique_classes = sorted(set(labels_raw))
    class_to_idx   = {c: i for i, c in enumerate(unique_classes)}
    idx_to_class   = {i: c for c, i in class_to_idx.items()}
    
    y = np.array([class_to_idx[c] for c in labels_raw])
    

    logger.info(f"Classes ({len(idx_to_class)}):")
    unique, counts = np.unique(y, return_counts=True)
    for idx, count in zip(unique, counts):
        logger.info(f"  {idx_to_class[idx]}: {count}")
    

    if cpg_list_path and Path(cpg_list_path).exists():
        with open(cpg_list_path) as f:
            reference_cpgs = json.load(f)
        
        cpg_set    = set(cpg_ids)
        common_cpgs = [c for c in reference_cpgs if c in cpg_set]
        
        if len(common_cpgs) < len(cpg_ids):
            logger.info(f"Using {len(common_cpgs)} CpGs from reference list")
            
            cpg_to_idx = {c: i for i, c in enumerate(cpg_ids)}
            indices    = [cpg_to_idx[c] for c in common_cpgs]
            X          = X[:, indices]
            cpg_ids    = common_cpgs
    
    logger.info(f"Final: {X.shape[0]} samples, {X.shape[1]} CpGs, {len(idx_to_class)} classes")
    
    return X, y, sample_ids, cpg_ids, idx_to_class




def filter_classes(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: List[str],
    idx_to_class: Dict[int, str],
    min_samples: int = 10,
    exclude_classes: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[int, str]]:

    if exclude_classes is None:
        exclude_classes = []
    
  
    exclude_lower = [c.lower().strip() for c in exclude_classes]
    
    unique, counts = np.unique(y, return_counts=True)
    
    keep_classes = []
    for idx, count in zip(unique, counts):
        name = idx_to_class.get(idx, "")
        name_lower = name.lower().strip()
        
        # Check exclusion criteria
        should_exclude = any(excl in name_lower for excl in exclude_lower)
        
        if count >= min_samples and not should_exclude:
            keep_classes.append(idx)
        else:
            reason = "excluded by name" if should_exclude else f"<{min_samples} samples"
            logger.info(f"  Dropping: {name} ({count} samples) - {reason}")
    
    # Create mask and remap labels to contiguous indices
    keep_mask = np.isin(y, keep_classes)
    old_to_new = {old: new for new, old in enumerate(sorted(keep_classes))}
    new_idx_to_class = {new: idx_to_class[old] for old, new in old_to_new.items()}
    
    X_filtered = X[keep_mask]
    y_filtered = np.array([old_to_new[yi] for yi in y[keep_mask]])
    sample_ids_filtered = [sid for sid, keep in zip(sample_ids, keep_mask) if keep]
    
    logger.info(f"Filtered: {len(idx_to_class)} -> {len(new_idx_to_class)} classes")
    logger.info(f"Samples: {len(y)} -> {len(y_filtered)}")
    
    return X_filtered, y_filtered, sample_ids_filtered, new_idx_to_class




def upsample_rare_classes(
    X: np.ndarray,
    y: np.ndarray,
    upsample_to: int = 50,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:

    unique, counts = np.unique(y, return_counts=True)
    
    X_list = [X]
    y_list = [y]
    
    for cls, count in zip(unique, counts):
        if count < upsample_to:
            # Get samples of this class
            cls_mask = y == cls
            X_cls = X[cls_mask]
            y_cls = y[cls_mask]
            
            # Bootstrap resample
            n_new = upsample_to - count
            X_new, y_new = resample(
                X_cls, y_cls,
                n_samples=n_new,
                replace=True,
                random_state=random_state,
            )
            
            X_list.append(X_new)
            y_list.append(y_new)
            
            logger.info(f"  Upsampled class {cls}: {count} -> {upsample_to}")
    
    X_upsampled = np.vstack(X_list)
    y_upsampled = np.concatenate(y_list)
    
    logger.info(f"After upsampling: {len(y)} -> {len(y_upsampled)} samples")

    return X_upsampled, y_upsampled


def upsample_with_masking_augmentation(
    X: np.ndarray,
    y: np.ndarray,
    target_count: int = 0,
    mask_ratios: Optional[List[float]] = None,
    fill_value: float = 0.5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    
    rng = np.random.RandomState(random_state)

    if mask_ratios is None:
        mask_ratios = [0.3, 0.5, 0.7, 0.85, 0.9]

    unique, counts = np.unique(y, return_counts=True)

    if target_count <= 0:
        target_count = counts.max()

    X_list = [X.copy()]
    y_list = [y.copy()]

    n_features = X.shape[1]

    for cls, count in zip(unique, counts):
        if count >= target_count:
            continue

        cls_mask = y == cls
        X_cls = X[cls_mask]
        n_new = target_count - count

        X_new = np.empty((n_new, n_features), dtype=np.float32)

        for i in range(n_new):
            # Randomly pick a source sample
            src_idx = rng.randint(0, count)
            sample = X_cls[src_idx].copy()

            # Randomly pick a mask ratio
            ratio = mask_ratios[rng.randint(0, len(mask_ratios))]

            # Create mask and apply
            mask = rng.random(n_features) < ratio
            sample[mask] = fill_value

            X_new[i] = sample

        y_new = np.full(n_new, cls, dtype=y.dtype)

        X_list.append(X_new)
        y_list.append(y_new)

        logger.info(
            f"  Augmented class {cls}: {count} -> {target_count} "
            f"(+{n_new} masked variants)"
        )

    X_augmented = np.vstack(X_list)
    y_augmented = np.concatenate(y_list)

    logger.info(
        f"After masking augmentation: {len(y)} -> {len(y_augmented)} samples"
    )

    return X_augmented, y_augmented



def create_holdout_split(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: List[str],
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str], np.ndarray, np.ndarray]:

    sample_ids_arr = np.array(sample_ids)
    indices = np.arange(len(y))

    train_idx, holdout_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_holdout = X[holdout_idx]
    y_holdout = y[holdout_idx]
    train_ids = sample_ids_arr[train_idx].tolist()
    holdout_ids = sample_ids_arr[holdout_idx].tolist()

    logger.info(f"Train/Holdout split: {len(train_idx)} train, {len(holdout_idx)} holdout")
    logger.info(f"  Train class distribution:")
    for cls in np.unique(y_train):
        logger.info(f"    Class {cls}: {(y_train == cls).sum()}")
    logger.info(f"  Holdout class distribution:")
    for cls in np.unique(y_holdout):
        logger.info(f"    Class {cls}: {(y_holdout == cls).sum()}")

    return X_train, y_train, X_holdout, y_holdout, train_ids, holdout_ids, train_idx, holdout_idx


# =============================================================================
# Feature Selection
# =============================================================================

def calculate_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    n_subsample: int = 1000,
    random_state: int = 42,
) -> np.ndarray:

    np.random.seed(random_state)
    n_features = X.shape[1]
    scores = np.zeros(n_features)
    
    logger.info("Calculating feature importance...")
    
    # Subsample for computational efficiency
    if len(X) > n_subsample:
        idx = np.random.choice(len(X), n_subsample, replace=False)
        X_sub = X[idx]
        y_sub = y[idx]
    else:
        X_sub = X
        y_sub = y
    
    # Method 1: Variance
    logger.info("  - Variance")
    var_scores = np.var(X, axis=0)
    var_scores = _normalize(var_scores)
    scores += var_scores
    
    # Method 2: Mutual Information
    logger.info("  - Mutual Information")
    mi_scores = mutual_info_classif(X_sub, y_sub, random_state=random_state, n_jobs=-1)
    mi_scores = _normalize(mi_scores)
    scores += mi_scores * 2
    
    # Method 3: F-score (ANOVA)
    logger.info("  - F-score (ANOVA)")
    f_scores, _ = f_classif(X, y)
    f_scores = np.nan_to_num(f_scores, nan=0)
    f_scores = _normalize(f_scores)
    scores += f_scores
    
    # Method 4: Random Forest
    logger.info("  - Random Forest")
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=random_state, 
        n_jobs=-1
    )
    rf.fit(X_sub, y_sub)
    rf_scores = rf.feature_importances_
    rf_scores = _normalize(rf_scores)
    scores += rf_scores * 2
    
    logger.info("Feature importance calculation complete")
    
    return scores


def _normalize(x: np.ndarray) -> np.ndarray:
    """Min-max normalize array to [0, 1] range."""
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-8:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def select_top_features(
    X: np.ndarray,
    importance: np.ndarray,
    cpg_ids: List[str],
    n_features: int = 20000,
) -> Tuple[np.ndarray, List[str], np.ndarray]:

    n_features = min(n_features, X.shape[1])
    top_indices = np.argsort(importance)[::-1][:n_features]
    X_selected = X[:, top_indices]
    selected_cpgs = [cpg_ids[i] for i in top_indices]
    
    logger.info(f"Selected top {n_features} features")
    
    return X_selected, selected_cpgs, top_indices


# =============================================================================
# ONT Sample Loading
# =============================================================================

def load_ont_samples(
    ont_dir: str,
    selected_cpgs: List[str],
    missing_value: float = 0.5,
) -> Tuple[np.ndarray, List[str], List[float]]:

    ont_dir = Path(ont_dir)
    ont_files = sorted(ont_dir.glob("*.csv"))
    
    if not ont_files:
        raise ValueError(f"No CSV files found in {ont_dir}")
    
    n_samples = len(ont_files)
    n_features = len(selected_cpgs)
    
    # Create lookup for fast CpG matching
    cpg_to_idx = {cpg: i for i, cpg in enumerate(selected_cpgs)}
    
    X_ont = np.full((n_samples, n_features), missing_value, dtype=np.float32)
    sample_ids = []
    coverages = []
    
    logger.info(f"Loading {n_samples} ONT samples...")
    
    for i, f in enumerate(ont_files):
        df = pd.read_csv(f, index_col=0)
        sample_id = f.stem
        sample_ids.append(sample_id)
        
        row = df.iloc[0]
        matched = 0
        
        for probe in row.index:
            if probe in cpg_to_idx:
                val = row[probe]
                if not np.isnan(val):
                    idx = cpg_to_idx[probe]
                    # Clip extreme values to valid beta range
                    if val <= 0:
                        X_ont[i, idx] = 0.05
                    elif val >= 1:
                        X_ont[i, idx] = 0.95
                    else:
                        X_ont[i, idx] = val
                    matched += 1
        
        coverage = matched / n_features * 100
        coverages.append(coverage)
        
        if (i + 1) % 10 == 0:
            logger.info(f"  Loaded {i+1}/{n_samples} samples")
    
    logger.info(f"Average coverage: {np.mean(coverages):.1f}%")
    
    return X_ont, sample_ids, coverages

