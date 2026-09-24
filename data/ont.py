"""
SPARSH-next: reading one ONT sample.

One loader is used everywhere (prediction, evaluation, tests), so training
and inference cannot drift apart again.

Expected file: a CSV in wide format with exactly one data row. The first
column is the row name; every other column is a CpG probe ID (cg...) and the
value is the fraction of reads methylated at that CpG, between 0 and 1.
Missing CpGs may be absent or empty.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def read_ont_csv(path: str, cpg_ids: List[str]) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Return (x, info).

    x    : float32 vector aligned to cpg_ids; NaN where the CpG has no data.
    info : n_columns, n_matched (CpG columns that are model CpGs),
           n_observed (matched CpGs with a value), coverage (n_observed / len(cpg_ids)).

    Raises ValueError for anything that would otherwise give a silent wrong
    prediction: more than one row, duplicated probe names, non-numeric values,
    or values outside [0, 1] (for example percentages).
    """
    path = Path(path)
    df = pd.read_csv(path, index_col=0)
    if df.shape[0] != 1:
        raise ValueError(
            f"{path.name}: expected one row (one column per CpG), found {df.shape[0]} rows. "
            "A long-format file (one row per CpG) must be pivoted first."
        )
    names = pd.Index([str(c).strip() for c in df.columns])
    if names.duplicated().any():
        raise ValueError(f"{path.name}: duplicated probe names, e.g. {list(names[names.duplicated()][:3])}; "
                         "collapse strands or replicate probes upstream")
    values = pd.to_numeric(pd.Series(df.iloc[0].to_numpy(), index=names), errors="coerce")
    bad = df.iloc[0].notna().to_numpy() & values.isna().to_numpy()
    if bad.any():
        raise ValueError(f"{path.name}: {int(bad.sum())} non-numeric values")
    finite = values[np.isfinite(values)]
    if len(finite) and finite.max() > 1.0 + 1e-6:
        raise ValueError(f"{path.name}: values up to {finite.max():.2f}. Expected methylated fractions "
                         "between 0 and 1; if these are percentages, divide by 100 upstream.")
    if len(finite) and finite.min() < -1e-6:
        raise ValueError(f"{path.name}: negative values")

    aligned = values.reindex(cpg_ids)
    x = aligned.to_numpy(dtype=np.float32)
    x[~np.isfinite(x)] = np.nan
    n_matched = int(names.isin(cpg_ids).sum())
    n_observed = int(np.isfinite(x).sum())
    info = {
        "n_columns": int(len(names)),
        "n_matched": n_matched,
        "n_observed": n_observed,
        "coverage": n_observed / max(1, len(cpg_ids)),
    }
    return x, info
