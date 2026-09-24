"""
SPARSH-next: reading one ONT sample.

One loader is used everywhere (prediction, evaluation, tests), so training
and inference cannot drift apart again.

Expected file: a CSV in wide format with exactly one data row. The first
column is the row name (sample ID); every other column is a CpG probe ID
(cg...) and the value is the fraction of reads methylated at that CpG,
between 0 and 1. Missing CpGs may be absent, empty, NA or NaN.

Repeated probe IDs (the same cg... in more than one column) are combined by
the `duplicates` rule:
  mean  : the mean of the copies that have a value (default);
  first : the first column only, as a pandas reader does implicitly (it renames
          later copies cg....1, cg....2, which then match no model CpG);
  error : the file is refused.
Whatever the rule, info reports how many probes were repeated and how the
copies relate, so the rule can be checked against the data.
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

MISSING_TOKENS = {"", "na", "nan", "null", "none"}
DUPLICATE_RULES = ("mean", "first", "error")


def read_ont_csv(path: str, cpg_ids: List[str], duplicates: str = "mean") -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Return (x, info).

    x    : float32 vector aligned to cpg_ids; NaN where the CpG has no data.
    info : n_columns (CpG columns in the file), n_matched (distinct model CpGs among them),
           n_observed (model CpGs with a value), coverage (n_observed / len(cpg_ids)),
           n_repeated_probes (probe IDs that occur in more than one column), max_copies,
           n_repeated_multi_observed (repeated probes with a value in two or more copies),
           n_repeated_disagree (of those, copies with different values),
           n_observed_first (model CpGs with a value in their first column).

    Raises ValueError for anything that would otherwise give a silent wrong
    prediction: no row-name column, more than one row, non-numeric values,
    values outside [0, 1] (for example percentages), and repeated probe names
    when duplicates="error".
    """
    if duplicates not in DUPLICATE_RULES:
        raise ValueError(f"duplicates must be one of {DUPLICATE_RULES}")
    path = Path(path)
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        rows = [r for r in reader if any(cell.strip() for cell in r)]
    if not header:
        raise ValueError(f"{path.name}: empty file")
    if header[0].strip().startswith("cg"):
        raise ValueError(f"{path.name}: the first column is {header[0]!r}; the first column must be the row name "
                         "(sample ID), followed by one column per CpG")
    if len(rows) != 1:
        raise ValueError(
            f"{path.name}: expected one data row (one column per CpG), found {len(rows)} rows. "
            "A long-format file (one row per CpG) must be pivoted first."
        )
    row = rows[0]
    if len(row) != len(header):
        raise ValueError(f"{path.name}: the data row has {len(row)} cells but the header has {len(header)}")
    index = pd.Index([h.strip() for h in header[1:]])
    repeated = index.duplicated(keep=False)
    if repeated.any() and duplicates == "error":
        raise ValueError(f"{path.name}: repeated probe names, e.g. {list(index[repeated].unique()[:3])}; "
                         "choose how to combine them (duplicates='mean' or 'first')")

    cells = pd.Series([c.strip() for c in row[1:]], index=index)
    missing = cells.str.lower().isin(MISSING_TOKENS)
    values = pd.to_numeric(cells.where(~missing), errors="coerce")
    bad = ~missing & values.isna()
    if bad.any():
        raise ValueError(f"{path.name}: {int(bad.sum())} non-numeric values, e.g. {cells[bad].iloc[0]!r}")
    finite = values[np.isfinite(values)]
    if len(finite) and finite.max() > 1.0 + 1e-6:
        raise ValueError(f"{path.name}: values up to {finite.max():.2f}. Expected methylated fractions "
                         "between 0 and 1; if these are percentages, divide by 100 upstream.")
    if len(finite) and finite.min() < -1e-6:
        raise ValueError(f"{path.name}: negative values")
    values = values.where(np.isfinite(values))           # inf -> NaN before combining copies

    first = values[~index.duplicated(keep="first")]
    stats = {"n_repeated_probes": 0, "max_copies": 1, "n_repeated_multi_observed": 0, "n_repeated_disagree": 0}
    if repeated.any():
        groups = values[repeated].groupby(level=0, sort=False)
        n_obs = groups.count()
        multi = n_obs >= 2
        spread = groups.max() - groups.min()
        stats = {"n_repeated_probes": int(len(n_obs)), "max_copies": int(groups.size().max()),
                 "n_repeated_multi_observed": int(multi.sum()),
                 "n_repeated_disagree": int((multi & (spread > 1e-6)).sum())}
        combined = values.groupby(level=0, sort=False).mean() if duplicates == "mean" else first
    else:
        combined = values

    x = combined.reindex(cpg_ids).to_numpy(dtype=np.float32)
    x[~np.isfinite(x)] = np.nan
    n_observed = int(np.isfinite(x).sum())
    info = {
        "n_columns": int(len(index)),
        "n_matched": int(index.unique().isin(cpg_ids).sum()),
        "n_observed": n_observed,
        "coverage": n_observed / max(1, len(cpg_ids)),
        **stats,
        "n_observed_first": int(np.isfinite(first.reindex(cpg_ids).to_numpy(dtype=np.float64)).sum()),
    }
    return x, info
