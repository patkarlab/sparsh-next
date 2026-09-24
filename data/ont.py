"""
SPARSH-next: reading one ONT sample.

One loader is used everywhere (prediction, evaluation, tests), so training
and inference cannot drift apart again.

Expected file: a CSV in wide format with exactly one data row. The first
column is the row name (sample ID); every other column is a CpG probe ID
(cg...) and the value is the fraction of reads methylated at that CpG,
between 0 and 1. Missing CpGs may be absent, empty, NA or NaN.
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

MISSING_TOKENS = {"", "na", "nan", "null", "none"}


def read_ont_csv(path: str, cpg_ids: List[str]) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Return (x, info).

    x    : float32 vector aligned to cpg_ids; NaN where the CpG has no data.
    info : n_columns, n_matched (CpG columns that are model CpGs),
           n_observed (matched CpGs with a value), coverage (n_observed / len(cpg_ids)).

    Raises ValueError for anything that would otherwise give a silent wrong
    prediction: no row-name column, more than one row, duplicated probe names,
    non-numeric values, or values outside [0, 1] (for example percentages).
    """
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
    names = [h.strip() for h in header[1:]]
    seen = set()
    dups = [n for n in names if n in seen or seen.add(n)]
    if dups:
        raise ValueError(f"{path.name}: duplicated probe names, e.g. {dups[:3]}; "
                         "collapse strands or replicate probes upstream")

    cells = pd.Series([c.strip() for c in row[1:]], index=names)
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

    x = values.reindex(cpg_ids).to_numpy(dtype=np.float32)
    x[~np.isfinite(x)] = np.nan
    n_matched = int(pd.Index(names).isin(cpg_ids).sum())
    n_observed = int(np.isfinite(x).sum())
    info = {
        "n_columns": len(names),
        "n_matched": n_matched,
        "n_observed": n_observed,
        "coverage": n_observed / max(1, len(cpg_ids)),
    }
    return x, info
