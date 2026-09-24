"""
SPARSH-next: names of the evaluation conditions and the summary rows built on them.

A condition is one way of presenting the outer-fold samples to a model:
  dense                  the array profile as measured
  <sim>_<f>              simulation <sim> at observed fraction f, e.g. binary_0.30
  <sim>-err<pct>_<f>     the same with a per-read call error of <pct> percent,
                         e.g. binary-err10_0.30 (see models/corruption.py)
Names without an error part are the ones earlier runs used, so tables of old
and new runs line up.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from models.corruption import READ_SIMS


def condition_name(sim: str, fraction: float, call_error: float = 0.0) -> str:
    label = sim if not call_error else f"{sim}-err{round(100.0 * float(call_error), 4):g}"
    return f"{label}_{float(fraction):.2f}"


def split_condition(name: str) -> Tuple[str, Optional[float]]:
    """'binary-err10_0.30' -> ('binary-err10', 0.30); 'dense' -> ('dense', None)."""
    label, _, cov = str(name).rpartition("_")
    try:
        return label, float(cov)
    except ValueError:
        return str(name), None


def parse_label(label: str) -> Tuple[str, float]:
    """'binary-err10' -> ('binary', 0.10); 'binary' -> ('binary', 0.0)."""
    sim, sep, err = str(label).partition("-err")
    if not sep:
        return sim, 0.0
    try:
        return sim, float(err) / 100.0
    except ValueError:
        return str(label), 0.0


def parse_condition(name: str) -> Tuple[str, float, Optional[float]]:
    """(simulation, call error, observed fraction) of a condition name."""
    label, cov = split_condition(name)
    sim, err = parse_label(label)
    return sim, err, cov


def is_nanopore(name: str) -> bool:
    """True for conditions that simulate nanopore reads (reads, binary, oneread; any call error)."""
    sim, _, cov = parse_condition(name)
    return sim in READ_SIMS and cov is not None


def nanopore_labels(names) -> list:
    """Distinct '<sim>' or '<sim>-err<pct>' labels among the nanopore conditions, in a stable order."""
    labels = {split_condition(n)[0] for n in names if is_nanopore(n)}
    return sorted(labels, key=lambda lab: (READ_SIMS.index(parse_label(lab)[0]), parse_label(lab)[1]))


def add_summary_rows(table: pd.DataFrame, ont_cov, with_mean: bool = True) -> pd.DataFrame:
    """
    For every nanopore label (simulation and call error), add the mean over its coverages and,
    if ont_cov (observed fractions of real samples) is given, the value expected on those samples:
    each sample's coverage is placed between the two nearest evaluated coverages and the metric
    interpolated there (the nearest end outside the evaluated range). Only coverage is used.
    """
    rows = {}
    for label in nanopore_labels(table.index):
        members = [(c, split_condition(c)[1]) for c in table.index
                   if is_nanopore(c) and split_condition(c)[0] == label]
        members.sort(key=lambda m: m[1])
        names, covs = [m[0] for m in members], np.array([m[1] for m in members])
        if with_mean:
            rows[f"mean over {label}_*"] = table.loc[names].mean()
        if ont_cov is not None and len(ont_cov):
            expected = {}
            for run in table.columns:
                values = table.loc[names, run].to_numpy(dtype=float)
                ok = np.isfinite(values)
                expected[run] = float(np.mean(np.interp(ont_cov, covs[ok], values[ok]))) if ok.any() else np.nan
            rows[f"your ONT samples, {label}_*"] = pd.Series(expected)
    for label, row in rows.items():
        table.loc[label] = row
    return table


def read_ont_coverage(path: str) -> np.ndarray:
    """Observed fractions (0-1) from the CSV written by scripts/ont_coverage.py."""
    oc = pd.read_csv(path)
    if "coverage_pct" not in oc.columns:
        raise ValueError(f"{path} has no coverage_pct column (write it with scripts/ont_coverage.py)")
    return pd.to_numeric(oc["coverage_pct"], errors="coerce").dropna().to_numpy() / 100.0
