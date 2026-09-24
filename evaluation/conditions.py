"""
SPARSH-next: names of the evaluation conditions and the summary rows built on them.

A condition is one way of presenting the outer-fold samples to a model:
  dense                         the array profile as measured
  <sim>_<f>                     simulation <sim> at observed fraction f, e.g. binary_0.30
  <sim>-err<pct>_<f>            the same with a per-read call error of <pct> percent,
                                e.g. binary-err10_0.30 (see models/corruption.py)
  <sim>-blast<pct>_<f>          leukaemia samples diluted with normal marrow to <pct>
                                percent blasts before the reads are simulated, e.g.
                                binary-blast30_0.30 (see models/dilution.py)
  <sim>-err<pct>-blast<pct>_<f> both
Names without these parts are the ones earlier runs used, so tables of old and
new runs line up.
"""

from typing import NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from models.corruption import READ_SIMS


class ConditionParts(NamedTuple):
    sim: str
    call_error: float
    blast: float
    fraction: Optional[float]


def _pct(value: float) -> str:
    return f"{round(100.0 * float(value), 4):g}"


def condition_name(sim: str, fraction: float, call_error: float = 0.0, blast: float = 1.0) -> str:
    label = sim
    if call_error:
        label += f"-err{_pct(call_error)}"
    if blast < 1.0:
        label += f"-blast{_pct(blast)}"
    return f"{label}_{float(fraction):.2f}"


def split_condition(name: str) -> Tuple[str, Optional[float]]:
    """'binary-err10_0.30' -> ('binary-err10', 0.30); 'dense' -> ('dense', None)."""
    label, _, cov = str(name).rpartition("_")
    try:
        return label, float(cov)
    except ValueError:
        return str(name), None


def parse_label(label: str) -> Tuple[str, float, float]:
    """'binary-err10-blast30' -> ('binary', 0.10, 0.30); 'binary' -> ('binary', 0.0, 1.0)."""
    parts = str(label).split("-")
    sim, err, blast = parts[0], 0.0, 1.0
    # simulation names contain no '-'; anything else keeps the whole label as the simulation
    try:
        for part in parts[1:]:
            if part.startswith("err"):
                err = float(part[3:]) / 100.0
            elif part.startswith("blast"):
                blast = float(part[5:]) / 100.0
            else:
                return str(label), 0.0, 1.0
    except ValueError:
        return str(label), 0.0, 1.0
    return sim, err, blast


def parse_condition(name: str) -> ConditionParts:
    """Simulation, call error, blast fraction and observed fraction of a condition name."""
    label, cov = split_condition(name)
    sim, err, blast = parse_label(label)
    return ConditionParts(sim, err, blast, cov)


def is_nanopore(name: str) -> bool:
    """True for conditions that simulate nanopore reads (reads, binary, oneread; any call error or dilution)."""
    p = parse_condition(name)
    return p.sim in READ_SIMS and p.fraction is not None


def nanopore_labels(names) -> list:
    """Distinct labels ('binary', 'binary-err10', 'binary-blast30', ...) among the nanopore conditions."""
    labels = {split_condition(n)[0] for n in names if is_nanopore(n)}

    def key(lab):
        sim, err, blast = parse_label(lab)
        return READ_SIMS.index(sim), err, -blast

    return sorted(labels, key=key)


def add_summary_rows(table: pd.DataFrame, ont_cov, with_mean: bool = True) -> pd.DataFrame:
    """
    For every nanopore label (simulation, call error, dilution), add the mean over its coverages and,
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
