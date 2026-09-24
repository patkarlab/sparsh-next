"""
SPARSH-next: reporting at the level the model is confident about.

The class hierarchy (configs/class_hierarchy.json) places every model class in a
lineage (AML, B-ALL, T-ALL, ...) and, optionally, in one family of related
subtypes. Probabilities of a family or a lineage are the sums of its classes'
calibrated probabilities. Each sample is reported at the most specific level
that reaches the threshold:

  subtype     the top class reaches the threshold
  no_subtype  the top class reaches it but is a background class such as
              AML_other: reported as "<lineage>, no specific subtype"
  family      no class does, but the summed probability of a family does
  lineage     no family does, but a lineage does: "<lineage>, subtype undetermined"
  none        nothing reaches the threshold (or coverage is too low)

A reported call is correct when the true class is the reported subtype, lies in
the reported family (any configured member, including classes the model lacks),
or belongs to the reported lineage. For no_subtype calls,
reported_correct requires the true class to be that background class (strict),
and lineage_correct records whether at least the lineage is right. True labels
that are not model classes still get a lineage from the prefix rules, so
out-of-scheme samples can be scored at the lineage level.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LEVELS = ("subtype", "no_subtype", "family", "lineage", "none")


class Hierarchy:
    def __init__(self, classes: Sequence[str], lineage_of: Dict[str, str], family_of: Dict[str, str],
                 families: Dict[str, List[str]], no_subtype: Sequence[str], prefix_rules: List[List[str]],
                 overrides: Dict[str, str], source: str = "", family_members: Optional[Dict[str, List[str]]] = None):
        self.classes = list(classes)
        self.lineage_of = lineage_of
        self.family_of = family_of
        self.families = families                         # members this model has: summed for the family call
        self.family_members = family_members or families  # every configured member: used to score a family call
        self.no_subtype = set(no_subtype)
        self.prefix_rules = prefix_rules
        self.overrides = overrides
        self.source = source
        self.lineages = list(dict.fromkeys(lineage_of[c] for c in self.classes))

    def lineage(self, name: str) -> str:
        """Lineage of any label, including labels that are not model classes (prefix rules)."""
        return _lineage_for(str(name), self.overrides, self.prefix_rules) or str(name)

    def describe(self) -> str:
        lines = [f"Class hierarchy from {self.source or 'defaults'}:"]
        for lin in self.lineages:
            members = [c for c in self.classes if self.lineage_of[c] == lin]
            lines.append(f"  {lin}: {len(members)} classes")
        for fam, members in self.families.items():
            lines.append(f"  family '{fam}': {', '.join(members)}")
        if self.no_subtype:
            lines.append(f"  reported as '<lineage>, no specific subtype': {', '.join(sorted(self.no_subtype))}")
        return "\n".join(lines)


def _lineage_for(name: str, overrides: Dict[str, str], prefix_rules: List[List[str]]) -> Optional[str]:
    if name in overrides:
        return overrides[name]
    for prefix, lineage in prefix_rules:
        if name.startswith(prefix):
            return lineage
    return None


def load_hierarchy(path: Optional[str], classes: Sequence[str]) -> Hierarchy:
    """
    Read the hierarchy file and fit it to a model's classes. Family members that are not
    model classes (for example a class left out of this model) are ignored with a note;
    a class in two families, or a family whose members are of different lineages, is an error.
    With path None every class is its own lineage group by prefix and there are no families.
    """
    content = {}
    if path is not None:
        with open(path) as f:
            content = json.load(f)
    prefix_rules = [[str(p), str(lin)] for p, lin in content.get("lineage_by_prefix", [])]
    overrides = {str(k): str(v) for k, v in content.get("lineage", {}).items()}
    lineage_of = {}
    for c in classes:
        lin = _lineage_for(c, overrides, prefix_rules)
        if lin is None:
            logger.warning(f"  hierarchy: no lineage rule for class {c!r}; it forms its own lineage group")
            lin = c
        lineage_of[c] = lin

    family_of, families, family_members = {}, {}, {}
    present = set(classes)
    for fam, members in content.get("families", {}).items():
        kept = [m for m in members if m in present]
        dropped = [m for m in members if m not in present]
        if dropped:
            logger.info(f"  hierarchy: family '{fam}': {dropped} not in this model, ignored")
        if len(kept) < 2:
            logger.info(f"  hierarchy: family '{fam}' has fewer than two classes in this model, ignored")
            continue
        lins = {lineage_of[m] for m in kept}
        if len(lins) > 1:
            raise ValueError(f"family '{fam}' mixes lineages {sorted(lins)}; a family must sit within one lineage")
        for m in kept:
            if m in family_of:
                raise ValueError(f"class {m!r} is in two families: '{family_of[m]}' and '{fam}'")
            family_of[m] = fam
        families[fam] = kept
        family_members[fam] = [str(m) for m in members]

    no_subtype = [c for c in content.get("no_subtype_classes", []) if c in present]
    return Hierarchy(classes, lineage_of, family_of, families, no_subtype, prefix_rules, overrides,
                     source=str(path) if path else "", family_members=family_members)


def _group_sums(probs: np.ndarray, classes: Sequence[str], group_of: Dict[str, str]):
    names = list(dict.fromkeys(group_of[c] for c in classes if c in group_of))
    if not names:
        return names, np.zeros((len(probs), 0))
    index = {g: i for i, g in enumerate(names)}
    member = np.zeros((len(classes), len(names)))
    for j, c in enumerate(classes):
        if c in group_of:
            member[j, index[group_of[c]]] = 1.0
    return names, probs @ member


def hierarchical_calls(probs: np.ndarray, classes: Sequence[str], h: Hierarchy, threshold: float,
                       callable_mask: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    One row per sample: the reported call and level; the family (with its summed probability)
    that was reported, or else the family of the top class, if any; the most probable lineage.
    """
    probs = np.asarray(probs, dtype=float)
    n = len(probs)
    top = probs.argmax(axis=1)
    top_p = probs[np.arange(n), top]
    fam_names, fam_p = _group_sums(probs, classes, h.family_of)
    fam_index = {f: j for j, f in enumerate(fam_names)}
    lin_names, lin_p = _group_sums(probs, classes, h.lineage_of)
    best_fam = fam_p.argmax(axis=1) if fam_p.shape[1] else np.zeros(n, dtype=int)
    best_lin = lin_p.argmax(axis=1)

    out = {"family": [], "family_prob": [], "lineage": [], "lineage_prob": [], "reported_call": [],
           "reported_level": []}
    for i in range(n):
        cls = classes[top[i]]
        own = h.family_of.get(cls)
        fname, fprob = (own, float(fam_p[i, fam_index[own]])) if own else ("", float("nan"))
        lname, lprob = lin_names[best_lin[i]], float(lin_p[i, best_lin[i]])
        best_name = fam_names[best_fam[i]] if fam_p.shape[1] else ""
        best_prob = float(fam_p[i, best_fam[i]]) if fam_p.shape[1] else float("nan")
        if callable_mask is not None and not callable_mask[i]:
            call, level = "not callable (coverage)", "none"
        elif top_p[i] >= threshold and cls in h.no_subtype:
            call, level = f"{h.lineage_of[cls]}, no specific subtype", "no_subtype"
        elif top_p[i] >= threshold:
            call, level = cls, "subtype"
        elif fam_p.shape[1] and best_prob >= threshold:
            call, level = best_name, "family"
            fname, fprob = best_name, best_prob
        elif lprob >= threshold:
            call, level = f"{lname}, subtype undetermined", "lineage"
        else:
            call, level = "not callable", "none"
        for key, value in zip(out, (fname, fprob, lname, lprob, call, level)):
            out[key].append(value)
    return pd.DataFrame(out)


def score_calls(true_names: Sequence[str], predictions: Sequence[str], calls: pd.DataFrame,
                h: Hierarchy) -> pd.DataFrame:
    """reported_correct and lineage_correct per sample (NaN where nothing was reported)."""
    rc, lc = [], []
    for t, p, lin, level, fam in zip(true_names, predictions, calls["lineage"], calls["reported_level"],
                                     calls["family"]):
        t = str(t)
        if level == "none":
            rc.append(np.nan)
            lc.append(np.nan)
            continue
        if level in ("subtype", "no_subtype"):
            rc.append(float(t == p))
            lc.append(float(h.lineage(t) == h.lineage_of.get(p, h.lineage(p))))
        elif level == "family":
            # every configured member counts, including classes this model lacks (the family name lists them)
            rc.append(float(t in h.family_members.get(fam, [])))
            lc.append(float(h.lineage(t) == h.lineage_of[h.families[fam][0]]))
        else:
            rc.append(float(h.lineage(t) == lin))
            lc.append(rc[-1])
    return pd.DataFrame({"reported_correct": rc, "lineage_correct": lc})


def summarize_calls(calls: pd.DataFrame, scored: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """Share of samples at each level; with scores, accuracy of what was reported."""
    n = len(calls)
    level = calls["reported_level"]
    out = {"n": int(n)}
    for lev in LEVELS:
        out[f"share_{lev}"] = float((level == lev).mean()) if n else float("nan")
    out["share_reported_any_level"] = float((level != "none").mean()) if n else float("nan")
    if scored is not None:
        rep = level != "none"
        out["accuracy_reported"] = float(scored.loc[rep, "reported_correct"].mean()) if rep.any() else float("nan")
        out["lineage_accuracy_reported"] = float(scored.loc[rep, "lineage_correct"].mean()) if rep.any() else float("nan")
        for lev in ("subtype", "no_subtype", "family", "lineage"):
            m = level == lev
            out[f"accuracy_{lev}"] = float(scored.loc[m, "reported_correct"].mean()) if m.any() else float("nan")
    return out


def default_hierarchy_path() -> Optional[Path]:
    path = Path(__file__).resolve().parent.parent / "configs" / "class_hierarchy.json"
    return path if path.exists() else None
