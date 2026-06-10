"""
Disease-specific column selection for the geometric extractor.

Layered logic:
  feature_disease_mapping.json[<disease>].features  (list of UPPER_SNAKE feature_groups)
    -> hpo_direction_codes.csv[feature_group -> csv_column]
      -> family expansion (same prefix after stripping {_r,_l,_mean,_asym,_x,_y})
        -> intersect with the extractor's actual FEATURE_COLUMNS
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Suffixes recognized for family expansion. Order matters: longer suffixes
# must be tested before shorter ones that are prefixes of them (none today,
# but kept consistent with future additions).
_FAMILY_SUFFIXES = ("_mean", "_asym", "_r", "_l", "_x", "_y")

# A single suffix-stripping regex builds the stem.
_SUFFIX_RE = re.compile(
    r"(?:" + "|".join(re.escape(s) for s in _FAMILY_SUFFIXES) + r")$"
)


def _strip_suffix(col: str) -> str:
    """Remove one trailing family suffix; return ``col`` unchanged if none."""
    return _SUFFIX_RE.sub("", col)


# ----------------------------------------------------------------------
# Loading & indexing
# ----------------------------------------------------------------------

def load_feature_disease_mapping(path: Path) -> Dict[str, List[str]]:
    """Return ``{disease_name: [feature_group, ...]}`` from the JSON file."""
    with open(path, "r") as fh:
        data = json.load(fh)
    by_disease = data.get("by_disease", {})
    return {
        name: list(info.get("features", []))
        for name, info in by_disease.items()
    }


def build_feature_group_to_columns(hpo_codes_path: Path) -> Dict[str, Set[str]]:
    """``feature_group -> {csv_column, ...}`` from ``hpo_direction_codes.csv``.

    ``NOT_APPLICABLE`` is excluded entirely (per PLAN).
    """
    out: Dict[str, Set[str]] = {}
    with open(hpo_codes_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            grp = (row.get("feature_group") or "").strip()
            col = (row.get("csv_column") or "").strip()
            if not grp or not col:
                continue
            if grp == "NOT_APPLICABLE":
                continue
            out.setdefault(grp, set()).add(col)
    return out


def build_family_expansion(all_feature_columns: List[str]) -> Dict[str, Set[str]]:
    """Build a ``csv_column -> {family columns}`` map.

    A "family" is the set of columns sharing the stem (col with one
    family suffix stripped). Each column maps to its own family — including
    itself. Columns with a single representative (e.g. ``mouth_width``) map
    to ``{"mouth_width"}``.
    """
    # 1. Group columns by stem (= column with at most one family suffix stripped).
    stem_to_cols: Dict[str, Set[str]] = {}
    for col in all_feature_columns:
        stem = _strip_suffix(col)
        # Both the stem-as-stem and the column itself are members of the family.
        stem_to_cols.setdefault(stem, set()).add(col)
        # A "stem" that itself exists as a column counts too (e.g. nose_length).
        if stem in all_feature_columns:
            stem_to_cols[stem].add(stem)

    # 2. For each column, pick its family by stem.
    expansion: Dict[str, Set[str]] = {}
    for col in all_feature_columns:
        stem = _strip_suffix(col)
        expansion[col] = set(stem_to_cols.get(stem, {col}))
    return expansion


# Cache (compute once per session). Keyed by the absolute path of the inputs
# AND the tuple of all_feature_columns (extractor's canonical column list).
@lru_cache(maxsize=8)
def _cached_indexes(
    hpo_codes_path_str: str,
    feature_columns_key: Tuple[str, ...],
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    fg_to_cols = build_feature_group_to_columns(Path(hpo_codes_path_str))
    family = build_family_expansion(list(feature_columns_key))
    return fg_to_cols, family


def _get_indexes(
    hpo_codes_path: Path,
    all_feature_columns: List[str],
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    return _cached_indexes(
        str(Path(hpo_codes_path).resolve()),
        tuple(all_feature_columns),
    )


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def select_columns_for_disease(
    disease_features: List[str],
    all_feature_columns: List[str],
    *,
    hpo_codes_path: Path,
) -> Set[str]:
    """Return the set of CSV columns to keep for one disease.

    :param disease_features: ``by_disease[<name>].features`` — UPPER_SNAKE
        feature_group names from ``feature_disease_mapping.json``.
    :param all_feature_columns: the extractor's canonical column list
        (e.g. ``GeometricFeatureExtractor.FEATURE_COLUMNS``).
    :param hpo_codes_path: path to ``hpo_direction_codes.csv``.
    """
    fg_to_cols, family = _get_indexes(hpo_codes_path, all_feature_columns)
    all_set = set(all_feature_columns)
    keep: Set[str] = set()
    for grp in disease_features:
        if grp == "NOT_APPLICABLE":
            continue
        rep_cols = fg_to_cols.get(grp)
        if not rep_cols:
            # Group present in JSON but absent from HPO CSV — silently skip,
            # consistent with how unmatched groups can't contribute columns.
            continue
        for rep in rep_cols:
            keep.update(family.get(rep, {rep}))
    return keep & all_set


# ----------------------------------------------------------------------
# User-customizable mapping (extract-features-custom)
# ----------------------------------------------------------------------

@dataclass
class UserDiseaseSpec:
    """One disease entry from a user mapping JSON.

    All three fields are optional individually, but at least one must be
    non-empty per disease (validated in ``load_user_disease_mapping``).
    """

    feature_groups: Tuple[str, ...] = ()
    extra_columns: Tuple[str, ...] = ()
    custom_features: Tuple[str, ...] = ()


def load_user_disease_mapping(path: Path) -> Dict[str, UserDiseaseSpec]:
    """Parse a user mapping JSON into ``{cohort_name: UserDiseaseSpec}``.

    Schema::

        {"by_disease": {"<cohort>": {"feature_groups": [...],
                                     "extra_columns": [...],
                                     "custom_features": [...]}}}

    Each disease must have at least one of the three fields non-empty.
    """
    with open(path, "r") as fh:
        data = json.load(fh)
    by_disease = data.get("by_disease", {})
    if not isinstance(by_disease, dict):
        raise ValueError(f"{path}: top-level 'by_disease' must be an object")
    out: Dict[str, UserDiseaseSpec] = {}
    for name, info in by_disease.items():
        if not isinstance(info, dict):
            raise ValueError(f"{path}: by_disease[{name!r}] must be an object")
        fg = tuple(info.get("feature_groups") or ())
        xc = tuple(info.get("extra_columns") or ())
        cf = tuple(info.get("custom_features") or ())
        if not (fg or xc or cf):
            raise ValueError(
                f"{path}: by_disease[{name!r}] must have at least one of "
                f"'feature_groups', 'extra_columns', 'custom_features' non-empty"
            )
        out[name] = UserDiseaseSpec(feature_groups=fg, extra_columns=xc, custom_features=cf)
    return out


def select_columns_for_user_disease(
    spec: UserDiseaseSpec,
    all_feature_columns: List[str],
    *,
    hpo_codes_path: Optional[Path],
    disease_name: Optional[str] = None,
) -> Set[str]:
    """Return the CSV columns to keep for one user-defined disease.

    Combines three paths:
      1. Family-expanded ``feature_groups`` (via ``select_columns_for_disease``).
         Requires ``hpo_codes_path``; if it is ``None`` and the spec has
         non-empty ``feature_groups``, raises ``ValueError``.
      2. Raw ``extra_columns`` (must already exist in ``all_feature_columns``).
      3. Plugin csv_columns from ``custom_features`` (looked up in the
         plugin registry; the registry must already contain them).

    :param disease_name: optional cohort label used only for clearer
        error messages when ``hpo_codes_path`` is missing.
    """
    from facekit.api import get_registered

    keep: Set[str] = set()
    if spec.feature_groups:
        if hpo_codes_path is None:
            who = f" (got: {disease_name})" if disease_name else ""
            raise ValueError(
                f"hpo_codes_path is required when a disease uses "
                f"'feature_groups'{who}"
            )
        keep |= select_columns_for_disease(
            list(spec.feature_groups),
            all_feature_columns,
            hpo_codes_path=hpo_codes_path,
        )

    all_set = set(all_feature_columns)
    unknown_extras = [c for c in spec.extra_columns if c not in all_set]
    if unknown_extras:
        raise ValueError(
            f"unknown extra_columns (not in extractor schema): {unknown_extras}"
        )
    keep.update(spec.extra_columns)

    registry = get_registered()
    for grp in spec.custom_features:
        plugin = registry.get(grp)
        if plugin is None:
            raise ValueError(
                f"custom_features entry {grp!r} not found in plugin registry "
                f"(available: {sorted(registry)})"
            )
        keep.update(plugin.csv_columns)
    return keep
