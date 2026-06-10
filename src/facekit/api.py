"""
Plugin registry for user-defined geometric features.

Used by the ``extract-features-custom`` command to let users add new
``(landmarks, scales) -> dict`` formulas without editing the system
extractor. Plugins register via the ``@register_feature`` decorator at
import time; the custom-batch driver imports the user file before
locking ``GeometricFeatureExtractor.FEATURE_COLUMNS`` so the new columns
appear in the canonical schema.

Path-injection contract
-----------------------
Conflict-checking against ``hpo_direction_codes.csv`` requires the
custom-batch driver to set ``facekit.api._HPO_CODES_PATH`` BEFORE
importing the user plugin file. If unset, the CSV-side conflict check
is skipped with a stderr warning (lets unit tests use the registry
without needing the CSV).
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Optional path to ``hpo_direction_codes.csv``; set by the custom-batch
# driver before importing user plugin files. Module-level for cheap
# access during decorator evaluation.
_HPO_CODES_PATH: Optional[Path] = None


@dataclass(frozen=True)
class PluginFeatureSpec:
    group: str
    func: Callable
    csv_columns: Tuple[str, ...]
    hpo_terms: Tuple[Tuple[str, str], ...]


_PLUGIN_REGISTRY: Dict[str, PluginFeatureSpec] = {}
_PLUGIN_COLUMN_ORDER: List[str] = []


def _read_hpo_feature_groups(path: Path) -> set:
    groups: set = set()
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            grp = (row.get("feature_group") or "").strip()
            if grp:
                groups.add(grp)
    return groups


def _validate_no_conflict(group: str, csv_columns: Tuple[str, ...]) -> None:
    """Raise ``ValueError`` if ``group`` or any csv_column collides.

    Checks (in order):
      1. Already-registered plugin group name.
      2. Already-registered plugin csv_columns.
      3. Base 125 feature columns (via ``GeometricFeatureExtractor``).
      4. Existing ``feature_group`` names in ``hpo_direction_codes.csv``
         (skipped with warning if ``_HPO_CODES_PATH`` is unset).
    """
    if group in _PLUGIN_REGISTRY:
        raise ValueError(
            f"plugin group {group!r} already registered "
            f"(existing csv_columns={list(_PLUGIN_REGISTRY[group].csv_columns)})"
        )
    existing_plugin_cols = set(_PLUGIN_COLUMN_ORDER)
    for col in csv_columns:
        if col in existing_plugin_cols:
            raise ValueError(
                f"plugin csv_column {col!r} already registered by another plugin"
            )

    # Local import to avoid circular dep — extractor.py will import api.py
    # for runtime dispatch; api.py imports extractor only inside this fn.
    from facekit.core.geometric.extractor import GeometricFeatureExtractor

    base_cols = set(GeometricFeatureExtractor._init_feature_columns())
    for col in csv_columns:
        if col in base_cols:
            raise ValueError(
                f"plugin csv_column {col!r} collides with a base feature column"
            )

    if _HPO_CODES_PATH is None:
        sys.stderr.write(
            "[FaceKit] warning: facekit.api._HPO_CODES_PATH unset; "
            "skipping feature_group conflict check against hpo_direction_codes.csv\n"
        )
        return
    hpo_groups = _read_hpo_feature_groups(Path(_HPO_CODES_PATH))
    if group in hpo_groups:
        raise ValueError(
            f"plugin group {group!r} collides with feature_group in "
            f"{_HPO_CODES_PATH}"
        )


def register_feature(
    *,
    group: str,
    csv_columns: List[str],
    hpo_terms: Optional[List[Tuple[str, str]]] = None,
) -> Callable:
    """Decorator: register a plugin feature function.

    The decorated function must have signature
    ``(lm: np.ndarray, scales: dict) -> dict[str, float]`` and return a
    dict whose keys equal ``csv_columns``. Validation of return-key
    consistency is the user's responsibility (mismatches surface as NaN
    at runtime).
    """
    if not isinstance(group, str) or not group:
        raise ValueError(f"group must be a non-empty str, got {group!r}")
    if not isinstance(csv_columns, list) or not csv_columns:
        raise ValueError(f"csv_columns must be a non-empty list, got {csv_columns!r}")
    for col in csv_columns:
        if not isinstance(col, str) or not col:
            raise ValueError(f"csv_columns entries must be non-empty str, got {col!r}")
    cols = tuple(csv_columns)
    terms = tuple((str(a), str(b)) for a, b in (hpo_terms or ()))

    def _decorator(func: Callable) -> Callable:
        _validate_no_conflict(group, cols)
        spec = PluginFeatureSpec(group=group, func=func, csv_columns=cols, hpo_terms=terms)
        _PLUGIN_REGISTRY[group] = spec
        _PLUGIN_COLUMN_ORDER.extend(cols)
        return func

    return _decorator


def get_registered() -> Dict[str, PluginFeatureSpec]:
    """Read-only view of the plugin registry."""
    return dict(_PLUGIN_REGISTRY)


def plugin_columns_in_order() -> List[str]:
    """Plugin csv_columns in registration order (for ``extend_feature_columns``)."""
    return list(_PLUGIN_COLUMN_ORDER)


def clear_registry() -> None:
    """Test helper: drop all registered plugins."""
    _PLUGIN_REGISTRY.clear()
    _PLUGIN_COLUMN_ORDER.clear()
