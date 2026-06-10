"""
Disease-specific masking pipeline.

Glues ``disease_resolver`` + ``feature_selector`` to the Phase 1 wide CSV.
The mask is applied row-wise: leading meta columns are preserved; feature
columns not selected for that row's cohort are NaN'd.

Failure policy: any cohort that fails to resolve to a MONDO id, OR that
resolves but has no entry in the JSON disease mapping, raises
``ValueError`` and aborts. The resolution check is intended to run BEFORE
extraction so that a missing cohort doesn't waste a full extraction pass;
``resolve_and_select`` is the entry point for that pre-extraction step,
``apply_disease_specific_mask`` is the post-extraction row-wise NaN-pad.
We do NOT silently produce partial output.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from facekit.core.geometric.disease_resolver import resolve_all
from facekit.core.geometric.extractor import GeometricFeatureExtractor
from facekit.core.geometric.feature_selector import (
    load_feature_disease_mapping,
    select_columns_for_disease,
)

logger = logging.getLogger(__name__)

LEADING_COLUMNS = (
    "disease", "image_id", "frontal_ok",
    "pose_yaw", "pose_pitch", "pose_roll",
)


@dataclass
class DiseaseSelectionPlan:
    """Pre-extraction plan: which columns survive for which cohort.

    Built by ``resolve_and_select`` from cohort folder names + the JSON
    disease mapping. Consumed by ``apply_disease_specific_mask`` after
    extraction to perform the row-wise NaN-pad and write the audit report.
    """

    cohort_to_mondo: Dict[str, str] = field(default_factory=dict)
    cohort_to_json: Dict[str, str] = field(default_factory=dict)
    cohort_to_selected: Dict[str, Set[str]] = field(default_factory=dict)


def _format_unresolved(label: str, names: List[str]) -> str:
    if not names:
        return ""
    bullet = "\n  - "
    return f"\n{label} ({len(names)}):" + bullet + bullet.join(sorted(names))


def resolve_and_select(
    cohort_names: List[str],
    *,
    output_dir: Path,
    mondo_obo_path: Optional[Union[str, Path]],
    feature_mapping_path: Optional[Union[str, Path]],
    hpo_codes_path: Optional[Union[str, Path]],
) -> DiseaseSelectionPlan:
    """Resolve cohorts + JSON keys and build the per-cohort selection map.

    This is intended to run BEFORE feature extraction starts. Any unresolved
    cohort or cohort-without-JSON-entry raises ``ValueError`` immediately so
    the run aborts before any MediaPipe inference happens.

    Writes / updates the ``disease_resolution.csv`` cache in ``output_dir``.
    Does NOT write the selection report (that happens post-extraction in
    ``apply_disease_specific_mask`` once the dataframe is in hand).
    """
    if mondo_obo_path is None or feature_mapping_path is None or hpo_codes_path is None:
        raise ValueError(
            "disease-specific mode requires mondo_obo_path, "
            "feature_mapping_path, and hpo_codes_path"
        )

    output_dir = Path(output_dir)
    mondo_obo_path = Path(mondo_obo_path)
    feature_mapping_path = Path(feature_mapping_path)
    hpo_codes_path = Path(hpo_codes_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "disease_resolution.csv"

    if not feature_mapping_path.exists():
        raise FileNotFoundError(f"feature mapping not found: {feature_mapping_path}")
    if not hpo_codes_path.exists():
        raise FileNotFoundError(f"hpo_codes not found: {hpo_codes_path}")

    cohort_names = sorted({str(c) for c in cohort_names if c})
    if not cohort_names:
        raise ValueError(
            "disease-specific mode found no cohort names to resolve "
            "(check input_path: subfolders should be named after cohorts, "
            "or JSONL records should carry a 'disease' field)."
        )

    disease_to_features = load_feature_disease_mapping(feature_mapping_path)
    json_keys = sorted(disease_to_features.keys())

    # Resolve folder names AND JSON keys; join on mondo_id.
    union_names = sorted(set(cohort_names) | set(json_keys))
    resolved, unresolved = resolve_all(
        union_names,
        mondo_obo_path=mondo_obo_path,
        cache_path=cache_path,
    )

    unresolved_set = set(unresolved)
    cohort_unresolved = [n for n in cohort_names if n in unresolved_set]
    json_unresolved = [n for n in json_keys if n in unresolved_set]

    # MONDO ids that *would* have matched if their JSON-side name had resolved
    # cleanly. We can't compute these directly (the unresolved name has no
    # MONDO id) — so the only signal we have is "JSON-side name failed
    # resolution". When attributing a cohort failure, we cannot say which
    # specific JSON entry would have matched, but we can at least split the
    # failure bucket into "JSON has no entry for your disease" vs "JSON has
    # entries but some failed to resolve, so the gap may be a JSON-side
    # cleanup issue, not a missing folder."
    json_unresolved_count = len(json_unresolved)

    cohort_to_mondo: Dict[str, str] = {
        n: resolved[n].mondo_id for n in cohort_names if n in resolved
    }
    mondo_to_json: Dict[str, str] = {}
    for jkey in json_keys:
        if jkey not in resolved:
            continue
        mid = resolved[jkey].mondo_id
        # First-write wins (deterministic because json_keys is sorted).
        mondo_to_json.setdefault(mid, jkey)

    # Per-cohort mapping & failure detection. Split the "no JSON entry"
    # bucket by whether any JSON-side names failed to resolve.
    cohort_to_json: Dict[str, str] = {}
    cohorts_no_json_clean: List[Tuple[str, str]] = []
    cohorts_no_json_jsonside: List[Tuple[str, str]] = []
    for cohort, mid in cohort_to_mondo.items():
        jkey = mondo_to_json.get(mid)
        if jkey is None:
            if json_unresolved_count > 0:
                cohorts_no_json_jsonside.append((cohort, mid))
            else:
                cohorts_no_json_clean.append((cohort, mid))
        else:
            cohort_to_json[cohort] = jkey

    if cohort_unresolved or cohorts_no_json_clean or cohorts_no_json_jsonside:
        msg_parts = ["disease-specific mode aborting: cohort resolution failures."]
        if cohort_unresolved:
            msg_parts.append(_format_unresolved(
                "Cohorts unresolved to any MONDO id (folder name -> MONDO failed)",
                cohort_unresolved,
            ))
        if cohorts_no_json_clean:
            details = [f"{c} -> {m} (no JSON entry for this disease)"
                       for c, m in cohorts_no_json_clean]
            msg_parts.append(_format_unresolved(
                "Cohorts resolved cleanly but no JSON mapping entry exists",
                details,
            ))
        if cohorts_no_json_jsonside:
            details = [
                f"{c} -> {m} (folder is fine; JSON-side disease name failed "
                f"MONDO lookup -- JSON cleanup needed)"
                for c, m in cohorts_no_json_jsonside
            ]
            msg_parts.append(_format_unresolved(
                "Cohorts may match a JSON entry whose own name failed to resolve",
                details,
            ))
        if json_unresolved:
            msg_parts.append(_format_unresolved(
                "JSON disease keys themselves unresolved (fix these in the "
                "JSON mapping or the resolution cache)",
                json_unresolved,
            ))
        raise ValueError("".join(msg_parts))

    # Build per-cohort selected column sets.
    feature_cols: List[str] = list(GeometricFeatureExtractor._init_feature_columns())
    cohort_to_selected: Dict[str, Set[str]] = {}
    for cohort, jkey in cohort_to_json.items():
        sel = select_columns_for_disease(
            disease_to_features[jkey],
            feature_cols,
            hpo_codes_path=hpo_codes_path,
        )
        cohort_to_selected[cohort] = sel

    return DiseaseSelectionPlan(
        cohort_to_mondo=cohort_to_mondo,
        cohort_to_json=cohort_to_json,
        cohort_to_selected=cohort_to_selected,
    )


def apply_disease_specific_mask(
    df: pd.DataFrame,
    plan: DiseaseSelectionPlan,
    *,
    output_dir: Path,
) -> pd.DataFrame:
    """Apply ``plan`` to ``df`` row-wise and write the audit report.

    Expects the resolution + selection step (``resolve_and_select``) to have
    already succeeded. Writes ``disease_specific_selection_report.csv`` to
    ``output_dir``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "disease" not in df.columns:
        raise ValueError("disease-specific mode requires a 'disease' column")

    feature_cols: List[str] = list(GeometricFeatureExtractor._init_feature_columns())
    df = df.copy()
    for cohort, selected in plan.cohort_to_selected.items():
        mask_cols = [c for c in feature_cols if c not in selected]
        if not mask_cols:
            continue
        row_mask = (df["disease"].astype(str) == cohort)
        if not row_mask.any():
            continue
        df.loc[row_mask, mask_cols] = np.nan

    # Write the audit report.
    report_path = output_dir / "disease_specific_selection_report.csv"
    report_rows = []
    for cohort in sorted(plan.cohort_to_json):
        jkey = plan.cohort_to_json[cohort]
        sel = plan.cohort_to_selected[cohort]
        report_rows.append({
            "cohort_name": cohort,
            "mondo_id": plan.cohort_to_mondo[cohort],
            "json_disease_key": jkey,
            "n_columns_selected": len(sel),
            "columns_selected": ";".join(sorted(sel)),
        })
    pd.DataFrame(report_rows, columns=[
        "cohort_name", "mondo_id", "json_disease_key",
        "n_columns_selected", "columns_selected",
    ]).to_csv(report_path, index=False)
    logger.info("Wrote disease-specific selection report to %s", report_path)

    return df
