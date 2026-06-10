"""
Custom batch driver for ``extract-features-custom``.

Skips MONDO normalization: cohort folder name is used directly as the
key into the user mapping JSON. Plugin features are imported from an
optional user Python file BEFORE the extractor's column schema is
locked, so plugin csv_columns appear at the tail of the output CSV.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Set, Union

import numpy as np
import pandas as pd

from facekit.core.geometric.batch import (
    _discover_cohort_names,
    _iter_image_records,
    _iter_jsonl_records,
    _jsonl_has_any_matrix,
    _records_to_frame,
)
from facekit.core.geometric.extractor import GeometricFeatureExtractor
from facekit.core.geometric.feature_selector import (
    UserDiseaseSpec,
    load_user_disease_mapping,
    select_columns_for_user_disease,
)

logger = logging.getLogger(__name__)


def _import_user_features(path: Path) -> None:
    """Import the user plugin file by absolute path (triggers decorators)."""
    spec = importlib.util.spec_from_file_location(
        f"_facekit_user_features_{path.stem}", str(path)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load user plugin from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


def run_batch_custom(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    user_mapping_path: Union[str, Path],
    user_features_path: Optional[Union[str, Path]] = None,
    frontal_check: bool = False,
    model_path: Optional[Union[str, Path]] = None,
    hpo_codes_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Extract features under a user-supplied disease->feature mapping.

    Writes ``phenotypes_custom.csv`` and ``custom_selection_report.csv``
    to ``output_dir``. The CSV schema is base 125 features +
    plugin-registered cols (in registration order). Per-row, columns
    not selected for that row's cohort are NaN.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    user_mapping_path = Path(user_mapping_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    is_jsonl = input_path.is_file() and input_path.suffix.lower() == ".jsonl"
    is_dir = input_path.is_dir()
    if not (is_jsonl or is_dir):
        raise ValueError(
            f"input_path must be a directory or a .jsonl file, got {input_path!r}"
        )

    # --- Pre-extraction: register plugins, lock schema, build selection -----
    import facekit.api as _api

    if hpo_codes_path is not None:
        _api._HPO_CODES_PATH = Path(hpo_codes_path)

    if user_features_path is not None:
        _import_user_features(Path(user_features_path))

    plugin_cols = _api.plugin_columns_in_order()
    GeometricFeatureExtractor.extend_feature_columns(plugin_cols)
    feature_cols = list(GeometricFeatureExtractor.FEATURE_COLUMNS)

    user_mapping: Dict[str, UserDiseaseSpec] = load_user_disease_mapping(user_mapping_path)

    cohort_names = _discover_cohort_names(input_path)
    missing = [c for c in cohort_names if c not in user_mapping]
    if missing:
        raise ValueError(
            f"cohorts missing from user mapping {user_mapping_path}: {missing}"
        )

    if hpo_codes_path is None:
        # Family-expansion path needs the HPO CSV; if no feature_groups
        # are used by any cohort, skip the requirement.
        any_feature_groups = any(
            user_mapping[c].feature_groups for c in cohort_names
        )
        if any_feature_groups:
            raise ValueError(
                "hpo_codes_path is required when any cohort uses 'feature_groups' "
                "(family expansion reads hpo_direction_codes.csv)"
            )

    cohort_to_selected: Dict[str, Set[str]] = {}
    for cohort in cohort_names:
        cohort_to_selected[cohort] = select_columns_for_user_disease(
            user_mapping[cohort],
            feature_cols,
            hpo_codes_path=Path(hpo_codes_path) if hpo_codes_path else None,
            disease_name=cohort,
        )

    # --- Extraction --------------------------------------------------------
    extractor = GeometricFeatureExtractor(model_path=model_path)
    if is_jsonl:
        if frontal_check and not _jsonl_has_any_matrix(input_path):
            raise ValueError(
                f"frontal_check=True requires transformation_matrix in JSONL "
                f"records; none found in {input_path}"
            )
        record_iter = _iter_jsonl_records(extractor, input_path, frontal_check)
    else:
        record_iter = _iter_image_records(extractor, input_path, frontal_check)

    df = _records_to_frame(record_iter)

    # --- Row-wise NaN mask -------------------------------------------------
    if "disease" not in df.columns:
        raise ValueError("custom mode requires a 'disease' column on the dataframe")
    df = df.copy()
    for cohort, selected in cohort_to_selected.items():
        mask_cols = [c for c in feature_cols if c not in selected]
        if not mask_cols:
            continue
        row_mask = (df["disease"].astype(str) == cohort)
        if not row_mask.any():
            continue
        df.loc[row_mask, mask_cols] = np.nan

    # --- Write outputs -----------------------------------------------------
    out_csv = output_dir / "phenotypes_custom.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %d rows to %s", len(df), out_csv)

    report_path = output_dir / "custom_selection_report.csv"
    rows = []
    for cohort in sorted(cohort_to_selected):
        sel = cohort_to_selected[cohort]
        rows.append({
            "cohort_name": cohort,
            "n_columns_selected": len(sel),
            "columns_selected": ";".join(sorted(sel)),
        })
    pd.DataFrame(rows, columns=[
        "cohort_name", "n_columns_selected", "columns_selected",
    ]).to_csv(report_path, index=False)
    logger.info("Wrote custom selection report to %s", report_path)

    return df
