"""
Batch driver for geometric feature extraction.

Two input modes:
  * Image directory (cohort-aware: subfolders = cohorts; flat folder =
    single cohort named after the folder).
  * JSONL landmark file (one JSON object per line; schema below).

JSONL schema (per line)
-----------------------
    image_id              : str
    disease               : str (optional; default "")
    landmarks_3d          : list[list[float]] (478 x 3, normalized [0,1]
                            coordinates as MediaPipe returns)
    image_size            : list[int] [width, height]
    transformation_matrix : list[list[float]] (4x4, optional;
                            required if frontal_check is enabled)

Output: a single CSV with columns
    [disease, image_id, frontal_ok, pose_yaw, pose_pitch, pose_roll,
     <feature columns in stable order>]
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from facekit.core.geometric.extractor import GeometricFeatureExtractor

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _discover_cohort_names(input_path: Path) -> List[str]:
    """List cohort names without running extraction.

    For an image directory: subfolders are cohorts; flat folder is a single
    cohort named after the folder. For a JSONL file: scans the ``disease``
    field across every line. Used by disease-specific mode to abort BEFORE
    any MediaPipe inference if cohort resolution will fail.
    """
    if input_path.is_dir():
        sub = sorted(d.name for d in input_path.iterdir() if d.is_dir())
        return sub if sub else [input_path.name]

    if input_path.is_file() and input_path.suffix.lower() == ".jsonl":
        names: set = set()
        with open(input_path, "r") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                d = obj.get("disease")
                if d:
                    names.add(str(d))
        return sorted(names)

    raise ValueError(
        f"input_path must be a directory or a .jsonl file, got {input_path!r}"
    )


def _iter_image_records(
    extractor: GeometricFeatureExtractor,
    input_dir: Path,
    frontal_check: bool,
) -> Iterator[dict]:
    """Walk a directory (cohort-aware) and yield one record per image with a face."""
    disease_folders = sorted(d for d in input_dir.iterdir() if d.is_dir())
    if disease_folders:
        cohorts = [(d.name, d) for d in disease_folders]
    else:
        cohorts = [(input_dir.name, input_dir)]

    for disease_name, folder in cohorts:
        img_files = sorted(
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not img_files:
            logger.warning("No images found in %s", folder)
            continue

        for img_path in tqdm(img_files, desc=disease_name, leave=False):
            try:
                feats = extractor.extract(str(img_path), frontal_check=frontal_check)
            except Exception as e:
                logger.warning("Error processing %s: %s", img_path, e)
                continue
            if feats is None:
                continue
            record = {"disease": disease_name, "image_id": img_path.stem}
            record.update(feats)
            yield record


def _jsonl_has_any_matrix(jsonl_path: Path) -> bool:
    """Return True iff at least one JSONL line carries a transformation_matrix."""
    with open(jsonl_path, "r") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if obj.get("transformation_matrix") is not None:
                return True
    return False


def _iter_jsonl_records(
    extractor: GeometricFeatureExtractor,
    jsonl_path: Path,
    frontal_check: bool,
) -> Iterator[dict]:
    """Stream a JSONL file and yield one record per line with a valid face."""
    with open(jsonl_path, "r") as fh:
        for line_no, raw in enumerate(tqdm(fh, desc=jsonl_path.name, leave=False), start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed JSONL line %d: %s", line_no, e)
                continue

            image_id = obj.get("image_id")
            if image_id is None:
                logger.warning("Skipping JSONL line %d: missing image_id", line_no)
                continue
            disease = obj.get("disease", "")

            try:
                lm_norm = np.asarray(obj["landmarks_3d"], dtype=np.float64)
            except (KeyError, ValueError) as e:
                logger.warning("Skipping %s: bad landmarks (%s)", image_id, e)
                continue

            img_size = obj.get("image_size")
            if (not isinstance(img_size, (list, tuple))
                    or len(img_size) != 2
                    or not all(isinstance(v, (int, float)) for v in img_size)):
                logger.warning(
                    "Skipping %s: image_size must be [width, height] of numbers, got %r",
                    image_id, img_size,
                )
                continue
            w, h = img_size

            # Convert normalized -> pixel coordinates. The extractor's math
            # is 2D, so z is dropped silently.
            lm_px = lm_norm.copy()
            lm_px[:, 0] *= w
            lm_px[:, 1] *= h

            mat_raw = obj.get("transformation_matrix")
            matrix = np.asarray(mat_raw, dtype=np.float64) if mat_raw is not None else None

            if frontal_check and matrix is None:
                # Hard fail: per-row missing matrix when frontal_check is on
                # is a contract violation (start-of-batch scan should have
                # already caught zero-matrix files; reaching here means the
                # JSONL is inconsistent across rows).
                raise ValueError(
                    f"frontal_check=True but record {image_id!r} has no "
                    f"transformation_matrix; JSONL must include matrices on "
                    f"every row when frontal_check is enabled."
                )

            try:
                feats = extractor.extract_from_landmarks(
                    lm_px,
                    transformation_matrix=matrix,
                    frontal_check=frontal_check,
                )
            except ValueError:
                # Contract violations from extract_from_landmarks (e.g.
                # malformed landmarks shape) should abort the batch, not be
                # silently dropped.
                raise
            except Exception as e:
                logger.warning("Error processing %s: %s", image_id, e)
                continue
            if feats is None:
                continue
            record = {"disease": disease, "image_id": image_id}
            record.update(feats)
            yield record


LEADING_COLUMNS = [
    "disease", "image_id", "frontal_ok",
    "pose_yaw", "pose_pitch", "pose_roll",
]


def _records_to_frame(records: Iterable[dict]) -> pd.DataFrame:
    """Build a DataFrame with a deterministic column ordering.

    Order: ``LEADING_COLUMNS`` + ``GeometricFeatureExtractor.FEATURE_COLUMNS``
    (the canonical dispatch order from ``_compute_all``). Rows missing a
    column get NaN, so all-non-frontal batches still produce a full-width
    schema. Any unexpected extra keys (none today) are appended at the end
    in first-seen order to preserve forward compatibility.
    """
    records = list(records)
    feature_cols = list(GeometricFeatureExtractor._init_feature_columns())
    expected = set(LEADING_COLUMNS) | set(feature_cols)

    extra_cols: List[str] = []
    seen_extra: set = set()
    for rec in records:
        for k in rec:
            if k not in expected and k not in seen_extra:
                extra_cols.append(k)
                seen_extra.add(k)

    columns = LEADING_COLUMNS + feature_cols + extra_cols
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records, columns=columns)


_OUTPUT_FILENAMES = {
    "all": "phenotypes_all.csv",
    "disease-specific": "phenotypes_disease_specific.csv",
}


def run_batch(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    mode: str = "all",
    frontal_check: bool = False,
    blendshapes: bool = False,
    model_path: Optional[Union[str, Path]] = None,
    num_faces: int = 1,
    mondo_obo_path: Optional[Union[str, Path]] = None,
    feature_mapping_path: Optional[Union[str, Path]] = None,
    hpo_codes_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Run geometric feature extraction over a directory or JSONL file.

    :param input_path: image directory OR a ``.jsonl`` landmark file.
    :param output_dir: destination directory. The CSV is auto-named
        ``phenotypes_all.csv`` or ``phenotypes_disease_specific.csv``
        depending on ``mode``; Phase 2 also writes a side-artifact
        ``disease_specific_selection_report.csv`` here.
    :param mode: ``"all"`` or ``"disease-specific"``.
    :param frontal_check: skip faces with out-of-range yaw/pitch/roll.
    :param blendshapes: reserved for future use; currently a no-op.
    :param model_path: optional path to ``face_landmarker.task``.
    :param num_faces: max faces to detect per image (currently fixed to 1
        downstream; argument retained for forward compatibility).
    :param mondo_obo_path: path to local ``mondo.obo`` (Phase 2 only).
    :param feature_mapping_path: path to ``feature_disease_mapping.json``
        (Phase 2 only).
    :param hpo_codes_path: path to ``hpo_direction_codes.csv`` (Phase 2
        only).
    :returns: the assembled DataFrame.
    """
    if mode not in _OUTPUT_FILENAMES:
        raise ValueError(
            f"mode must be 'all' or 'disease-specific', got {mode!r}"
        )

    # blendshapes / num_faces are accepted for forward compatibility;
    # they are not used by the geometric pipeline today.
    del blendshapes, num_faces

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / _OUTPUT_FILENAMES[mode]

    is_jsonl = input_path.is_file() and input_path.suffix.lower() == ".jsonl"
    is_dir = input_path.is_dir()
    if not (is_jsonl or is_dir):
        raise ValueError(
            f"input_path must be a directory or a .jsonl file, got {input_path!r}"
        )

    # --- Pre-extraction: disease-specific resolution & selection -----------
    # Resolve cohorts BEFORE running MediaPipe so that a missing cohort or
    # missing JSON entry aborts in seconds, not after a full extraction pass.
    plan = None
    if mode == "disease-specific":
        from facekit.core.geometric.disease_specific import resolve_and_select
        cohort_names = _discover_cohort_names(input_path)
        plan = resolve_and_select(
            cohort_names,
            output_dir=output_dir,
            mondo_obo_path=mondo_obo_path,
            feature_mapping_path=feature_mapping_path,
            hpo_codes_path=hpo_codes_path,
        )

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

    if mode == "disease-specific":
        from facekit.core.geometric.disease_specific import (
            apply_disease_specific_mask,
        )
        df = apply_disease_specific_mask(df, plan, output_dir=output_dir)

    df.to_csv(output_csv, index=False)
    logger.info("Wrote %d rows to %s", len(df), output_csv)
    return df
