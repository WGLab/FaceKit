"""End-to-end smoke + fail-fast tests for extract-features-custom."""
from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import facekit.api as api
from facekit.core.geometric.custom_batch import run_batch_custom
from facekit.core.geometric.extractor import GeometricFeatureExtractor


DEFAULT_IMAGES = (
    "/vast/projects/kai/multimodal-machine-learn/hongzhuo/mm_fusion_top10/"
    "train_images_top50/ADNP-related_multiple_congenital_anomalies_-_"
    "intellectual_disability_-_autism_spectrum_disorder"
)
DEFAULT_MODEL = (
    "/vast/projects/kai/multimodal-machine-learn/hongzhuo/mm_fusion_top10/"
    "preprocess_scripts/face_landmarker.task"
)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@pytest.fixture(autouse=True)
def _reset_state():
    api.clear_registry()
    api._HPO_CODES_PATH = None
    GeometricFeatureExtractor.FEATURE_COLUMNS = None
    yield
    api.clear_registry()
    api._HPO_CODES_PATH = None
    GeometricFeatureExtractor.FEATURE_COLUMNS = None


def _hpo_csv(tmp_path: Path) -> Path:
    p = tmp_path / "hpo.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "hpo_id", "hpo_name", "feature_group", "csv_column",
            "expected_direction", "confidence", "rationale", "n_diseases",
        ])
        w.writerow(["HP:0000154", "Wide mouth", "MOUTH_WIDTH", "mouth_width", "1", "HIGH", "", "0"])
    return p


def _stage_cohort(tmp_path: Path, cohort_name: str, n_imgs: int = 5) -> Path:
    images_dir = Path(os.environ.get("FACEKIT_PARITY_IMAGES", DEFAULT_IMAGES))
    if not images_dir.is_dir():
        pytest.skip(f"image dir missing: {images_dir}")
    src_imgs = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)[:n_imgs]
    if not src_imgs:
        pytest.skip(f"no images in {images_dir}")
    in_root = tmp_path / "in"
    cohort = in_root / cohort_name
    cohort.mkdir(parents=True)
    for p in src_imgs:
        shutil.copy2(p, cohort / p.name)
    return in_root


def test_cohort_not_in_mapping_fails_fast(tmp_path):
    in_root = _stage_cohort(tmp_path, "cohort_A")
    mapping = {"by_disease": {"different_cohort": {"feature_groups": ["MOUTH_WIDTH"]}}}
    map_path = tmp_path / "m.json"
    map_path.write_text(json.dumps(mapping))

    with pytest.raises(ValueError, match="missing from user mapping"):
        run_batch_custom(
            input_path=in_root,
            output_dir=tmp_path / "out",
            user_mapping_path=map_path,
            hpo_codes_path=_hpo_csv(tmp_path),
        )


def test_end_to_end_with_plugin(tmp_path):
    model_path = Path(os.environ.get("FACEKIT_PARITY_MODEL", DEFAULT_MODEL))
    if not model_path.is_file():
        pytest.skip(f"model missing: {model_path}")

    in_root = _stage_cohort(tmp_path, "cohort_A")

    # Plugin file: jaw_slenderness = bizyg / face_h.
    plugin_path = tmp_path / "my_features.py"
    plugin_path.write_text(
        "from facekit.api import register_feature\n"
        "@register_feature(group='JAW_SLENDERNESS', csv_columns=['jaw_slenderness'])\n"
        "def jaw_slenderness(lm, scales):\n"
        "    return {'jaw_slenderness': float(scales['bizyg'] / (scales['face_h'] + 1e-12))}\n"
    )

    mapping = {
        "by_disease": {
            "cohort_A": {
                "feature_groups": ["MOUTH_WIDTH"],
                "extra_columns": ["face_aspect_ratio"],
                "custom_features": ["JAW_SLENDERNESS"],
            }
        }
    }
    map_path = tmp_path / "m.json"
    map_path.write_text(json.dumps(mapping))

    out_dir = tmp_path / "out"
    df = run_batch_custom(
        input_path=in_root,
        output_dir=out_dir,
        user_mapping_path=map_path,
        user_features_path=plugin_path,
        model_path=model_path,
        hpo_codes_path=_hpo_csv(tmp_path),
    )

    # Schema includes the plugin column at the tail.
    assert "jaw_slenderness" in df.columns
    assert df.columns[-1] == "jaw_slenderness"

    # Selected cols (mouth_width, face_aspect_ratio, jaw_slenderness) must
    # be non-NaN; non-selected cols must be NaN. We pick a clearly
    # non-selected base col to verify.
    selected = {"mouth_width", "face_aspect_ratio", "jaw_slenderness"}
    a_row = df.iloc[0]
    for col in selected:
        assert pd.notna(a_row[col]), f"selected col {col} unexpectedly NaN"
    not_selected_sample = "eye_fissure_length_r"
    assert pd.isna(a_row[not_selected_sample])

    # CSV + report files exist with the right schema.
    csv_path = out_dir / "phenotypes_custom.csv"
    rep_path = out_dir / "custom_selection_report.csv"
    assert csv_path.is_file()
    assert rep_path.is_file()

    on_disk = pd.read_csv(csv_path)
    assert "jaw_slenderness" in on_disk.columns
    rep = pd.read_csv(rep_path)
    assert list(rep["cohort_name"]) == ["cohort_A"]
    assert int(rep.iloc[0]["n_columns_selected"]) == len(selected)
