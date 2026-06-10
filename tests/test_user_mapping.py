"""Tests for the user-mapping loader and selector."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import facekit.api as api
from facekit.core.geometric.feature_selector import (
    UserDiseaseSpec,
    load_user_disease_mapping,
    select_columns_for_user_disease,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    api.clear_registry()
    api._HPO_CODES_PATH = None
    yield
    api.clear_registry()
    api._HPO_CODES_PATH = None


def _hpo_csv(tmp_path: Path) -> Path:
    p = tmp_path / "hpo.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "hpo_id", "hpo_name", "feature_group", "csv_column",
            "expected_direction", "confidence", "rationale", "n_diseases",
        ])
        # MOUTH_WIDTH -> mouth_width (a real base column)
        w.writerow(["HP:0000154", "Wide mouth", "MOUTH_WIDTH", "mouth_width", "1", "HIGH", "", "0"])
    return p


BASE_COLS = [
    "mouth_width", "eye_fissure_length_r", "eye_fissure_length_l",
    "eye_fissure_length_mean", "eye_fissure_length_asym",
    "face_aspect_ratio",
]


def test_load_all_three_fields(tmp_path):
    mapping = {
        "by_disease": {
            "cohort_X": {
                "feature_groups": ["MOUTH_WIDTH"],
                "extra_columns": ["face_aspect_ratio"],
                "custom_features": ["MY_GROUP"],
            }
        }
    }
    p = tmp_path / "m.json"
    p.write_text(json.dumps(mapping))
    out = load_user_disease_mapping(p)
    assert "cohort_X" in out
    spec = out["cohort_X"]
    assert spec.feature_groups == ("MOUTH_WIDTH",)
    assert spec.extra_columns == ("face_aspect_ratio",)
    assert spec.custom_features == ("MY_GROUP",)


def test_load_all_empty_raises(tmp_path):
    mapping = {
        "by_disease": {
            "cohort_X": {
                "feature_groups": [],
                "extra_columns": [],
                "custom_features": [],
            }
        }
    }
    p = tmp_path / "m.json"
    p.write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="at least one"):
        load_user_disease_mapping(p)


def test_select_unknown_extra_column_raises(tmp_path):
    spec = UserDiseaseSpec(extra_columns=("does_not_exist",))
    with pytest.raises(ValueError, match="unknown extra_columns"):
        select_columns_for_user_disease(
            spec, BASE_COLS, hpo_codes_path=_hpo_csv(tmp_path),
        )


def test_select_unknown_custom_feature_raises(tmp_path):
    spec = UserDiseaseSpec(custom_features=("UNREGISTERED_GROUP",))
    with pytest.raises(ValueError, match="not found in plugin registry"):
        select_columns_for_user_disease(
            spec, BASE_COLS, hpo_codes_path=_hpo_csv(tmp_path),
        )


def test_select_combines_all_three(tmp_path):
    api._HPO_CODES_PATH = _hpo_csv(tmp_path)

    @api.register_feature(group="MY_GROUP", csv_columns=["my_plugin_col"])
    def f(lm, scales):
        return {"my_plugin_col": 0.0}

    spec = UserDiseaseSpec(
        feature_groups=("MOUTH_WIDTH",),
        extra_columns=("face_aspect_ratio",),
        custom_features=("MY_GROUP",),
    )
    sel = select_columns_for_user_disease(
        spec, BASE_COLS, hpo_codes_path=_hpo_csv(tmp_path),
    )
    assert "mouth_width" in sel  # from feature_groups
    assert "face_aspect_ratio" in sel  # from extra_columns
    assert "my_plugin_col" in sel  # from custom_features


def test_select_extras_only_with_none_hpo_path_succeeds():
    """No feature_groups + hpo_codes_path=None must work cleanly.

    Cohorts that only use extra_columns / custom_features should not
    require an hpo_direction_codes.csv path.
    """
    spec = UserDiseaseSpec(
        extra_columns=("face_aspect_ratio",),
    )
    sel = select_columns_for_user_disease(
        spec, BASE_COLS, hpo_codes_path=None,
    )
    assert sel == {"face_aspect_ratio"}


def test_select_feature_groups_with_none_hpo_path_raises():
    """feature_groups + hpo_codes_path=None must raise a clear error."""
    spec = UserDiseaseSpec(feature_groups=("MOUTH_WIDTH",))
    with pytest.raises(ValueError, match="hpo_codes_path is required"):
        select_columns_for_user_disease(
            spec, BASE_COLS, hpo_codes_path=None, disease_name="cohort_X",
        )
