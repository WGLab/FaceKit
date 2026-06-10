"""Tests for the plugin registry in ``facekit.api``."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

import facekit.api as api
from facekit.core.geometric.extractor import GeometricFeatureExtractor


@pytest.fixture(autouse=True)
def _reset_registry():
    api.clear_registry()
    api._HPO_CODES_PATH = None
    yield
    api.clear_registry()
    api._HPO_CODES_PATH = None


def _hpo_csv(tmp_path: Path, groups: list) -> Path:
    p = tmp_path / "hpo.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "hpo_id", "hpo_name", "feature_group", "csv_column",
            "expected_direction", "confidence", "rationale", "n_diseases",
        ])
        for grp in groups:
            w.writerow(["HP:0000000", "x", grp, f"{grp.lower()}_col", "1", "HIGH", "x", "0"])
    return p


def test_register_and_retrieve():
    @api.register_feature(group="ZZZ_TEST_GROUP", csv_columns=["zzz_test_col"])
    def f(lm, scales):
        return {"zzz_test_col": 1.0}

    reg = api.get_registered()
    assert "ZZZ_TEST_GROUP" in reg
    assert reg["ZZZ_TEST_GROUP"].csv_columns == ("zzz_test_col",)
    assert api.plugin_columns_in_order() == ["zzz_test_col"]


def test_conflict_with_base_feature_column():
    # eye_fissure_length_r is one of the base 125.
    with pytest.raises(ValueError, match="base feature column"):
        @api.register_feature(
            group="MY_NEW_GROUP",
            csv_columns=["eye_fissure_length_r"],
        )
        def f(lm, scales):
            return {"eye_fissure_length_r": 0.0}


def test_conflict_with_hpo_feature_group(tmp_path):
    api._HPO_CODES_PATH = _hpo_csv(tmp_path, ["MOUTH_WIDTH", "EB_THICKNESS"])
    with pytest.raises(ValueError, match="collides with feature_group"):
        @api.register_feature(group="MOUTH_WIDTH", csv_columns=["my_col"])
        def f(lm, scales):
            return {"my_col": 0.0}


def test_duplicate_plugin_group():
    @api.register_feature(group="DUP_GROUP", csv_columns=["dup_a"])
    def f1(lm, scales):
        return {"dup_a": 0.0}

    with pytest.raises(ValueError, match="already registered"):
        @api.register_feature(group="DUP_GROUP", csv_columns=["dup_b"])
        def f2(lm, scales):
            return {"dup_b": 0.0}


def test_duplicate_plugin_csv_column():
    @api.register_feature(group="GRP_A", csv_columns=["shared_col"])
    def f1(lm, scales):
        return {"shared_col": 0.0}

    with pytest.raises(ValueError, match="already registered"):
        @api.register_feature(group="GRP_B", csv_columns=["shared_col"])
        def f2(lm, scales):
            return {"shared_col": 0.0}


def test_extend_with_different_cols_after_lock_raises():
    """Calling ``extend_feature_columns`` with a set that overlaps the
    already-locked plugin tail must raise.

    This documents the partial-extension guard inside
    ``extend_feature_columns``: once a suffix has been locked, you can
    only re-call the classmethod with the exact same suffix (idempotent)
    or a brand-new suffix appended via a fresh registration cycle.
    """
    @api.register_feature(group="LOCK_GROUP_A", csv_columns=["lock_col_a"])
    def f1(lm, scales):
        return {"lock_col_a": 0.0}
    GeometricFeatureExtractor.extend_feature_columns(api.plugin_columns_in_order())

    @api.register_feature(group="LOCK_GROUP_B", csv_columns=["lock_col_b"])
    def f2(lm, scales):
        return {"lock_col_b": 0.0}

    # Re-extend with a *different* set (overlapping the already-locked
    # tail) than the previous extension.
    with pytest.raises(RuntimeError, match="already locked"):
        GeometricFeatureExtractor.extend_feature_columns(["lock_col_a", "lock_col_b"])

    # Cleanup: reset FEATURE_COLUMNS so other tests aren't affected.
    GeometricFeatureExtractor.FEATURE_COLUMNS = None


def test_register_after_lock_then_extend_raises_with_clear_message():
    """Document register-after-lock contract.

    Behavior (a): ``register_feature`` itself succeeds (it appends to
    ``_PLUGIN_REGISTRY`` and ``_PLUGIN_COLUMN_ORDER`` only); the failure
    surfaces at the next ``extend_feature_columns(plugin_columns_in_order())``
    call, because the new plugin's columns can't be appended to a list
    whose tail was already locked with a strict prefix of the new list.
    """
    # 1. Register and lock plugin A.
    @api.register_feature(group="LOCK_GROUP_A", csv_columns=["lock_col_a"])
    def f1(lm, scales):
        return {"lock_col_a": 0.0}
    GeometricFeatureExtractor.extend_feature_columns(api.plugin_columns_in_order())
    locked_len = len(GeometricFeatureExtractor.FEATURE_COLUMNS)

    # 2. register_feature for plugin B AFTER lock — this itself succeeds.
    @api.register_feature(group="LOCK_GROUP_B", csv_columns=["lock_col_b"])
    def f2(lm, scales):
        return {"lock_col_b": 0.0}

    assert "LOCK_GROUP_B" in api.get_registered()
    assert api.plugin_columns_in_order() == ["lock_col_a", "lock_col_b"]
    # FEATURE_COLUMNS is unchanged at this point.
    assert len(GeometricFeatureExtractor.FEATURE_COLUMNS) == locked_len

    # 3. The user-facing call that surfaces the error is
    # extend_feature_columns(plugin_columns_in_order()). Message must
    # clearly attribute the cause to register-after-lock.
    with pytest.raises(RuntimeError) as exc_info:
        GeometricFeatureExtractor.extend_feature_columns(
            api.plugin_columns_in_order()
        )
    msg = str(exc_info.value)
    assert "already locked" in msg
    assert "lock_col_a" in msg  # already-locked plugin column listed
    assert "lock_col_b" in msg  # newly-attempted column listed
    assert "BEFORE" in msg  # remediation hint

    # Cleanup: reset FEATURE_COLUMNS so other tests aren't affected.
    GeometricFeatureExtractor.FEATURE_COLUMNS = None


def test_extend_feature_columns_idempotent():
    @api.register_feature(group="IDEM_GROUP", csv_columns=["idem_col"])
    def f(lm, scales):
        return {"idem_col": 0.0}

    GeometricFeatureExtractor.extend_feature_columns(["idem_col"])
    n1 = len(GeometricFeatureExtractor.FEATURE_COLUMNS)
    GeometricFeatureExtractor.extend_feature_columns(["idem_col"])
    n2 = len(GeometricFeatureExtractor.FEATURE_COLUMNS)
    assert n1 == n2

    GeometricFeatureExtractor.FEATURE_COLUMNS = None


def test_plugin_dispatch_in_compute_all():
    @api.register_feature(group="PROBE_GROUP", csv_columns=["probe_col"])
    def f(lm, scales):
        return {"probe_col": 42.0}

    GeometricFeatureExtractor.extend_feature_columns(["probe_col"])
    rng = np.random.default_rng(0)
    synthetic = rng.uniform(0.0, 1000.0, size=(478, 2))
    probe = GeometricFeatureExtractor.__new__(GeometricFeatureExtractor)
    lm_c, scales = probe._canonicalize(synthetic)
    feats = probe._compute_all(lm_c, scales)
    assert feats["probe_col"] == 42.0
    assert "probe_col" in GeometricFeatureExtractor.FEATURE_COLUMNS

    GeometricFeatureExtractor.FEATURE_COLUMNS = None
