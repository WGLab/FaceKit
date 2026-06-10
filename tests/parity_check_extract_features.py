"""
Parity + CLI regression check for ``facekit extract-features``.

Three suites:

1. **Per-image parity**: GeometricFeatureExtractor (new) vs the legacy
   ``mm_fusion_top10/preprocess_scripts/facekit_feature_extractor.py``.
   Asserts every shared feature column matches within ``ATOL`` (default 1e-9).

2. **CLI / run_batch regression**: invokes ``run_batch(mode="all", ...)``
   on a small image directory and verifies:
     - the CSV lands at ``<output_dir>/phenotypes_all.csv``
     - column count = 6 leading + 125 features = 131
     - feature columns are in ``GeometricFeatureExtractor.FEATURE_COLUMNS``
       order
     - one row's feature values numerically match calling
       ``legacy.FaceFeatureExtractor.extract`` on the same image (1e-9).

3. **JSONL roundtrip parity** (Phase 3): runs
   ``extract-landmarks --format jsonl`` on a small directory, then feeds
   the JSONL to ``run_batch``, and verifies that every feature column
   matches the direct-from-images CSV within ``ATOL``. Pose columns are
   compared with a looser tolerance (``POSE_ATOL``) because the
   transformation matrix is float32 in MediaPipe but rehydrated as float64
   from JSONL — both representations are valid; only the feature math is
   contractually byte-identical.

Disease-specific CLI is NOT exercised here: oaklib + a populated
``mondo.obo`` are required to drive that path, and they are not assumed
available on this cluster. Validate that path manually after running
``scripts/download_mondo.sh`` + ``facekit resolve-diseases``.

Configuration via environment variables:
  FACEKIT_PARITY_IMAGES   : directory of test images (cohort folder is fine)
  FACEKIT_PARITY_MODEL    : path to face_landmarker.task
  FACEKIT_PARITY_LEGACY   : path to mm_fusion_top10/preprocess_scripts dir
                            (must contain facekit_feature_extractor.py)
  FACEKIT_PARITY_NIMG     : number of images to check (default 5)

Usage:
    cd /vast/projects/kai/multimodal-machine-learn/hongzhuo/facekit
    PYTHONPATH=src python tests/parity_check_extract_features.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# --- defaults from this user's cluster layout -------------------------------
DEFAULT_IMAGES = (
    "/vast/projects/kai/multimodal-machine-learn/hongzhuo/mm_fusion_top10/"
    "train_images_top50/ADNP-related_multiple_congenital_anomalies_-_"
    "intellectual_disability_-_autism_spectrum_disorder"
)
DEFAULT_MODEL = (
    "/vast/projects/kai/multimodal-machine-learn/hongzhuo/mm_fusion_top10/"
    "preprocess_scripts/face_landmarker.task"
)
DEFAULT_LEGACY_DIR = (
    "/vast/projects/kai/multimodal-machine-learn/hongzhuo/mm_fusion_top10/"
    "preprocess_scripts"
)
ATOL = 1e-9
# Pose columns derive from the MediaPipe facial transformation matrix,
# which is float32. The JSONL roundtrip rehydrates it as float64 from JSON,
# producing pose values that differ by ~1e-6 vs the direct path. This is
# not a math bug — both pipelines are valid; only the feature math is
# contractually byte-identical.
POSE_ATOL = 1e-5
POSE_COLS = {"pose_yaw", "pose_pitch", "pose_roll"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def run_per_image_parity(
    images: list,
    legacy,
    new,
) -> tuple:
    """Compare per-image extract() outputs. Returns (n_pass, n_fail)."""
    print(f"[parity] comparing {len(images)} images at atol={ATOL:g}")
    n_pass, n_fail = 0, 0
    for img in images:
        # Legacy default is frontal_check=True; we want both pipelines to run
        # the full geometry. The legacy minimal-dict path returns only pose if
        # the image is non-frontal — bypass that by passing False.
        leg_feats = legacy.extract(str(img), frontal_check=False)
        new_feats = new.extract(str(img), frontal_check=False)
        if leg_feats is None or new_feats is None:
            print(f"[parity] {img.name}: skipped (legacy={leg_feats is None}, new={new_feats is None})")
            continue

        shared = sorted(set(leg_feats) & set(new_feats))
        only_legacy = sorted(set(leg_feats) - set(new_feats))
        only_new = sorted(set(new_feats) - set(leg_feats))

        worst_key, worst_diff = None, 0.0
        mismatches = []
        for k in shared:
            a, b = leg_feats[k], new_feats[k]
            if isinstance(a, bool) or isinstance(b, bool):
                if bool(a) != bool(b):
                    mismatches.append((k, a, b))
                continue
            if a is None or b is None:
                if a is not b:
                    mismatches.append((k, a, b))
                continue
            d = abs(float(a) - float(b))
            if d > worst_diff:
                worst_diff, worst_key = d, k
            if not np.isfinite(a) and not np.isfinite(b):
                continue
            if d > ATOL:
                mismatches.append((k, a, b))

        status = "OK" if not mismatches else "FAIL"
        print(
            f"[parity] {img.name}: {status} "
            f"(shared={len(shared)} only_legacy={len(only_legacy)} only_new={len(only_new)} "
            f"worst_diff={worst_diff:g} @ {worst_key})"
        )
        if only_legacy:
            print(f"    legacy-only keys: {only_legacy[:6]}{' ...' if len(only_legacy) > 6 else ''}")
        if only_new:
            print(f"    new-only keys   : {only_new[:6]}{' ...' if len(only_new) > 6 else ''}")
        if mismatches:
            n_fail += 1
            for k, a, b in mismatches[:5]:
                print(f"    {k}: legacy={a!r} new={b!r}")
            if len(mismatches) > 5:
                print(f"    ... and {len(mismatches) - 5} more")
        else:
            n_pass += 1

    print(f"[parity] summary: {n_pass} pass, {n_fail} fail (of {n_pass + n_fail})")
    return n_pass, n_fail


def run_cli_regression(
    images_dir: Path,
    model_path: Path,
    legacy,
    n_imgs: int,
) -> int:
    """Smoke-test the run_batch CLI path (mode=all, image directory).

    Verifies output filename, schema width / order, and a one-row numerical
    sanity check vs the legacy extractor.
    """
    import shutil
    import tempfile

    import pandas as pd

    from facekit.core.geometric import GeometricFeatureExtractor
    from facekit.core.geometric.batch import LEADING_COLUMNS, run_batch

    # Stage a fresh tiny cohort dir (avoid pulling in the full source folder).
    src_imgs = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)[:n_imgs]
    if not src_imgs:
        print(f"[cli-regress] no images in {images_dir}")
        return 1

    with tempfile.TemporaryDirectory(prefix="facekit_cli_regress_") as tmp:
        tmp = Path(tmp)
        in_root = tmp / "in"
        in_root.mkdir()
        cohort = in_root / "cohort_A"
        cohort.mkdir()
        for p in src_imgs:
            shutil.copy2(p, cohort / p.name)
        out_dir = tmp / "out"

        df = run_batch(
            input_path=in_root,
            output_dir=out_dir,
            mode="all",
            frontal_check=False,
            model_path=str(model_path),
        )

        out_csv = out_dir / "phenotypes_all.csv"
        if not out_csv.is_file():
            print(f"[cli-regress] FAIL: expected {out_csv} not written")
            return 1

        # Schema width.
        feature_cols = list(GeometricFeatureExtractor._init_feature_columns())
        expected_cols = list(LEADING_COLUMNS) + feature_cols
        on_disk = pd.read_csv(out_csv)
        if list(on_disk.columns) != expected_cols:
            print("[cli-regress] FAIL: column order mismatch")
            print(f"  expected[:8]  = {expected_cols[:8]}")
            print(f"  got[:8]       = {list(on_disk.columns)[:8]}")
            mismatches = [
                (i, e, g) for i, (e, g) in enumerate(zip(expected_cols, on_disk.columns))
                if e != g
            ]
            for i, e, g in mismatches[:5]:
                print(f"  col {i}: expected={e!r} got={g!r}")
            return 1
        if len(on_disk.columns) != 6 + 125:
            print(f"[cli-regress] FAIL: column count {len(on_disk.columns)} != 131")
            return 1

        # Numerical sanity: pick the first non-empty row that matches a real
        # image_id, recompute via legacy, compare shared columns.
        if len(on_disk) == 0:
            print("[cli-regress] FAIL: empty dataframe")
            return 1

        sample_row = on_disk.iloc[0]
        image_id = str(sample_row["image_id"])
        matching = [p for p in src_imgs if p.stem == image_id]
        if not matching:
            print(f"[cli-regress] WARN: image_id {image_id} not found in source list, skipping numeric sanity")
        else:
            leg_feats = legacy.extract(str(matching[0]), frontal_check=False)
            if leg_feats is None:
                print("[cli-regress] WARN: legacy returned None on sample image, skipping numeric sanity")
            else:
                worst_key, worst_diff = None, 0.0
                mismatches = []
                for k in feature_cols:
                    if k not in leg_feats:
                        continue
                    a = leg_feats[k]
                    b = sample_row[k]
                    if a is None or b is None:
                        continue
                    try:
                        af, bf = float(a), float(b)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(af) and not np.isfinite(bf):
                        continue
                    d = abs(af - bf)
                    if d > worst_diff:
                        worst_diff, worst_key = d, k
                    if d > ATOL:
                        mismatches.append((k, a, b))
                if mismatches:
                    print(f"[cli-regress] FAIL: {len(mismatches)} numeric mismatches on row {image_id}")
                    for k, a, b in mismatches[:5]:
                        print(f"    {k}: legacy={a!r} csv={b!r}")
                    return 1
                print(
                    f"[cli-regress] numeric sanity row={image_id} OK "
                    f"(worst_diff={worst_diff:g} @ {worst_key})"
                )

        # Sanity-confirm the in-memory df shape matches on-disk shape.
        if df.shape != on_disk.shape:
            print(f"[cli-regress] FAIL: in-memory df {df.shape} != on-disk {on_disk.shape}")
            return 1

        print(
            f"[cli-regress] OK: {out_csv.name} written, "
            f"{on_disk.shape[0]} rows x {on_disk.shape[1]} cols, schema matches"
        )
        return 0


def run_jsonl_roundtrip(
    images_dir: Path,
    model_path: Path,
    n_imgs: int,
) -> int:
    """Phase 3: extract-landmarks --format jsonl, then extract-features on it.

    Verifies that the feature columns of the JSONL-driven CSV match the
    direct-from-images CSV exactly (max diff = 0). Pose columns are
    checked with the looser ``POSE_ATOL`` because the matrix is float32 in
    MediaPipe and float64 after JSON roundtrip.
    """
    import shutil
    import tempfile

    import pandas as pd

    from facekit.commands.extract_landmarks import extract_landmarks
    from facekit.core.geometric.batch import run_batch

    src_imgs = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)[:n_imgs]
    if not src_imgs:
        print(f"[jsonl-roundtrip] SKIPPED: no images in {images_dir}")
        return 0

    with tempfile.TemporaryDirectory(prefix="facekit_jsonl_roundtrip_") as tmp:
        tmp = Path(tmp)
        in_root = tmp / "in"
        cohort = in_root / "cohort_A"
        cohort.mkdir(parents=True)
        for p in src_imgs:
            shutil.copy2(p, cohort / p.name)

        # Step 1: write JSONL via the new --format jsonl path. We invoke the
        # underlying typer-decorated function with explicit args rather than
        # spawning a subprocess to keep the test in-process.
        jsonl_out = tmp / "lm_jsonl"
        try:
            extract_landmarks(
                input_dir=in_root,
                output_dir=jsonl_out,
                blendshapes=False,
                transform=True,           # so JSONL carries the matrix
                visualize=False,
                visualize_style="mesh",
                num_faces=1,
                model=model_path,
                format="jsonl",
            )
        except SystemExit as e:
            # typer raises Exit on errors; surface as a test failure.
            print(f"[jsonl-roundtrip] FAIL: extract-landmarks exited {e}")
            return 1
        jsonl_path = jsonl_out / f"{in_root.name}_landmarks.jsonl"
        if not jsonl_path.is_file():
            print(f"[jsonl-roundtrip] FAIL: expected {jsonl_path} not written")
            return 1

        # Quick well-formedness check.
        import json as _json
        n_lines = 0
        with open(jsonl_path) as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                obj = _json.loads(raw)
                for k in ("image_id", "disease", "landmarks_3d", "image_size"):
                    if k not in obj:
                        print(f"[jsonl-roundtrip] FAIL: missing key {k!r} in line")
                        return 1
                if len(obj["landmarks_3d"]) != 478:
                    print(f"[jsonl-roundtrip] FAIL: landmarks_3d len != 478")
                    return 1
                n_lines += 1
        if n_lines != len(src_imgs):
            print(f"[jsonl-roundtrip] FAIL: jsonl has {n_lines} lines, expected {len(src_imgs)}")
            return 1

        # Step 2: run extract-features twice — once on the dir, once on the JSONL.
        out_dir_dir = tmp / "out_dir"
        out_dir_jsonl = tmp / "out_jsonl"
        run_batch(
            input_path=in_root,
            output_dir=out_dir_dir,
            mode="all",
            frontal_check=False,
            model_path=str(model_path),
        )
        run_batch(
            input_path=jsonl_path,
            output_dir=out_dir_jsonl,
            mode="all",
            frontal_check=False,
            model_path=str(model_path),
        )

        a = pd.read_csv(out_dir_dir / "phenotypes_all.csv").sort_values("image_id").reset_index(drop=True)
        b = pd.read_csv(out_dir_jsonl / "phenotypes_all.csv").sort_values("image_id").reset_index(drop=True)

        if a.shape != b.shape:
            print(f"[jsonl-roundtrip] FAIL: shape mismatch {a.shape} vs {b.shape}")
            return 1
        if list(a.columns) != list(b.columns):
            print("[jsonl-roundtrip] FAIL: column order mismatch")
            return 1
        if list(a.image_id) != list(b.image_id):
            print("[jsonl-roundtrip] FAIL: image_id order mismatch")
            return 1

        worst_feat, worst_feat_diff = None, 0.0
        worst_pose, worst_pose_diff = None, 0.0
        n_feat_above_atol = 0
        n_pose_above_atol = 0
        for col in a.columns:
            if col in {"disease", "image_id", "frontal_ok"}:
                continue
            av = pd.to_numeric(a[col], errors="coerce").to_numpy()
            bv = pd.to_numeric(b[col], errors="coerce").to_numpy()
            mask = np.isfinite(av) & np.isfinite(bv)
            if not mask.any():
                continue
            diff = np.abs(av[mask] - bv[mask])
            md = float(diff.max()) if diff.size else 0.0
            if col in POSE_COLS:
                if md > worst_pose_diff:
                    worst_pose_diff, worst_pose = md, col
                if md > POSE_ATOL:
                    n_pose_above_atol += int((diff > POSE_ATOL).sum())
            else:
                if md > worst_feat_diff:
                    worst_feat_diff, worst_feat = md, col
                if md > ATOL:
                    n_feat_above_atol += int((diff > ATOL).sum())

        ok = n_feat_above_atol == 0 and n_pose_above_atol == 0
        status = "OK" if ok else "FAIL"
        print(
            f"[jsonl-roundtrip] {status}: feat worst={worst_feat_diff:.3e} @ {worst_feat} "
            f"(>{ATOL:g}: {n_feat_above_atol}) | "
            f"pose worst={worst_pose_diff:.3e} @ {worst_pose} (>{POSE_ATOL:g}: {n_pose_above_atol})"
        )
        return 0 if ok else 1


def main() -> int:
    images_dir = Path(os.environ.get("FACEKIT_PARITY_IMAGES", DEFAULT_IMAGES))
    model_path = Path(os.environ.get("FACEKIT_PARITY_MODEL", DEFAULT_MODEL))
    legacy_dir = Path(os.environ.get("FACEKIT_PARITY_LEGACY", DEFAULT_LEGACY_DIR))
    n_imgs = int(os.environ.get("FACEKIT_PARITY_NIMG", "5"))

    if not images_dir.is_dir():
        print(f"[parity] image dir missing: {images_dir}")
        return 2
    if not model_path.is_file():
        print(f"[parity] model missing: {model_path}")
        return 2
    if not (legacy_dir / "facekit_feature_extractor.py").is_file():
        print(f"[parity] legacy extractor missing: {legacy_dir}/facekit_feature_extractor.py")
        return 2

    sys.path.insert(0, str(legacy_dir))
    from facekit_feature_extractor import FaceFeatureExtractor  # type: ignore

    from facekit.core.geometric import GeometricFeatureExtractor

    legacy = FaceFeatureExtractor(model_path=str(model_path))
    new = GeometricFeatureExtractor(model_path=str(model_path))

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)[:n_imgs]
    if not images:
        print(f"[parity] no images in {images_dir}")
        return 2

    n_pass, n_fail = run_per_image_parity(images, legacy, new)
    cli_rc = run_cli_regression(images_dir, model_path, legacy, n_imgs)
    jsonl_rc = run_jsonl_roundtrip(images_dir, model_path, n_imgs)

    print("[disease-specific-CLI] SKIPPED: requires oaklib + populated mondo.obo "
          "(see scripts/download_mondo.sh + facekit resolve-diseases)")
    return 0 if (n_fail == 0 and cli_rc == 0 and jsonl_rc == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
