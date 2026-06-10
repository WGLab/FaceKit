#!/usr/bin/env python3
"""Regenerate the three FaceKit illustration figures in ``figures/``.

All three figures are built from real GestaltMatcher Database (GMDB) source
images that live OUTSIDE this repository; the raw GMDB jpgs are never copied
in, only the rendered PNG outputs. NOTE: average-face.png is an aggregate,
de-identified average; extract-landmarks.png and extract-features.png overlay
annotations on an identifiable patient face -- confirm GMDB consent/DUA before
publishing these. The public cannot re-run this script (it needs GMDB access)
-- that is expected, the PNGs are committed.

Exact command used to generate the committed figures::

    python scripts/make_figures.py \
        --crouzon-image /vast/projects/kai/multimodal-machine-learn/hongzhuo/mm_fusion_top50/visualization_images/pat8721_img13755.jpg \
        --williams-dir  /vast/projects/kai/multimodal-machine-learn/hongzhuo/mm_fusion_top50/combined_data/images/Williams_syndrome

Produces (one per illustrated command):
    figures/average-face.png      ``facekit average-face`` over the Williams cohort
    figures/extract-landmarks.png ``facekit extract-landmarks`` mesh on a Crouzon face
    figures/extract-features.png  four geometric measurement overlays on that face
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"

DEFAULT_CROUZON = Path(
    "/vast/projects/kai/multimodal-machine-learn/hongzhuo/mm_fusion_top50"
    "/visualization_images/pat8721_img13755.jpg"
)
DEFAULT_WILLIAMS = Path(
    "/vast/projects/kai/multimodal-machine-learn/hongzhuo/mm_fusion_top50"
    "/combined_data/images/Williams_syndrome"
)

# Landmark-pair segments for the four facekit features, with indices copied
# verbatim from src/facekit/core/geometric/extractor.py. Each feature value is
# euclidean(pair) / euclidean(BIZYG), i.e. normalized by bizygomatic width.
FEATURES = [
    ("inter_pupillary_distance", 468, 473, "#e6194b"),  # iris centers
    ("inter_canthal_distance", 133, 362, "#3cb44b"),    # inner canthi
    ("nose_base_width", 98, 327, "#4363d8"),            # alae
    ("mouth_width", 61, 291, "#f58231"),                # mouth commissures
]
BIZYG = (234, 454)
DPI = 200


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def make_average_face(williams_dir: Path, tmp: Path) -> None:
    """Run ``facekit average-face`` on the Williams cohort ONLY."""
    cohorts = tmp / "cohorts"
    cohorts.mkdir()
    (cohorts / "Williams_syndrome").symlink_to(williams_dir)
    out = tmp / "avg_out"
    run(["facekit", "average-face", "-i", str(cohorts), "-o", str(out)])
    produced = next(out.rglob("Williams_syndrome*avg*.png"))
    shutil.copy(produced, FIGURES / "average-face.png")


def make_extract_landmarks(crouzon_image: Path, tmp: Path) -> None:
    """Run ``facekit extract-landmarks --visualize`` on the Crouzon face."""
    demo = tmp / "images" / "demo"
    demo.mkdir(parents=True)
    staged = demo / crouzon_image.name
    shutil.copy(crouzon_image, staged)
    out = tmp / "lm_out"
    # --visualize is incompatible with --format jsonl, so use --format json.
    run([
        "facekit", "extract-landmarks", "-i", str(tmp / "images"),
        "-o", str(out), "--visualize", "--visualize-style", "mesh",
        "--format", "json",
    ])
    vis = out / "demo" / f"{staged.stem}_vis.png"
    shutil.copy(vis, FIGURES / "extract-landmarks.png")


def detect_landmarks(image_path: Path, model_path: Path | None):
    """Return (rgb_image, landmarks_px (N, 2)) via MediaPipe Face Landmarker.

    Mirrors the detect step of mm_fusion_top50/scripts/plot_ipd_face_overlay.py;
    resolves the model through facekit's ``ensure_model`` (auto-download)."""
    from facekit.core.morph.landmarks import ensure_model
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model = ensure_model(model_path)
    base = mp_python.BaseOptions(model_asset_path=str(model))
    opts = mp_vision.FaceLandmarkerOptions(base_options=base, num_faces=1)
    detector = mp_vision.FaceLandmarker.create_from_options(opts)
    image = mp.Image.create_from_file(str(image_path))
    res = detector.detect(image)
    if not res.face_landmarks:
        raise RuntimeError(f"No face detected in {image_path}")
    rgb = image.numpy_view()
    h, w = rgb.shape[:2]
    lm = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], dtype=np.float64)
    return rgb, lm


def make_extract_features(crouzon_image: Path, model_path: Path | None) -> None:
    """Overlay the four facekit feature segments on the Crouzon face."""
    rgb, lm = detect_landmarks(crouzon_image, model_path)
    h, w = rgb.shape[:2]

    # Tight crop around all landmarks (same idea as the reference script).
    x_min, y_min = lm.min(axis=0)
    x_max, y_max = lm.max(axis=0)
    x0, y0 = max(0, int(x_min)), max(0, int(y_min))
    x1, y1 = min(w, int(x_max)), min(h, int(y_max))
    crop = rgb[y0:y1, x0:x1]
    off = np.array([x0, y0])

    bz = np.linalg.norm(lm[BIZYG[0]] - lm[BIZYG[1]])

    ch, cw = crop.shape[:2]
    fig, ax = plt.subplots(figsize=(3.8, 3.8 * ch / cw), dpi=DPI)
    ax.imshow(crop)

    handles = []
    for name, a, b, color in FEATURES:
        pa, pb = lm[a] - off, lm[b] - off
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=color, lw=2.2,
                solid_capstyle="round", zorder=4)
        ax.scatter([pa[0], pb[0]], [pa[1], pb[1]], s=20, color=color,
                   edgecolors="white", linewidths=0.7, zorder=5)
        val = np.linalg.norm(lm[a] - lm[b]) / bz
        handles.append(Line2D([0], [0], color=color, lw=2.2,
                              label=f"{name}  ({val:.2f})"))

    ax.legend(handles=handles, loc="upper left", fontsize=6.5,
              framealpha=0.92, handlelength=1.3, borderpad=0.5,
              labelspacing=0.4, edgecolor="#888888")

    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(pad=0.1)
    fig.savefig(FIGURES / "extract-features.png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--crouzon-image", type=Path, default=DEFAULT_CROUZON)
    p.add_argument("--williams-dir", type=Path, default=DEFAULT_WILLIAMS)
    p.add_argument("--model-path", type=Path, default=None,
                   help="MediaPipe face_landmarker.task (auto-downloaded if absent).")
    args = p.parse_args()

    FIGURES.mkdir(exist_ok=True)
    for path in (args.crouzon_image, args.williams_dir):
        if not path.exists():
            raise SystemExit(f"missing source asset: {path}")

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        make_average_face(args.williams_dir, tmp)
        make_extract_landmarks(args.crouzon_image, tmp)
    make_extract_features(args.crouzon_image, args.model_path)

    print("wrote 3 figures to", FIGURES)


if __name__ == "__main__":
    main()
