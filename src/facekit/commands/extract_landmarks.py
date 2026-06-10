"""
FaceKit CLI command: extract-landmarks

Extract MediaPipe facial landmarks (478 points) from face images, with
optional blendshapes, head pose transformation matrices, and mesh
visualizations.

Automatically handles two input layouts (same as ``facekit average-face``):

  Layout A (multi-cohort):
      images/
      ├── cdls/       -> outputs/landmarks/cdls/*_landmarks.json
      ├── 22q/        -> outputs/landmarks/22q/*_landmarks.json

  Layout B (single folder):
      images/*.jpg    -> outputs/landmarks/*_landmarks.json

Output formats (``--format``):
  * ``json`` (default): one JSON file per image (legacy behavior).
  * ``jsonl``: a single file ``<output_dir>/<input_dir_name>_landmarks.jsonl``,
    one image per line, matching the schema consumed by
    ``facekit extract-features``.

Example:
    # Basic: just extract landmarks
    facekit extract-landmarks

    # With blendshapes, pose matrix, and visualization
    facekit extract-landmarks --blendshapes --transform --visualize

    # JSONL intermediate for `facekit extract-features`
    facekit extract-landmarks --format jsonl --transform

    # Custom paths
    facekit extract-landmarks -i /data/gmdb -o /scratch/landmarks
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
import typer

from facekit.constants import DEFAULT_IMAGE_DIR, DEFAULT_OUTPUT_DIR
from facekit.core.morph import list_imgpaths
from facekit.core.morph.landmarks import (
    MediaPipeLandmarkExtractor,
    draw_face_landmarks,
)


DEFAULT_LANDMARK_OUTPUT = DEFAULT_OUTPUT_DIR / "landmarks"


def _has_images(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    try:
        return any(list_imgpaths(folder))
    except (FileNotFoundError, NotADirectoryError):
        return False


def _find_cohort_dirs(input_dir: Path) -> List[Path]:
    return sorted(
        d for d in input_dir.iterdir()
        if d.is_dir() and _has_images(d)
    )


def _process_image(
    image_path: Path,
    out_dir: Path,
    extractor: MediaPipeLandmarkExtractor,
    visualize: Optional[str] = None,
) -> bool:
    """Extract landmarks for one image and save JSON (+ optional visualization).

    :returns: True on success, False if no face detected or read error.
    """
    # Rich extraction (includes any optional fields enabled on extractor)
    result = extractor.extract_full(image_path)
    if result is None:
        return False

    # Write JSON
    stem = image_path.stem
    json_path = out_dir / f"{stem}_landmarks.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    # Optional visualization: re-run detector to get raw landmark objects
    # (extract_full returns plain dicts which lose the MediaPipe structure)
    if visualize:
        mp_image = mp.Image.create_from_file(str(image_path))
        detection = extractor.detector.detect(mp_image)
        if detection.face_landmarks:
            rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            annotated = draw_face_landmarks(
                rgb, detection.face_landmarks, style=visualize
            )
            vis_path = out_dir / f"{stem}_vis.png"
            cv2.imwrite(str(vis_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    return True


def _process_image_jsonl(
    image_path: Path,
    disease: str,
    extractor: MediaPipeLandmarkExtractor,
    out_fh,
) -> bool:
    """Run MediaPipe on one image and append a single JSONL line.

    Schema (locked by Phase 1 of extract-features integration):
        {
          "image_id": str,
          "disease": str,
          "landmarks_3d": [[x, y, z], ...],   # 478 entries, MediaPipe-normalized [0,1]
          "image_size": [w, h],
          "transformation_matrix": [[...]],   # 4x4 floats, present iff MediaPipe returned one
          "blendshapes": [52 floats]          # present iff blendshapes were enabled
        }

    :returns: True if a face was detected and written, False otherwise.
    """
    mp_image = mp.Image.create_from_file(str(image_path))
    result = extractor.detector.detect(mp_image)
    if not result.face_landmarks:
        return False

    h, w = mp_image.height, mp_image.width
    face_landmarks = result.face_landmarks[0]
    record = {
        "image_id": image_path.stem,
        "disease": disease,
        "landmarks_3d": [
            [float(lm.x), float(lm.y), float(lm.z)] for lm in face_landmarks
        ],
        "image_size": [int(w), int(h)],
    }
    if extractor.output_transformation_matrixes and result.facial_transformation_matrixes:
        mat = result.facial_transformation_matrixes[0]
        record["transformation_matrix"] = np.asarray(mat, dtype=float).tolist()
    if extractor.output_blendshapes and result.face_blendshapes:
        record["blendshapes"] = [float(bs.score) for bs in result.face_blendshapes[0]]

    out_fh.write(json.dumps(record))
    out_fh.write("\n")
    return True


def extract_landmarks(
    input_dir: Path = typer.Option(
        DEFAULT_IMAGE_DIR, "--input", "-i",
        exists=True, file_okay=False, dir_okay=True,
        help=f"Input folder (default: {DEFAULT_IMAGE_DIR}).",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_LANDMARK_OUTPUT, "--output", "-o",
        file_okay=False, dir_okay=True,
        help=f"Output directory (default: {DEFAULT_LANDMARK_OUTPUT}).",
    ),
    blendshapes: bool = typer.Option(
        False, "--blendshapes",
        help="Include 52 facial expression blendshape scores.",
    ),
    transform: bool = typer.Option(
        False, "--transform",
        help="Include 4x4 facial transformation matrix (head pose).",
    ),
    visualize: bool = typer.Option(
        False, "--visualize",
        help="Also save an annotated image alongside each JSON.",
    ),
    visualize_style: str = typer.Option(
        "mesh", "--visualize-style",
        help="Visualization style: 'mesh' (tesselation + contours + irises) or 'dots'.",
    ),
    num_faces: int = typer.Option(
        1, "--num-faces",
        help="Max number of faces to detect per image.",
    ),
    model: Optional[Path] = typer.Option(
        None, "--model",
        help="Path to MediaPipe face_landmarker.task (auto-downloaded if absent).",
    ),
    format: str = typer.Option(
        "json", "--format",
        help=(
            "Output format: 'json' (one file per image, default) or 'jsonl' "
            "(single file <output_dir>/<input_dir_name>_landmarks.jsonl, "
            "consumed directly by `facekit extract-features`)."
        ),
    ),
):
    """Extract MediaPipe facial landmarks from face images."""
    if visualize_style not in {"mesh", "dots"}:
        typer.secho(
            f"--visualize-style must be 'mesh' or 'dots', got {visualize_style!r}",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(2)
    if format not in {"json", "jsonl"}:
        typer.secho(
            f"--format must be 'json' or 'jsonl', got {format!r}",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(2)
    if format == "jsonl" and visualize:
        typer.secho(
            "--visualize is not supported with --format jsonl "
            "(JSONL mode emits a single intermediate file, not per-image artifacts).",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(2)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect cohort layout
    cohort_dirs = _find_cohort_dirs(input_dir)
    has_direct_images = _has_images(input_dir)

    if cohort_dirs and has_direct_images:
        typer.secho(
            f"[FaceKit] Warning: {input_dir} has both images and subfolders. "
            f"Using subfolder mode; loose images will be ignored.",
            fg=typer.colors.YELLOW,
        )

    if cohort_dirs:
        typer.echo(
            f"[FaceKit] Found {len(cohort_dirs)} cohort(s): "
            f"{', '.join(d.name for d in cohort_dirs)}"
        )
    elif has_direct_images:
        cohort_dirs = [input_dir]
        typer.echo(
            f"[FaceKit] No subfolders detected; treating {input_dir} as a single cohort"
        )
    else:
        typer.secho(
            f"[FaceKit] No images or image subfolders found in {input_dir}",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(1)

    # Build extractor once, reuse for all images
    typer.echo("[FaceKit] Loading MediaPipe Face Landmarker...")
    typer.echo(
        f"[FaceKit] Outputs: landmarks"
        + (" + blendshapes" if blendshapes else "")
        + (" + transformation matrix" if transform else "")
        + (f" + visualizations ({visualize_style})" if visualize else "")
    )
    extractor = MediaPipeLandmarkExtractor(
        model_path=model,
        num_faces=num_faces,
        output_blendshapes=blendshapes,
        output_transformation_matrixes=transform,
    )

    vis_style = visualize_style if visualize else None
    total_ok = 0
    total_images = 0

    if format == "jsonl":
        # Single JSONL output file. ``disease`` is the cohort folder name
        # in multi-cohort mode; "" when the input is a flat folder (because
        # in that case there is no enclosing cohort label to attach).
        jsonl_path = output_dir / f"{input_dir.name}_landmarks.jsonl"
        with open(jsonl_path, "w") as out_fh:
            for cohort in cohort_dirs:
                disease = cohort.name if cohort != input_dir else ""
                images = list(list_imgpaths(cohort))
                total_images += len(images)
                typer.echo(f"[FaceKit] {cohort.name}: processing {len(images)} images")

                ok = 0
                for img_path in images:
                    if _process_image_jsonl(
                        Path(img_path), disease, extractor, out_fh
                    ):
                        ok += 1
                    else:
                        typer.secho(
                            f"[FaceKit]   ✗ No face: {Path(img_path).name}",
                            fg=typer.colors.YELLOW,
                        )
                typer.echo(f"[FaceKit]   ✓ {ok}/{len(images)} -> {jsonl_path.name}")
                total_ok += ok

        typer.echo(
            f"[FaceKit] Done: {total_ok}/{total_images} images processed "
            f"across {len(cohort_dirs)} cohort(s) -> {jsonl_path}"
        )
        return

    for cohort in cohort_dirs:
        cohort_out = (
            output_dir / cohort.name
            if cohort != input_dir else output_dir
        )
        cohort_out.mkdir(parents=True, exist_ok=True)

        images = list(list_imgpaths(cohort))
        total_images += len(images)
        typer.echo(f"[FaceKit] {cohort.name}: processing {len(images)} images")

        ok = 0
        for img_path in images:
            if _process_image(Path(img_path), cohort_out, extractor, vis_style):
                ok += 1
            else:
                typer.secho(
                    f"[FaceKit]   ✗ No face: {Path(img_path).name}",
                    fg=typer.colors.YELLOW,
                )
        typer.echo(f"[FaceKit]   ✓ {ok}/{len(images)} -> {cohort_out}")
        total_ok += ok

    typer.echo(
        f"[FaceKit] Done: {total_ok}/{total_images} images processed "
        f"across {len(cohort_dirs)} cohort(s)"
    )