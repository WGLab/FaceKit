from pathlib import Path
from typing import List, Optional

import typer

from facekit.constants import DEFAULT_IMAGE_DIR, DEFAULT_OUTPUT_DIR
from facekit.core.morph import average_faces, list_imgpaths
from facekit.core.morph.landmarks import MediaPipeLandmarkExtractor


def _has_images(folder: Path) -> bool:
    """Return True if folder contains at least one supported image file."""
    if not folder.is_dir():
        return False
    try:
        return any(list_imgpaths(folder))
    except (FileNotFoundError, NotADirectoryError):
        return False


def _find_cohort_dirs(input_dir: Path) -> List[Path]:
    """Return sorted subfolders of ``input_dir`` that contain images."""
    return sorted(
        d for d in input_dir.iterdir()
        if d.is_dir() and _has_images(d)
    )


def average_face(
    input_dir: Path = typer.Option(
        DEFAULT_IMAGE_DIR, "--input", "-i",
        exists=True, file_okay=False, dir_okay=True,
        help=(
            f"Input folder (default: {DEFAULT_IMAGE_DIR}). "
            "Subfolders with images become separate cohorts; "
            "otherwise the folder itself is treated as one cohort."
        ),
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR, "--output", "-o",
        file_okay=False, dir_okay=True,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    ),
    suffix: str = typer.Option(
        "_avg.png", "--suffix",
        help="Suffix appended to cohort name for output filenames.",
    ),
    width: int = typer.Option(500, help="Output width in pixels."),
    height: int = typer.Option(600, help="Output height in pixels."),
    blur_edges: bool = typer.Option(
        False, "--blur-edges", help="Blur the face mask edges."
    ),
    alpha: bool = typer.Option(
        False, "--alpha", help="Save with transparent background."
    ),
    dest_image: Optional[Path] = typer.Option(
        None, "--dest-image",
        help="Optional destination face to overlay each average onto.",
    ),
    model: Optional[Path] = typer.Option(
        None, "--model",
        help="Path to MediaPipe face_landmarker.task (auto-downloaded if absent).",
    ),
):
    """Generate an average face per cohort."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect layout: subfolders with images, or flat
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

    # Load the MediaPipe model once and reuse across all cohorts
    typer.echo("[FaceKit] Loading MediaPipe Face Landmarker...")
    extractor = MediaPipeLandmarkExtractor(model_path=model)

    failed: List[str] = []
    for cohort in cohort_dirs:
        paths = list(list_imgpaths(cohort))
        out_file = output_dir / f"{cohort.name}{suffix}"
        typer.echo(
            f"[FaceKit] {cohort.name}: averaging {len(paths)} images -> {out_file}"
        )
        try:
            average_faces(
                paths,
                dest_filename=str(dest_image) if dest_image else None,
                width=width,
                height=height,
                alpha=alpha,
                blur_edges=blur_edges,
                out_filename=str(out_file),
                extractor=extractor,
            )
        except Exception as e:
            typer.secho(
                f"[FaceKit] Failed on {cohort.name}: {e}",
                fg=typer.colors.YELLOW, err=True,
            )
            failed.append(cohort.name)

    n_ok = len(cohort_dirs) - len(failed)
    typer.echo(f"[FaceKit] Done: {n_ok}/{len(cohort_dirs)} cohort(s) processed")
    if failed:
        typer.secho(
            f"[FaceKit] Failed cohorts: {', '.join(failed)}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)