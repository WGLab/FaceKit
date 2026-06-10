"""
FaceKit CLI command: extract-features

Run the geometric feature extractor (~125 columns) over an image directory
or a JSONL landmark file, and write a phenotype CSV.

Examples
--------
    # Image directory (cohort-aware: subfolders = cohorts)
    facekit extract-features -i images/ -o results/

    # JSONL landmark file (skip MediaPipe)
    facekit extract-features -i landmarks.jsonl -o results/

    # Disease-specific mask
    facekit extract-features -i images/ -o results/ --mode disease-specific

    # Enable frontal-pose filtering
    facekit extract-features -i images/ -o results/ --frontal-check
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from facekit.core.geometric.batch import run_batch
from facekit.core.geometric.defaults import (
    DEFAULT_FEATURE_MAPPING,
    DEFAULT_HPO_CODES,
    DEFAULT_MONDO_OBO,
)


def extract_features(
    input_path: Path = typer.Option(
        ...,
        "--input", "-i",
        exists=True, dir_okay=True, file_okay=True,
        help="Input: an image directory OR a .jsonl landmark file.",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output", "-o",
        file_okay=False, dir_okay=True,
        help=(
            "Output directory. CSV is auto-named "
            "'phenotypes_all.csv' or 'phenotypes_disease_specific.csv'."
        ),
    ),
    mode: str = typer.Option(
        "all", "--mode",
        help="'all' (every feature column) or 'disease-specific' (per-cohort mask).",
    ),
    frontal_check: bool = typer.Option(
        False, "--frontal-check/--no-frontal-check",
        help="Skip faces with out-of-range yaw/pitch/roll.",
    ),
    blendshapes: bool = typer.Option(
        False, "--blendshapes/--no-blendshapes",
        help="(Reserved) include MediaPipe blendshapes; currently no-op.",
    ),
    num_faces: int = typer.Option(
        1, "--num-faces",
        help="Max faces to detect per image.",
    ),
    model: Optional[Path] = typer.Option(
        None, "--model",
        help="Path to MediaPipe face_landmarker.task (auto-downloaded if absent).",
    ),
    mondo_obo: Path = typer.Option(
        DEFAULT_MONDO_OBO, "--mondo-obo",
        help="Path to a local mondo.obo (disease-specific mode only).",
    ),
    feature_mapping: Path = typer.Option(
        DEFAULT_FEATURE_MAPPING, "--feature-mapping",
        help="Path to feature_disease_mapping.json (disease-specific mode only).",
    ),
    hpo_codes: Path = typer.Option(
        DEFAULT_HPO_CODES, "--hpo-codes",
        help="Path to hpo_direction_codes.csv (disease-specific mode only).",
    ),
):
    """Extract geometric phenotype features into a CSV.

    Output schema: ``[disease, image_id, frontal_ok, pose_yaw, pose_pitch,
    pose_roll, <feature columns in canonical order>]``.

    ``frontal_ok`` is tri-state: ``True`` (frontal pose verified),
    ``False`` (pose checked and out-of-range, feature columns are NaN), or
    ``NaN`` (pose unknown — JSONL row had no transformation matrix and
    ``--frontal-check`` was off, so the check was neither passed nor failed).

    In ``--mode disease-specific`` the schema is identical to ``--mode all``;
    columns not selected for a row's cohort are NaN-padded. A side artifact
    ``disease_specific_selection_report.csv`` is written next to the main CSV
    documenting the selected columns per cohort.
    """
    if mode not in {"all", "disease-specific"}:
        typer.secho(
            f"--mode must be 'all' or 'disease-specific', got {mode!r}",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(2)

    output_dir = output_dir.expanduser()
    mondo_obo = mondo_obo.expanduser()
    feature_mapping = feature_mapping.expanduser()
    hpo_codes = hpo_codes.expanduser()

    typer.echo(f"[FaceKit] extract-features: mode={mode}")
    typer.echo(f"[FaceKit] Input    : {input_path}")
    typer.echo(f"[FaceKit] Output   : {output_dir}")

    try:
        df = run_batch(
            input_path=input_path,
            output_dir=output_dir,
            mode=mode,
            frontal_check=frontal_check,
            blendshapes=blendshapes,
            model_path=model,
            num_faces=num_faces,
            mondo_obo_path=mondo_obo,
            feature_mapping_path=feature_mapping,
            hpo_codes_path=hpo_codes,
        )
    except (NotImplementedError, ValueError, FileNotFoundError, ImportError) as e:
        typer.secho(f"[FaceKit] {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    feat_cols = [
        c for c in df.columns
        if c not in {"disease", "image_id", "frontal_ok",
                     "pose_yaw", "pose_pitch", "pose_roll"}
    ]
    typer.echo(
        f"[FaceKit] Done: {len(df)} rows, "
        f"{len(feat_cols)} feature columns -> {output_dir}"
    )
