"""
FaceKit CLI command: extract-features-custom

Run the geometric feature extractor under a user-supplied disease ->
feature mapping (no MONDO normalization). Cohort folder names are used
directly as keys into the user JSON.

Examples
--------
    # Mapping only (no plugin Python file)
    facekit extract-features-custom -i images/ -o results/ \\
        --user-mapping my_diseases.json

    # With a plugin Python file that defines new feature formulas
    facekit extract-features-custom -i images/ -o results/ \\
        --user-mapping my_diseases.json --user-features my_features.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from facekit.core.geometric.custom_batch import run_batch_custom
from facekit.core.geometric.defaults import DEFAULT_HPO_CODES


def extract_features_custom(
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
        help="Output directory. Writes 'phenotypes_custom.csv' and 'custom_selection_report.csv'.",
    ),
    user_mapping: Path = typer.Option(
        ...,
        "--user-mapping",
        exists=True, dir_okay=False, file_okay=True,
        help="User JSON mapping each cohort folder name to feature_groups, "
             "extra_columns, and/or custom_features.",
    ),
    user_features: Optional[Path] = typer.Option(
        None, "--user-features",
        help="Optional Python file registering new feature formulas via "
             "@facekit.api.register_feature. WARNING: executes arbitrary code.",
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
    hpo_codes: Path = typer.Option(
        DEFAULT_HPO_CODES, "--hpo-codes",
        help="Path to hpo_direction_codes.csv (used for family expansion AND "
             "for plugin feature_group conflict checks).",
    ),
):
    """Extract features under a user-customizable disease->feature mapping."""
    del blendshapes, num_faces  # accepted for symmetry with extract-features

    output_dir = output_dir.expanduser()
    user_mapping = user_mapping.expanduser()
    hpo_codes = hpo_codes.expanduser()
    if user_features is not None:
        user_features = user_features.expanduser()
        typer.secho(
            "[FaceKit] Loading user Python plugin: this executes arbitrary code. "
            "Only use trusted files.",
            fg=typer.colors.YELLOW, err=True,
        )

    typer.echo(f"[FaceKit] extract-features-custom")
    typer.echo(f"[FaceKit] Input    : {input_path}")
    typer.echo(f"[FaceKit] Output   : {output_dir}")
    typer.echo(f"[FaceKit] Mapping  : {user_mapping}")
    if user_features is not None:
        typer.echo(f"[FaceKit] Plugin   : {user_features}")

    try:
        df = run_batch_custom(
            input_path=input_path,
            output_dir=output_dir,
            user_mapping_path=user_mapping,
            user_features_path=user_features,
            frontal_check=frontal_check,
            model_path=model,
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
