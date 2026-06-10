"""
FaceKit CLI command: resolve-diseases

Standalone helper that runs the MONDO-backed name resolver against either
the 49-disease JSON mapping, a folder of cohort directories, or both.
Updates a positive-only cache CSV and prints a diagnostic summary.

Examples
--------
    # Resolve all 49 keys in the canonical mapping
    facekit resolve-diseases --from-mapping

    # Resolve cohort folder names
    facekit resolve-diseases --from-folder /path/to/train_images_top50

    # Both, into a custom cache
    facekit resolve-diseases --from-mapping --from-folder /path/to/imgs \
        --cache ./disease_resolution.csv
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from facekit.core.geometric.defaults import (
    DEFAULT_FEATURE_MAPPING,
    DEFAULT_MONDO_OBO,
)
from facekit.core.geometric.disease_resolver import (
    resolve_all,
    suggest_neighbors,
)
from facekit.core.geometric.feature_selector import load_feature_disease_mapping


def _folder_cohort_names(folder: Path) -> List[str]:
    return sorted(d.name for d in folder.iterdir() if d.is_dir())


def resolve_diseases(
    from_mapping: bool = typer.Option(
        False, "--from-mapping/--no-from-mapping",
        help="Resolve all keys in the disease-mapping JSON (default OFF; pass --from-mapping to enable).",
    ),
    from_folder: Optional[Path] = typer.Option(
        None, "--from-folder",
        exists=True, dir_okay=True, file_okay=False,
        help="Resolve all subfolder names of this directory.",
    ),
    feature_mapping: Path = typer.Option(
        DEFAULT_FEATURE_MAPPING, "--feature-mapping",
        help="Path to feature_disease_mapping.json (only used with --from-mapping).",
    ),
    mondo_obo: Path = typer.Option(
        DEFAULT_MONDO_OBO, "--mondo-obo",
        help="Path to a local mondo.obo.",
    ),
    cache: Path = typer.Option(
        Path("disease_resolution.csv"), "--cache",
        help="Cache CSV path (resolved entries are written/updated here).",
    ),
):
    """Resolve disease names to MONDO IDs and refresh the cache CSV."""
    feature_mapping = feature_mapping.expanduser()
    mondo_obo = mondo_obo.expanduser()
    cache = cache.expanduser()

    if not (from_mapping or from_folder):
        typer.secho(
            "[FaceKit] Provide --from-mapping and/or --from-folder.",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(2)

    names: List[str] = []
    if from_mapping:
        if not feature_mapping.exists():
            typer.secho(
                f"[FaceKit] feature_mapping not found: {feature_mapping}",
                fg=typer.colors.RED, err=True,
            )
            raise typer.Exit(2)
        json_keys = sorted(load_feature_disease_mapping(feature_mapping).keys())
        typer.echo(f"[FaceKit] Loaded {len(json_keys)} disease keys from {feature_mapping}")
        names.extend(json_keys)
    if from_folder is not None:
        folder_keys = _folder_cohort_names(from_folder)
        typer.echo(f"[FaceKit] Found {len(folder_keys)} subfolders in {from_folder}")
        names.extend(folder_keys)

    # Deduplicate while preserving sorted order for stable diagnostics.
    names = sorted(set(names))

    typer.echo(f"[FaceKit] Resolving {len(names)} unique names against {mondo_obo} ...")

    try:
        resolved, unresolved = resolve_all(
            names, mondo_obo_path=mondo_obo, cache_path=cache,
        )
    except (ImportError, FileNotFoundError) as e:
        typer.secho(f"[FaceKit] {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    typer.echo(
        f"[FaceKit] Resolved : {len(resolved)} / {len(names)}  "
        f"(cache: {cache})"
    )
    typer.echo(f"[FaceKit] Unresolved: {len(unresolved)}")

    if unresolved:
        typer.secho("[FaceKit] Unresolved names:", fg=typer.colors.YELLOW)
        for n in sorted(unresolved):
            suggestions = suggest_neighbors(n, mondo_obo, k=3)
            tail = (f"  did you mean: {', '.join(suggestions)}"
                    if suggestions else "")
            typer.echo(f"  - {n}{tail}")
