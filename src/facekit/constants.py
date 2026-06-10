"""Shared constants for FaceKit.

Default paths can be overridden via environment variables, which is handy
on HPC / shared filesystems:

    export FACEKIT_IMAGE_DIR=/lab/shared/gmdb
    export FACEKIT_OUTPUT_DIR=/scratch/tomorrow/facekit_out
"""
import os
from pathlib import Path

# Default input directory. Expected structure:
#   images/
#   ├── <cohort_a>/*.jpg
#   ├── <cohort_b>/*.png
#   └── ...
# If no subfolders are present, images/ itself is treated as a single cohort.
DEFAULT_IMAGE_DIR = Path(os.environ.get("FACEKIT_IMAGE_DIR", "./images"))

# Default output directory. All generated files land here.
DEFAULT_OUTPUT_DIR = Path(os.environ.get("FACEKIT_OUTPUT_DIR", "./outputs"))