"""
Default file-system paths for the geometric / disease-specific pipeline.

Each can be overridden on the CLI. The defaults reflect the user's cluster
layout documented in PLAN.md; production deployments should pin them to
project-tracked copies.
"""
from pathlib import Path

# Local MONDO ontology (downloaded by ``scripts/download_mondo.sh``).
DEFAULT_MONDO_OBO = Path("~/data/ontologies/mondo.obo").expanduser()

# 49-disease mapping JSON (feature_group lists per disease).
DEFAULT_FEATURE_MAPPING = Path(
    "/vast/projects/kai/multimodal-machine-learn/hongzhuo/"
    "mm_fusion_top50/feature_disease_mapping.json"
)

# HPO term -> feature_group / csv_column codes.
DEFAULT_HPO_CODES = Path(
    "/vast/projects/kai/multimodal-machine-learn/hongzhuo/"
    "mm_fusion_top50/hpo_direction_codes.csv"
)
