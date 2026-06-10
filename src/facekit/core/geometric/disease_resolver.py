"""
Disease-name -> MONDO ID resolver, backed by a local ``mondo.obo`` and a
positive-only CSV cache.

The cache stores only resolved entries. Unresolved queries are returned in a
separate list so callers can decide policy (warn / fail-fast / fall back).
"""
from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CACHE_COLUMNS = [
    "query_name",
    "mondo_id",
    "mondo_name",
    "omim_xrefs",
    "matched_via",
    "manual_override",
]


@dataclass
class DiseaseResolution:
    """One resolved query: query string -> MONDO record."""

    query_name: str
    mondo_id: str
    mondo_name: str
    omim_xrefs: List[str] = field(default_factory=list)
    matched_via: str = "exact"  # exact | synonym | fuzzy | manual

    def to_cache_row(self, manual_override: bool = False) -> dict:
        return {
            "query_name": self.query_name,
            "mondo_id": self.mondo_id,
            "mondo_name": self.mondo_name,
            "omim_xrefs": ";".join(self.omim_xrefs),
            "matched_via": self.matched_via,
            "manual_override": "true" if manual_override else "false",
        }


# ----------------------------------------------------------------------
# Cache I/O
# ----------------------------------------------------------------------

def _read_cache(cache_path: Path) -> Tuple[Dict[str, DiseaseResolution], Dict[str, bool]]:
    """Return (query_name -> resolution, query_name -> manual_override flag)."""
    if not cache_path.exists():
        return {}, {}
    cached: Dict[str, DiseaseResolution] = {}
    overrides: Dict[str, bool] = {}
    with open(cache_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            q = row.get("query_name", "").strip()
            if not q:
                continue
            xrefs_raw = row.get("omim_xrefs") or ""
            xrefs = [x for x in xrefs_raw.split(";") if x]
            cached[q] = DiseaseResolution(
                query_name=q,
                mondo_id=row.get("mondo_id", "").strip(),
                mondo_name=row.get("mondo_name", "").strip(),
                omim_xrefs=xrefs,
                matched_via=row.get("matched_via", "manual").strip() or "manual",
            )
            overrides[q] = (row.get("manual_override", "false").strip().lower()
                            == "true")
    return cached, overrides


def _write_cache(
    cache_path: Path,
    cached: Dict[str, DiseaseResolution],
    overrides: Dict[str, bool],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CACHE_COLUMNS)
        writer.writeheader()
        for q in sorted(cached):
            writer.writerow(cached[q].to_cache_row(manual_override=overrides.get(q, False)))


# ----------------------------------------------------------------------
# OAK adapter (lazy)
# ----------------------------------------------------------------------

class _OakAdapter:
    """Thin wrapper around ``oaklib`` so we import it only when needed."""

    def __init__(self, mondo_obo_path: Path) -> None:
        # Local import: keeps the rest of facekit usable when the
        # [geometric] extras aren't installed.
        try:
            from oaklib import get_adapter
        except ImportError as exc:
            raise ImportError(
                "oaklib is required for disease resolution. Install via "
                "'pip install -e \".[geometric]\"' or 'pip install oaklib'."
            ) from exc

        if not mondo_obo_path.exists():
            raise FileNotFoundError(
                f"mondo.obo not found at {mondo_obo_path}. Run "
                f"scripts/download_mondo.sh to fetch it."
            )

        # oaklib accepts e.g. "simpleobo:/path/to/mondo.obo".
        self._adapter = get_adapter(f"simpleobo:{mondo_obo_path}")
        self._labels: Optional[Dict[str, str]] = None
        self._lower_label_to_id: Optional[Dict[str, str]] = None
        self._lower_synonym_to_id: Optional[Dict[str, str]] = None

    # -- low-level lookups -------------------------------------------------

    def _ensure_label_index(self) -> None:
        if self._labels is not None:
            return
        labels: Dict[str, str] = {}
        lower_label: Dict[str, str] = {}
        # Sort entities by CURIE so duplicate-label collisions resolve to the
        # same MONDO id across runs (cache stability across oaklib versions).
        for curie in sorted(self._adapter.entities()):
            if not curie.startswith("MONDO:"):
                continue
            lbl = self._adapter.label(curie)
            if not lbl:
                continue
            labels[curie] = lbl
            lower_label.setdefault(lbl.strip().lower(), curie)
        self._labels = labels
        self._lower_label_to_id = lower_label

    def _ensure_synonym_index(self) -> None:
        if self._lower_synonym_to_id is not None:
            return
        self._ensure_label_index()
        syn_idx: Dict[str, str] = {}
        # Sort curies so duplicate-synonym collisions resolve deterministically
        # (cache stability across oaklib versions / OBO file orderings).
        sorted_curies = sorted(self._labels.keys())  # type: ignore[union-attr]
        # ``entity_alias_map`` returns {curie: [synonyms...]} for OBO sources.
        try:
            alias_map = self._adapter.entity_alias_map(sorted_curies)
        except Exception:
            # Fallback: per-term call.
            alias_map = {}
            for curie in sorted_curies:
                try:
                    alias_map[curie] = list(self._adapter.entity_aliases(curie))
                except Exception:
                    alias_map[curie] = []
        for curie in sorted_curies:
            aliases = alias_map.get(curie) or []
            if not curie.startswith("MONDO:"):
                continue
            for syn in aliases:
                syn_idx.setdefault(syn.strip().lower(), curie)
        self._lower_synonym_to_id = syn_idx

    def lookup_exact_label(self, name: str) -> Optional[str]:
        self._ensure_label_index()
        return self._lower_label_to_id.get(name.strip().lower())  # type: ignore[union-attr]

    def lookup_synonym(self, name: str) -> Optional[str]:
        self._ensure_synonym_index()
        return self._lower_synonym_to_id.get(name.strip().lower())  # type: ignore[union-attr]

    def label_of(self, curie: str) -> str:
        self._ensure_label_index()
        return self._labels.get(curie, "")  # type: ignore[union-attr]

    def omim_xrefs_of(self, curie: str) -> List[str]:
        try:
            xrefs = list(self._adapter.entity_xrefs(curie))
        except Exception:
            xrefs = []
        return [x for x in xrefs if x.startswith("OMIM:")]


# ----------------------------------------------------------------------
# Fuzzy normalization
# ----------------------------------------------------------------------

# Normalize a name: lowercase, strip whitespace, drop trailing numeric
# variants like " 1", " type 1", " -1".
_TRAILING_NUM_RE = re.compile(
    r"\s*(?:[-,]\s*\d+|\btype\s+\d+|\d+)\s*$",
    re.IGNORECASE,
)


def _fuzzy_variants(name: str) -> List[str]:
    """Return progressively-more-aggressive normalizations of ``name``."""
    base = name.strip()
    variants = [base]
    low = base.lower()
    if low != base:
        variants.append(low)
    stripped = _TRAILING_NUM_RE.sub("", low).strip()
    if stripped and stripped != low:
        variants.append(stripped)
    # Also try collapsing internal whitespace.
    collapsed = re.sub(r"\s+", " ", stripped or low).strip()
    if collapsed and collapsed not in variants:
        variants.append(collapsed)
    return variants


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def resolve_disease(
    name: str,
    *,
    mondo_obo_path: Path,
    cache_path: Path,
) -> Optional[DiseaseResolution]:
    """Resolve a single name. Convenience wrapper around ``resolve_all``."""
    resolved, _ = resolve_all(
        [name], mondo_obo_path=mondo_obo_path, cache_path=cache_path,
    )
    return resolved.get(name)


def resolve_all(
    names: List[str],
    *,
    mondo_obo_path: Path,
    cache_path: Path,
) -> Tuple[Dict[str, DiseaseResolution], List[str]]:
    """Resolve a batch of disease names.

    :returns: (``{query_name: DiseaseResolution}``, ``[unresolved query names]``).
        The cache CSV at ``cache_path`` is updated in-place with newly
        resolved entries; unresolved queries are NOT written to the cache.
    """
    mondo_obo_path = Path(mondo_obo_path)
    cache_path = Path(cache_path)

    cached, overrides = _read_cache(cache_path)

    resolved: Dict[str, DiseaseResolution] = {}
    unresolved: List[str] = []
    new_hits = 0

    # Names we still need to resolve via oaklib (cache miss & not override).
    pending: List[str] = []
    for name in names:
        if name in cached:
            resolved[name] = cached[name]
        else:
            pending.append(name)

    if pending:
        try:
            adapter = _OakAdapter(mondo_obo_path)
        except (ImportError, FileNotFoundError):
            # Hard failure: nothing in pending can be resolved. Surface
            # to caller; cached hits already populated above.
            raise

        for name in pending:
            hit = _resolve_one_via_oak(name, adapter)
            if hit is None:
                unresolved.append(name)
                continue
            resolved[name] = hit
            cached[name] = hit
            overrides.setdefault(name, False)
            new_hits += 1

    if new_hits:
        _write_cache(cache_path, cached, overrides)
        logger.info("disease_resolver: cached %d new entries -> %s",
                    new_hits, cache_path)

    return resolved, unresolved


def _resolve_one_via_oak(name: str, adapter: _OakAdapter) -> Optional[DiseaseResolution]:
    """Walk the lookup ladder for one name."""
    # 1. exact label
    curie = adapter.lookup_exact_label(name)
    if curie:
        return _build_resolution(name, curie, "exact", adapter)

    # 2. exact synonym
    curie = adapter.lookup_synonym(name)
    if curie:
        return _build_resolution(name, curie, "synonym", adapter)

    # 3. fuzzy normalizations against label and synonym indexes
    for variant in _fuzzy_variants(name)[1:]:  # variants[0] is the unchanged input
        curie = adapter.lookup_exact_label(variant)
        if curie:
            return _build_resolution(name, curie, "fuzzy", adapter)
        curie = adapter.lookup_synonym(variant)
        if curie:
            return _build_resolution(name, curie, "fuzzy", adapter)

    return None


def _build_resolution(
    query: str,
    curie: str,
    matched_via: str,
    adapter: _OakAdapter,
) -> DiseaseResolution:
    return DiseaseResolution(
        query_name=query,
        mondo_id=curie,
        mondo_name=adapter.label_of(curie),
        omim_xrefs=adapter.omim_xrefs_of(curie),
        matched_via=matched_via,
    )


def suggest_neighbors(name: str, mondo_obo_path: Path, k: int = 3) -> List[str]:
    """Best-effort 'did you mean' suggestions for an unresolved name.

    Uses simple substring containment over the MONDO label index. Returns up
    to ``k`` candidate labels. Silent fallback to ``[]`` on any error so we
    don't blow up the user's CLI when oaklib is unavailable.
    """
    try:
        adapter = _OakAdapter(mondo_obo_path)
        adapter._ensure_label_index()
    except Exception:
        return []

    target = name.strip().lower()
    target_stripped = _TRAILING_NUM_RE.sub("", target).strip()
    candidates: List[str] = []
    for low_label, _curie in (adapter._lower_label_to_id or {}).items():  # type: ignore[union-attr]
        if target_stripped and target_stripped in low_label:
            candidates.append(low_label)
        elif target and target in low_label:
            candidates.append(low_label)
        if len(candidates) >= k * 4:
            break
    return candidates[:k]
