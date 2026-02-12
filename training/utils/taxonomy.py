"""ESCO and O*NET taxonomy loading and skill-hierarchy utilities.

This module provides a unified interface for two widely-used occupational
taxonomies:

* **ESCO** (European Skills, Competences, Qualifications and Occupations)
  -- published by the European Commission as CSV / JSON-LD.
* **O*NET** (Occupational Information Network) -- published by the US
  Department of Labor, typically as tab-delimited or Excel files.

The functions here normalise both sources into a common in-memory
representation so that downstream code (e.g., skill matching, curriculum
alignment) can work taxonomy-agnostically.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Skill:
    """A single skill / competence node in the taxonomy graph."""

    id: str
    label: str
    description: str = ""
    source: str = ""  # "esco" | "onet"
    parent_ids: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)
    alt_labels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Skill(id={self.id!r}, label={self.label!r}, source={self.source!r})"


@dataclass
class TaxonomyGraph:
    """A directed acyclic graph of :class:`Skill` nodes.

    Provides O(1) lookup by skill ID and convenience traversal methods.
    """

    skills: Dict[str, Skill] = field(default_factory=dict)
    source: str = ""

    # -- query helpers -----------------------------------------------------

    def __len__(self) -> int:
        return len(self.skills)

    def __contains__(self, skill_id: str) -> bool:
        return skill_id in self.skills

    def __getitem__(self, skill_id: str) -> Skill:
        return self.skills[skill_id]

    def get(self, skill_id: str) -> Optional[Skill]:
        return self.skills.get(skill_id)

    def roots(self) -> List[Skill]:
        """Return all skills that have no parents."""
        return [s for s in self.skills.values() if not s.parent_ids]

    def leaves(self) -> List[Skill]:
        """Return all skills that have no children."""
        return [s for s in self.skills.values() if not s.child_ids]

    def all_labels(self) -> List[str]:
        """Return a flat list of primary labels (useful for fuzzy matching)."""
        return [s.label for s in self.skills.values()]


# ---------------------------------------------------------------------------
# ESCO loading
# ---------------------------------------------------------------------------


def load_esco_taxonomy(
    path: Union[str, Path],
    *,
    id_column: str = "conceptUri",
    label_column: str = "preferredLabel",
    description_column: str = "description",
    alt_label_column: str = "altLabels",
    broader_column: str = "broaderConceptUri",
    delimiter: Optional[str] = None,
) -> TaxonomyGraph:
    """Load an ESCO skills file (CSV or JSON) into a :class:`TaxonomyGraph`.

    The function auto-detects CSV vs JSON based on the file extension.  For
    CSV, the column names default to the standard ESCO bulk-download schema
    but can be overridden.

    Parameters
    ----------
    path:
        Path to the ESCO data file (``.csv``, ``.tsv``, or ``.json``).
    id_column:
        Column / key for the unique skill identifier.
    label_column:
        Column / key for the preferred human-readable label.
    description_column:
        Column / key for the skill description.
    alt_label_column:
        Column / key containing alternative labels (newline- or
        comma-separated in CSV; list in JSON).
    broader_column:
        Column / key pointing to the parent / broader concept URI.
    delimiter:
        Explicit CSV delimiter.  When ``None``, the function uses ``,``
        for ``.csv`` and ``\\t`` for ``.tsv``.

    Returns
    -------
    TaxonomyGraph
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"ESCO file not found: {filepath}")

    suffix = filepath.suffix.lower()

    if suffix == ".json":
        records = _load_json_records(filepath)
    elif suffix in (".csv", ".tsv"):
        sep = delimiter if delimiter is not None else ("\t" if suffix == ".tsv" else ",")
        records = _load_csv_records(filepath, sep)
    else:
        raise ValueError(
            f"Unsupported file extension {suffix!r}.  Expected .csv, .tsv, or .json."
        )

    graph = TaxonomyGraph(source="esco")

    for rec in records:
        skill_id = str(rec.get(id_column, "")).strip()
        if not skill_id:
            continue

        alt_raw = rec.get(alt_label_column, "")
        if isinstance(alt_raw, list):
            alt_labels = [a.strip() for a in alt_raw if a.strip()]
        else:
            alt_labels = [
                a.strip()
                for a in str(alt_raw).replace("\n", ",").split(",")
                if a.strip()
            ]

        broader_raw = rec.get(broader_column, "")
        parent_ids = (
            [b.strip() for b in str(broader_raw).split(",") if b.strip()]
            if broader_raw
            else []
        )

        skill = Skill(
            id=skill_id,
            label=str(rec.get(label_column, "")).strip(),
            description=str(rec.get(description_column, "")).strip(),
            source="esco",
            parent_ids=parent_ids,
            alt_labels=alt_labels,
            metadata={k: v for k, v in rec.items() if k not in {
                id_column, label_column, description_column,
                alt_label_column, broader_column,
            }},
        )
        graph.skills[skill_id] = skill

    # Back-fill child_ids from parent_ids.
    _backfill_children(graph)

    logger.info("Loaded ESCO taxonomy with %d skills from %s", len(graph), filepath)
    return graph


# ---------------------------------------------------------------------------
# O*NET loading
# ---------------------------------------------------------------------------


def load_onet_taxonomy(
    path: Union[str, Path],
    *,
    id_column: str = "Element ID",
    label_column: str = "Element Name",
    description_column: str = "Description",
    scale_column: str = "Scale ID",
    data_value_column: str = "Data Value",
    delimiter: Optional[str] = None,
) -> TaxonomyGraph:
    """Load an O*NET content-model file into a :class:`TaxonomyGraph`.

    O*NET distributes skills as tab-delimited text files (e.g.,
    ``Skills.txt``, ``Knowledge.txt``, ``Abilities.txt``).  The hierarchy
    is implicit in the element IDs (e.g., ``2.A.1.a``).

    Parameters
    ----------
    path:
        Path to the O*NET data file (``.txt``, ``.csv``, or ``.tsv``).
    id_column:
        Column name for the element identifier.
    label_column:
        Column name for the human-readable element name.
    description_column:
        Column name for the element description.
    scale_column:
        Column name for the measurement scale (kept in metadata).
    data_value_column:
        Column name for the data value (kept in metadata).
    delimiter:
        Explicit delimiter.  Defaults to tab for ``.txt``/``.tsv`` and
        comma for ``.csv``.

    Returns
    -------
    TaxonomyGraph
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"O*NET file not found: {filepath}")

    suffix = filepath.suffix.lower()

    if suffix == ".json":
        records = _load_json_records(filepath)
    else:
        sep = delimiter if delimiter is not None else (
            "," if suffix == ".csv" else "\t"
        )
        records = _load_csv_records(filepath, sep)

    # O*NET files often contain duplicate element IDs (one row per
    # occupation x scale combination).  We deduplicate by element ID,
    # aggregating metadata.
    seen: Dict[str, Skill] = {}

    for rec in records:
        element_id = str(rec.get(id_column, "")).strip()
        if not element_id:
            continue

        if element_id in seen:
            # Append extra metadata (e.g., different scale values).
            existing = seen[element_id]
            scale = rec.get(scale_column, "")
            value = rec.get(data_value_column, "")
            if scale or value:
                existing.metadata.setdefault("scales", []).append(
                    {"scale": str(scale), "value": str(value)}
                )
            continue

        skill = Skill(
            id=element_id,
            label=str(rec.get(label_column, "")).strip(),
            description=str(rec.get(description_column, "")).strip(),
            source="onet",
            parent_ids=_infer_onet_parent(element_id),
            metadata={},
        )
        scale = rec.get(scale_column, "")
        value = rec.get(data_value_column, "")
        if scale or value:
            skill.metadata["scales"] = [
                {"scale": str(scale), "value": str(value)}
            ]

        seen[element_id] = skill

    graph = TaxonomyGraph(skills=seen, source="onet")
    _backfill_children(graph)

    logger.info("Loaded O*NET taxonomy with %d skills from %s", len(graph), filepath)
    return graph


def _infer_onet_parent(element_id: str) -> List[str]:
    """Derive the parent element ID from an O*NET dotted identifier.

    O*NET element IDs follow a hierarchical dotted notation, e.g.:
    ``2.A.1.a`` -> parent ``2.A.1`` -> parent ``2.A`` -> root ``2``.

    Returns a single-element list with the parent ID, or an empty list
    if *element_id* is already a top-level element.
    """
    parts = element_id.rsplit(".", 1)
    if len(parts) == 2 and parts[0]:
        return [parts[0]]
    return []


# ---------------------------------------------------------------------------
# Hierarchy construction
# ---------------------------------------------------------------------------


def build_skill_hierarchy(
    taxonomy_data: TaxonomyGraph,
) -> Dict[str, List[str]]:
    """Return an adjacency-list representation of parent -> children.

    This is a convenience function for algorithms that prefer a plain dict
    over the :class:`TaxonomyGraph` object model.

    Parameters
    ----------
    taxonomy_data:
        A loaded taxonomy graph.

    Returns
    -------
    dict[str, list[str]]
        Mapping from each skill ID to its list of direct child IDs.
    """
    hierarchy: Dict[str, List[str]] = defaultdict(list)
    for skill in taxonomy_data.skills.values():
        # Ensure every node appears as a key, even if it has no children.
        hierarchy.setdefault(skill.id, [])
        for pid in skill.parent_ids:
            hierarchy[pid].append(skill.id)
    return dict(hierarchy)


# ---------------------------------------------------------------------------
# Related-skill traversal
# ---------------------------------------------------------------------------


def get_related_skills(
    skill: Union[str, Skill],
    taxonomy: TaxonomyGraph,
    max_depth: int = 2,
    *,
    include_ancestors: bool = True,
    include_descendants: bool = True,
    include_siblings: bool = True,
) -> List[Skill]:
    """Find skills related to *skill* within *max_depth* hops in the hierarchy.

    Traversal can go **up** (ancestors), **down** (descendants), and
    **sideways** (siblings = other children of the same parent).

    Parameters
    ----------
    skill:
        A :class:`Skill` instance or a skill ID string.
    taxonomy:
        The taxonomy graph to search.
    max_depth:
        Maximum number of hops from the starting skill.  Must be >= 1.
    include_ancestors:
        Whether to traverse upward to parent skills.
    include_descendants:
        Whether to traverse downward to child skills.
    include_siblings:
        Whether to include siblings (children of the same parents).

    Returns
    -------
    list[Skill]
        Related skills (excluding the starting skill itself), ordered by
        discovery (breadth-first).

    Raises
    ------
    KeyError
        If the skill ID is not found in the taxonomy.
    ValueError
        If *max_depth* < 1.
    """
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}")

    skill_id = skill.id if isinstance(skill, Skill) else skill
    if skill_id not in taxonomy:
        raise KeyError(f"Skill {skill_id!r} not found in taxonomy")

    visited: Set[str] = {skill_id}
    result: List[Skill] = []
    queue: deque[tuple[str, int]] = deque()

    # Seed the BFS with immediate neighbours.
    start_node = taxonomy[skill_id]

    if include_ancestors:
        for pid in start_node.parent_ids:
            if pid in taxonomy and pid not in visited:
                queue.append((pid, 1))
                visited.add(pid)

    if include_descendants:
        for cid in start_node.child_ids:
            if cid in taxonomy and cid not in visited:
                queue.append((cid, 1))
                visited.add(cid)

    if include_siblings:
        for pid in start_node.parent_ids:
            parent = taxonomy.get(pid)
            if parent is None:
                continue
            for sibling_id in parent.child_ids:
                if sibling_id not in visited and sibling_id in taxonomy:
                    queue.append((sibling_id, 1))
                    visited.add(sibling_id)

    while queue:
        current_id, depth = queue.popleft()
        current_node = taxonomy.get(current_id)
        if current_node is None:
            continue

        result.append(current_node)

        if depth >= max_depth:
            continue

        # Expand neighbours.
        if include_ancestors:
            for pid in current_node.parent_ids:
                if pid not in visited and pid in taxonomy:
                    queue.append((pid, depth + 1))
                    visited.add(pid)

        if include_descendants:
            for cid in current_node.child_ids:
                if cid not in visited and cid in taxonomy:
                    queue.append((cid, depth + 1))
                    visited.add(cid)

        if include_siblings:
            for pid in current_node.parent_ids:
                parent = taxonomy.get(pid)
                if parent is None:
                    continue
                for sibling_id in parent.child_ids:
                    if sibling_id not in visited and sibling_id in taxonomy:
                        queue.append((sibling_id, depth + 1))
                        visited.add(sibling_id)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_csv_records(filepath: Path, delimiter: str) -> List[Dict[str, str]]:
    """Read a delimited file into a list of row dicts."""
    records: List[Dict[str, str]] = []
    with open(filepath, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        for row in reader:
            records.append(dict(row))
    return records


def _load_json_records(filepath: Path) -> List[Dict[str, Any]]:
    """Read a JSON file that is either a list of objects or a single object."""
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Common ESCO JSON-LD pattern: top-level wrapper with a list inside.
        for key in ("skills", "hasTopConcept", "narrower", "data", "results"):
            if key in data and isinstance(data[key], list):
                return data[key]
        # Single-record fallback.
        return [data]
    raise ValueError(f"Unexpected JSON root type: {type(data).__name__}")


def _backfill_children(graph: TaxonomyGraph) -> None:
    """Populate ``child_ids`` from ``parent_ids`` across the entire graph."""
    for skill in graph.skills.values():
        for pid in skill.parent_ids:
            parent = graph.skills.get(pid)
            if parent is not None and skill.id not in parent.child_ids:
                parent.child_ids.append(skill.id)
