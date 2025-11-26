"""Knowledge-graph helpers for GraphRAG-lite retrieval."""

from __future__ import annotations

import logging
import re
from itertools import combinations
from typing import Iterable, List, Optional, Set

import yake
from bson import ObjectId
from bson.errors import InvalidId
from pymongo import ReturnDocument
from pymongo.collection import Collection

logger = logging.getLogger(__name__)

_KG_STOPWORDS: Set[str] = {
    # Generic structure words/headings
    "section",
    "chapter",
    "page",
    "pages",
    "figure",
    "fig",
    "table",
    "appendix",
    "contents",
    "overview",
    "summary",
    # Common boilerplate/empty markers
    "blank",
    "blank page",
    "intentionally blank",
    "blank for pagination",
    "number description",
    "not used",
    "reserved",
    # Month abbreviations often mis-read as headings
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
}

_KG_BAD_PATTERNS = [
    r"\bpage\s+\d+(\s+of\s+\d+)?\b",
    r"\bsection\s+\d+(\.\d+)*\b",
    r"\bappendix\s+[a-z0-9]+\b",
    r"\btable\s+\d+(\.\d+)*\b",
    r"\bfigure\s+\d+(\.\d+)*\b",
    r"\bintentionally\s+blank\b",
    r"\bblank\s+for\s+pagination\b",
    r"\bnumber\s+description\b",
]

def extract_keyphrases(
    text: str, max_k: int = 8, stopwords: Optional[Set[str]] = None
) -> List[str]:
    """Extract salient keyphrases from free text.

    Args:
        text: Raw text from which phrases will be extracted.
        max_k: Maximum number of phrases to return.
        stopwords: Optional set of phrases to ignore.

    Returns:
        list[str]: Keyphrases sorted by importance.
    """
    stops = stopwords or _KG_STOPWORDS
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    extractor = yake.KeywordExtractor(
        lan="en", top=max(10, max_k * 2), n=3, dedupLim=0.9
    )
    phrases: List[str] = []
    seen_labels: Set[str] = set()
    for phrase, _ in extractor.extract_keywords(cleaned):
        normalised = phrase.strip()
        if not normalised:
            continue
        label = _normalise_phrase(normalised)
        if (
            not label
            or len(label) < 3
            or label in seen_labels
            or label in stops
            or label.replace(" ", "").isdigit()
            or sum(ch.isalpha() for ch in label) < 3
        ):
            continue
        if any(re.search(pattern, label) for pattern in _KG_BAD_PATTERNS):
            continue
        seen_labels.add(label)
        phrases.append(normalised)
        if len(phrases) >= max_k:
            break
    logger.debug("Extracted %s keyphrases.", len(phrases))
    return phrases


def upsert_nodes_edges(
    nodes: Collection,
    edges: Collection,
    chunk_id: ObjectId,
    phrases: Iterable[str],
) -> None:
    """Ensure nodes and edges exist for the supplied phrases, linking back to chunks.

    Args:
        nodes: Knowledge graph nodes collection.
        edges: Knowledge graph edges collection.
        chunk_id: Identifier of the chunk being associated.
        phrases: Iterable of phrases describing the chunk.

    Returns:
        None: Persists updates to MongoDB collections.
    """
    node_ids: List[ObjectId] = []
    for phrase in phrases:
        label = _normalise_phrase(phrase)
        if not label:
            continue
        node = nodes.find_one_and_update(
            {"label": label},
            {"$setOnInsert": {"label": label, "type": "concept", "aliases": []}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        if not node:
            continue
        nodes.update_one({"_id": node["_id"]}, {"$addToSet": {"aliases": phrase}})
        edges.update_one(
            {"from_id": node["_id"], "to_id": node["_id"], "rel": "mentions"},
            {
                "$addToSet": {"chunk_ids": chunk_id},
                "$inc": {"weight": 1},
            },
            upsert=True,
        )
        node_ids.append(node["_id"])

    for left_id, right_id in combinations(node_ids, 2):
        _upsert_association(edges, left_id, right_id, chunk_id)
        _upsert_association(edges, right_id, left_id, chunk_id)
    if node_ids:
        logger.debug("Linked %s keyphrases to chunk '%s'.", len(node_ids), chunk_id)


def expand_candidate_chunk_ids(
    nodes: Collection,
    edges: Collection,
    phrases: Iterable[str],
    max_hops: int,
    max_candidates: int,
) -> Set[ObjectId]:
    """Expand phrases to a set of chunk identifiers via the knowledge graph.

    Args:
        nodes: Knowledge graph nodes collection.
        edges: Knowledge graph edges collection.
        phrases: Seed phrases derived from the user query.
        max_hops: Maximum number of hops when traversing associations.
        max_candidates: Maximum number of chunk identifiers to return.

    Returns:
        set[ObjectId]: Candidate chunk identifiers.
    """
    labels = {_normalise_phrase(phrase) for phrase in phrases if phrase.strip()}
    chunk_ids: Set[ObjectId] = set()
    frontier: Set[ObjectId] = set()
    for label in labels:
        node = nodes.find_one({"label": label})
        if not node:
            continue
        node_id = node["_id"]
        frontier.add(node_id)
        chunk_ids.update(_collect_chunk_ids(edges, node_id))
        if len(chunk_ids) >= max_candidates:
            capped = set(list(chunk_ids)[:max_candidates])
            logger.debug(
                "Graph expansion hit candidate cap at initial step: %s items.",
                len(capped),
            )
            return capped

    if max_hops <= 0:
        logger.debug(
            "Graph expansion produced %s chunk ids with zero hops.", len(chunk_ids)
        )
        return chunk_ids

    visited = set(frontier)
    hops = 0
    while frontier and hops < max_hops and len(chunk_ids) < max_candidates:
        hops += 1
        next_frontier: Set[ObjectId] = set()
        for node_id in frontier:
            for edge in edges.find({"from_id": node_id, "rel": "associated_with"}):
                target = edge.get("to_id")
                if target and target not in visited:
                    next_frontier.add(target)
                chunk_ids.update(edge.get("chunk_ids", []))
                if len(chunk_ids) >= max_candidates:
                    capped = set(list(chunk_ids)[:max_candidates])
                    logger.debug(
                        "Graph expansion reached cap while traversing associations: %s items.",
                        len(capped),
                    )
                    return capped
        frontier = next_frontier - visited
        visited.update(frontier)
        for node_id in frontier:
            chunk_ids.update(_collect_chunk_ids(edges, node_id))
            if len(chunk_ids) >= max_candidates:
                capped = set(list(chunk_ids)[:max_candidates])
                logger.debug(
                    "Graph expansion reached cap after collecting mentions: %s items.",
                    len(capped),
                )
                return capped
    logger.debug("Graph expansion produced %s chunk ids.", len(chunk_ids))
    return chunk_ids


def _collect_chunk_ids(edges: Collection, node_id: ObjectId) -> Set[ObjectId]:
    """Return chunk identifiers mentioned by the node.

    Args:
        edges: Knowledge graph edges collection.
        node_id: Identifier for the node of interest.

    Returns:
        set[ObjectId]: Chunk identifiers connected via mentions.
    """
    edge = edges.find_one({"from_id": node_id, "to_id": node_id, "rel": "mentions"})
    if not edge:
        return set()
    chunk_ids: Set[ObjectId] = set()
    for chunk in edge.get("chunk_ids", []):
        if isinstance(chunk, ObjectId):
            chunk_ids.add(chunk)
            continue
        try:
            chunk_ids.add(ObjectId(chunk))
        except InvalidId:
            continue
    logger.debug("Node '%s' references %s chunks.", node_id, len(chunk_ids))
    return chunk_ids


def _upsert_association(
    edges: Collection, from_id: ObjectId, to_id: ObjectId, chunk_id: ObjectId
) -> None:
    """Create or strengthen an association edge between two nodes.

    Args:
        edges: Knowledge graph edges collection.
        from_id: Source node identifier.
        to_id: Target node identifier.
        chunk_id: Chunk identifier providing the evidence.

    Returns:
        None: Persists changes to the edges collection.
    """
    edges.update_one(
        {"from_id": from_id, "to_id": to_id, "rel": "associated_with"},
        {
            "$inc": {"weight": 1},
            "$addToSet": {"chunk_ids": chunk_id},
        },
        upsert=True,
    )


def _normalise_phrase(phrase: str) -> str:
    """Normalise phrases for key comparisons.

    Args:
        phrase: Raw phrase string.

    Returns:
        str: Lowercased whitespace-normalised phrase.
    """
    return " ".join(phrase.lower().split())
