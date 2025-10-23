"""Knowledge-graph helpers for GraphRAG-lite retrieval."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, List, Set

import yake
from bson import ObjectId
from bson.errors import InvalidId
from pymongo import ReturnDocument
from pymongo.collection import Collection


def extract_keyphrases(text: str, max_k: int = 8) -> List[str]:
    """Extract salient keyphrases from free text."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    extractor = yake.KeywordExtractor(
        lan="en", top=max(10, max_k * 2), n=3, dedupLim=0.9
    )
    phrases: List[str] = []
    for phrase, _score in extractor.extract_keywords(cleaned):
        normalised = phrase.strip()
        if not normalised:
            continue
        lower = _normalise_phrase(normalised)
        if lower in {_normalise_phrase(p) for p in phrases}:
            continue
        phrases.append(normalised)
        if len(phrases) >= max_k:
            break
    return phrases


def upsert_nodes_edges(
    nodes: Collection,
    edges: Collection,
    chunk_id: ObjectId,
    phrases: Iterable[str],
) -> None:
    """Ensure nodes and edges exist for the supplied phrases, linking back to chunks."""
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
                "$setOnInsert": {"weight": 1},
            },
            upsert=True,
        )
        node_ids.append(node["_id"])

    for left_id, right_id in combinations(node_ids, 2):
        _upsert_association(edges, left_id, right_id, chunk_id)
        _upsert_association(edges, right_id, left_id, chunk_id)


def expand_candidate_chunk_ids(
    nodes: Collection,
    edges: Collection,
    phrases: Iterable[str],
    max_hops: int,
    max_candidates: int,
) -> Set[ObjectId]:
    """Expand phrases to a set of chunk identifiers via the knowledge graph."""
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
            return set(list(chunk_ids)[:max_candidates])

    if max_hops <= 0:
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
                    return set(list(chunk_ids)[:max_candidates])
        frontier = next_frontier - visited
        visited.update(frontier)
        for node_id in frontier:
            chunk_ids.update(_collect_chunk_ids(edges, node_id))
            if len(chunk_ids) >= max_candidates:
                return set(list(chunk_ids)[:max_candidates])
    return chunk_ids


def _collect_chunk_ids(edges: Collection, node_id: ObjectId) -> Set[ObjectId]:
    """Return chunk identifiers mentioned by the node."""
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
    return chunk_ids


def _upsert_association(
    edges: Collection, from_id: ObjectId, to_id: ObjectId, chunk_id: ObjectId
) -> None:
    """Create or strengthen an association edge between two nodes."""
    edges.update_one(
        {"from_id": from_id, "to_id": to_id, "rel": "associated_with"},
        {
            "$inc": {"weight": 1},
            "$addToSet": {"chunk_ids": chunk_id},
        },
        upsert=True,
    )


def _normalise_phrase(phrase: str) -> str:
    """Normalise phrases for key comparisons."""
    return " ".join(phrase.lower().split())
