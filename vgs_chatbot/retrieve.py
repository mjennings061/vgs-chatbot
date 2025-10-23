"""GraphRAG-lite retrieval helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from bson import ObjectId
from pymongo.collection import Collection

from vgs_chatbot.config import get_settings
from vgs_chatbot.embeddings import FastEmbedder
from vgs_chatbot.kg import expand_candidate_chunk_ids, extract_keyphrases


@dataclass
class RetrievedChunk:
    """Container for a retrieved chunk and fused score."""

    chunk_id: ObjectId
    doc_id: ObjectId
    doc_title: str
    section_title: str
    page_start: int
    page_end: int
    text: str
    score: float
    source_scores: Dict[str, float] = field(default_factory=dict)

    def as_citation(self) -> Dict[str, str | int]:
        """Return citation metadata for UI rendering."""
        return {
            "chunk_id": str(self.chunk_id),
            "document": self.doc_title,
            "section": self.section_title,
            "page": self.page_start,
        }


def retrieve_chunks(
    *,
    doc_chunks: Collection,
    kg_nodes: Collection,
    kg_edges: Collection,
    embedder: FastEmbedder,
    query: str,
) -> List[RetrievedChunk]:
    """Run GraphRAG-lite retrieval for a user query."""
    if not query.strip():
        return []

    settings = get_settings()
    phrases = extract_keyphrases(query, max_k=5)
    candidate_ids = expand_candidate_chunk_ids(
        nodes=kg_nodes,
        edges=kg_edges,
        phrases=phrases,
        max_hops=settings.graph_max_hops,
        max_candidates=settings.graph_max_candidates,
    )

    query_vector = embedder.embed_query(query)
    filter_expr = None
    if 0 < len(candidate_ids) <= settings.graph_max_candidates:
        filter_expr = {"_id": {"$in": list(candidate_ids)}}

    vector_hits = _vector_search(
        collection=doc_chunks,
        query_vector=query_vector,
        limit=settings.retrieval_top_k,
        num_candidates=settings.retrieval_num_candidates,
        filter_expr=filter_expr,
    )
    text_hits = _text_search(
        collection=doc_chunks,
        query=query,
        limit=settings.retrieval_top_k,
    )
    fused = _fuse_results(
        vector_hits=vector_hits,
        text_hits=text_hits,
        kg_bonus_ids=candidate_ids,
    )
    return fused[: settings.retrieval_top_k]


def _vector_search(
    *,
    collection: Collection,
    query_vector: Sequence[float],
    limit: int,
    num_candidates: int,
    filter_expr: Optional[dict],
) -> List[dict]:
    settings = get_settings()
    stage = {
        "$vectorSearch": {
            "index": settings.mongodb_vector_index,
            "path": "embedding",
            "queryVector": query_vector,
            "numCandidates": num_candidates,
            "limit": limit,
        }
    }
    if filter_expr:
        stage["$vectorSearch"]["filter"] = filter_expr
    pipeline = [
        stage,
        {
            "$project": {
                "_id": 1,
                "doc_id": 1,
                "doc_title": 1,
                "section_title": 1,
                "section_id": 1,
                "page_start": 1,
                "page_end": 1,
                "text": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(collection.aggregate(pipeline))


def _text_search(
    *,
    collection: Collection,
    query: str,
    limit: int,
) -> List[dict]:
    settings = get_settings()
    pipeline = [
        {
            "$search": {
                "index": settings.mongodb_search_index,
                "text": {
                    "query": query,
                    "path": ["text", "section_title", "doc_title"],
                },
            }
        },
        {"$limit": limit},
        {
            "$project": {
                "_id": 1,
                "doc_id": 1,
                "doc_title": 1,
                "section_title": 1,
                "section_id": 1,
                "page_start": 1,
                "page_end": 1,
                "text": 1,
                "score": {"$meta": "searchScore"},
            }
        },
    ]
    return list(collection.aggregate(pipeline))


def _fuse_results(
    *,
    vector_hits: List[dict],
    text_hits: List[dict],
    kg_bonus_ids: Set[ObjectId],
) -> List[RetrievedChunk]:
    fused: Dict[ObjectId, RetrievedChunk] = {}

    for item in vector_hits:
        chunk = _make_chunk(item)
        chunk.score = item.get("score", 0.0) * 0.7
        chunk.source_scores["vector"] = item.get("score", 0.0)
        fused[chunk.chunk_id] = chunk

    for item in text_hits:
        chunk_id = item.get("_id")
        chunk = fused.get(chunk_id)
        if chunk is None:
            chunk = _make_chunk(item)
            fused[chunk.chunk_id] = chunk
        chunk.source_scores["text"] = item.get("score", 0.0)
        chunk.score += item.get("score", 0.0) * 0.3

    for chunk_id in kg_bonus_ids:
        if chunk_id in fused:
            fused[chunk_id].source_scores["kg"] = (
                fused[chunk_id].source_scores.get("kg", 0.0) + 1.0
            )
            fused[chunk_id].score += 0.1

    return sorted(fused.values(), key=lambda chunk: chunk.score, reverse=True)


def _make_chunk(item: dict) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=item.get("_id"),
        doc_id=item.get("doc_id"),
        doc_title=item.get("doc_title", ""),
        section_title=item.get("section_title", ""),
        page_start=item.get("page_start", 0),
        page_end=item.get("page_end", 0),
        text=item.get("text", ""),
        score=0.0,
    )
