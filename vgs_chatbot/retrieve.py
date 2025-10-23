"""GraphRAG-lite retrieval helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from bson import ObjectId
from pymongo.collection import Collection

from vgs_chatbot.config import get_settings
from vgs_chatbot.embeddings import FastEmbedder
from vgs_chatbot.kg import expand_candidate_chunk_ids, extract_keyphrases

logger = logging.getLogger(__name__)


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
        """Return citation metadata for UI rendering.

        Returns:
            dict[str, str | int]: Minimal metadata for citation display.
        """
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
    """Run GraphRAG-lite retrieval for a user query.

    Args:
        doc_chunks: Collection containing chunk documents.
        kg_nodes: Knowledge graph nodes collection.
        kg_edges: Knowledge graph edges collection.
        embedder: Embedder used for vector search queries.
        query: User question driving retrieval.

    Returns:
        list[RetrievedChunk]: Ranked retrieval results.
    """
    if not query.strip():
        logger.warning("retrieve_chunks invoked with empty query.")
        return []

    settings = get_settings()
    phrases = extract_keyphrases(query, max_k=5)
    logger.debug("Query '%s' produced %s keyphrases.", query, len(phrases))
    candidate_ids = expand_candidate_chunk_ids(
        nodes=kg_nodes,
        edges=kg_edges,
        phrases=phrases,
        max_hops=settings.graph_max_hops,
        max_candidates=settings.graph_max_candidates,
    )
    logger.debug("Candidate set contains %s chunk ids.", len(candidate_ids))

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
    limited = fused[: settings.retrieval_top_k]
    logger.info(
        "Retrieve returned %s chunks (vector=%s, text=%s, query='%s').",
        len(limited),
        len(vector_hits),
        len(text_hits),
        query,
    )
    return limited


def _vector_search(
    *,
    collection: Collection,
    query_vector: Sequence[float],
    limit: int,
    num_candidates: int,
    filter_expr: Optional[dict],
) -> List[dict]:
    """Run MongoDB Atlas Vector Search and return raw hits.

    Args:
        collection: MongoDB collection containing chunk documents.
        query_vector: Embedding vector for the user query.
        limit: Maximum number of results to return.
        num_candidates: Number of candidate vectors examined by Atlas.
        filter_expr: Optional filter to restrict searched documents.

    Returns:
        list[dict]: Aggregation results with vector scores.
    """
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
    results = list(collection.aggregate(pipeline))
    logger.debug(
        "Vector search returned %s hits (limit=%s, candidates=%s).",
        len(results),
        limit,
        num_candidates,
    )
    return results


def _text_search(
    *,
    collection: Collection,
    query: str,
    limit: int,
) -> List[dict]:
    """Execute Atlas full-text search and return raw hits.

    Args:
        collection: MongoDB collection containing chunk documents.
        query: User query string used for text search.
        limit: Maximum number of results to return.

    Returns:
        list[dict]: Aggregation results with text scores.
    """
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
    results = list(collection.aggregate(pipeline))
    logger.debug("Text search returned %s hits (limit=%s).", len(results), limit)
    return results


def _fuse_results(
    *,
    vector_hits: List[dict],
    text_hits: List[dict],
    kg_bonus_ids: Set[ObjectId],
) -> List[RetrievedChunk]:
    """Fuse vector, text, and knowledge graph signals.

    Args:
        vector_hits: Ranked results from vector search.
        text_hits: Ranked results from text search.
        kg_bonus_ids: Candidate chunk identifiers from the knowledge graph.

    Returns:
        list[RetrievedChunk]: Deduplicated and scored retrieval results.
    """
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

    fused_list = sorted(fused.values(), key=lambda chunk: chunk.score, reverse=True)
    logger.debug(
        "Fusion produced %s combined results (vector=%s, text=%s, kg_bonus=%s).",
        len(fused_list),
        len(vector_hits),
        len(text_hits),
        len(kg_bonus_ids),
    )
    return fused_list


def _make_chunk(item: dict) -> RetrievedChunk:
    """Create a `RetrievedChunk` instance from a MongoDB document.

    Args:
        item: Raw MongoDB result.

    Returns:
        RetrievedChunk: Structured retrieval item with default scores.
    """
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
