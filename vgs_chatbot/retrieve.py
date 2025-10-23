"""GraphRAG-lite retrieval helpers.

This module implements a lightweight, explainable retrieval stack used by the
Streamlit app:

- Extract keyphrases from the user query and expand to candidate chunk ids via
  a tiny knowledge-graph (GraphRAG-lite prior).
- Run Atlas Vector Search (semantic) and Atlas Search (BM25 text) in parallel.
- Optionally use a small lexical regex fallback when Atlas Search returns no
  hits (e.g. short queries like "wind limits").
- Apply targeted domain rewrites to help the embedder/text search match VGS
  phrasing (e.g. "launch"/"flight", GS).
- Fuse scores with simple, transparent weights and return ranked chunks for
  grounding the answer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Set, cast

from bson import ObjectId
from pymongo.collection import Collection

from vgs_chatbot.config import get_settings
from vgs_chatbot.embeddings import FastEmbedder
from vgs_chatbot.kg import expand_candidate_chunk_ids, extract_keyphrases
from vgs_chatbot.utils_text import clean_title

logger = logging.getLogger(__name__)

# Minimal stopword set used only for building compact lexical variants and
# fallback filters; not intended to be a comprehensive list.
_STOPWORDS: Set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "if",
    "in",
    "is",
    "it",
    "may",
    "of",
    "on",
    "or",
    "should",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


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
    source_scores: dict[str, float] = field(default_factory=dict)

    def as_citation(self) -> dict[str, str | int]:
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
) -> list[RetrievedChunk]:
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
    # 1) Build a small set of query phrases and get a KG-based prior over
    #    potentially-relevant chunks (fast set membership, not ranking).
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

    # 2) Semantic retrieval – steer the embedding slightly toward common domain
    #    phrasing before calling Vector Search.
    vector_query = _rewrite_query_for_vector(query)
    query_vector = embedder.embed_query(vector_query)

    vector_hits = _vector_search(
        collection=doc_chunks,
        query_vector=query_vector,
        limit=settings.retrieval_top_k,
        num_candidates=settings.retrieval_num_candidates,
        # Do not filter vector search to KG candidates; use KG only for scoring bonus
        filter_expr=None,
    )
    # 3) Textual retrieval – expand the query with simple domain synonyms and a
    #    condensed keyword version; then Atlas Search (BM25).
    expanded_terms = _expand_query_for_text(query)
    text_hits = _text_search(
        collection=doc_chunks,
        query=expanded_terms,
        limit=settings.retrieval_top_k,
    )
    # 3b) Lexical fallback – on very short queries Atlas Search can miss.
    #     Fall back to a strict AND regex over the chunk text to recover
    #     obvious matches (e.g. "wind limits").
    if not text_hits:
        text_hits = _lexical_search_fallback(
            collection=doc_chunks,
            query=query,
            limit=settings.retrieval_top_k,
        )
    # 4) Heuristic keyword pass – only for day-based limits to avoid unrelated
    #    noise. This helps questions like "how many flights in one day?".
    kw_hits: list[dict[str, Any]] = []
    if _should_apply_day_limit_heuristic(query):
        kw_hits = _keyword_hits(
            collection=doc_chunks,
            limit=max(12, settings.retrieval_top_k),
        )

    # 5) Score fusion – combine signals with simple weights and a small graph
    #    bonus without hard-filtering to the KG candidates.
    fused = _fuse_results(
        vector_hits=vector_hits,
        text_hits=text_hits,
        kw_hits=kw_hits,
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
    filter_expr: Optional[Mapping[str, Any]],
) -> list[dict[str, Any]]:
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
    results: list[dict[str, Any]] = list(collection.aggregate(pipeline))
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
    query: str | Sequence[str],
    limit: int,
) -> list[dict[str, Any]]:
    """Execute Atlas full-text search and return raw hits.

    Args:
        collection: MongoDB collection containing chunk documents.
        query: User query string used for text search.
        limit: Maximum number of results to return.

    Returns:
        list[dict]: Aggregation results with text scores.
    """
    settings = get_settings()
    # Support passing multiple query variants to Atlas Search
    query_input: str | list[str]
    if isinstance(query, str):
        query_input = query
    else:
        # Deduplicate while preserving order; keep the list short
        seen: Set[str] = set()
        unique: list[str] = []
        for q in query:
            qn = q.strip()
            if not qn or qn.lower() in seen:
                continue
            seen.add(qn.lower())
            unique.append(qn)
        query_input = unique or ""
    pipeline = [
        {
            "$search": {
                "index": settings.mongodb_search_index,
                "text": {
                    "query": query_input,
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
    results: list[dict[str, Any]] = list(collection.aggregate(pipeline))
    logger.debug("Text search returned %s hits (limit=%s).", len(results), limit)
    return results


def _keyword_hits(*, collection: Collection, limit: int) -> list[dict[str, Any]]:
    """Heuristic keyword matches for day-based launch/flight limits.

    Targets patterns like "one day"/"per day" co-occurring with "launch"/"flight".
    Uses regex find to avoid dependence on Atlas Search availability/synonyms.
    """
    # Two-stage AND: both day-phrase and launch/flight must appear in text.
    day_regex = {"$regex": r"(one day|1 day|per day)", "$options": "i"}
    lf_regex = {"$regex": r"(launch|flight)s?", "$options": "i"}
    cursor = collection.find(
        {"$and": [{"text": day_regex}, {"text": lf_regex}]},
        {
            "doc_id": 1,
            "doc_title": 1,
            "section_title": 1,
            "section_id": 1,
            "page_start": 1,
            "page_end": 1,
            "text": 1,
        },
    ).limit(limit)
    results: list[dict[str, Any]] = []
    for doc in cursor:
        # Provide a neutral base score; fusion will weight appropriately
        doc["score"] = 1.0
        results.append(doc)
    logger.debug("Keyword hits returned %s items (limit=%s).", len(results), limit)
    return results


def _fuse_results(
    *,
    vector_hits: list[dict[str, Any]],
    text_hits: list[dict[str, Any]],
    kw_hits: list[dict[str, Any]],
    kg_bonus_ids: Set[ObjectId],
) -> list[RetrievedChunk]:
    """Fuse vector, text, and knowledge graph signals.

    Args:
        vector_hits: Ranked results from vector search.
        text_hits: Ranked results from text search.
        kg_bonus_ids: Candidate chunk identifiers from the knowledge graph.

    Returns:
        list[RetrievedChunk]: Deduplicated and scored retrieval results.
    """
    fused: dict[ObjectId, RetrievedChunk] = {}

    # Prefer semantic similarity but keep weights transparent so the UI can
    # expose per-source contributions if desired.
    for item in vector_hits:
        chunk = _make_chunk(item)
        chunk.score = item.get("score", 0.0) * 0.7
        chunk.source_scores["vector"] = item.get("score", 0.0)
        fused[chunk.chunk_id] = chunk

    for item in text_hits:
        chunk_id = item.get("_id")
        if not isinstance(chunk_id, ObjectId):
            continue
        chunk = fused.get(chunk_id)
        if chunk is None:
            chunk = _make_chunk(item)
            fused[chunk.chunk_id] = chunk
        chunk.source_scores["text"] = item.get("score", 0.0)
        chunk.score += item.get("score", 0.0) * 0.3

    # Add keyword hits with a modest weight to promote clear day-limit rules.
    for item in kw_hits:
        chunk_id = item.get("_id")
        if not isinstance(chunk_id, ObjectId):
            continue
        chunk = fused.get(chunk_id)
        if chunk is None:
            chunk = _make_chunk(item)
            fused[chunk.chunk_id] = chunk
        chunk.source_scores["kw"] = item.get("score", 0.0)
        # Weight tuned to surface day-limit rules alongside vector hits.
        chunk.score += item.get("score", 0.0) * 0.7

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


def _make_chunk(item: dict[str, Any]) -> RetrievedChunk:
    """Create a `RetrievedChunk` instance from a MongoDB document.

    Args:
        item: Raw MongoDB result.

    Returns:
        RetrievedChunk: Structured retrieval item with default scores.
    """
    return RetrievedChunk(
        chunk_id=cast(ObjectId, item.get("_id")),
        doc_id=cast(ObjectId, item.get("doc_id")),
        doc_title=clean_title(item.get("doc_title", "")),
        section_title=clean_title(item.get("section_title", "")),
        page_start=item.get("page_start", 0),
        page_end=item.get("page_end", 0),
        text=item.get("text", ""),
        score=0.0,
    )


def _expand_query_for_text(query: str) -> list[str]:
    """Return simple synonym-expanded query variants for Atlas Search.

    The goal is to bridge domain phrasing (e.g., trainee vs student, flights vs launches).
    Keeps the list compact to avoid diluting scores.
    """
    base = query.strip()
    variants: list[str] = [base]

    # Include a condensed "keyword-only" variant (no stopwords/punctuation)
    # which often helps when the source uses compact headings or tables.
    focus_terms = _extract_focus_terms(base)
    if focus_terms:
        condensed = " ".join(focus_terms)
        if condensed and condensed.lower() != base.lower():
            variants.append(condensed)

    # Define targeted replacements (case-insensitive semantics applied manually)
    replacements = [
        ("student", "trainee"),
        ("trainee", "student"),
        ("launches", "flights"),
        ("launch", "flight"),
        ("flights", "launches"),
        ("flight", "launch"),
        (" gs ", " gliding scholarship "),
        ("gs ", "gliding scholarship "),
        (" gs", " gliding scholarship"),
    ]

    def ci_replace(text: str, old: str, new: str) -> str:
        return re.sub(re.escape(old), new, text, flags=re.IGNORECASE)

    # Build a few sensible single-replacement variants
    for old, new in replacements:
        v = ci_replace(base, old, new)
        if v != base:
            variants.append(v)

    # Add a compressed variant to help match briefer chunks
    compact = " ".join(base.split())
    if compact != base:
        variants.append(compact)

    # Limit to a small set to avoid noise
    max_variants = 8
    return variants[:max_variants]


def _rewrite_query_for_vector(query: str) -> str:
    """Apply targeted synonym rewrites for vector search.

    This steers the embedding toward common domain phrasing
    (e.g., trainee/flights/Gliding Scholarship) that aligns with source text.
    """
    text = query
    rules = [
        ("student", "trainee"),
        ("launches", "flights"),
        ("launch", "flight"),
        (r"\bGS\b", "Gliding Scholarship"),
    ]
    for old, new in rules:
        text = re.sub(old, new, text, flags=re.IGNORECASE)
    return text


def _should_apply_day_limit_heuristic(query: str) -> bool:
    """Return True when the keyword heuristic should run for day-based limits.

    Avoids over-triggering on unrelated queries by requiring both a day-style
    phrase and flight/launch terminology.
    """
    lowered = query.lower()
    day_signals = (
        "per day",
        "in a day",
        "one day",
        "1 day",
        "per-day",
        "perday",
        "daily",
    )
    if not any(signal in lowered for signal in day_signals):
        return False
    return any(
        term in lowered
        for term in ("launch", "launches", "flight", "flights", "sortie", "sorties")
    )


def _extract_focus_terms(query: str, max_terms: int = 4) -> list[str]:
    """Return meaningful tokens from a query for lexical search variants.

    Keeps only short, high-signal tokens and caps the count to avoid noisy
    regex fallbacks.
    """
    cleaned = query.strip().lower()
    if not cleaned:
        return []
    # Replace punctuation with spaces to isolate words
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    candidates = cleaned.split()
    terms: list[str] = []
    for token in candidates:
        if not token or token in _STOPWORDS or len(token) <= 2:
            continue
        if token not in terms:
            terms.append(token)
        if len(terms) >= max_terms:
            break
    return terms


def _lexical_search_fallback(
    *,
    collection: Collection,
    query: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Apply a simple regex AND search when Atlas Search yields no results.

    This is intentionally conservative and only considers up to the first
    three focus tokens. It scores by simple term-frequency to feed fusion.
    """
    terms = _extract_focus_terms(query)
    if not terms:
        return []
    projection = {
        "_id": 1,
        "doc_id": 1,
        "doc_title": 1,
        "section_title": 1,
        "section_id": 1,
        "page_start": 1,
        "page_end": 1,
        "text": 1,
    }
    # Attempt progressively less strict matches if necessary (3-term AND → 2 → 1).
    results: list[dict[str, Any]] = []
    for take in range(min(len(terms), 3), 0, -1):
        selected = terms[:take]
        regex_filters = []
        compiled: list[re.Pattern[str]] = []
        for term in selected:
            pattern = rf"\b{re.escape(term)}\b"
            regex_filters.append({"text": {"$regex": pattern, "$options": "i"}})
            compiled.append(re.compile(pattern, flags=re.IGNORECASE))
        if not regex_filters:
            continue
        if len(regex_filters) == 1:
            mongo_filter: Mapping[str, Any] = regex_filters[0]
        else:
            mongo_filter = {"$and": regex_filters}
        cursor = collection.find(mongo_filter, projection).limit(limit)
        docs = list(cursor)
        if not docs:
            continue
        for doc in docs:
            text = doc.get("text", "")
            score = 0.0
            for pattern in compiled:
                score += len(pattern.findall(text))
            doc["score"] = score or float(len(selected))
        results = docs
        break
    return results
