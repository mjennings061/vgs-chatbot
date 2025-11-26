"""Document ingestion pipeline."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from typing import Callable, List, Optional, Sequence, Tuple

from bson import ObjectId
from gridfs import GridFS
from pymongo.collection import Collection

from vgs_chatbot.config import get_settings
from vgs_chatbot.embeddings import FastEmbedder
from vgs_chatbot.kg import extract_keyphrases, upsert_nodes_edges
from vgs_chatbot.utils_text import (
    chunk_text,
    clean_title,
    detect_sections,
    Section,
    read_docx,
    read_pdf,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Summary of an ingestion run."""

    document_id: ObjectId
    chunk_count: int
    page_count: int


def _extract_phrases_job(args: Tuple[str, int]) -> List[str]:
    """Worker helper for multiprocessing-safe phrase extraction."""
    text, max_k = args
    try:
        return extract_keyphrases(text, max_k=max_k)
    except Exception:  # noqa: BLE001 - worker safety
        logger.exception("Keyphrase extraction failed in worker.")
        return []


def ingest_file(
    *,
    fs: GridFS,
    documents: Collection,
    doc_chunks: Collection,
    kg_nodes: Collection,
    kg_edges: Collection,
    embedder: FastEmbedder,
    file_bytes: bytes,
    filename: str,
    content_type: str,
    uploaded_by: str,
    progress_callback: Optional[
        Callable[[str, Optional[int], Optional[int]], None]
    ] = None,
) -> IngestResult:
    """Store a document, chunk it, embed it, and update the knowledge graph.

    Args:
        fs: GridFS bucket that stores the uploaded document bytes.
        documents: MongoDB collection for document metadata.
        doc_chunks: Collection storing chunked content and embeddings.
        kg_nodes: Knowledge graph nodes collection.
        kg_edges: Knowledge graph edges collection.
        embedder: Embedder used for passage embeddings.
        file_bytes: Raw bytes of the uploaded document.
        filename: Human-readable file name.
        content_type: MIME type captured at upload time.
        uploaded_by: Username responsible for the upload.
        progress_callback: Optional progress hook for UI updates.

    Returns:
        IngestResult: Summary of the ingestion run.
    """

    def notify(
        stage: str, current: Optional[int] = None, total: Optional[int] = None
    ) -> None:
        """Forward progress updates to optional callback and log transitions."""
        if progress_callback:
            progress_callback(stage, current, total)
        if total is not None and current is not None:
            logger.debug("Ingestion stage '%s': %s/%s", stage, current, total)
        else:
            logger.debug("Ingestion stage '%s' updated.", stage)

    doc_type = _detect_doc_type(filename, content_type)
    settings = get_settings()
    title = filename.rsplit(".", 1)[0]
    title_key = clean_title(title).lower()
    logger.info(
        "Starting ingestion for '%s' detected as '%s' uploaded by '%s'.",
        filename,
        doc_type,
        uploaded_by,
    )
    notify("Loading pages")
    pages = _load_pages(doc_type, file_bytes)
    total_pages = len(pages)

    combined_text = _combine_pages_with_markers(pages)
    sections = detect_sections(combined_text)
    notify("Processing pages", total_pages, total_pages if total_pages else None)

    chunk_inputs: List[Tuple[str, Optional[str], str, str, int, int]] = []
    for section in sections or []:
        section_key = section.title.strip() or "General"
        section_page_start, section_page_end = _section_page_range(section)
        for fragment in chunk_text(
            blocks=section.blocks,
            target_chars=settings.chunk_target_chars,
            overlap=settings.chunk_overlap_chars,
        ):
            if not fragment.text:
                continue
            page_start = fragment.page_start
            page_end = fragment.page_end
            if page_start is None and section_page_start is not None:
                page_start = section_page_start
            if page_end is None and section_page_end is not None:
                page_end = section_page_end
            if page_start is None and page_end is not None:
                page_start = page_end
            if page_end is None and page_start is not None:
                page_end = page_start
            if page_start is None and page_end is None:
                page_start = page_end = 0

            if page_start is None:
                page_start = 0
            if page_end is None:
                page_end = 0

            chunk_inputs.append(
                (
                    section_key,
                    section.order_code,
                    fragment.kind,
                    fragment.text,
                    page_start,
                    page_end,
                )
            )

    total_chunks = len(chunk_inputs)
    notify("Chunking document", 0, total_chunks if total_chunks else None)

    chunk_docs: List[dict] = []
    section_ids: dict[str, ObjectId] = {}
    for chunk_index, (
        section_key,
        order_code,
        chunk_kind,
        chunk_text_value,
        page_start,
        page_end,
    ) in enumerate(chunk_inputs, start=1):
        section_id = section_ids.setdefault(section_key, ObjectId())
        chunk_docs.append(
            {
                "doc_title": title,
                "section_id": section_id,
                "section_title": section_key,
                "section_path": [section_key],
                "order_code": order_code,
                "chunk_type": chunk_kind,
                "page_start": page_start,
                "page_end": page_end,
                "text": chunk_text_value,
            }
        )
        notify("Chunking document", chunk_index, total_chunks)

    notify("Embedding chunks", 0, total_chunks if total_chunks else None)
    embeddings = embedder.embed_passages([doc["text"] for doc in chunk_docs])
    for index, (doc, vector) in enumerate(
        zip(chunk_docs, embeddings, strict=False), start=1
    ):
        doc["embedding"] = vector
        notify("Embedding chunks", index, total_chunks)

    grid_id: Optional[ObjectId] = None
    doc_id: Optional[ObjectId] = None
    chunk_ids: List[ObjectId] = []

    worker_count = max(1, (os.cpu_count() or 2) - 1)

    try:
        existing = documents.find_one({"title_key": title_key})
        if existing:
            doc_id = existing["_id"]
            chunk_ids = [
                chunk["_id"]
                for chunk in doc_chunks.find({"doc_id": doc_id}, {"_id": 1})
            ]
            if chunk_ids:
                doc_chunks.delete_many({"_id": {"$in": chunk_ids}})
                kg_edges.update_many(
                    {"chunk_ids": {"$in": chunk_ids}},
                    {"$pull": {"chunk_ids": {"$in": chunk_ids}}},
                )
                kg_edges.delete_many({"chunk_ids": {"$size": 0}})
            if existing.get("gridfs_id"):
                try:
                    fs.delete(existing["gridfs_id"])
                except Exception:  # noqa: BLE001 - best effort cleanup
                    logger.warning(
                        "Failed to delete previous GridFS file '%s'.",
                        existing["gridfs_id"],
                    )
            documents.delete_one({"_id": doc_id})

        notify("Uploading document")
        grid_id = fs.put(file_bytes, filename=filename, content_type=content_type)
        document = {
            "title": title,
            "title_key": title_key,
            "filename": filename,
            "doc_type": doc_type,
            "uploaded_by": uploaded_by,
            "uploaded_at": datetime.now(tz=timezone.utc),
            "gridfs_id": grid_id,
        }
        if doc_id:
            document["_id"] = doc_id
        doc_id = documents.insert_one(document).inserted_id

        if chunk_docs:
            for chunk_doc in chunk_docs:
                chunk_doc["doc_id"] = doc_id

        notify("Saving chunks", 0, total_chunks if total_chunks else None)
        result = doc_chunks.insert_many(chunk_docs) if chunk_docs else None
        if result:
            chunk_ids = list(result.inserted_ids)
        notify("Saving chunks", total_chunks, total_chunks if total_chunks else None)

        phrase_results: List[List[str]] = []
        if chunk_docs:
            jobs = [(doc["text"], settings.kg_max_phrases) for doc in chunk_docs]
            # Parallelise phrase extraction; DB writes remain serial to avoid contention.
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                phrase_results = list(executor.map(_extract_phrases_job, jobs))

        if chunk_ids and phrase_results:
            for index, (chunk_id, phrases) in enumerate(
                zip(chunk_ids, phrase_results, strict=False), start=1
            ):
                if phrases:
                    upsert_nodes_edges(kg_nodes, kg_edges, chunk_id, phrases)
                notify("Updating knowledge graph", index, total_chunks)

    except Exception:
        logger.exception("Ingestion failed; rolling back partial state for '%s'.", filename)
        _cleanup_failed_ingest(
            fs=fs,
            documents=documents,
            doc_chunks=doc_chunks,
            kg_edges=kg_edges,
            doc_id=doc_id,
            gridfs_id=grid_id,
            chunk_ids=chunk_ids,
        )
        raise

    notify("Completed ingestion")
    logger.info(
        "Completed ingestion for '%s': pages=%s chunks=%s",
        filename,
        len(pages),
        len(chunk_docs),
    )

    if not doc_id:
        raise RuntimeError("Ingestion completed without a valid document ID.")

    return IngestResult(
        document_id=doc_id,
        chunk_count=len(chunk_docs),
        page_count=len(pages),
    )


def _cleanup_failed_ingest(
    *,
    fs: GridFS,
    documents: Collection,
    doc_chunks: Collection,
    kg_edges: Collection,
    doc_id: Optional[ObjectId],
    gridfs_id: Optional[ObjectId],
    chunk_ids: List[ObjectId],
) -> None:
    """Best-effort rollback when persistence fails mid-ingestion."""

    if chunk_ids:
        logger.info("Removing %s chunk records after failure.", len(chunk_ids))
        doc_chunks.delete_many({"_id": {"$in": chunk_ids}})
        kg_edges.update_many(
            {"chunk_ids": {"$in": chunk_ids}},
            {"$pull": {"chunk_ids": {"$in": chunk_ids}}},
        )
        kg_edges.delete_many({"chunk_ids": {"$size": 0}})

    if doc_id:
        logger.info("Removing document metadata '%s' after failure.", doc_id)
        documents.delete_one({"_id": doc_id})

    if gridfs_id:
        try:
            logger.info("Deleting GridFS file '%s' after failure.", gridfs_id)
            fs.delete(gridfs_id)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            logger.warning("Failed to delete GridFS file '%s' during cleanup.", gridfs_id)


def _detect_doc_type(filename: str, content_type: str) -> str:
    """Infer document type from filename or MIME type."""
    lowered = filename.lower()
    if lowered.endswith(".pdf") or content_type == "application/pdf":
        return "pdf"
    if lowered.endswith(".docx") or "word" in content_type:
        return "docx"
    return "unknown"


def _load_pages(doc_type: str, file_bytes: bytes) -> List[Tuple[int, str]]:
    """Load pages based on document type."""
    if doc_type == "pdf":
        return read_pdf(file_bytes)
    if doc_type == "docx":
        return read_docx(file_bytes)
    # Unknown fallback: treat as plain text
    text = file_bytes.decode("utf-8", errors="ignore")
    return [(1, text)]


def _combine_pages_with_markers(pages: Sequence[Tuple[int, str]]) -> str:
    """Join page texts with explicit markers to preserve continuity."""
    parts: List[str] = []
    for page_no, text in pages:
        parts.append(f"[[[PAGE_BREAK_{page_no}]]]")
        parts.append(text.strip())
    return "\n".join(parts)


def _section_page_range(section: Section) -> tuple[int | None, int | None]:
    """Return min/max page range covered by a section."""
    pages: List[int] = []
    for block in section.blocks:
        if block.page_start is not None:
            pages.append(block.page_start)
        if block.page_end is not None:
            pages.append(block.page_end)
    if not pages:
        return (None, None)
    return (min(pages), max(pages))
