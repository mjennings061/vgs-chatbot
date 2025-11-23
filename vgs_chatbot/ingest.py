"""Document ingestion pipeline."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from typing import Callable, List, Optional, Tuple

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


def _process_page_job(args: Tuple[int, str, int, int]) -> List[Tuple[int, str, str]]:
    """Worker helper to detect sections and chunk a page."""
    page_no, text, target_chars, overlap = args
    sections = detect_sections(text) or [("Page", text)]
    chunks: List[Tuple[int, str, str]] = []
    for section_title, body in sections:
        section_key = section_title.strip() or f"Page {page_no}"
        for chunk in chunk_text(body, target_chars=target_chars, overlap=overlap):
            if chunk:
                chunks.append((page_no, section_key, chunk))
    return chunks


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
        """Forward progress updates to optional callback and log transitions.

        Args:
            stage: Name of the ingestion stage being reported.
            current: Current progress step for the stage.
            total: Total steps expected for the stage.

        Returns:
            None: Performs logging and optional callback invocation.
        """
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
    notify("Processing pages", 0, total_pages if total_pages else None)
    worker_count = max(1, (os.cpu_count() or 2) - 1)

    chunk_inputs: List[Tuple[int, str, str]] = []
    page_jobs = [
        (page_no, text, settings.chunk_target_chars, settings.chunk_overlap_chars)
        for page_no, text in pages
    ]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        for page_index, chunks_for_page in enumerate(
            executor.map(_process_page_job, page_jobs), start=1
        ):
            notify("Processing pages", page_index, total_pages if total_pages else None)
            if chunks_for_page:
                chunk_inputs.extend(chunks_for_page)

    total_chunks = len(chunk_inputs)
    notify("Chunking document", 0, total_chunks if total_chunks else None)

    chunk_docs: List[dict] = []
    section_ids: dict[str, ObjectId] = {}
    for chunk_index, (page_no, section_key, chunk_text_value) in enumerate(
        chunk_inputs, start=1
    ):
        section_id = section_ids.setdefault(section_key, ObjectId())
        chunk_docs.append(
            {
                "doc_title": title,
                "section_id": section_id,
                "section_title": section_key,
                "page_start": page_no,
                "page_end": page_no,
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
    """Infer document type from filename or MIME type.

    Args:
        filename: Name of the uploaded file.
        content_type: MIME type reported by the uploader.

    Returns:
        str: Normalised document type indicator.
    """
    lowered = filename.lower()
    if lowered.endswith(".pdf") or content_type == "application/pdf":
        return "pdf"
    if lowered.endswith(".docx") or "word" in content_type:
        return "docx"
    return "unknown"


def _load_pages(doc_type: str, file_bytes: bytes) -> List[Tuple[int, str]]:
    """Load pages based on document type.

    Args:
        doc_type: Normalised document type (pdf, docx, unknown).
        file_bytes: Raw file contents.

    Returns:
        list[tuple[int, str]]: Sequence of page numbers and extracted text.
    """
    if doc_type == "pdf":
        return read_pdf(file_bytes)
    if doc_type == "docx":
        return read_docx(file_bytes)
    # Unknown fallback: treat as plain text
    text = file_bytes.decode("utf-8", errors="ignore")
    return [(1, text)]
