"""Document ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, List, Optional, Tuple

from bson import ObjectId
from gridfs import GridFS
from pymongo.collection import Collection

from vgs_chatbot.embeddings import FastEmbedder
from vgs_chatbot.kg import extract_keyphrases, upsert_nodes_edges
from vgs_chatbot.utils_text import chunk_text, detect_sections, read_docx, read_pdf


@dataclass
class IngestResult:
    """Summary of an ingestion run."""

    document_id: ObjectId
    chunk_count: int
    page_count: int


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
    """Store a document, chunk it, embed it, and update the knowledge graph."""

    def notify(
        stage: str, current: Optional[int] = None, total: Optional[int] = None
    ) -> None:
        if progress_callback:
            progress_callback(stage, current, total)

    doc_type = _detect_doc_type(filename, content_type)
    title = filename.rsplit(".", 1)[0]
    notify("Uploading document")
    grid_id = fs.put(file_bytes, filename=filename, content_type=content_type)
    document = {
        "title": title,
        "filename": filename,
        "doc_type": doc_type,
        "uploaded_by": uploaded_by,
        "uploaded_at": datetime.now(tz=timezone.utc),
        "gridfs_id": grid_id,
    }
    doc_id = documents.insert_one(document).inserted_id

    notify("Loading pages")
    pages = _load_pages(doc_type, file_bytes)
    total_pages = len(pages)
    notify("Processing pages", 0, total_pages if total_pages else None)

    chunk_inputs: List[Tuple[int, str, str]] = []
    for page_index, (page_no, text) in enumerate(pages, start=1):
        notify("Processing pages", page_index, total_pages if total_pages else None)
        sections = detect_sections(text) or [("Page", text)]
        for section_title, body in sections:
            section_key = section_title.strip() or f"Page {page_no}"
            for chunk in chunk_text(body):
                if not chunk:
                    continue
                chunk_inputs.append((page_no, section_key, chunk))

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
                "doc_id": doc_id,
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

    notify("Saving chunks", 0, total_chunks if total_chunks else None)
    result = doc_chunks.insert_many(chunk_docs) if chunk_docs else None
    notify("Saving chunks", total_chunks, total_chunks if total_chunks else None)

    if result:
        for index, (chunk_id, chunk_doc) in enumerate(
            zip(result.inserted_ids, chunk_docs, strict=False), start=1
        ):
            phrases = extract_keyphrases(chunk_doc["text"], max_k=8)
            if phrases:
                upsert_nodes_edges(kg_nodes, kg_edges, chunk_id, phrases)
            notify("Updating knowledge graph", index, total_chunks)

    notify("Completed ingestion")

    return IngestResult(
        document_id=doc_id,
        chunk_count=len(chunk_docs),
        page_count=len(pages),
    )


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
