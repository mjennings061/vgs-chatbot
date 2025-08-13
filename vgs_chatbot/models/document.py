"""Document data models."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class Document(BaseModel):
    """Raw document model."""

    id: str | None = None
    name: str
    file_path: str | None = None  # Optional for MongoDB stored documents
    file_type: str
    size: int | None = None
    modified_date: datetime | None = None
    directory_path: str | None = None  # Optional for MongoDB stored documents
    file_content: bytes | None = None  # New field for storing file content in MongoDB
    uploaded_at: datetime | None = None  # Track upload timestamp

    model_config = {"str_strip_whitespace": True}


class ProcessedDocument(BaseModel):
    """Processed document model for RAG."""

    id: str
    original_document: Document
    content: str
    chunks: list[str]
    embeddings: list[list[float]] | None = None
    metadata: dict[str, Any] = {}
    processed_at: datetime

    model_config = {"arbitrary_types_allowed": True}


class DocumentChunk(BaseModel):
    """Document chunk model."""

    id: str
    document_id: str
    chunk_index: int
    content: str
    section_title: str | None = None
    page_number: int | None = None
    embedding: list[float] | None = None
    metadata: dict[str, Any] = {}
