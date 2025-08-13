"""Vector document models for MongoDB storage."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class VectorDocument(BaseModel):
    """Vector document model for MongoDB storage."""

    id: str | None = Field(default=None, alias="_id")
    document_id: str
    chunk_id: str
    content: str
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)
    section_title: str | None = None
    page_number: int | None = None
    chunk_index: int | None = None
    key_terms: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    class Config:
        """Pydantic configuration."""
        populate_by_name = True
        arbitrary_types_allowed = True


class DocumentSummary(BaseModel):
    """Document summary model for MongoDB storage."""

    id: str | None = Field(default=None, alias="_id")
    document_id: str
    summary_text: str
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    class Config:
        """Pydantic configuration."""
        populate_by_name = True
        arbitrary_types_allowed = True


class VectorSearchResult(BaseModel):
    """Vector search result model."""

    document_id: str
    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    section_title: str | None = None
    page_number: int | None = None
