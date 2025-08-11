"""Chat data models."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class SourceReference(BaseModel):
    """Source reference model for chat responses."""

    document_name: str
    section_title: str | None = None
    page_number: int | None = None


class MessageRole(str, Enum):
    """Chat message roles."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """Chat message model."""

    id: str | None = None
    role: MessageRole
    content: str
    timestamp: datetime
    user_id: int | None = None
    sources: list[str] = []
    source_references: list[SourceReference] = []


class ChatResponse(BaseModel):
    """Chat response model."""

    message: str
    sources: list[str] = []
    source_references: list[SourceReference] = []
    confidence: float | None = None
    processing_time: float | None = None


class ChatSession(BaseModel):
    """Chat session model."""

    id: str
    user_id: int
    messages: list[ChatMessage] = []
    created_at: datetime
    updated_at: datetime
    is_active: bool = True
