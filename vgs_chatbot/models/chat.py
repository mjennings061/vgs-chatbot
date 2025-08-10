"""Chat data models."""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class MessageRole(str, Enum):
    """Chat message roles."""
    
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """Chat message model."""
    
    id: Optional[str] = None
    role: MessageRole
    content: str
    timestamp: datetime
    user_id: Optional[int] = None
    

class ChatResponse(BaseModel):
    """Chat response model."""
    
    message: str
    sources: List[str] = []
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    

class ChatSession(BaseModel):
    """Chat session model."""
    
    id: str
    user_id: int
    messages: List[ChatMessage] = []
    created_at: datetime
    updated_at: datetime
    is_active: bool = True