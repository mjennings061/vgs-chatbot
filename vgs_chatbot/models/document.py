"""Document data models."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, HttpUrl


class Document(BaseModel):
    """Raw document model."""
    
    id: Optional[str] = None
    name: str
    url: HttpUrl
    file_type: str
    size: Optional[int] = None
    modified_date: Optional[datetime] = None
    directory_path: str
    

class ProcessedDocument(BaseModel):
    """Processed document model for RAG."""
    
    id: str
    original_document: Document
    content: str
    chunks: List[str]
    embeddings: Optional[List[List[float]]] = None
    metadata: dict = {}
    processed_at: datetime
    
    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True


class DocumentChunk(BaseModel):
    """Document chunk model."""
    
    id: str
    document_id: str
    chunk_index: int
    content: str
    embedding: Optional[List[float]] = None
    metadata: dict = {}