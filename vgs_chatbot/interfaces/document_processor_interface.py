"""Document processor interface for RAG pipeline."""

from abc import ABC, abstractmethod
from typing import List

from vgs_chatbot.models.document import Document, ProcessedDocument


class DocumentProcessorInterface(ABC):
    """Abstract interface for document processors."""
    
    @abstractmethod
    async def process_documents(self, documents: List[Document]) -> List[ProcessedDocument]:
        """Process documents into searchable format.
        
        Args:
            documents: List of raw documents
            
        Returns:
            List of processed documents
        """
        pass
    
    @abstractmethod
    async def search_documents(self, query: str, top_k: int = 5) -> List[ProcessedDocument]:
        """Search processed documents using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of relevant processed documents
        """
        pass
    
    @abstractmethod
    async def index_documents(self, processed_docs: List[ProcessedDocument]) -> None:
        """Index processed documents for fast retrieval.
        
        Args:
            processed_docs: List of processed documents to index
        """
        pass