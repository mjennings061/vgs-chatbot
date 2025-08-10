"""Document connector interface for SharePoint integration."""

from abc import ABC, abstractmethod
from typing import List

from vgs_chatbot.models.document import Document


class DocumentConnectorInterface(ABC):
    """Abstract interface for document connectors."""
    
    @abstractmethod
    async def connect(self, username: str, password: str) -> bool:
        """Connect to document source.
        
        Args:
            username: Username for authentication
            password: Password for authentication
            
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def list_documents(self, directory_urls: List[str]) -> List[Document]:
        """List all documents from specified directories.
        
        Args:
            directory_urls: List of directory URLs to search
            
        Returns:
            List of Document objects
        """
        pass
    
    @abstractmethod
    async def download_document(self, document: Document) -> bytes:
        """Download document content.
        
        Args:
            document: Document to download
            
        Returns:
            Document content as bytes
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from document source."""
        pass