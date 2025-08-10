"""SharePoint connector service implementation."""

import mimetypes
from datetime import datetime
from typing import List
from urllib.parse import urlparse

from office365.runtime.auth.authentication_context import AuthenticationContext
from office365.sharepoint.client_context import ClientContext
from office365.sharepoint.files.file import File
from pydantic import HttpUrl

from vgs_chatbot.interfaces.document_connector_interface import DocumentConnectorInterface
from vgs_chatbot.models.document import Document


class SharePointConnector(DocumentConnectorInterface):
    """SharePoint document connector implementation."""
    
    def __init__(self) -> None:
        """Initialize SharePoint connector."""
        self.client_context: ClientContext = None
        self.is_connected = False
    
    async def connect(self, username: str, password: str) -> bool:
        """Connect to SharePoint using user credentials.
        
        Args:
            username: SharePoint username
            password: SharePoint password
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # This will be set when we have actual SharePoint URLs
            site_url = "https://your-sharepoint-site.com"
            
            auth_context = AuthenticationContext(site_url)
            if auth_context.acquire_token_for_user(username, password):
                self.client_context = ClientContext(site_url, auth_context)
                self.is_connected = True
                return True
            
            return False
        
        except Exception:
            self.is_connected = False
            return False
    
    async def list_documents(self, directory_urls: List[str]) -> List[Document]:
        """List all documents from specified SharePoint directories.
        
        Args:
            directory_urls: List of SharePoint directory URLs
            
        Returns:
            List of Document objects
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to SharePoint")
        
        documents = []
        
        for directory_url in directory_urls:
            try:
                documents.extend(await self._list_documents_in_directory(directory_url))
            except Exception:
                # Log error but continue with other directories
                continue
        
        return documents
    
    async def _list_documents_in_directory(self, directory_url: str) -> List[Document]:
        """List documents in a specific directory.
        
        Args:
            directory_url: SharePoint directory URL
            
        Returns:
            List of documents in directory
        """
        documents = []
        parsed_url = urlparse(directory_url)
        relative_url = parsed_url.path
        
        # Get folder from SharePoint
        folder = self.client_context.web.get_folder_by_server_relative_url(relative_url)
        files = folder.files
        self.client_context.load(files)
        self.client_context.execute_query()
        
        for file in files:
            # Load file properties
            self.client_context.load(file)
            self.client_context.execute_query()
            
            # Get file type from name
            file_type = mimetypes.guess_type(file.properties["Name"])[0] or "unknown"
            
            document = Document(
                id=file.properties.get("UniqueId"),
                name=file.properties["Name"],
                url=HttpUrl(f"{self.client_context.service_root_url()}{file.serverRelativeUrl}"),
                file_type=file_type,
                size=file.properties.get("Length"),
                modified_date=self._parse_sharepoint_date(file.properties.get("TimeLastModified")),
                directory_path=directory_url
            )
            
            documents.append(document)
        
        # Recursively get subdirectories
        subfolders = folder.folders
        self.client_context.load(subfolders)
        self.client_context.execute_query()
        
        for subfolder in subfolders:
            if not subfolder.properties["Name"].startswith("_"):  # Skip system folders
                subfolder_url = f"{directory_url}/{subfolder.properties['Name']}"
                documents.extend(await self._list_documents_in_directory(subfolder_url))
        
        return documents
    
    async def download_document(self, document: Document) -> bytes:
        """Download document content from SharePoint.
        
        Args:
            document: Document to download
            
        Returns:
            Document content as bytes
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to SharePoint")
        
        # Extract relative URL from full URL
        parsed_url = urlparse(str(document.url))
        relative_url = parsed_url.path
        
        # Get file from SharePoint
        file = self.client_context.web.get_file_by_server_relative_url(relative_url)
        content = file.read()
        self.client_context.execute_query()
        
        return content
    
    async def disconnect(self) -> None:
        """Disconnect from SharePoint."""
        self.client_context = None
        self.is_connected = False
    
    def _parse_sharepoint_date(self, date_string: str) -> datetime:
        """Parse SharePoint date string to datetime.
        
        Args:
            date_string: SharePoint date string
            
        Returns:
            Parsed datetime object
        """
        if not date_string:
            return None
        
        try:
            # SharePoint typically uses ISO format
            return datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        except ValueError:
            return None