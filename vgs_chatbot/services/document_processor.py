"""Document processor service implementation with RAG pipeline."""

import uuid
from datetime import datetime
from typing import List, Dict, Any
from io import BytesIO

import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document as DocxDocument
import openpyxl

from vgs_chatbot.interfaces.document_processor_interface import DocumentProcessorInterface
from vgs_chatbot.models.document import Document, ProcessedDocument, DocumentChunk


class RAGDocumentProcessor(DocumentProcessorInterface):
    """RAG-based document processor implementation."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2") -> None:
        """Initialize document processor.
        
        Args:
            embedding_model: Name of sentence transformer model for embeddings
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chroma_client = chromadb.Client()
        # Delete existing collection if it exists to avoid schema conflicts
        try:
            self.chroma_client.delete_collection(name="documents")
        except:
            pass  # Collection might not exist
        
        self.collection = self.chroma_client.create_collection(
            name="documents"
        )
        self.chunk_size = 1000
        self.chunk_overlap = 200
    
    async def process_documents(self, documents: List[Document]) -> List[ProcessedDocument]:
        """Process documents into searchable format.
        
        Args:
            documents: List of raw documents
            
        Returns:
            List of processed documents
        """
        processed_docs = []
        
        for document in documents:
            try:
                processed_doc = await self._process_single_document(document)
                processed_docs.append(processed_doc)
            except Exception as e:
                # Log error but continue with other documents
                print(f"Error processing document {document.name}: {e}")
                continue
        
        return processed_docs
    
    async def _process_single_document(self, document: Document) -> ProcessedDocument:
        """Process a single document.
        
        Args:
            document: Document to process
            
        Returns:
            Processed document
        """
        # Extract text content based on file type
        content = await self._extract_text_content(document)
        
        # Split into chunks
        chunks = self._split_text_into_chunks(content)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks).tolist()
        
        processed_doc = ProcessedDocument(
            id=str(uuid.uuid4()),
            original_document=document,
            content=content,
            chunks=chunks,
            embeddings=embeddings,
            processed_at=datetime.utcnow()
        )
        
        return processed_doc
    
    async def _extract_text_content(self, document: Document) -> str:
        """Extract text content from document based on file type.
        
        Args:
            document: Document to extract text from
            
        Returns:
            Extracted text content
        """
        # This would normally download the document content
        # For now, we'll simulate with placeholder text
        content = f"Document: {document.name}\nPath: {document.directory_path}\n"
        
        if document.file_type == "application/pdf":
            content += await self._extract_pdf_text(b"")  # Placeholder
        elif document.file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
            content += await self._extract_docx_text(b"")  # Placeholder
        elif document.file_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
            content += await self._extract_excel_text(b"")  # Placeholder
        else:
            content += "Text document content"
        
        return content
    
    async def _extract_pdf_text(self, content: bytes) -> str:
        """Extract text from PDF content.
        
        Args:
            content: PDF file content as bytes
            
        Returns:
            Extracted text
        """
        if not content:
            return "PDF content placeholder"
        
        try:
            reader = PdfReader(BytesIO(content))
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        except Exception:
            return "Error extracting PDF text"
    
    async def _extract_docx_text(self, content: bytes) -> str:
        """Extract text from DOCX content.
        
        Args:
            content: DOCX file content as bytes
            
        Returns:
            Extracted text
        """
        if not content:
            return "DOCX content placeholder"
        
        try:
            doc = DocxDocument(BytesIO(content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception:
            return "Error extracting DOCX text"
    
    async def _extract_excel_text(self, content: bytes) -> str:
        """Extract text from Excel content.
        
        Args:
            content: Excel file content as bytes
            
        Returns:
            Extracted text
        """
        if not content:
            return "Excel content placeholder"
        
        try:
            workbook = openpyxl.load_workbook(BytesIO(content))
            text = ""
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                text += f"Sheet: {sheet_name}\n"
                for row in sheet.iter_rows(values_only=True):
                    row_text = " ".join(str(cell) for cell in row if cell is not None)
                    if row_text.strip():
                        text += row_text + "\n"
            return text
        except Exception:
            return "Error extracting Excel text"
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)
        
        return chunks
    
    async def search_documents(self, query: str, top_k: int = 5) -> List[ProcessedDocument]:
        """Search processed documents using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of relevant processed documents
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Convert results back to ProcessedDocument objects
        # This is simplified - in practice you'd store and retrieve full document data
        processed_docs = []
        for doc_id in results["ids"][0]:
            # Retrieve full document data from storage
            # For now, return placeholder
            processed_docs.append(ProcessedDocument(
                id=doc_id,
                original_document=Document(
                    name="Retrieved Document",
                    file_path="/search/results/retrieved_document",
                    file_type="text/plain",
                    directory_path="/search/results"
                ),
                content="Retrieved content",
                chunks=["Retrieved chunk"],
                processed_at=datetime.utcnow()
            ))
        
        return processed_docs
    
    async def index_documents(self, processed_docs: List[ProcessedDocument]) -> None:
        """Index processed documents for fast retrieval.
        
        Args:
            processed_docs: List of processed documents to index
        """
        for doc in processed_docs:
            # Store document chunks with embeddings in ChromaDB
            for i, (chunk, embedding) in enumerate(zip(doc.chunks, doc.embeddings)):
                chunk_id = f"{doc.id}_chunk_{i}"
                
                self.collection.add(
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        "document_id": doc.id,
                        "document_name": doc.original_document.name,
                        "chunk_index": i,
                        "file_type": doc.original_document.file_type,
                        "directory_path": doc.original_document.directory_path
                    }],
                    ids=[chunk_id]
                )