"""Tests for document processor."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from vgs_chatbot.services.document_processor import RAGDocumentProcessor
from vgs_chatbot.models.document import Document


@pytest.fixture
def document_processor():
    """Document processor instance."""
    with patch('vgs_chatbot.services.document_processor.SentenceTransformer'), \
         patch('vgs_chatbot.services.document_processor.chromadb'):
        processor = RAGDocumentProcessor()
        processor.embedding_model.encode = MagicMock(return_value=[[0.1, 0.2, 0.3]])
        return processor


@pytest.fixture
def sample_document():
    """Sample document for testing."""
    return Document(
        id="test-doc-1",
        name="test-document.txt",
        url="https://example.com/doc.txt",
        file_type="text/plain",
        size=1024,
        modified_date=datetime.utcnow(),
        directory_path="/test/path"
    )


@pytest.mark.asyncio
async def test_process_documents(document_processor, sample_document):
    """Test document processing."""
    # Arrange
    documents = [sample_document]
    
    # Mock text extraction
    document_processor._extract_text_content = AsyncMock(return_value="Test content")
    
    # Act
    result = await document_processor.process_documents(documents)
    
    # Assert
    assert len(result) == 1
    assert result[0].original_document == sample_document
    assert result[0].content == "Test content"
    assert len(result[0].chunks) > 0


@pytest.mark.asyncio
async def test_search_documents(document_processor):
    """Test document search."""
    # Arrange
    query = "test query"
    
    # Mock ChromaDB query
    document_processor.collection.query = MagicMock(return_value={
        "ids": [["doc1", "doc2"]],
        "documents": [["chunk1", "chunk2"]],
        "metadatas": [[{"document_name": "doc1.txt"}, {"document_name": "doc2.txt"}]]
    })
    
    # Act
    result = await document_processor.search_documents(query, top_k=2)
    
    # Assert
    assert len(result) == 2
    document_processor.collection.query.assert_called_once()


def test_split_text_into_chunks(document_processor):
    """Test text chunking."""
    # Arrange
    text = " ".join([f"word{i}" for i in range(2000)])  # Create long text
    
    # Act
    chunks = document_processor._split_text_into_chunks(text)
    
    # Assert
    assert len(chunks) > 1
    assert len(chunks[0].split()) <= document_processor.chunk_size