"""Tests for document processor."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vgs_chatbot.models.document import Document
from vgs_chatbot.services.document_processor import RAGDocumentProcessor


@pytest.fixture
def document_processor():
    """Document processor instance."""
    import numpy as np
    with patch('vgs_chatbot.services.document_processor.SentenceTransformer') as mock_st, \
         patch('vgs_chatbot.services.document_processor.chromadb') as mock_chroma:

        # Setup mocks
        mock_model = MagicMock()
        mock_model.encode = MagicMock(return_value=np.array([[0.1, 0.2, 0.3]]))
        mock_st.return_value = mock_model

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_client.delete_collection = MagicMock()
        mock_chroma.Client.return_value = mock_client

        processor = RAGDocumentProcessor()
        return processor


@pytest.fixture
def sample_document():
    """Sample document for testing."""
    return Document(
        name="test-document.txt",
        file_path="/test/path/test-document.txt",
        file_type="text/plain",
        size=1024,
        modified_date=datetime.now(UTC),
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
        "ids": [["doc1_chunk_0", "doc2_chunk_0"]],
        "documents": [["chunk1 content", "chunk2 content"]],
        "metadatas": [[
            {
                "document_id": "doc1",
                "document_name": "doc1.txt",
                "file_type": "text/plain",
                "directory_path": "/test/path"
            },
            {
                "document_id": "doc2",
                "document_name": "doc2.txt",
                "file_type": "text/plain",
                "directory_path": "/test/path"
            }
        ]]
    })
    document_processor.collection.count = MagicMock(return_value=5)

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
