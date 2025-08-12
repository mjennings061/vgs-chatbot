"""Integration test for wind limits query that should not return 'sorry' response."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from vgs_chatbot.models.chat import ChatMessage, MessageRole
from vgs_chatbot.models.document import Document
from vgs_chatbot.services.chat_service import LLMChatService
from vgs_chatbot.services.document_processor import RAGDocumentProcessor


@pytest.fixture
def sample_wind_limits_document():
    """Create a sample document with wind limits information."""
    return Document(
        name="2FTS DHOs Issue 3.pdf",
        file_path="/test/documents/2FTS DHOs Issue 3.pdf",
        file_type="application/pdf",
        directory_path="/test/documents"
    )


@pytest.fixture
def sample_wind_limits_content():
    """Sample content containing wind limits information."""
    return """
--- PAGE 15 ---
WEATHER CONDITIONS AND LIMITS

Wind Limits for VGS Operations:

Maximum crosswind component: 15 knots
Maximum total wind speed: 25 knots
Gusting conditions: Not to exceed 20% above steady wind
Turbulence: Light turbulence acceptable, moderate and above prohibited

Additional Weather Criteria:
- Visibility minimum: 5km
- Cloud base minimum: 1500 feet AGL
- No precipitation during operations

--- PAGE 16 ---
EMERGENCY PROCEDURES

In the event of weather deterioration beyond limits:
1. Cease flying operations immediately
2. Secure all aircraft
3. Report to duty instructor

The wind limits are established for safety and must not be exceeded under any circumstances.
Maximum demonstrated crosswind component during training: 12 knots
"""


class TestWindLimitsIntegration:
    """Integration tests for wind limits queries."""

    @pytest.mark.asyncio
    async def test_wind_limits_query_success(self, sample_wind_limits_document, sample_wind_limits_content):
        """Test that wind limits query returns specific information without 'sorry'."""

        # Mock the document processor to return our test content
        with patch('vgs_chatbot.services.document_processor.RAGDocumentProcessor._extract_text_content') as mock_extract:
            mock_extract.return_value = sample_wind_limits_content

            # Initialize document processor
            doc_processor = RAGDocumentProcessor()

            # Process the document
            processed_docs = await doc_processor.process_documents([sample_wind_limits_document])

            # Verify we have processed documents
            assert len(processed_docs) == 1
            processed_doc = processed_docs[0]

            # Verify content contains wind limits
            assert "wind limit" in processed_doc.content.lower()
            assert "25 knots" in processed_doc.content
            assert "15 knots" in processed_doc.content

            # Verify chunks contain relevant information
            wind_chunks = [chunk for chunk in processed_doc.chunks if "wind" in chunk.lower()]
            assert len(wind_chunks) > 0, "Should have chunks containing wind information"

            # Index documents
            await doc_processor.index_documents(processed_docs)

            # Test search functionality
            search_results = await doc_processor.search_documents("wind limits", top_k=3)
            assert len(search_results) > 0, "Should find relevant documents for wind limits query"

            # Mock the LLM response to simulate proper reasoning
            with patch('vgs_chatbot.services.chat_service.ChatOpenAI') as mock_llm:
                # Mock LLM to return a proper answer about wind limits
                mock_response = AsyncMock()
                mock_response.content = """Based on the VGS documentation, the wind limits for operations are:

- Maximum crosswind component: 15 knots
- Maximum total wind speed: 25 knots
- Gusting conditions: Not to exceed 20% above steady wind
- Maximum demonstrated crosswind component during training: 12 knots

These limits are established for safety and must not be exceeded under any circumstances. In moderate to severe turbulence, operations are prohibited."""

                mock_llm_instance = AsyncMock()
                mock_llm_instance.ainvoke = AsyncMock(return_value=mock_response)
                mock_llm.return_value = mock_llm_instance

                # Initialize chat service
                chat_service = LLMChatService("fake_key")

                # Create test message
                test_message = ChatMessage(
                    role=MessageRole.USER,
                    content="What are the wind limits?",
                    timestamp=datetime.now(UTC)
                )

                # Generate response
                response = await chat_service.generate_response(
                    messages=[test_message],
                    context_documents=search_results
                )

                # Print response for debugging
                print(f"Generated response: {response.message}")
                print(f"Response confidence: {response.confidence}")

                # Verify response does not contain 'sorry'
                assert "sorry" not in response.message.lower(), f"Response should not contain 'sorry': {response.message}"

                # Verify response contains specific wind limit information
                assert "15 knots" in response.message, f"Response should contain crosswind limit: {response.message}"
                assert "25 knots" in response.message, f"Response should contain total wind speed limit: {response.message}"

                # Verify response has confidence > 0
                assert response.confidence > 0, "Response should have confidence > 0"

                # Verify sources are included
                assert len(response.sources) > 0, "Response should include sources"
                assert "2FTS DHOs Issue 3.pdf" in response.sources[0], "Should reference the correct document"

    @pytest.mark.asyncio
    async def test_document_chunking_preserves_wind_limits(self, sample_wind_limits_content):
        """Test that document chunking preserves wind limits information together."""

        doc_processor = RAGDocumentProcessor()
        chunks = doc_processor._split_text_into_chunks(sample_wind_limits_content)

        # Find chunks containing wind limits
        wind_limit_chunks = []
        for chunk in chunks:
            if "wind limit" in chunk.lower() and ("15 knots" in chunk or "25 knots" in chunk):
                wind_limit_chunks.append(chunk)

        assert len(wind_limit_chunks) > 0, "Should have chunks that preserve wind limits information"

        # Verify at least one chunk has both crosswind and total wind limits
        complete_chunks = []
        for chunk in wind_limit_chunks:
            if "15 knots" in chunk and "25 knots" in chunk:
                complete_chunks.append(chunk)

        assert len(complete_chunks) > 0, "Should have at least one chunk with complete wind limits"

    @pytest.mark.asyncio
    async def test_metadata_extraction_identifies_wind_terms(self, sample_wind_limits_content):
        """Test that metadata extraction identifies wind-related terms."""

        doc_processor = RAGDocumentProcessor()
        metadata = doc_processor._extract_metadata(sample_wind_limits_content, "test_doc.pdf")

        # Verify wind-related key terms are identified
        key_terms = metadata.get("key_terms", [])
        expected_terms = ["wind limit", "wind speed", "crosswind", "gusts", "knots"]

        found_expected = [term for term in expected_terms if term in key_terms]
        assert len(found_expected) >= 3, f"Should identify at least 3 wind-related terms, found: {key_terms}"

        # Verify page count is extracted
        assert metadata.get("total_pages", 0) > 0, "Should extract page count from content"
