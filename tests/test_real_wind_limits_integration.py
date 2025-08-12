"""Real integration test for wind limits without mocking the LLM."""

import os
from datetime import UTC, datetime
from unittest.mock import patch

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
def comprehensive_wind_limits_content():
    """More comprehensive sample content containing wind limits information."""
    return """
--- PAGE 15 ---
CHAPTER 3: WEATHER CONDITIONS AND OPERATIONAL LIMITS

3.1 Wind Limits for VGS Operations

The following wind limits shall be observed during all VGS flying operations:

a) Maximum crosswind component: 15 knots
b) Maximum total wind speed: 25 knots
c) Gusting conditions: Wind gusts shall not exceed 20% above the steady wind speed
d) Turbulence limits: Operations are prohibited in moderate or severe turbulence

3.2 Additional Weather Minima

- Visibility minimum: 5 kilometers
- Cloud base minimum: 1500 feet AGL
- No precipitation during operations
- Temperature limits: -10°C to +40°C

--- PAGE 16 ---

3.3 Wind Assessment Procedures

Before commencement of flying, the duty instructor must:

1. Check local wind conditions using calibrated equipment
2. Assess crosswind component against aircraft limits
3. Monitor for gusting conditions throughout operations
4. Cease operations if conditions exceed established limits

Maximum demonstrated crosswind component during training: 12 knots

The wind limits specified in this document are mandatory and must not be exceeded under any circumstances. These limits ensure safe operations within the demonstrated performance envelope of VGS aircraft.

3.4 Emergency Procedures

In the event of weather deterioration beyond operational limits:
1. Cease flying operations immediately
2. Secure all aircraft in designated parking areas
3. Report conditions to duty instructor
4. Resume operations only when conditions improve within limits

--- PAGE 17 ---

3.5 Record Keeping

All weather observations and decisions regarding wind limits must be recorded in the daily flying log. Include:
- Wind speed and direction at time of decision
- Crosswind component calculation
- Reason for cessation if applicable
- Time operations resumed

Note: These procedures apply to both K21 and Vigilant aircraft operations.
Wind speed measurements should be taken at regular intervals, minimum every 30 minutes during active operations.
"""


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - skipping real LLM integration test"
)
class TestRealWindLimitsIntegration:
    """Real integration tests for wind limits queries using actual LLM."""

    @pytest.mark.asyncio
    async def test_real_wind_limits_query_with_knots_response(
        self,
        sample_wind_limits_document,
        comprehensive_wind_limits_content
    ):
        """Test that wind limits query returns response containing 'knots' using real LLM."""

        # Mock the document processor to return our test content
        with patch('vgs_chatbot.services.document_processor.RAGDocumentProcessor._extract_text_content') as mock_extract:
            mock_extract.return_value = comprehensive_wind_limits_content

            # Initialize document processor
            doc_processor = RAGDocumentProcessor()

            # Process the document
            processed_docs = await doc_processor.process_documents([sample_wind_limits_document])

            # Verify we have processed documents
            assert len(processed_docs) == 1
            processed_doc = processed_docs[0]

            print(f"Processed document content length: {len(processed_doc.content)}")
            print(f"Number of chunks: {len(processed_doc.chunks)}")
            print(f"Metadata key terms: {processed_doc.metadata.get('key_terms', [])}")

            # Verify content contains wind limits and knots
            assert "wind limit" in processed_doc.content.lower()
            assert "knots" in processed_doc.content.lower()
            assert "25 knots" in processed_doc.content
            assert "15 knots" in processed_doc.content

            # Index documents
            await doc_processor.index_documents(processed_docs)

            # Test search functionality
            search_results = await doc_processor.search_documents("wind limits", top_k=3)
            assert len(search_results) > 0, "Should find relevant documents for wind limits query"

            print(f"Search found {len(search_results)} relevant documents")
            for i, result in enumerate(search_results):
                print(f"Result {i+1} content preview: {result.content[:200]}...")

            # Initialize chat service with real API key
            chat_service = LLMChatService(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                model="gpt-4o-mini"
            )

            # Create test message
            test_message = ChatMessage(
                role=MessageRole.USER,
                content="What are the wind limits?",
                timestamp=datetime.now(UTC)
            )

            # Generate response using real LLM
            response = await chat_service.generate_response(
                messages=[test_message],
                context_documents=search_results
            )

            # Print response for debugging
            print("\n=== REAL LLM RESPONSE ===")
            print(f"Message: {response.message}")
            print(f"Confidence: {response.confidence}")
            print(f"Sources: {response.sources}")
            print("=" * 50)

            # Critical assertions - response must not be "sorry" and must contain knots/kts
            assert "sorry" not in response.message.lower(), f"Response should not contain 'sorry': {response.message}"

            # Check for knots or kts in response
            response_lower = response.message.lower()
            has_knots = "knots" in response_lower or "kts" in response_lower
            assert has_knots, f"Response should contain 'knots' or 'kts': {response.message}"

            # Verify response has some confidence
            assert response.confidence > 0, "Response should have confidence > 0"

            # Verify sources are included
            assert len(response.sources) > 0, "Response should include sources"


@pytest.mark.asyncio
async def test_document_processing_extracts_wind_terms():
    """Test that document processing correctly identifies and extracts wind-related terms."""

    # Sample content with various wind-related terms
    content = """
    Wind limits for gliding operations:
    - Maximum wind speed: 30 knots
    - Crosswind limit: 12 kts
    - Gust factor: not to exceed 50% of steady wind
    """

    doc_processor = RAGDocumentProcessor()
    metadata = doc_processor._extract_metadata(content, "test_doc.pdf")

    key_terms = metadata.get("key_terms", [])
    print(f"Extracted key terms: {key_terms}")

    # Should identify wind-related terms
    expected_terms = ["wind speed", "crosswind", "knots"]
    found_terms = [term for term in expected_terms if term in key_terms]

    assert len(found_terms) >= 2, f"Should identify wind terms, found: {key_terms}"

    # Test chunking preserves wind information
    chunks = doc_processor._split_text_into_chunks(content)
    print(f"Generated chunks: {chunks}")

    # At least one chunk should contain wind limits info
    wind_chunks = [chunk for chunk in chunks if "wind" in chunk.lower()]
    assert len(wind_chunks) > 0, "Should have chunks containing wind information"


if __name__ == "__main__":
    # Run the test directly if API key is available
    import asyncio

    if os.getenv("OPENAI_API_KEY"):
        print("Running real integration test with OpenAI API...")
        asyncio.run(test_document_processing_extracts_wind_terms())
    else:
        print("OPENAI_API_KEY not set - skipping real LLM test")
