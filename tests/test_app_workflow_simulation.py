"""Test that simulates the exact application workflow to identify why 'sorry' responses occur."""

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from vgs_chatbot.models.chat import ChatMessage, MessageRole
from vgs_chatbot.models.document import Document
from vgs_chatbot.services.chat_service import LLMChatService
from vgs_chatbot.services.document_processor import RAGDocumentProcessor


class TestAppWorkflowSimulation:
    """Test that simulates the exact same workflow as the VGS Chatbot app."""

    @pytest.mark.asyncio
    async def test_complete_app_workflow_with_wind_limits(self):
        """Test the complete workflow from document processing to chat response."""

        # Create test content with wind limits
        wind_limits_content = """
--- PAGE 15 ---
WEATHER CONDITIONS AND LIMITS

Wind Limits for VGS Operations:

Maximum crosswind component: 15 knots
Maximum total wind speed: 25 knots
Gusting conditions: Not to exceed 20% above steady wind
Turbulence: Light turbulence acceptable, moderate and above prohibited

--- PAGE 16 ---
EMERGENCY PROCEDURES

The wind limits are established for safety and must not be exceeded under any circumstances.
Maximum demonstrated crosswind component during training: 12 knots
"""

        # Create a temporary file to simulate a real document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(wind_limits_content)
            temp_file_path = temp_file.name

        try:
            # Step 1: Create document objects (simulating what app.py does)
            documents = []
            doc = Document(
                name="2FTS DHOs Issue 3.pdf",
                file_path=temp_file_path,
                file_type="text/plain",
                directory_path=str(Path(temp_file_path).parent)
            )
            documents.append(doc)

            print(f"Created {len(documents)} documents")

            # Step 2: Initialize services (exactly like app.py)
            document_processor = RAGDocumentProcessor()

            # Step 3: Process documents (like app.py does)
            processed_documents = await document_processor.process_documents(documents)

            print(f"Processed {len(processed_documents)} documents")
            assert len(processed_documents) == 1

            processed_doc = processed_documents[0]
            print(f"Document content length: {len(processed_doc.content)}")
            print(f"Number of chunks: {len(processed_doc.chunks)}")
            print(f"Metadata: {processed_doc.metadata}")

            # Verify the content was processed correctly
            assert "wind limit" in processed_doc.content.lower()
            assert "15 knots" in processed_doc.content
            assert "25 knots" in processed_doc.content

            # Step 4: Index documents (like app.py does)
            await document_processor.index_documents(processed_documents)
            print("Documents indexed successfully")

            # Step 5: Perform search (like app.py does)
            query = "What are the wind limits?"
            relevant_docs = await document_processor.search_documents(query, top_k=3)
            context_documents = relevant_docs if relevant_docs else processed_documents[:2]

            print(f"Search found {len(relevant_docs)} relevant documents")
            print(f"Using {len(context_documents)} context documents")

            for i, doc in enumerate(context_documents):
                print(f"Context doc {i+1} content preview: {doc.content[:100]}...")

            # Step 6: Create chat message (like app.py does)
            messages = [ChatMessage(
                role=MessageRole.USER,
                content=query,
                timestamp=datetime.now(UTC)
            )]

            # Step 7: Mock the LLM service but test the context building
            with patch('vgs_chatbot.services.chat_service.ChatOpenAI') as mock_llm:
                # Mock LLM to return what it receives as context
                mock_response = AsyncMock()

                # Create a mock that captures the context and returns it
                def capture_context(prompt_text):
                    print("\n=== LLM PROMPT ===")
                    print(prompt_text)
                    print("=" * 50)

                    # Check if context contains wind limits
                    if "15 knots" in prompt_text and "25 knots" in prompt_text:
                        mock_response.content = "Based on the VGS documentation, maximum crosswind component is 15 knots and maximum total wind speed is 25 knots."
                    else:
                        mock_response.content = "I'm sorry, but the provided documents do not contain information regarding wind limits."

                    return mock_response

                mock_llm_instance = AsyncMock()
                mock_llm_instance.ainvoke = AsyncMock(side_effect=capture_context)
                mock_llm.return_value = mock_llm_instance

                # Initialize chat service
                chat_service = LLMChatService("fake_key")

                # Step 8: Generate response (like app.py does)
                response = await chat_service.generate_response(messages, context_documents)

                print("\n=== FINAL RESPONSE ===")
                print(f"Message: {response.message}")
                print(f"Confidence: {response.confidence}")
                print(f"Sources: {response.sources}")
                print("=" * 50)

                # The critical test - does the context contain the wind limits?
                # If our improvements worked, the LLM should receive proper context
                assert "sorry" not in response.message.lower(), f"Response should not be 'sorry': {response.message}"
                assert ("knots" in response.message or "kts" in response.message), f"Response should contain knots: {response.message}"

        finally:
            # Clean up temp file
            Path(temp_file_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_document_chunking_and_search_quality(self):
        """Test that our chunking and search improvements are working."""

        content_with_scattered_wind_info = """
--- PAGE 10 ---
GENERAL INFORMATION

This document covers various operational procedures.

--- PAGE 15 ---
WEATHER CONDITIONS

Wind Limits for Operations:
- Maximum crosswind component: 15 knots
- Maximum total wind speed: 25 knots

--- PAGE 20 ---
OTHER PROCEDURES

Various other procedures are covered here.

--- PAGE 25 ---
EMERGENCY WIND LIMITS

In emergency conditions:
- Demonstrated crosswind limit: 12 knots
- Maximum gusting factor: 20% above steady wind
"""

        doc_processor = RAGDocumentProcessor()

        # Test chunking preserves related content
        chunks = doc_processor._split_text_into_chunks(content_with_scattered_wind_info)

        print(f"Generated {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: {chunk[:100]}...")

        # Find chunks with wind information
        wind_chunks = [chunk for chunk in chunks if "wind" in chunk.lower()]
        print(f"Found {len(wind_chunks)} chunks containing wind information")

        # Should have chunks with wind limits
        assert len(wind_chunks) >= 2, f"Should have multiple chunks with wind info, got {len(wind_chunks)}"

        # Test metadata extraction
        metadata = doc_processor._extract_metadata(content_with_scattered_wind_info, "test.pdf")
        key_terms = metadata.get("key_terms", [])
        print(f"Extracted key terms: {key_terms}")

        # Should identify relevant aviation terms
        aviation_terms_found = [term for term in ["wind limit", "wind speed", "crosswind", "knots"] if term in key_terms]
        assert len(aviation_terms_found) >= 3, f"Should find aviation terms, got: {aviation_terms_found}"

        print("Chunking and metadata extraction working correctly!")


if __name__ == "__main__":
    import asyncio

    # Run the workflow test directly
    test_instance = TestAppWorkflowSimulation()
    asyncio.run(test_instance.test_complete_app_workflow_with_wind_limits())
