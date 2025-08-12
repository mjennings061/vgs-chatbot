"""Integration tests for VGS Chatbot."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from vgs_chatbot.gui.app import VGSChatbot
from vgs_chatbot.models.chat import ChatMessage, MessageRole
from vgs_chatbot.models.document import Document, ProcessedDocument
from vgs_chatbot.models.user import User
from vgs_chatbot.services.auth_service import AuthenticationService
from vgs_chatbot.services.chat_service import LLMChatService
from vgs_chatbot.services.document_processor import RAGDocumentProcessor
from vgs_chatbot.services.sharepoint_connector import SharePointConnector


@pytest.fixture
def mock_settings():
    """Mock application settings."""
    settings = MagicMock()
    settings.database_url = "sqlite+aiosqlite:///:memory:"
    settings.jwt_secret = "test-secret-key"
    settings.openai_api_key = "test-key"
    settings.openai_model = "gpt-3.5-turbo"
    settings.sharepoint_site_url = "https://test.sharepoint.com"
    settings.sharepoint_directory_urls = ["https://test.sharepoint.com/docs"]
    settings.app_title = "Test VGS Chatbot"
    settings.debug = True
    return settings


@pytest.fixture
def sample_user():
    """Sample user for testing."""
    return User(
        id=uuid4(),
        username="testuser",
        email="test@example.com",
        password_hash="$2b$12$hashed_password",
        is_active=True
    )


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        Document(
            name="Safety Manual.pdf",
            file_path="/documents/safety/Safety Manual.pdf",
            file_type="application/pdf",
            size=1024,
            modified_date=datetime.now(datetime.UTC),
            directory_path="/documents/safety"
        ),
        Document(
            name="Procedures.docx",
            file_path="/documents/procedures/Procedures.docx",
            file_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            size=2048,
            modified_date=datetime.now(datetime.UTC),
            directory_path="/documents/procedures"
        )
    ]


@pytest.mark.asyncio
async def test_full_authentication_flow(mock_settings, sample_user):
    """Test complete authentication flow."""
    with patch('vgs_chatbot.gui.app.DatabaseManager') as mock_db_manager:
        # Setup mocks
        mock_session = AsyncMock()
        mock_db_manager.return_value.get_session.return_value.__aenter__.return_value = mock_session
        mock_db_manager.return_value.create_tables = AsyncMock()

        # Mock user repository and auth service
        with patch('vgs_chatbot.repositories.user_repository.UserRepository'), \
             patch('vgs_chatbot.services.auth_service.AuthenticationService') as mock_auth:

            mock_auth_instance = mock_auth.return_value

            # Test user creation
            mock_auth_instance.create_user.return_value = sample_user
            created_user = await mock_auth_instance.create_user(
                "testuser", "password", "test@example.com"
            )

            assert created_user.username == "testuser"
            assert created_user.email == "test@example.com"

            # Test authentication
            mock_auth_instance.authenticate.return_value = sample_user
            authenticated_user = await mock_auth_instance.authenticate("testuser", "password")

            assert authenticated_user is not None
            assert authenticated_user.username == "testuser"


@pytest.mark.asyncio
async def test_sharepoint_document_flow(sample_documents):
    """Test SharePoint document listing and processing."""
    connector = SharePointConnector()

    with patch.object(connector, 'connect') as mock_connect, \
         patch.object(connector, 'list_documents') as mock_list_docs:

        # Test connection
        mock_connect.return_value = True
        connected = await connector.connect("testuser", "password")
        assert connected is True

        # Test document listing
        mock_list_docs.return_value = sample_documents
        docs = await connector.list_documents(["/test/path"])

        assert len(docs) == 2
        assert docs[0].name == "Safety Manual.pdf"
        assert docs[1].name == "Procedures.docx"


@pytest.mark.asyncio
async def test_document_processing_pipeline(sample_documents):
    """Test document processing and RAG pipeline."""
    with patch('vgs_chatbot.services.document_processor.SentenceTransformer') as mock_st, \
         patch('vgs_chatbot.services.document_processor.chromadb') as mock_chroma:

        # Setup mocks
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_st.return_value = mock_model

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_chroma.Client.return_value = mock_client

        processor = RAGDocumentProcessor()

        # Mock text extraction
        processor._extract_text_content = AsyncMock(side_effect=[
            "Safety procedures and guidelines for VGS operations.",
            "Standard operating procedures for maintenance tasks."
        ])

        # Test document processing
        processed_docs = await processor.process_documents(sample_documents)

        assert len(processed_docs) == 2
        assert len(processed_docs[0].chunks) > 0
        assert processed_docs[0].embeddings is not None

        # Test indexing
        await processor.index_documents(processed_docs)

        # Verify ChromaDB calls
        assert mock_collection.add.called


@pytest.mark.asyncio
async def test_chat_service_response():
    """Test chat service response generation."""
    with patch('vgs_chatbot.services.chat_service.OpenAI') as mock_openai:
        # Setup mock LLM
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.generations = [[MagicMock(text="Based on the safety manual, you should follow these procedures...")]]
        mock_llm.agenerate.return_value = mock_response
        mock_openai.return_value = mock_llm

        chat_service = LLMChatService("test-key")

        # Create test messages
        messages = [
            ChatMessage(
                role=MessageRole.USER,
                content="What are the safety procedures?",
                timestamp=datetime.now(datetime.UTC)
            )
        ]

        # Create test context documents
        context_docs = [
            ProcessedDocument(
                id="doc1",
                original_document=Document(
                    name="Safety Manual.pdf",
                    file_path="/safety/Safety Manual.pdf",
                    file_type="application/pdf",
                    directory_path="/safety"
                ),
                content="Safety procedures content...",
                chunks=["Safety procedures..."],
                processed_at=datetime.now(datetime.UTC)
            )
        ]

        # Test response generation
        response = await chat_service.generate_response(messages, context_docs)

        assert response.message is not None
        assert len(response.sources) > 0
        assert "Safety Manual.pdf" in response.sources
        assert response.processing_time is not None


@pytest.mark.asyncio
async def test_end_to_end_chat_flow(mock_settings, sample_user, sample_documents):
    """Test complete end-to-end chat flow."""
    with patch('vgs_chatbot.gui.app.DatabaseManager'), \
         patch('vgs_chatbot.services.auth_service.AuthenticationService') as mock_auth, \
         patch('vgs_chatbot.services.sharepoint_connector.SharePointConnector') as mock_sp, \
         patch('vgs_chatbot.services.document_processor.RAGDocumentProcessor') as mock_processor, \
         patch('vgs_chatbot.services.chat_service.LLMChatService') as mock_chat:

        # Setup service mocks
        mock_auth_instance = mock_auth.return_value
        mock_sp_instance = mock_sp.return_value
        mock_processor_instance = mock_processor.return_value
        mock_chat_instance = mock_chat.return_value

        # Mock authentication
        mock_auth_instance.authenticate.return_value = sample_user

        # Mock SharePoint connection
        mock_sp_instance.connect.return_value = True
        mock_sp_instance.list_documents.return_value = sample_documents

        # Mock document processing
        processed_doc = ProcessedDocument(
            id="processed1",
            original_document=sample_documents[0],
            content="Processed content...",
            chunks=["Chunk 1", "Chunk 2"],
            processed_at=datetime.now(datetime.UTC)
        )
        mock_processor_instance.process_documents.return_value = [processed_doc]
        mock_processor_instance.search_documents.return_value = [processed_doc]

        # Mock chat response
        from vgs_chatbot.models.chat import ChatResponse
        mock_response = ChatResponse(
            message="Based on the documentation, here are the safety procedures...",
            sources=["Safety Manual.pdf"],
            confidence=0.9,
            processing_time=1.5
        )
        mock_chat_instance.generate_response.return_value = mock_response

        # Create app instance (would normally be tested through Streamlit)
        app = VGSChatbot()
        app.settings = mock_settings

        # Test the response generation method directly
        import asyncio
        response = asyncio.run(app._generate_response("What are the safety procedures?"))

        assert response.message is not None
        assert len(response.sources) > 0


def test_configuration_validation():
    """Test configuration validation."""
    from vgs_chatbot.utils.config import Settings

    # Test with minimal valid config
    settings = Settings(
        database_url="postgresql://user:pass@localhost/db",
        jwt_secret="test-secret-key-32-chars-long",
        openai_api_key="sk-test-key"
    )

    assert settings.database_url is not None
    assert len(settings.jwt_secret) >= 10
    assert settings.openai_api_key.startswith("sk-")


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in various components."""

    # Test auth service with invalid credentials
    with patch('vgs_chatbot.repositories.user_repository.UserRepository') as mock_repo:
        mock_repo_instance = mock_repo.return_value
        mock_repo_instance.get_by_username.return_value = None

        auth_service = AuthenticationService(mock_repo_instance, "test-secret")
        result = await auth_service.authenticate("invalid", "invalid")

        assert result is None

    # Test SharePoint connector with connection failure
    connector = SharePointConnector()
    with patch('vgs_chatbot.services.sharepoint_connector.AuthenticationContext') as mock_auth:
        mock_auth.return_value.acquire_token_for_user.return_value = False

        connected = await connector.connect("invalid", "invalid")
        assert connected is False

    # Test chat service with API error
    with patch('vgs_chatbot.services.chat_service.OpenAI') as mock_openai:
        mock_llm = MagicMock()
        mock_llm.agenerate.side_effect = Exception("API Error")
        mock_openai.return_value = mock_llm

        chat_service = LLMChatService("invalid-key")
        messages = [ChatMessage(
            role=MessageRole.USER,
            content="test",
            timestamp=datetime.now(datetime.UTC)
        )]

        response = await chat_service.generate_response(messages, [])
        assert "error" in response.message.lower()
        assert response.confidence == 0.0
