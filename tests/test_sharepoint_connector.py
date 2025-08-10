"""Tests for SharePoint connector."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from vgs_chatbot.services.sharepoint_connector import SharePointConnector


@pytest.fixture
def sharepoint_connector():
    """SharePoint connector instance."""
    return SharePointConnector()


@pytest.mark.asyncio
async def test_connect_success(sharepoint_connector):
    """Test successful SharePoint connection."""
    with patch('vgs_chatbot.services.sharepoint_connector.AuthenticationContext') as mock_auth, \
         patch('vgs_chatbot.services.sharepoint_connector.ClientContext') as mock_client:
        
        # Arrange
        mock_auth_instance = mock_auth.return_value
        mock_auth_instance.acquire_token_for_user.return_value = True
        
        # Act
        result = await sharepoint_connector.connect("testuser", "password")
        
        # Assert
        assert result is True
        assert sharepoint_connector.is_connected is True


@pytest.mark.asyncio
async def test_connect_failure(sharepoint_connector):
    """Test failed SharePoint connection."""
    with patch('vgs_chatbot.services.sharepoint_connector.AuthenticationContext') as mock_auth:
        
        # Arrange
        mock_auth_instance = mock_auth.return_value
        mock_auth_instance.acquire_token_for_user.return_value = False
        
        # Act
        result = await sharepoint_connector.connect("testuser", "wrongpassword")
        
        # Assert
        assert result is False
        assert sharepoint_connector.is_connected is False


@pytest.mark.asyncio
async def test_list_documents_not_connected(sharepoint_connector):
    """Test listing documents when not connected."""
    with pytest.raises(RuntimeError, match="Not connected to SharePoint"):
        await sharepoint_connector.list_documents(["/test/path"])


@pytest.mark.asyncio
async def test_disconnect(sharepoint_connector):
    """Test SharePoint disconnection."""
    # Arrange
    sharepoint_connector.is_connected = True
    sharepoint_connector.client_context = MagicMock()
    
    # Act
    await sharepoint_connector.disconnect()
    
    # Assert
    assert sharepoint_connector.client_context is None
    assert sharepoint_connector.is_connected is False