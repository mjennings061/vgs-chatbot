"""Tests for authentication service."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from vgs_chatbot.services.auth_service import AuthenticationService
from vgs_chatbot.models.user import User


@pytest.fixture
def mock_user_repository():
    """Mock user repository."""
    return AsyncMock()


@pytest.fixture
def auth_service(mock_user_repository):
    """Authentication service with mocked repository."""
    return AuthenticationService(
        user_repository=mock_user_repository,
        jwt_secret="test-secret"
    )


@pytest.mark.asyncio
async def test_authenticate_valid_user(auth_service, mock_user_repository):
    """Test authentication with valid user."""
    # Arrange
    user = User(
        id=1,
        username="testuser",
        email="test@example.com",
        password_hash="$2b$12$hashed_password",
        is_active=True
    )
    mock_user_repository.get_by_username.return_value = user
    mock_user_repository.update_last_login = AsyncMock()
    
    # Mock bcrypt verification
    auth_service._verify_password = MagicMock(return_value=True)
    
    # Act
    result = await auth_service.authenticate("testuser", "password")
    
    # Assert
    assert result is not None
    assert result.username == "testuser"
    mock_user_repository.update_last_login.assert_called_once_with(1)


@pytest.mark.asyncio
async def test_authenticate_invalid_password(auth_service, mock_user_repository):
    """Test authentication with invalid password."""
    # Arrange
    user = User(
        id=1,
        username="testuser",
        email="test@example.com",
        password_hash="$2b$12$hashed_password",
        is_active=True
    )
    mock_user_repository.get_by_username.return_value = user
    
    # Mock bcrypt verification to return False
    auth_service._verify_password = MagicMock(return_value=False)
    
    # Act
    result = await auth_service.authenticate("testuser", "wrong_password")
    
    # Assert
    assert result is None


@pytest.mark.asyncio
async def test_create_user(auth_service, mock_user_repository):
    """Test user creation."""
    # Arrange
    created_user = User(
        id=1,
        username="newuser",
        email="new@example.com",
        password_hash="$2b$12$hashed_password",
        is_active=True
    )
    mock_user_repository.create.return_value = created_user
    
    # Mock password hashing
    auth_service._hash_password = MagicMock(return_value="$2b$12$hashed_password")
    
    # Act
    result = await auth_service.create_user("newuser", "password", "new@example.com")
    
    # Assert
    assert result.username == "newuser"
    assert result.email == "new@example.com"
    auth_service._hash_password.assert_called_once_with("password")