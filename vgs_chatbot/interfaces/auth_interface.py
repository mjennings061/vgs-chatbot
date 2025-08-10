"""Authentication interface for dependency inversion."""

from abc import ABC, abstractmethod
from typing import Optional

from vgs_chatbot.models.user import User


class AuthenticationInterface(ABC):
    """Abstract interface for authentication services."""
    
    @abstractmethod
    async def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password.
        
        Args:
            username: User's username
            password: User's password
            
        Returns:
            User object if authentication successful, None otherwise
        """
        pass
    
    @abstractmethod
    async def create_user(self, username: str, password: str, email: str) -> User:
        """Create a new user account.
        
        Args:
            username: Unique username
            password: User's password
            email: User's email address
            
        Returns:
            Created User object
        """
        pass
    
    @abstractmethod
    async def validate_token(self, token: str) -> Optional[User]:
        """Validate authentication token.
        
        Args:
            token: JWT token
            
        Returns:
            User object if token valid, None otherwise
        """
        pass