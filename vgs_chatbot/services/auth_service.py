"""Authentication service implementation."""

import jwt
from datetime import datetime, timedelta
from typing import Optional

import bcrypt

from vgs_chatbot.interfaces.auth_interface import AuthenticationInterface
from vgs_chatbot.models.user import User
from vgs_chatbot.repositories.user_repository import UserRepository


class AuthenticationService(AuthenticationInterface):
    """PostgreSQL-based authentication service."""
    
    def __init__(self, user_repository: UserRepository, jwt_secret: str) -> None:
        """Initialize authentication service.
        
        Args:
            user_repository: User repository for database operations
            jwt_secret: Secret key for JWT token generation
        """
        self.user_repository = user_repository
        self.jwt_secret = jwt_secret
    
    async def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password.
        
        Args:
            username: User's username
            password: User's password
            
        Returns:
            User object if authentication successful, None otherwise
        """
        user = await self.user_repository.get_by_username(username)
        
        if not user or not user.is_active:
            return None
        
        if not self._verify_password(password, user.password_hash):
            return None
        
        await self.user_repository.update_last_login(user.id)
        return user
    
    async def create_user(self, username: str, password: str, email: str) -> User:
        """Create a new user account.
        
        Args:
            username: Unique username
            password: User's password
            email: User's email address
            
        Returns:
            Created User object
        """
        password_hash = self._hash_password(password)
        
        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            created_at=datetime.utcnow()
        )
        
        return await self.user_repository.create(user)
    
    async def validate_token(self, token: str) -> Optional[User]:
        """Validate authentication token.
        
        Args:
            token: JWT token
            
        Returns:
            User object if token valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            user_id = payload.get("user_id")
            
            if not user_id:
                return None
            
            return await self.user_repository.get_by_id(user_id)
        
        except jwt.InvalidTokenError:
            return None
    
    def generate_token(self, user: User) -> str:
        """Generate JWT token for user.
        
        Args:
            user: User to generate token for
            
        Returns:
            JWT token string
        """
        payload = {
            "user_id": user.id,
            "username": user.username,
            "exp": datetime.utcnow() + timedelta(hours=24)
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash.
        
        Args:
            password: Plain text password
            password_hash: Stored password hash
            
        Returns:
            True if password matches, False otherwise
        """
        return bcrypt.checkpw(password.encode(), password_hash.encode())