"""User data models."""

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, EmailStr


class User(BaseModel):
    """User model."""
    id: UUID = uuid4()
    username: str
    email: EmailStr
    password_hash: str
    created_at: datetime = datetime.now()
    last_login: datetime = datetime.now()
    is_active: bool = True

    class Config:
        """Pydantic configuration."""

        from_attributes = True


class UserCreate(BaseModel):
    """User creation model."""

    username: str
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    """User login model."""

    username: str
    password: str
