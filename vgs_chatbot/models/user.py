"""User data models."""

from datetime import UTC, datetime

from pydantic import BaseModel, EmailStr, Field


class User(BaseModel):
    """User model."""
    id: str | None = Field(default=None, alias="_id")
    email: EmailStr
    password_hash: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_login: datetime | None = None
    is_active: bool = True

    class Config:
        """Pydantic configuration."""
        populate_by_name = True
        arbitrary_types_allowed = True


class UserCreate(BaseModel):
    """User creation model."""

    email: EmailStr
    password: str


class UserLogin(BaseModel):
    """User login model."""

    email: EmailStr
    password: str
