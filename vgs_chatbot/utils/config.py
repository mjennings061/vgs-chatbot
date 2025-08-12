"""Configuration utilities."""

import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Database
    database_url: str = "postgresql+asyncpg://user:password@localhost/vgs_chatbot"

    # JWT
    jwt_secret: str = "your-secret-key-change-in-production"

    # OpenAI
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    openai_model: str = os.environ.get("OPENAI_MODEL", "gpt-4.1-nano")

    # Document Storage
    documents_dir: str = os.environ.get("DOCUMENTS_DIR", "data/documents")
    vectors_dir: str = os.environ.get("VECTORS_DIR", "data/vectors")

    # App
    app_title: str = os.environ.get("APP_TITLE", "VGS Chatbot")
    debug: bool = os.environ.get("DEBUG", "False").lower() in ("1", "true", "yes", "on")

    # Admin credentials (set via environment variables only)
    admin_username: str = os.environ.get("ADMIN_USERNAME", "")
    admin_password: str = os.environ.get("ADMIN_PASSWORD", "")

    # Test credentials (set via environment variables only)
    test_username: str = os.environ.get("TEST_USERNAME", "")
    test_password: str = os.environ.get("TEST_PASSWORD", "")

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    """Get application settings.

    Returns:
        Application settings instance
    """
    return Settings()
