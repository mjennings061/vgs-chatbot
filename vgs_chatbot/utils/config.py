"""Configuration utilities."""

import os
from typing import List

from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Database
    database_url: str = "postgresql+asyncpg://user:password@localhost/vgs_chatbot"
    
    # JWT
    jwt_secret: str = "your-secret-key-change-in-production"
    
    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-3.5-turbo"
    
    # SharePoint
    sharepoint_site_url: str = ""
    sharepoint_directory_urls: List[str] = []
    
    # App
    app_title: str = "VGS Chatbot"
    debug: bool = False
    
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