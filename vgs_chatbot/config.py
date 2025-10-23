"""Application configuration helpers."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env early so local development mirrors deployed environments.
load_dotenv()

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Centralised configuration backed by environment variables."""

    mongodb_host: str = Field(..., alias="MONGODB_HOST")
    mongodb_db: str = Field("vgs", alias="MONGODB_DB")
    mongodb_vector_index: str = Field("vgs_vector", alias="MONGODB_VECTOR_INDEX")
    mongodb_search_index: str = Field("vgs_text", alias="MONGODB_SEARCH_INDEX")

    app_login_user: str = Field("test", alias="APP_LOGIN_USER")
    app_login_pass: str = Field("test_user", alias="APP_LOGIN_PASS")

    embedding_model_name: str = Field(
        "snowflake/snowflake-arctic-embed-xs", alias="EMBEDDING_MODEL_NAME"
    )
    openai_api_key: Optional[str] = Field(None, alias="OPENAI_API_KEY")

    retrieval_top_k: int = Field(10, ge=1, lt=50)
    retrieval_num_candidates: int = Field(400, ge=10, lt=1200)
    graph_max_hops: int = Field(1, ge=0, le=2)
    graph_max_candidates: int = Field(300, ge=0, le=2000)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="",
    )

    def build_srv_uri(self, username: str, password: str) -> str:
        """Return an SRV connection string for MongoDB Atlas.

        Args:
            username: Database username supplied by the end user.
            password: Database password matching the username.

        Returns:
            str: MongoDB SRV connection string targeting the configured host.
        """
        return (
            f"mongodb+srv://{username}:{password}@{self.mongodb_host}"
            "/?retryWrites=true&w=majority&appName=vgs-chatbot"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance.

    Returns:
        Settings: Singleton application settings object.
    """
    logger.debug("Fetching application settings.")
    # Settings is populated from environment variables by pydantic BaseSettings at runtime,
    # but static type checkers may still require constructor arguments for the fields;
    # silence that with a type-ignore for arg-type.
    return Settings()  # type: ignore[arg-type]
