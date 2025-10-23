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

    # Keep only necessary environment variables
    mongodb_host: str = Field(..., alias="MONGODB_HOST")
    openai_api_key: Optional[str] = Field(None, alias="OPENAI_API_KEY")

    # Local dev convenience (optional, not required in production envs)
    app_login_user: str = Field("test", alias="APP_LOGIN_USER")
    app_login_pass: str = Field("test_user", alias="APP_LOGIN_PASS")

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

    # Read-only properties backed by module-level constants below. These are
    # not environment-driven and represent non-secret application defaults.

    @property
    def mongodb_db(self) -> str:  # database name
        return DEFAULT_DB_NAME

    @property
    def mongodb_vector_index(self) -> str:  # Atlas Vector Search index name
        return DEFAULT_VECTOR_INDEX

    @property
    def mongodb_search_index(self) -> str:  # Atlas Search (text) index name
        return DEFAULT_SEARCH_INDEX

    @property
    def embedding_model_name(self) -> str:  # FastEmbed model identifier
        return DEFAULT_EMBEDDING_MODEL

    @property
    def retrieval_top_k(self) -> int:
        return DEFAULT_RETRIEVAL_TOP_K

    @property
    def retrieval_num_candidates(self) -> int:
        return DEFAULT_RETRIEVAL_NUM_CANDIDATES

    @property
    def graph_max_hops(self) -> int:
        return DEFAULT_GRAPH_MAX_HOPS

    @property
    def graph_max_candidates(self) -> int:
        return DEFAULT_GRAPH_MAX_CANDIDATES


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


# ------------------
# Non-secret defaults
# ------------------

# MongoDB names
DEFAULT_DB_NAME = "vgs"
DEFAULT_VECTOR_INDEX = "vgs_vector"
DEFAULT_SEARCH_INDEX = "vgs_text"

# Embeddings
DEFAULT_EMBEDDING_MODEL = "snowflake/snowflake-arctic-embed-xs"

# Retrieval parameters
DEFAULT_RETRIEVAL_TOP_K = 10
DEFAULT_RETRIEVAL_NUM_CANDIDATES = 400
DEFAULT_GRAPH_MAX_HOPS = 1
DEFAULT_GRAPH_MAX_CANDIDATES = 300
