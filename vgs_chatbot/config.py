"""Application configuration: lightweight and Streamlit-friendly.

This module provides a very small `Settings` object whose attributes are loaded
from Streamlit secrets when available, otherwise from environment variables.
Local development can still use a `.env` file (loaded via python-dotenv).
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

# Load .env for local development so env vars are available via os.getenv
load_dotenv()

logger = logging.getLogger(__name__)


def _get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Return a config value from st.secrets or the environment.

    Streamlit Community Cloud exposes secrets via `st.secrets`. Locally, or when
    not running under Streamlit, fall back to environment variables.
    """
    try:  # Avoid hard dependency on Streamlit
        import streamlit as st  # type: ignore

        if key in st.secrets:
            value = st.secrets.get(key)
            if value is not None:
                return str(value)
    except Exception:
        pass
    return os.getenv(key, default)


class Settings:
    """Minimal settings holder with convenient defaults."""

    def __init__(self) -> None:
        self.mongodb_host: Optional[str] = _get_secret("MONGODB_HOST")
        if not self.mongodb_host:
            raise ValueError(
                "MONGODB_HOST is not set. Provide it via Streamlit secrets or env."
            )

        self.openai_api_key: Optional[str] = _get_secret("OPENAI_API_KEY")

        # Local dev convenience (optional). Defaults are blank strings.
        self.app_login_user: str = _get_secret("APP_LOGIN_USER", "") or ""
        self.app_login_pass: str = _get_secret("APP_LOGIN_PASS", "") or ""

    # Read-only properties backed by module-level constants below.
    @property
    def log_level(self) -> str:
        return "DEBUG"

    @property
    def mongodb_db(self) -> str:
        return "chatbot"

    @property
    def mongodb_vector_index(self) -> str:
        return "vector_index"

    @property
    def mongodb_search_index(self) -> str:
        return "vgs_text"

    @property
    def embedding_model_name(self) -> str:
        return "snowflake/snowflake-arctic-embed-xs"

    @property
    def retrieval_top_k(self) -> int:
        return 10

    @property
    def retrieval_num_candidates(self) -> int:
        return 400

    @property
    def graph_max_hops(self) -> int:
        return 1

    @property
    def graph_max_candidates(self) -> int:
        return 300

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
    """Return cached settings instance."""
    logger.debug("Fetching application settings.")
    return Settings()
