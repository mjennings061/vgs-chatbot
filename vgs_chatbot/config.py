"""Application configuration: lightweight and Streamlit-friendly.

This module provides a very small `Settings` object whose attributes are loaded
from Streamlit secrets when available, otherwise from environment variables.
Local development can still use a `.env` file (loaded via python-dotenv).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

from vgs_chatbot import logger

# Load .env for local development so env vars are available via os.getenv
load_dotenv()


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
        # Required; validated immediately to avoid Optional typing downstream.
        self.mongo_uri: str = _get_secret("MONGO_URI") or ""
        if not self.mongo_uri:
            raise ValueError(
                "MONGO_URI is not set. Provide it via Streamlit secrets or env."
            )

        self.openai_api_key: Optional[str] = _get_secret("OPENAI_API_KEY")

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
    def retrieval_require_kg_filter(self) -> bool:
        # When True, retrieval stages are hard-filtered to KG candidates; useful for small, curated corpora.
        return False

    @property
    def chunk_target_chars(self) -> int:
        # Target chunk size (characters) when splitting documents; keep close to ingestion defaults.
        return 3000

    @property
    def chunk_overlap_chars(self) -> int:
        # Overlap when splitting long paragraphs; improves continuity across chunks.
        return 120

    @property
    def kg_max_phrases(self) -> int:
        # Cap the number of keyphrases per chunk to avoid KG bloat.
        return 12

    @property
    def graph_max_hops(self) -> int:
        return 1

    @property
    def graph_max_candidates(self) -> int:
        return 300


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    logger.debug("Fetching application settings.")
    return Settings()
