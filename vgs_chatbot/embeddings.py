"""Embedding utilities using FastEmbed."""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Sequence

from fastembed import TextEmbedding

from vgs_chatbot.config import get_settings


class FastEmbedder:
    """Thin wrapper around FastEmbed with simple query caching."""

    def __init__(
        self, model_name: str, batch_size: int = 16, cache_size: int = 256
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_size = cache_size
        self._model = TextEmbedding(model_name=model_name)
        self._query_cache: OrderedDict[str, List[float]] = OrderedDict()

    def embed_passages(self, passages: Sequence[str]) -> List[List[float]]:
        """Embed passages for storage in Vector Search."""
        if not passages:
            return []
        prefixed = (f"passage: {text.strip()}" for text in passages)
        embeddings: List[List[float]] = []
        for vector in self._model.embed(prefixed, batch_size=self.batch_size):
            embeddings.append(list(vector))
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embed a user query with a lightweight LRU cache."""
        key = query.strip()
        if key in self._query_cache:
            self._query_cache.move_to_end(key)
            return self._query_cache[key]
        vector_iter = self._model.embed((f"query: {key}",), batch_size=1)
        vector = list(next(vector_iter))
        self._query_cache[key] = vector
        if len(self._query_cache) > self.cache_size:
            self._query_cache.popitem(last=False)
        return vector


_EMBEDDER: FastEmbedder | None = None


def get_embedder() -> FastEmbedder:
    """Return a cached embedder instance."""
    global _EMBEDDER  # noqa: PLW0603  # keep singleton for performance
    if _EMBEDDER is None:
        settings = get_settings()
        _EMBEDDER = FastEmbedder(settings.embedding_model_name)
    return _EMBEDDER
