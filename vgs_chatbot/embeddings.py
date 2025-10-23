"""Embedding utilities using FastEmbed."""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Iterable, List, Sequence, cast

from fastembed import TextEmbedding

from vgs_chatbot.config import get_settings

logger = logging.getLogger(__name__)


def _to_python_floats(vector: Iterable[float]) -> List[float]:
    """Coerce numpy/array outputs into plain Python floats for serialization.

    Args:
        vector: Iterable of numeric values produced by the embedder.

    Returns:
        list[float]: Cleaned list suitable for MongoDB storage.
    """
    if hasattr(vector, "tolist"):
        # Cast to Any so static type checkers stop complaining about unknown
        # attributes on the declared Iterable type; at runtime this calls the
        # .tolist() method on array-like objects (e.g. numpy arrays).
        raw = cast(Any, vector).tolist()
        if isinstance(raw, list):
            return [float(value) for value in raw]
        return [float(raw)]
    return [float(value) for value in vector]


class FastEmbedder:
    """Thin wrapper around FastEmbed with simple query caching."""

    def __init__(
        self, model_name: str, batch_size: int = 16, cache_size: int = 256
    ) -> None:
        """Initialise the embedder wrapper and configure caching.

        Args:
            model_name: Name of the FastEmbed model to load.
            batch_size: Maximum batch size used for passage embedding.
            cache_size: Maximum number of cached query embeddings.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_size = cache_size
        self._model = TextEmbedding(model_name=model_name)
        self._query_cache: OrderedDict[str, List[float]] = OrderedDict()
        logger.info("Loaded FastEmbed model '%s'.", model_name)

    def embed_passages(self, passages: Sequence[str]) -> List[List[float]]:
        """Embed passages for storage in Vector Search.

        Args:
            passages: Text passages to be embedded.

        Returns:
            list[list[float]]: Embedding vectors for each passage.
        """
        if not passages:
            return []
        prefixed = (f"passage: {text.strip()}" for text in passages)
        embeddings: List[List[float]] = []
        for vector in self._model.embed(prefixed, batch_size=self.batch_size):
            embeddings.append(_to_python_floats(vector))
        logger.debug("Embedded %s passages.", len(embeddings))
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embed a user query with a lightweight LRU cache.

        Args:
            query: Query string supplied by the user.

        Returns:
            list[float]: Embedding vector representing the query.
        """
        key = query.strip()
        if key in self._query_cache:
            self._query_cache.move_to_end(key)
            logger.debug("Cache hit for query embedding.")
            return self._query_cache[key]
        vector_iter = self._model.embed((f"query: {key}",), batch_size=1)
        vector = _to_python_floats(next(vector_iter))
        self._query_cache[key] = vector
        if len(self._query_cache) > self.cache_size:
            self._query_cache.popitem(last=False)
        logger.debug(
            "Cache miss for query embedding; cache size=%s.", len(self._query_cache)
        )
        return vector


_EMBEDDER: FastEmbedder | None = None


def get_embedder() -> FastEmbedder:
    """Return a cached embedder instance.

    Returns:
        FastEmbedder: Singleton embedder shared across the app.
    """
    global _EMBEDDER  # noqa: PLW0603  # keep singleton for performance
    if _EMBEDDER is None:
        settings = get_settings()
        _EMBEDDER = FastEmbedder(settings.embedding_model_name)
        logger.debug(
            "Created embedder singleton for model '%s'.",
            settings.embedding_model_name,
        )
    return _EMBEDDER
