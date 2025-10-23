"""Answer generation utilities."""

from __future__ import annotations

from typing import Iterable, List

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None

from vgs_chatbot.config import get_settings
from vgs_chatbot.retrieve import RetrievedChunk


def generate_answer(query: str, chunks: List[RetrievedChunk]) -> str:
    """Generate an answer constrained to supplied context."""
    if not chunks:
        return (
            "I do not currently hold any relevant training material. "
            "Please upload the document and try again."
        )

    settings = get_settings()
    context = _format_context(chunks)

    if settings.openai_api_key and OpenAI:
        client = OpenAI(api_key=settings.openai_api_key)
        prompt = (
            "You are an instructor for RAF 2FTS Viking trainees. "
            "Answer using only the provided context. "
            "If the answer is not present, say you do not know."
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                },
            ],
        )
        choice = response.choices[0]
        return choice.message.content.strip()

    return _extractive_fallback(chunks)


def _format_context(chunks: Iterable[RetrievedChunk]) -> str:
    """Render context for prompting."""
    parts = []
    for chunk in chunks:
        parts.append(
            f"[Document: {chunk.doc_title} | Section: {chunk.section_title} | Page: {chunk.page_start}]\n"
            f"{chunk.text}"
        )
    return "\n\n".join(parts)


def _extractive_fallback(chunks: List[RetrievedChunk]) -> str:
    """Return a conservative extract when no LLM is configured."""
    best = max(chunks, key=lambda chunk: chunk.score if chunk.score else 0.0)
    snippet = best.text.strip()
    if len(snippet) > 900:
        snippet = snippet[:900].rsplit(" ", 1)[0] + "…"
    return (
        "LLM responses are disabled. The most relevant extract is shown below:\n\n"
        f"{snippet}"
    )
