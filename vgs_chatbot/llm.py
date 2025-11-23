"""Answer generation utilities."""

from __future__ import annotations

import logging
from typing import Iterable, List, Sequence

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None

from vgs_chatbot.config import get_settings
from vgs_chatbot.retrieve import RetrievedChunk

logger = logging.getLogger(__name__)


def rewrite_question(
    question: str, history: Sequence[dict[str, str]] | None = None
) -> str:
    """Rewrite the question using brief chat history to steer retrieval.

    Keeps wording aligned to Volunteer Gliding Squadron (VGS) docs and avoids
    adding new facts; falls back to the original question when LLM access is unavailable.
    """
    settings = get_settings()
    if not settings.openai_api_key or not OpenAI:
        return question

    snippets: List[str] = []
    # Only include the last few turns to keep the rewrite prompt lean.
    for turn in (history or [])[-4:]:
        q = turn.get("question")
        a = turn.get("answer")
        if not q:
            continue
        snippet = f"Q: {q}"
        if a:
            snippet += f"\nA: {a}"
        snippets.append(snippet)
    history_block = "\n\n".join(snippets) if snippets else "None"

    client = OpenAI(api_key=settings.openai_api_key)
    prompt = """
You are assisting retrieval for Royal Air Force Volunteer Gliding Squadron (VGS) documents.
Rewrite the user's latest question to be specific, self-contained, and phrased like VGS policy/training material (Viking glider, winch launches, GIF/GS/FSC, CGS/CFS).
Do not invent facts. Return a single rewritten question only.
Recent chat (for context, may be empty):
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            temperature=0.2,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"{history_block}\n\nUser question: {question}",
                },
            ],
        )
        rewritten = response.choices[0].message.content
        if rewritten:
            return rewritten.strip()
    except Exception:  # noqa: BLE001 - best-effort rewrite
        logger.exception("Question rewrite failed; using original.")
    return question


def generate_answer(query: str, chunks: List[RetrievedChunk]) -> str:
    """Generate an answer constrained to supplied context.

    Args:
        query: Original user question.
        chunks: Retrieved context chunks relevant to the question.

    Returns:
        str: Answer text grounded in the provided context.
    """
    if not chunks:
        logger.warning("Attempted to answer query '%s' with no context.", query)
        return (
            "I do not currently hold any relevant training material. "
            "Please upload the document and try again."
        )

    settings = get_settings()
    context = _format_context(chunks)

    if settings.openai_api_key and OpenAI:
        logger.debug("Generating answer via OpenAI for query '%s'.", query)
        client = OpenAI(api_key=settings.openai_api_key)
        prompt = """
**Aim**
You are an retrieval agent for Royal Air Force (RAF) Gliding trainees and instructors on the Viking Glider.
You are to answer questions using the context provided from RAG searches.

**Organisation**
- A volunteer gliding squadron (VGS) operates mainly on weekends, delivering elementary gliding trainingto Air Cadets aged 13-20.
- Basic training is Glider Induction Flights (GIF), then Gliding Scholarships (GS).
- Cadets can continue at a VGS as Flight Staff Cadets (FSC).
- VGS are governed by Central Gliding School (CGS) and standardised by Central Flying School (CFS)
- A VGS primarily launch gliders using a winch with cables, rarely are aerotows used.
- Qualified Gliding Instructors (QGIs) are categories B2, B1, A2, A1 (including star e.g. A2*)
- Graded pilots are G2 and G1. Ungraded pilots are anyone else


**IMPORTANT**
- Answer using only the provided context.
- If the answer or relevant context are not present, say you do not know.
        """
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            temperature=0.1,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"# **CONTEXT**\n{context}\n\n# **QUESTION** \n{query}",
                },
            ],
        )
        choice = response.choices[0]
        logger.info("OpenAI answer generated for query '%s'.", query)

        message = choice.message.content
        if not message:
            logger.warning("No message returned")
            return "Sorry, I am having some issues."

        return message.strip()

    logger.info("Using extractive fallback for query '%s'.", query)
    return _extractive_fallback(chunks)


def _format_context(chunks: Iterable[RetrievedChunk]) -> str:
    """Render context for prompting.

    Args:
        chunks: Retrieved chunks included in the prompt.

    Returns:
        str: Formatted context block.
    """
    parts = []
    for chunk in chunks:
        parts.append(
            f"[Document: {chunk.doc_title} | Section: {chunk.section_title} | Page: {chunk.page_start}]\n"
            f"{chunk.text}"
        )
    return "\n\n".join(parts)


def _extractive_fallback(chunks: List[RetrievedChunk]) -> str:
    """Return a conservative extract when no LLM is configured.

    Args:
        chunks: Retrieved context chunks.

    Returns:
        str: Extractive summary to display to the user.
    """
    best = max(chunks, key=lambda chunk: chunk.score if chunk.score else 0.0)
    snippet = best.text.strip()
    if len(snippet) > 900:
        snippet = snippet[:900].rsplit(" ", 1)[0] + "…"
    logger.debug(
        "Extractive fallback selected chunk '%s' with score %.3f.",
        best.chunk_id,
        best.score or 0.0,
    )
    return (
        "LLM responses are disabled. The most relevant extract is shown below:\n\n"
        f"{snippet}"
    )
