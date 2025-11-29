"""Answer generation utilities."""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None

from vgs_chatbot import logger
from vgs_chatbot.config import get_settings
from vgs_chatbot.retrieve import RetrievedChunk

MODEL = "gpt-4.1-nano"


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
        past_question = turn.get("question")
        past_answer = turn.get("answer")
        if not past_question:
            continue
        snippet = f"Q: {past_question}"
        if past_answer:
            snippet += f"\nA: {past_answer}"
        snippets.append(snippet)
    history_block = "\n\n".join(snippets) if snippets else "None"

    client = OpenAI(api_key=settings.openai_api_key)
    prompt = """
## Instructions

You are assisting a RAG agent for Volunteer Gliding Squadron (VGS) documents.
- Rewrite the user's question to include relevant context from prior Q&A pairs, if they are relevant to the current question.
- Acronyms should be preserved in your rewrite without expansion.
- Use the provided conversation history to inform the rewrite.
- Do not invent facts. Return a single rewritten question only.
- The question may not need rewriting; if so, return it as-is.

## Examples

Q: What is a G2?
A: What is a G2 graded pilot?

Q: How many launches can I do?
A: As a pilot, what is the maximum number of launches I can do in a day?

Q: I am so confused I do not know if I can fly after drinking last night.
A: What are the alcohol limits for gliding?

## Conversation History

Recent chat (for context, may be empty):

"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
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
            logger.debug("Rewrote question '%s' to '%s'.", question, rewritten)
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
    logger.debug(
        "Prepared answer context: %s characters across %s chunks.",
        len(context),
        len(chunks),
    )

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
- Orders are ranked in importance, with RA being highest and least restrictive, GASOs being more restrictive and DHOs being the most restrictive.

**IMPORTANT**
- Answer using only the provided context.
- Cite the document sections you used to form your answer.
- Present all relevant information in your answer as clearly as possible, as some information may be critical for safety.
- If the answer or relevant context are not present, say you do not know.
        """
        response = client.chat.completions.create(
            model=MODEL,
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
        _log_token_usage(getattr(response, "usage", None))

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


def _log_token_usage(usage: Any) -> None:
    """Log token counts from an OpenAI response, handling older/newer schemas."""
    if not usage:
        return
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)

    # Newer SDKs may expose input/output fields instead.
    if prompt_tokens is None:
        prompt_tokens = getattr(usage, "input_tokens", None)
    if completion_tokens is None:
        completion_tokens = getattr(usage, "output_tokens", None)
    if (
        total_tokens is None
        and prompt_tokens is not None
        and completion_tokens is not None
    ):
        total_tokens = prompt_tokens + completion_tokens

    logger.info(
        "OpenAI token usage - prompt: %s, completion: %s, total: %s",
        prompt_tokens,
        completion_tokens,
        total_tokens,
    )
