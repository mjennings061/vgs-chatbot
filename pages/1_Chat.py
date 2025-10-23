"""Chat interface for querying Viking training materials."""

from __future__ import annotations

import logging

import streamlit as st

from vgs_chatbot.db import get_collections
from vgs_chatbot.embeddings import get_embedder
from vgs_chatbot.llm import generate_answer
from vgs_chatbot.retrieve import retrieve_chunks

logger = logging.getLogger(__name__)


def _require_login() -> None:
    """Guard access to the page for authenticated users only.

    Returns:
        None: Raises Streamlit stop exceptions when unauthenticated.
    """
    if not st.session_state.get("logged_in"):
        logger.warning("Unauthenticated access attempt to chat page.")
        st.error("Please return to the Home page and sign in first.")
        st.stop()
    if "mongo_client" not in st.session_state:
        logger.error("Session state missing MongoDB client.")
        st.error("Connection not found. Please sign in again.")
        st.stop()


def _clear_history() -> None:
    """Remove all previously asked questions from the chat history.

    Returns:
        None: Session state `chat_history` is reset.
    """
    logger.debug("Clearing chat history for current session.")
    st.session_state["chat_history"] = []


def main() -> None:
    """Render chat UI and orchestrate retrieval workflow.

    Returns:
        None: Streamlit handles the rendering pipeline.
    """
    _require_login()
    st.title("Viking Knowledge Chat")
    st.caption("Ask about RAF 2FTS Viking procedures and receive cited answers.")

    client = st.session_state["mongo_client"]
    collections = get_collections(client)
    embedder = get_embedder()
    st.session_state.setdefault("chat_history", [])

    with st.form("chat-form", clear_on_submit=True):
        question = st.text_area(
            "Question", placeholder="e.g. What are the wind limits?"
        )
        submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        clean_question = question.strip()
        logger.info("Received chat question: '%s'", clean_question)
        with st.spinner("Retrieving context…"):
            chunks = retrieve_chunks(
                doc_chunks=collections["doc_chunks"],
                kg_nodes=collections["kg_nodes"],
                kg_edges=collections["kg_edges"],
                embedder=embedder,
                query=clean_question,
            )
            logger.debug(
                "Retrieved %s chunks for question.",
                len(chunks),
            )
            answer = generate_answer(clean_question, chunks)
            citations = [chunk.as_citation() for chunk in chunks]
        st.session_state["chat_history"].append(
            {
                "question": clean_question,
                "answer": answer,
                "citations": citations,
            }
        )
        logger.debug(
            "Appended new chat turn; total turns: %s",
            len(st.session_state["chat_history"]),
        )

    if st.session_state["chat_history"]:
        st.button("Clear history", on_click=_clear_history)

    for turn in reversed(st.session_state["chat_history"]):
        st.chat_message("user").write(turn["question"])
        assistant = st.chat_message("assistant")
        assistant.write(turn["answer"])
        if turn["citations"]:
            with assistant.expander("Sources"):
                for citation in turn["citations"]:
                    st.write(
                        f"{citation['document']} · {citation['section']} · Page {citation['page']}"
                    )


if __name__ == "__main__":
    main()
