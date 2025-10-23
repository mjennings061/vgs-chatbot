"""Chat interface for querying Viking training materials."""

from __future__ import annotations

import streamlit as st

from vgs_chatbot.db import get_collections
from vgs_chatbot.embeddings import get_embedder
from vgs_chatbot.llm import generate_answer
from vgs_chatbot.retrieve import retrieve_chunks


def _require_login() -> None:
    if not st.session_state.get("logged_in"):
        st.error("Please return to the Home page and sign in first.")
        st.stop()
    if "mongo_client" not in st.session_state:
        st.error("Connection not found. Please sign in again.")
        st.stop()


def _clear_history() -> None:
    st.session_state["chat_history"] = []


def main() -> None:
    """Render chat UI."""
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
        with st.spinner("Retrieving context…"):
            chunks = retrieve_chunks(
                doc_chunks=collections["doc_chunks"],
                kg_nodes=collections["kg_nodes"],
                kg_edges=collections["kg_edges"],
                embedder=embedder,
                query=question.strip(),
            )
            answer = generate_answer(question.strip(), chunks)
            citations = [chunk.as_citation() for chunk in chunks]
        st.session_state["chat_history"].append(
            {
                "question": question.strip(),
                "answer": answer,
                "citations": citations,
            }
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
