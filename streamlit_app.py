"""Streamlit entry point providing login and navigation."""

from __future__ import annotations

import streamlit as st

from vgs_chatbot.config import get_settings
from vgs_chatbot.db import connect_with_user

st.set_page_config(
    page_title="VGS Chatbot",
    page_icon="✈️",
)


def _reset_session() -> None:
    client = st.session_state.get("mongo_client")
    if client:
        client.close()
    st.session_state.clear()


def _ensure_state() -> None:
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("username", "")
    st.session_state.setdefault("chat_history", [])


def main() -> None:
    """Render login screen then link to Chat/Admin pages."""
    _ensure_state()
    settings = get_settings()
    st.title("RAF 2FTS Knowledge Assistant")

    if st.session_state["logged_in"]:
        st.success("Signed in. Use the sidebar to open Chat or Admin.")
        if st.button("Sign out"):
            _reset_session()
            st.rerun()
        with st.expander("Connection details"):
            st.write(f"MongoDB host: `{settings.mongodb_host}`")
        st.stop()

    st.write(
        "Sign in with a MongoDB Atlas database user. "
        "Demo credentials default to the values in `.env`."
    )

    with st.form("login-form", clear_on_submit=False):
        username = st.text_input("Username", value=settings.app_login_user)
        password = st.text_input(
            "Password", type="password", value=settings.app_login_pass
        )
        submitted = st.form_submit_button("Connect")

    if submitted:
        try:
            client = connect_with_user(username.strip(), password.strip())
        except Exception as exc:  # noqa: BLE001 - surface friendly message
            st.error("Login failed. Please check the credentials and network rules.")
            st.caption(f"Reason: {exc}")
            return
        st.session_state["logged_in"] = True
        st.session_state["username"] = username.strip()
        st.session_state["mongo_client"] = client
        st.rerun()


if __name__ == "__main__":
    main()
