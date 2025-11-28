"""Login page for authenticated access to the chatbot."""

from __future__ import annotations

import logging

import streamlit as st

from vgs_chatbot.config import get_settings
from vgs_chatbot.db import (
    connect_default,
    find_user_by_email,
    update_last_login,
    verify_password,
)

logger = logging.getLogger(__name__)


def _ensure_state() -> None:
    """Initialise expected keys in the Streamlit session state."""
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("user_email", "")
    st.session_state.setdefault("user_role", "")
    st.session_state.setdefault("user_id", "")
    st.session_state.setdefault("chat_history", [])


def main() -> None:
    """Render login form and redirect authenticated users."""
    _ensure_state()
    try:
        # Load settings early to surface configuration issues to the user.
        get_settings()
    except Exception as exc:
        st.error(
            "Configuration error: missing or invalid settings.\n\n"
            "Set `MONGO_URI` (and optionally `OPENAI_API_KEY`). In Streamlit Cloud, add them to Secrets or Environment Variables."
        )
        st.caption(str(exc))
        st.stop()

    st.image("media/2fts.png", use_container_width=True)
    st.title("RAF 2FTS Knowledge Assistant")

    if st.session_state["logged_in"]:
        # Authenticated users should land on Chat immediately.
        try:
            st.switch_page("pages/1_Chat.py")
        except Exception:
            st.info("Signed in. Use the sidebar to open Chat or Admin.")
        return

    st.write("Sign in with your email and password registered for this service.")

    with st.form("login-form", clear_on_submit=False):
        email = st.text_input("Email address")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    if not submitted:
        return

    email_input = email.strip().lower()
    if not email_input or not password:
        st.warning("Please provide both an email address and password.")
        return

    client = None
    try:
        logger.info("Attempting login for user '%s'.", email_input)
        client = connect_default()
        user = find_user_by_email(client, email_input)
        if not user:
            st.error("Login failed. Please check the credentials.")
            client.close()
            return
        if not user.get("is_active", True):
            st.error("This account is inactive. Please contact an administrator.")
            client.close()
            return
        if not verify_password(password.strip(), user.get("password_hash", "")):
            st.error("Login failed. Please check the credentials.")
            client.close()
            return
        if user.get("_id"):
            update_last_login(client, user["_id"])
    except Exception as exc:  # noqa: BLE001 - surface friendly message
        if client:
            client.close()
        logger.warning(
            "Login failed for user '%s': %s", email_input, exc, exc_info=True
        )
        st.error("Login failed. Please check the credentials and network rules.")
        st.caption(f"Reason: {exc}")
        return

    logger.info("Login succeeded for user '%s'.", email_input)
    st.session_state["logged_in"] = True
    st.session_state["user_email"] = user.get("email", email_input)
    st.session_state["user_role"] = user.get("role", "user")
    st.session_state["user_id"] = str(user.get("_id", ""))
    st.session_state["mongo_client"] = client
    try:
        st.switch_page("pages/1_Chat.py")
    except Exception:
        st.rerun()


if __name__ == "__main__":
    main()
