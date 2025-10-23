"""Streamlit entry point providing login and navigation."""

from __future__ import annotations

import logging

import streamlit as st

from vgs_chatbot.config import get_settings
from vgs_chatbot.db import connect_with_user

st.set_page_config(
    page_title="VGS Chatbot",
    page_icon="✈️",
)


logger = logging.getLogger(__name__)


def _reset_session() -> None:
    """Clear session state and close the Mongo client if present.

    Returns:
        None: This function mutates Streamlit's session state in place.
    """
    client = st.session_state.get("mongo_client")
    if client:
        logger.debug("Closing MongoDB client on logout.")
        client.close()
    logger.debug("Clearing Streamlit session state.")
    st.session_state.clear()


def _ensure_state() -> None:
    """Initialise expected keys in the Streamlit session state.

    Returns:
        None: The session state gains defaults for required keys.
    """
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("username", "")
    st.session_state.setdefault("chat_history", [])
    # Tracks whether we've already redirected to Chat after a fresh login
    st.session_state.setdefault("redirected_after_login", False)


def main() -> None:
    """Render login screen and route authenticated users to other pages.

    Returns:
        None: The Streamlit framework handles rendering side effects.
    """
    _ensure_state()
    settings = get_settings()
    st.title("RAF 2FTS Knowledge Assistant")

    if st.session_state["logged_in"]:
        # After a successful login, automatically take users to the Chat page once.
        if not st.session_state.get("redirected_after_login"):
            st.session_state["redirected_after_login"] = True
            try:
                st.switch_page("pages/1_Chat.py")
            except Exception:
                # Fallback to existing behaviour if switching pages is unavailable.
                pass

        st.success("Signed in. Use the sidebar to open Chat or Admin.")
        if st.button("Sign out"):
            logger.info(
                "User '%s' requested sign out.", st.session_state.get("username")
            )
            _reset_session()
            st.rerun()
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
        username_input = username.strip()
        try:
            logger.info("Attempting login for user '%s'.", username_input)
            client = connect_with_user(username_input, password.strip())
        except Exception as exc:  # noqa: BLE001 - surface friendly message
            logger.warning(
                "Login failed for user '%s': %s", username_input, exc, exc_info=True
            )
            st.error("Login failed. Please check the credentials and network rules.")
            st.caption(f"Reason: {exc}")
            return
        logger.info("Login succeeded for user '%s'.", username_input)
        st.session_state["logged_in"] = True
        st.session_state["username"] = username_input
        st.session_state["mongo_client"] = client
        st.session_state["redirected_after_login"] = False
        st.rerun()


if __name__ == "__main__":
    # Configure root logging once when launched directly.
    # Honour LOG_LEVEL env var if set, defaulting to INFO for user-friendly output.
    import os

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    # Keep the root logger quiet to avoid third‑party noise; elevate our modules explicitly.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
    )
    project_level = getattr(logging, log_level, logging.INFO)
    for logger_name in ("streamlit_app", "pages", "vgs_chatbot"):
        logging.getLogger(logger_name).setLevel(project_level)
    main()
