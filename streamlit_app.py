"""Streamlit entry point configuring navigation for all pages."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="VGS Chatbot",
    page_icon="✈️",
)
def _ensure_state() -> None:
    """Initialise expected keys in the Streamlit session state."""
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("user_email", "")
    st.session_state.setdefault("user_role", "")
    st.session_state.setdefault("user_id", "")
    st.session_state.setdefault("chat_history", [])


def _build_pages() -> list[st.Page]:
    """Return the navigation structure based on authentication state."""
    login = st.Page("pages/0_Login.py", title="Sign in", icon=":material/login:")
    register = st.Page(
        "pages/0_Register.py",
        title="Register",
        icon=":material/person_add:",
    )
    chat = st.Page("pages/1_Chat.py", title="Chat", icon=":material/forum:")
    admin = st.Page(
        "pages/2_Admin.py",
        title="Admin",
        icon=":material/admin_panel_settings:",
    )

    if st.session_state["logged_in"]:
        pages = [chat]
        if st.session_state.get("user_role") == "admin":
            pages.append(admin)
    else:
        pages = [login, register]
    return pages


def main() -> None:
    """Create navigation and dispatch to the correct page."""
    _ensure_state()
    navigation = st.navigation(_build_pages())
    navigation.run()


if __name__ == "__main__":
    main()
