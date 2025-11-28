"""Registration interface for new users."""

from __future__ import annotations

import logging

import streamlit as st

from vgs_chatbot.db import (
    connect_default,
    create_user,
    find_user_by_email,
    update_last_login,
)

logger = logging.getLogger(__name__)


ALLOWED_DOMAINS = ("@mod.uk", "@rafac.mod.gov.uk")


def _email_allowed(email: str) -> bool:
    """Check whether an email address ends with an approved domain."""
    lowered = email.strip().lower()
    return any(lowered.endswith(domain) for domain in ALLOWED_DOMAINS)


def main() -> None:
    """Render the registration form for new users."""
    st.title("Create an account")
    st.caption("Only RAF or MOD email addresses are eligible.")

    if st.session_state.get("logged_in"):
        try:
            st.switch_page("pages/1_Chat.py")
        except Exception:
            st.info("You are already signed in.")
        return

    with st.form("registration-form", clear_on_submit=False):
        email = st.text_input("Email address")
        password = st.text_input("Password", type="password")
        password_confirm = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Register")

    if not submitted:
        return

    email_clean = email.strip().lower()
    if not _email_allowed(email_clean):
        st.error("Please use an email ending with @mod.uk or @rafac.mod.gov.uk.")
        return
    if not password or not password_confirm:
        st.error("Please enter and confirm your password.")
        return
    if password != password_confirm:
        st.error("Passwords do not match.")
        return
    if len(password) < 8:
        st.error("Please choose a password with at least 8 characters.")
        return

    client = None
    try:
        client = connect_default()
        if find_user_by_email(client, email_clean):
            st.error("An account with this email already exists. Please sign in.")
            client.close()
            return
        user = create_user(client, email_clean, password)
        if user.get("_id"):
            update_last_login(client, user["_id"])
    except Exception as exc:  # noqa: BLE001 - surface friendly message
        if client:
            client.close()
        logger.exception("Registration failed for '%s'.", email_clean)
        st.error("Registration failed. Please try again or contact an administrator.")
        st.caption(str(exc))
        return

    st.success("Account created. You are now signed in.")
    st.session_state["logged_in"] = True
    st.session_state["user_email"] = user.get("email", email_clean)
    st.session_state["user_role"] = user.get("role", "user")
    st.session_state["user_id"] = str(user.get("_id", ""))
    st.session_state["mongo_client"] = client
    st.rerun()


if __name__ == "__main__":
    main()
