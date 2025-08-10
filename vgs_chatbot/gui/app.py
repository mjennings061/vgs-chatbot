"""Main Streamlit application."""

from datetime import datetime
from typing import Optional

import streamlit as st

from vgs_chatbot.models.chat import ChatMessage, MessageRole
from vgs_chatbot.repositories.database import DatabaseManager
from vgs_chatbot.repositories.user_repository import UserRepository
from vgs_chatbot.services.auth_service import AuthenticationService
from vgs_chatbot.services.chat_service import LLMChatService
from vgs_chatbot.services.document_processor import RAGDocumentProcessor
from vgs_chatbot.services.sharepoint_connector import SharePointConnector
from vgs_chatbot.utils.config import get_settings


class ChatbotApp:
    """Main chatbot application."""

    def __init__(self) -> None:
        """Initialize chatbot application."""
        self.settings = get_settings()
        self.db_manager = DatabaseManager(self.settings.database_url)

        # Initialize services (will be done with dependency injection in production)
        self.auth_service: Optional[AuthenticationService] = None
        self.sharepoint_connector = SharePointConnector()
        self.document_processor = RAGDocumentProcessor()
        self.chat_service = LLMChatService(
            openai_api_key=self.settings.openai_api_key,
            model=self.settings.openai_model,
        )

    async def setup(self) -> None:
        """Setup application dependencies."""
        await self.db_manager.create_tables()

        # Initialize auth service with repository
        async for session in self.db_manager.get_session():
            user_repository = UserRepository(session)
            self.auth_service = AuthenticationService(
                user_repository=user_repository, jwt_secret=self.settings.jwt_secret
            )
            break

    def run(self) -> None:
        """Run the Streamlit application."""
        st.set_page_config(
            page_title=self.settings.app_title, page_icon="🤖", layout="wide"
        )

        # Initialize session state
        if "user" not in st.session_state:
            st.session_state.user = None
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "sharepoint_connected" not in st.session_state:
            st.session_state.sharepoint_connected = False

        # Main app logic
        if st.session_state.user is None:
            self._render_auth_page()
        else:
            self._render_chat_page()

    def _render_auth_page(self) -> None:
        """Render authentication page."""
        st.title(f"Welcome to {self.settings.app_title}")

        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            self._render_login_form()

        with tab2:
            self._render_register_form()

    def _render_login_form(self) -> None:
        """Render login form."""
        st.header("Login")

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

            if submit:
                if username and password:
                    # This would normally be async, simplified for Streamlit
                    st.success("Login functionality available - requires async setup")
                else:
                    st.error("Please enter both username and password")

    def _render_register_form(self) -> None:
        """Render registration form."""
        st.header("Register")

        with st.form("register_form"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit = st.form_submit_button("Register")

            if submit:
                if not all([username, email, password, confirm_password]):
                    st.error("Please fill in all fields")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    st.success(
                        "Registration functionality available - requires async setup"
                    )

    def _render_chat_page(self) -> None:
        """Render main chat page."""
        st.title(f"{self.settings.app_title} - Chat")

        # Sidebar
        self._render_sidebar()

        # Main chat interface
        self._render_chat_interface()

    def _render_sidebar(self) -> None:
        """Render sidebar with user info and settings."""
        with st.sidebar:
            st.header("User Info")
            st.write(f"Logged in as: {st.session_state.user.username}")

            if st.button("Logout"):
                st.session_state.user = None
                st.experimental_rerun()

            st.header("SharePoint Connection")
            if not st.session_state.sharepoint_connected:
                with st.form("sharepoint_form"):
                    sp_username = st.text_input("SharePoint Username")
                    sp_password = st.text_input("SharePoint Password", type="password")
                    connect = st.form_submit_button("Connect to SharePoint")

                    if connect and sp_username and sp_password:
                        st.session_state.sharepoint_connected = True
                        st.success("Connected to SharePoint!")
                        st.experimental_rerun()
            else:
                st.success("✅ Connected to SharePoint")
                if st.button("Disconnect"):
                    st.session_state.sharepoint_connected = False
                    st.experimental_rerun()

            st.header("Document Sources")
            if self.settings.sharepoint_directory_urls:
                for url in self.settings.sharepoint_directory_urls:
                    st.write(f"📁 {url}")
            else:
                st.write("No SharePoint directories configured")

    def _render_chat_interface(self) -> None:
        """Render chat interface."""
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message.role.value):
                st.write(message.content)
                if hasattr(message, "sources") and message.sources:
                    st.caption(f"Sources: {', '.join(message.sources)}")

        # Chat input
        if prompt := st.chat_input("Ask a question about VGS documentation..."):
            if not st.session_state.sharepoint_connected:
                st.warning("Please connect to SharePoint first to access documents.")
                return

            # Add user message
            user_message = ChatMessage(
                role=MessageRole.USER, content=prompt, timestamp=datetime.utcnow()
            )
            st.session_state.messages.append(user_message)

            # Display user message
            with st.chat_message("user"):
                st.write(prompt)

            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Searching documents and generating response..."):
                    response = self._generate_response(prompt)
                    st.write(response.message)

                    if response.sources:
                        st.caption(f"Sources: {', '.join(response.sources)}")

                    # Add assistant message
                    assistant_message = ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=response.message,
                        timestamp=datetime.utcnow(),
                    )
                    assistant_message.sources = response.sources
                    st.session_state.messages.append(assistant_message)

    def _generate_response(self, query: str) -> "ChatResponse":
        """Generate response to user query.

        Args:
            query: User query

        Returns:
            Chat response
        """
        # Simplified response generation for demo
        from vgs_chatbot.models.chat import ChatResponse

        return ChatResponse(
            message=f"I received your question: '{query}'. This is a demo response. In the full implementation, I would search the SharePoint documents and provide a relevant answer.",
            sources=["Demo Document 1", "Demo Document 2"],
            confidence=0.8,
            processing_time=1.0,
        )


def main() -> None:
    """Main entry point."""
    app = ChatbotApp()

    # Setup async components (simplified for demo)
    # In production, you'd properly handle async initialization

    app.run()


if __name__ == "__main__":
    main()
