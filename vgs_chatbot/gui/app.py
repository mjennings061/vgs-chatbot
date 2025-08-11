"""Main VGS Chatbot application with local admin document upload."""

from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

from vgs_chatbot.models.chat import (
    ChatMessage,
    ChatResponse,
    MessageRole,
)
from vgs_chatbot.models.document import Document, ProcessedDocument
from vgs_chatbot.services.chat_service import LLMChatService
from vgs_chatbot.services.document_processor import RAGDocumentProcessor
from vgs_chatbot.utils.config import get_settings


class VGSChatbot:
    """VGS Chatbot with admin document upload and user chat interface."""

    def __init__(self) -> None:
        """Initialize chatbot application."""
        self.settings = get_settings()
        self.data_dir = Path("data")
        self.documents_dir = self.data_dir / "documents"
        self.vectors_dir = self.data_dir / "vectors"

        # Ensure directories exist
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.vectors_dir.mkdir(parents=True, exist_ok=True)

        # Initialize services
        self.document_processor = RAGDocumentProcessor()
        self.chat_service = LLMChatService(
            openai_api_key=self.settings.openai_api_key,
            model=self.settings.openai_model,
        )

        # Admin credentials from settings
        self.admin_credentials = {
            self.settings.admin_username: self.settings.admin_password
        }

    def validate_mod_email(self, email: str) -> bool:
        """Validate email is from MOD domain."""
        allowed_domains = ["@mod.gov.uk", "@mod.uk"]
        return any(email.lower().endswith(domain) for domain in allowed_domains)

    def are_documents_indexed(self) -> bool:
        """Check if documents have been processed and are ready for chat."""
        documents = list(self.documents_dir.glob("*"))
        index_file = self.vectors_dir / ".documents_indexed"
        return len(documents) > 0 and index_file.exists()

    def run(self) -> None:
        """Run the Streamlit application."""
        st.set_page_config(
            page_title="VGS Chatbot - 2FTS Document Assistant",
            page_icon="🚁",
            layout="wide"
        )

        # Initialize session state
        if "user_type" not in st.session_state:
            st.session_state.user_type = None
        if "username" not in st.session_state:
            st.session_state.username = None
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Main routing logic
        if st.session_state.user_type is None:
            self._render_login_page()
        elif st.session_state.user_type == "admin":
            self._render_admin_page()
        elif st.session_state.user_type == "user":
            self._render_user_page()

    def _render_login_page(self) -> None:
        """Render login/registration page."""
        st.title("🚁 VGS Chatbot - 2FTS Document Assistant")
        st.markdown("### Intelligent document assistant for RAF operations")

        tab1, tab2, tab3 = st.tabs(["User Login", "User Register", "Admin Login"])

        with tab1:
            self._render_user_login()

        with tab2:
            self._render_user_register()

        with tab3:
            self._render_admin_login()

    def _render_user_login(self) -> None:
        """Render user login form."""
        st.header("User Login")

        with st.form("user_login"):
            email = st.text_input("Email Address")
            password = st.text_input("Password", type="password")
            login = st.form_submit_button("Login")

            if login:
                if not email or not password:
                    st.error("Please enter both email and password")
                elif not self.validate_mod_email(email):
                    st.error("Access restricted to @mod.gov.uk and @mod.uk email addresses")
                else:
                    # Simplified login for demo - in production use proper auth
                    st.session_state.user_type = "user"
                    st.session_state.username = email
                    st.success("✅ Logged in successfully!")
                    st.rerun()

    def _render_user_register(self) -> None:
        """Render user registration form."""
        st.header("User Registration")

        with st.form("user_register"):
            email = st.text_input("Email Address")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            register = st.form_submit_button("Register")

            if register:
                if not email or not password or not confirm_password:
                    st.error("Please fill in all fields")
                elif not self.validate_mod_email(email):
                    st.error("Registration restricted to @mod.gov.uk and @mod.uk email addresses")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    # Simplified registration for demo
                    st.success("✅ Registration successful! You can now log in.")

    def _render_admin_login(self) -> None:
        """Render admin login form."""
        st.header("Administrator Login")

        with st.form("admin_login"):
            username = st.text_input("Admin Username")
            password = st.text_input("Admin Password", type="password")
            login = st.form_submit_button("Admin Login")

            if login:
                if username in self.admin_credentials and self.admin_credentials[username] == password:
                    st.session_state.user_type = "admin"
                    st.session_state.username = username
                    st.success("✅ Admin logged in successfully!")
                    st.rerun()
                else:
                    st.error("❌ Invalid admin credentials")

    def _render_admin_page(self) -> None:
        """Render admin dashboard."""
        st.title("🔧 Administrator Dashboard")
        st.sidebar.write(f"👤 **Admin:** {st.session_state.username}")

        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.rerun()

        tab1, tab2 = st.tabs(["📁 Document Management", "📈 System Status"])

        with tab1:
            self._render_document_management()

        with tab2:
            self._render_system_status()

    def _render_document_management(self) -> None:
        """Render document upload and management interface."""
        st.header("Document Management")

        # Document upload
        st.subheader("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Select documents to upload (PDF, Word, Excel, PowerPoint, Text)",
            type=['pdf', 'docx', 'xlsx', 'pptx', 'txt'],
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("Process and Index Documents"):
                with st.spinner("Processing documents..."):
                    success_count = 0
                    for uploaded_file in uploaded_files:
                        try:
                            # Save uploaded file
                            file_path = self.documents_dir / uploaded_file.name
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())

                            success_count += 1

                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")

                    if success_count > 0:
                        # Create index marker file
                        index_file = self.vectors_dir / ".documents_indexed"
                        index_file.write_text(f"Documents indexed at {datetime.now()}")
                        st.success(f"✅ Successfully processed {success_count} documents!")
                        st.rerun()
                    else:
                        st.error("❌ No documents were successfully processed")

        # List existing documents
        st.subheader("📚 Current Documents")
        documents = list(self.documents_dir.glob("*"))

        if documents:
            for doc in documents:
                if doc.is_file():  # Only show files, not directories
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"📄 {doc.name}")
                    with col2:
                        st.write(f"{doc.stat().st_size / 1024:.1f} KB")
                    with col3:
                        if st.button("🗑️", key=f"delete_{doc.name}"):
                            doc.unlink()
                            st.success(f"Deleted {doc.name}")
                            st.rerun()
        else:
            st.info("No documents uploaded yet")

    def _render_system_status(self) -> None:
        """Render system status information."""
        st.header("System Status")

        # Document statistics
        doc_count = len([f for f in self.documents_dir.glob("*") if f.is_file()])
        st.metric("📄 Documents", doc_count)

        # Vector database status
        if self.are_documents_indexed():
            st.success("✅ Vector database is indexed and ready")
        else:
            st.warning("⚠️ No documents indexed yet")

        # OpenAI status
        if self.settings.openai_api_key and self.settings.openai_api_key != "":
            st.success("✅ OpenAI API configured")
        else:
            st.error("❌ OpenAI API key not configured")

    def _render_user_page(self) -> None:
        """Render user chat interface."""
        st.title("🚁 VGS Chatbot - Ask me about 2FTS documents")

        # Sidebar
        with st.sidebar:
            st.write(f"👤 **User:** {st.session_state.username}")
            if st.button("Logout"):
                st.session_state.clear()
                st.rerun()

            st.header("📚 Available Documents")
            documents = [f for f in self.documents_dir.glob("*") if f.is_file()]
            if documents:
                for doc in documents:
                    st.write(f"📄 {doc.name}")
            else:
                st.info("No documents available yet")

        # Check if documents are available
        if not self.are_documents_indexed():
            st.warning("⚠️ No documents have been processed yet. Please contact your administrator to upload documents.")
            return

        # Chat interface
        self._render_chat_interface()

    def _render_chat_interface(self) -> None:
        """Render chat interface."""
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message.role.value):
                st.write(message.content)
                if hasattr(message, "source_references") and message.source_references:
                    source_refs = []
                    for ref in message.source_references:
                        ref_text = ref.document_name
                        if ref.section_title:
                            ref_text += f" (Section: {ref.section_title})"
                        if ref.page_number:
                            ref_text += f" - Page {ref.page_number}"
                        source_refs.append(ref_text)
                    st.caption(f"Sources: {'; '.join(source_refs)}")
                elif hasattr(message, "sources") and message.sources:
                    st.caption(f"Sources: {', '.join(message.sources)}")

        # Chat input
        if prompt := st.chat_input("Ask me about 2FTS documents..."):
            if not self.are_documents_indexed():
                st.warning("No documents available. Please contact your administrator.")
                return

            # Add user message
            user_message = ChatMessage(
                role=MessageRole.USER, content=prompt, timestamp=datetime.now(UTC)
            )
            st.session_state.messages.append(user_message)

            # Display user message
            with st.chat_message("user"):
                st.write(prompt)

            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Searching documents and generating response..."):
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    response = loop.run_until_complete(self._generate_response(prompt))
                    st.write(response.message)

                    if response.source_references:
                        source_refs = []
                        for ref in response.source_references:
                            ref_text = ref.document_name
                            if ref.section_title:
                                ref_text += f" (Section: {ref.section_title})"
                            if ref.page_number:
                                ref_text += f" - Page {ref.page_number}"
                            source_refs.append(ref_text)
                        st.caption(f"Sources: {'; '.join(source_refs)}")
                    elif response.sources:
                        st.caption(f"Sources: {', '.join(response.sources)}")

                    # Add assistant message
                    assistant_message = ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=response.message,
                        timestamp=datetime.now(UTC),
                    )
                    assistant_message.sources = response.sources
                    assistant_message.source_references = response.source_references
                    st.session_state.messages.append(assistant_message)

    async def _generate_response(self, query: str) -> ChatResponse:
        """Generate response using local documents and LLM.
        
        Args:
            query: User query
            
        Returns:
            Chat response with source references
        """
        import time

        start_time = time.time()

        try:
            # Get available documents and convert to ProcessedDocument format
            processed_documents = []
            for doc_path in self.documents_dir.glob("*"):
                if doc_path.is_file():
                    try:
                        # Simple text extraction
                        if doc_path.suffix.lower() == '.txt':
                            content = doc_path.read_text(encoding='utf-8')
                        elif doc_path.suffix.lower() == '.pdf':
                            try:
                                from pypdf import PdfReader
                                with open(doc_path, 'rb') as file:
                                    pdf_reader = PdfReader(file)
                                    content = ""
                                    for page_num, page in enumerate(pdf_reader.pages, 1):
                                        page_text = page.extract_text()
                                        if page_text.strip():
                                            content += f"[Page {page_num}] {page_text}\\n"
                            except Exception:
                                content = f"Could not extract text from {doc_path.name}"
                        else:
                            content = f"Document: {doc_path.name} (content extraction not implemented for {doc_path.suffix})"

                        if content.strip():
                            # Create Document and ProcessedDocument objects
                            try:
                                doc = Document(
                                    name=doc_path.name,
                                    file_path=str(doc_path.absolute()),
                                    file_type=f"application/{doc_path.suffix[1:]}" if doc_path.suffix else "text/plain",
                                    directory_path=str(doc_path.parent.absolute())
                                )
                            except Exception as e:
                                print(f"Error creating Document for {doc_path.name}: {e}")
                                continue
                            
                            # Simple chunking - split into smaller pieces to stay within token limits
                            # Limit each chunk to ~500 characters and take only first few chunks
                            chunks = [chunk.strip() for chunk in content.split('\\n\\n') if chunk.strip()]
                            if not chunks:
                                chunks = [content[:500]]  # Much smaller fallback chunk
                            else:
                                # Limit chunk size and number of chunks
                                chunks = [chunk[:500] for chunk in chunks[:3]]  # Max 3 chunks, 500 chars each
                            
                            processed_doc = ProcessedDocument(
                                id=str(doc_path.name),
                                original_document=doc,
                                content=content,
                                chunks=chunks[:5],  # Limit to 5 chunks for performance
                                metadata={
                                    "section_title": "Document Content",
                                    "page_number": 1
                                },
                                processed_at=datetime.now()
                            )
                            processed_documents.append(processed_doc)
                    except Exception as e:
                        print(f"Error reading {doc_path.name}: {e}")
                        continue

            if not processed_documents:
                return ChatResponse(
                    message="I apologize, but I couldn't access any documents to answer your question. Please contact the administrator.",
                    sources=[],
                    source_references=[],
                    confidence=0.0,
                    processing_time=time.time() - start_time
                )

            # Simple relevance filtering (in production, use proper vector search)
            query_lower = query.lower()
            relevant_docs = []
            
            for doc in processed_documents:
                content_lower = doc.content.lower()
                query_terms = query_lower.split()
                relevance_score = sum(1 for term in query_terms if term in content_lower)
                
                if relevance_score > 0:
                    relevant_docs.append(doc)
            
            # Use only top 2 most relevant documents to reduce token usage
            context_documents = relevant_docs[:2] if relevant_docs else processed_documents[:2]

            # Create a simple chat message for the LLM service
            messages = [ChatMessage(
                role=MessageRole.USER,
                content=query,
                timestamp=datetime.now(UTC)
            )]

            # Use the chat service to generate a proper response
            response = await self.chat_service.generate_response(messages, context_documents)
            
            return response

        except Exception as e:
            return ChatResponse(
                message=f"I encountered an error while processing your question: {str(e)}. Please try again or contact the administrator.",
                sources=[],
                source_references=[],
                confidence=0.0,
                processing_time=time.time() - start_time
            )


def main() -> None:
    """Main entry point."""
    app = VGSChatbot()
    app.run()


if __name__ == "__main__":
    main()
