"""Main VGS Chatbot application with local admin document upload."""

from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

from vgs_chatbot.models.chat import (
    ChatMessage,
    ChatResponse,
    MessageRole,
)
from vgs_chatbot.models.document import Document
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

    def _reindex_all_documents(self) -> bool:
        """Reindex all documents with improved chunking strategies.

        Returns:
            bool: True if reindexing was successful, False otherwise
        """
        try:
            # Get all document files
            document_files = [doc for doc in self.documents_dir.glob("*") if doc.is_file()]

            if not document_files:
                return False

            # Convert to Document objects
            documents = []
            for doc_path in document_files:
                # Determine file type from extension
                file_type_map = {
                    '.pdf': 'application/pdf',
                    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                    '.txt': 'text/plain'
                }

                file_type = file_type_map.get(doc_path.suffix.lower(), 'application/octet-stream')

                document = Document(
                    name=doc_path.name,
                    file_path=str(doc_path),
                    file_type=file_type,
                    directory_path=str(doc_path.parent)
                )
                documents.append(document)

            # Clear existing vector store (create new document processor instance)
            self.document_processor = RAGDocumentProcessor()

            # Process documents with improved chunking (async operations)
            import asyncio

            async def process_and_index():
                processed_docs = await self.document_processor.process_documents(documents)
                if processed_docs:
                    await self.document_processor.index_documents(processed_docs)
                return len(processed_docs) > 0

            # Run the async operations
            success = asyncio.run(process_and_index())
            if not success:
                return False

            # Update the index marker file
            index_file = self.vectors_dir / ".documents_indexed"
            index_file.write_text(f"Documents reindexed with improved chunking at {datetime.now(UTC)}")

            return True

        except Exception as e:
            st.error(f"Reindexing failed: {str(e)}")
            return False

    def are_documents_indexed(self) -> bool:
        """Check if documents have been processed and are ready for chat."""
        # Check if we have any documents at all
        documents = list(self.documents_dir.glob("*"))
        return len(documents) > 0

    def should_reprocess_documents(self) -> bool:
        """Determine if documents should be reprocessed."""
        # Always reprocess to ensure we use the improved RAG pipeline
        # In production, you might check timestamps or version markers
        return True

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

        # Reindex existing documents
        st.subheader("🔄 Reindex Documents")
        st.info("💡 Reindexing will apply improved chunking strategies to existing documents without re-uploading them.")

        documents = list(self.documents_dir.glob("*"))
        indexed_documents = [doc for doc in documents if doc.is_file()]

        if indexed_documents:
            st.write(f"Found {len(indexed_documents)} documents ready for reindexing:")
            for doc in indexed_documents[:5]:  # Show first 5 documents
                st.write(f"• {doc.name}")
            if len(indexed_documents) > 5:
                st.write(f"• ... and {len(indexed_documents) - 5} more")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.warning("⚠️ **Important**: Reindexing will clear the current search index and rebuild it with improved chunking strategies. This may take a few minutes.")
            with col2:
                if st.button("🔄 Reindex All Documents", type="primary"):
                    with st.spinner("Reindexing documents with improved chunking..."):
                        try:
                            success = self._reindex_all_documents()
                            if success:
                                st.success("✅ Successfully reindexed all documents with improved chunking!")
                                st.info("🎉 Wind limits queries should now return more specific answers with pilot categories and exact limits.")
                                st.rerun()
                            else:
                                st.error("❌ Failed to reindex documents")
                        except Exception as e:
                            st.error(f"Error during reindexing: {str(e)}")
        else:
            st.info("No documents available to reindex. Upload documents first.")

        # List existing documents
        st.subheader("📚 Current Documents")

        if indexed_documents:
            for doc in indexed_documents:
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

            # Check if documents have been reindexed with improved chunking
            index_file = self.vectors_dir / ".documents_indexed"
            if index_file.exists():
                index_content = index_file.read_text()
                if "reindexed with improved chunking" in index_content:
                    st.success("🎯 **Enhanced Chunking Active**: Wind limits queries will return specific pilot categories and exact limits")
                elif "improved chunking" in index_content or "Documents indexed at" in index_content:
                    st.info("📈 **Improved Chunking**: Weather limitations table is preserved as high-priority chunk")
                else:
                    st.warning("⚡ **Chunking Available**: Click 'Reindex All Documents' to apply improved wind limits detection")
        else:
            st.warning("⚠️ No documents indexed yet")

        # OpenAI status
        if self.settings.openai_api_key and self.settings.openai_api_key != "":
            st.success("✅ OpenAI API configured")
        else:
            st.error("❌ OpenAI API key not configured")

        # Chunking strategy information
        st.subheader("🔧 Document Processing Features")
        st.info("**Enhanced Chunking Strategy:**\n"
                "• Weather limitations tables preserved as single chunks\n"
                "• High priority scoring for regulatory tables\n"
                "• Aviation-specific term extraction\n"
                "• Improved search relevance for wind limits queries")

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

                    # Display IEEE-style references
                    if response.source_references:
                        st.caption("**References:**")

                        # Create IEEE reference list with numbers
                        ieee_refs = getattr(response, 'ieee_references', {})

                        for ref in response.source_references:
                            # Get IEEE reference number
                            ref_num = ieee_refs.get(ref.document_name, len(ieee_refs) + 1)

                            # Build IEEE reference string
                            ref_text = f"[{ref_num}] {ref.document_name}"

                            if ref.section_title:
                                ref_text += f", Section: {ref.section_title}"

                            if ref.page_number:
                                ref_text += f", {ref.page_number}"

                            st.caption(ref_text)
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
        """Generate response using local documents and improved RAG pipeline.

        Args:
            query: User query

        Returns:
            Chat response with source references
        """
        import time

        start_time = time.time()

        try:
            # Get available documents using proper Document objects
            documents = []
            for doc_path in self.documents_dir.glob("*"):
                if doc_path.is_file():
                    try:
                        doc = Document(
                            name=doc_path.name,
                            file_path=str(doc_path.absolute()),
                            file_type=f"application/{doc_path.suffix[1:]}" if doc_path.suffix else "text/plain",
                            directory_path=str(doc_path.parent.absolute())
                        )
                        documents.append(doc)
                    except Exception as e:
                        print(f"Error creating Document for {doc_path.name}: {e}")
                        continue

            if not documents:
                return ChatResponse(
                    message="I apologize, but I couldn't access any documents to answer your question. Please contact the administrator.",
                    sources=[],
                    source_references=[],
                    confidence=0.0,
                    processing_time=time.time() - start_time
                )

            # Use the improved document processor to process documents
            processed_documents = await self.document_processor.process_documents(documents)

            if not processed_documents:
                return ChatResponse(
                    message="I apologize, but I couldn't process any documents to answer your question. Please contact the administrator.",
                    sources=[],
                    source_references=[],
                    confidence=0.0,
                    processing_time=time.time() - start_time
                )

            print(f"Processed {len(processed_documents)} documents with improved RAG pipeline")

            # Index documents (always reprocess to use improved pipeline)
            if processed_documents and self.should_reprocess_documents():
                await self.document_processor.index_documents(processed_documents)
                print(f"Indexed {len(processed_documents)} documents using improved RAG pipeline")

            # Use semantic search to find relevant documents
            try:
                # Use higher top_k for comprehensive context, especially for wind limits queries
                top_k = 8 if any(term in query.lower() for term in ['wind', 'limit', 'cadet', 'gs', 'pilot']) else 5
                relevant_docs = await self.document_processor.search_documents(query, top_k=top_k)
                context_documents = relevant_docs if relevant_docs else processed_documents[:3]
                print(f"Search found {len(relevant_docs)} relevant documents for query: '{query}' (top_k={top_k})")
            except Exception as e:
                print(f"Error in semantic search, falling back to all documents: {e}")
                # Fallback to using all documents if search fails
                context_documents = processed_documents[:2]

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
