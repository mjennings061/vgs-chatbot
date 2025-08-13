"""Main VGS Chatbot application with local admin document upload."""

from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

from vgs_chatbot.models.chat import ChatMessage, ChatResponse, MessageRole
from vgs_chatbot.models.document import Document
from vgs_chatbot.repositories.database import DatabaseManager
from vgs_chatbot.repositories.document_repository import DocumentRepository
from vgs_chatbot.repositories.user_repository import UserRepository
from vgs_chatbot.repositories.vector_repository import VectorRepository
from vgs_chatbot.services.auth_service import AuthenticationService
from vgs_chatbot.services.chat_service import LLMChatService
from vgs_chatbot.services.mongodb_document_processor import MongoDBDocumentProcessor
from vgs_chatbot.utils.config import get_settings


class VGSChatbot:
    """VGS Chatbot with admin document upload and user chat interface."""

    def __init__(self) -> None:
        """Initialize chatbot application."""
        self.settings = get_settings()
        self.data_dir = Path("data")
        self.vectors_dir = self.data_dir / "vectors"

        # Ensure vectors directory exists (for manifest)
        self.vectors_dir.mkdir(parents=True, exist_ok=True)

        # Initialize MongoDB database manager first
        self.db_manager = DatabaseManager(self.settings.mongo_uri)
        self.users_collection = self.db_manager.get_collection("users")
        self.uploads_collection = self.db_manager.get_collection("uploads")
        self.user_repository = UserRepository(self.users_collection)
        self.document_repository = DocumentRepository(self.uploads_collection)
        self.auth_service = AuthenticationService(
            self.user_repository,
            self.settings.jwt_secret
        )

        # Initialize / reuse document processor (persist across reruns & avoid re-embedding)
        if "document_processor" in st.session_state:
            self.document_processor = st.session_state.document_processor  # type: ignore[attr-defined]
        else:
            # Initialize MongoDB-based document processor
            documents_collection = self.db_manager.get_collection("documents")
            vector_repository = VectorRepository(documents_collection)
            self.document_processor = MongoDBDocumentProcessor(
                vector_repository=vector_repository,
                manifest_path="data/vectors/manifest.json"
            )
            st.session_state.document_processor = self.document_processor  # type: ignore[attr-defined]

        self.chat_service = LLMChatService(
            openai_api_key=self.settings.openai_api_key,
            model=self.settings.openai_model,
        )

        # Admin credentials from settings
        self.admin_credentials = {
            self.settings.admin_username: self.settings.admin_password
        }

    def validate_mod_email(self, email: str) -> bool:
        """Validate email is from RAF Air Cadets or MOD domain."""
        allowed_domains = ["@rafac.mod.gov.uk", "@mod.uk"]
        return any(email.lower().endswith(domain) for domain in allowed_domains)

    def _reindex_all_documents(self) -> bool:
        """Reindex all documents with improved chunking strategies.

        Returns:
            bool: True if reindexing was successful, False otherwise
        """
        try:
            # Get all documents from MongoDB uploads collection
            import asyncio
            documents = asyncio.run(self.document_repository.list_documents())

            if not documents:
                return False

            # Recreate processor (clears collection implicitly if needed)
            documents_collection = self.db_manager.get_collection("documents")
            vector_repository = VectorRepository(documents_collection)
            self.document_processor = MongoDBDocumentProcessor(
                vector_repository=vector_repository,
                manifest_path="data/vectors/manifest.json"
            )
            st.session_state.document_processor = (
                self.document_processor
            )  # keep in session

            # Process documents with improved chunking (async operations)
            async def process_and_index():
                processed_docs = await self.document_processor.process_documents(
                    documents
                )
                if processed_docs:
                    await self.document_processor.index_documents(processed_docs)
                return len(processed_docs) > 0

            # Run the async operations
            success = asyncio.run(process_and_index())
            if not success:
                return False

            # Update the index marker file
            index_file = self.vectors_dir / ".documents_indexed"
            index_file.write_text(
                f"Documents reindexed with improved chunking at {datetime.now(UTC)}"
            )

            return True

        except Exception as e:
            st.error(f"Reindexing failed: {str(e)}")
            return False

    def _reindex_changed_documents(self, documents: list[Document]) -> bool:
        """Reindex only changed documents.

        Args:
            documents: List of all documents to check for changes

        Returns:
            bool: True if reindexing was successful, False otherwise
        """
        try:
            import asyncio

            async def process_changed_only():
                processed_docs = await self.document_processor.process_changed_documents(
                    documents
                )
                if processed_docs:
                    await self.document_processor.index_documents(processed_docs)
                return len(processed_docs) > 0

            # Run the async operations
            success = asyncio.run(process_changed_only())
            if not success:
                return False

            # Update the index marker file
            index_file = self.vectors_dir / ".documents_indexed"
            index_file.write_text(
                f"Changed documents reindexed at {datetime.now(UTC)}"
            )

            return True

        except Exception as e:
            st.error(f"Changed document reindexing failed: {str(e)}")
            return False

    def are_documents_indexed(self) -> bool:
        """Check if documents have been processed and are ready for chat."""
        # Check if we have any documents in MongoDB
        import asyncio
        documents = asyncio.run(self.document_repository.list_documents())
        return len(documents) > 0

    def should_reprocess_documents(self) -> bool:
        """Determine if documents should be reprocessed."""
        return False

    def run(self) -> None:
        """Run the Streamlit application."""
        st.set_page_config(
            page_title="VGS Chatbot - 2FTS Document Assistant",
            page_icon="🚁",
            layout="wide",
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
                    st.error(
                        "Access restricted to @rafac.mod.gov.uk and @mod.uk email addresses"
                    )
                else:
                    # Use proper MongoDB authentication
                    try:
                        import asyncio
                        authenticated_user = asyncio.run(
                            self.auth_service.authenticate(email, password)
                        )
                        if authenticated_user:
                            st.session_state.user_type = "user"
                            st.session_state.username = email
                            st.session_state.user_id = authenticated_user.id
                            st.success("✅ Logged in successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Invalid email or password")
                    except Exception as e:
                        st.error(f"❌ Login failed: {str(e)}")

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
                    st.error(
                        "Registration restricted to @rafac.mod.gov.uk and @mod.uk email addresses"
                    )
                elif password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    # Use proper MongoDB authentication
                    try:
                        import asyncio
                        new_user = asyncio.run(
                            self.auth_service.create_user(
                                email=email,
                                password=password
                            )
                        )
                        if new_user:
                            st.success("✅ Registration successful! You can now log in.")
                        else:
                            st.error("❌ Registration failed")
                    except Exception as e:
                        if "duplicate key" in str(e).lower():
                            st.error("❌ User with this email already exists")
                        else:
                            st.error(f"❌ Registration failed: {str(e)}")

    def _render_admin_login(self) -> None:
        """Render admin login form."""
        st.header("Administrator Login")

        with st.form("admin_login"):
            email = st.text_input("Admin Email")
            password = st.text_input("Admin Password", type="password")
            login = st.form_submit_button("Admin Login")

            if login:
                if not email or not password:
                    st.error("Please enter both email and password")
                else:
                    # Try MongoDB authentication first
                    mongodb_auth_success = False
                    try:
                        import asyncio
                        authenticated_user = asyncio.run(
                            self.auth_service.authenticate(email, password)
                        )
                        if authenticated_user:
                            st.session_state.user_type = "admin"
                            st.session_state.username = email
                            st.session_state.user_id = authenticated_user.id
                            st.success("✅ Admin logged in successfully via MongoDB!")
                            mongodb_auth_success = True
                    except Exception as e:
                        print(f"MongoDB admin auth failed: {e}")

                    # Fallback to .env credentials if MongoDB auth fails
                    if not mongodb_auth_success:
                        # Check if email matches admin pattern and password matches .env
                        admin_email = f"{self.settings.admin_username}@rafac.mod.gov.uk"
                        if (email == admin_email and password == self.settings.admin_password):
                            st.session_state.user_type = "admin"
                            st.session_state.username = email
                            st.success("✅ Admin logged in successfully via .env credentials!")
                            mongodb_auth_success = True

                    if mongodb_auth_success:
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
            type=["pdf", "docx", "xlsx", "pptx", "txt"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            if st.button("Process and Index Documents"):
                with st.spinner("Processing documents..."):
                    saved_documents: list[Document] = []
                    for uploaded_file in uploaded_files:
                        try:
                            # Map extension to mime type
                            file_extension = Path(uploaded_file.name).suffix.lower()
                            file_type_map = {
                                ".pdf": "application/pdf",
                                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                ".txt": "text/plain",
                            }
                            file_type = file_type_map.get(
                                file_extension, "application/octet-stream"
                            )

                            # Check if document already exists
                            import asyncio
                            if asyncio.run(self.document_repository.document_exists(uploaded_file.name)):
                                st.warning(f"Document {uploaded_file.name} already exists. Skipping...")
                                continue

                            # Create document with file content stored in MongoDB
                            document = Document(
                                name=uploaded_file.name,
                                file_type=file_type,
                                size=uploaded_file.size,
                                file_content=bytes(uploaded_file.getbuffer()),
                                uploaded_at=datetime.now(UTC)
                            )

                            # Save to MongoDB uploads collection
                            saved_doc = asyncio.run(self.document_repository.save_document(document))
                            saved_documents.append(saved_doc)

                        except Exception as e:
                            st.error(f"Error saving {uploaded_file.name}: {str(e)}")

                    if not saved_documents:
                        st.error("❌ No documents were successfully saved")
                    else:
                        # Process & index asynchronously (mirrors reindex logic)
                        import asyncio

                        async def process_and_index():
                            processed_docs = (
                                await self.document_processor.process_documents(
                                    saved_documents
                                )
                            )
                            if processed_docs:
                                await self.document_processor.index_documents(
                                    processed_docs
                                )
                            return len(processed_docs)

                        try:
                            processed_count = asyncio.run(process_and_index())
                            if processed_count:
                                index_file = self.vectors_dir / ".documents_indexed"
                                index_file.write_text(
                                    f"Documents indexed at {datetime.now(UTC)}"
                                )
                                st.success(
                                    f"✅ Successfully processed & indexed {processed_count} documents!"
                                )
                                st.rerun()
                            else:
                                st.error(
                                    "❌ Failed to process documents (no text extracted?)"
                                )
                        except Exception as e:
                            st.error(f"Error generating embeddings: {e}")

        # Reindex existing documents
        st.subheader("🔄 Reindex Documents")
        st.info(
            "💡 Reindexing will apply improved chunking strategies to existing documents without re-uploading them."
        )

        # Get documents from MongoDB uploads collection
        import asyncio
        indexed_documents = asyncio.run(self.document_repository.list_documents())

        if indexed_documents:
            # Show manifest statistics
            manifest_stats = self.document_processor.get_manifest_stats()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Tracked Documents", manifest_stats["total_documents"])
            with col2:
                st.metric("📦 Tracked Chunks", manifest_stats["total_chunks"])
            with col3:
                model_name = manifest_stats["embedding_model"].split('/')[-1]
                st.metric("🧠 Model", model_name)

            st.write(f"Found {len(indexed_documents)} documents ready for reindexing:")
            for doc in indexed_documents[:5]:  # Show first 5 documents
                st.write(f"• {doc.name}")
            if len(indexed_documents) > 5:
                st.write(f"• ... and {len(indexed_documents) - 5} more")

            # Check for changed documents
            doc_objects = indexed_documents

            changed_docs = self.document_processor.manifest.get_changed_documents(doc_objects)

            # Show reindex options
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                if changed_docs:
                    st.info(f"💡 **{len(changed_docs)} documents have changed** and can be selectively reindexed.")
                else:
                    st.success("✅ All documents are up-to-date with the current index.")
                st.warning(
                    "⚠️ **Full reindex** will clear the current search index and rebuild everything."
                )

            with col2:
                if changed_docs and st.button("🔄 Reindex Changed Only", type="secondary"):
                    with st.spinner("Reindexing changed documents..."):
                        try:
                            success = self._reindex_changed_documents(doc_objects)
                            if success:
                                st.success(f"✅ Successfully reindexed {len(changed_docs)} changed documents!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to reindex changed documents")
                        except Exception as e:
                            st.error(f"Error during selective reindexing: {str(e)}")

            with col3:
                if st.button("🔄 Reindex All Documents", type="primary"):
                    with st.spinner("Reindexing all documents with improved chunking..."):
                        try:
                            success = self._reindex_all_documents()
                            if success:
                                st.success(
                                    "✅ Successfully reindexed all documents with improved chunking!"
                                )
                                st.info(
                                    "🎉 Wind limits queries should now return more specific answers with pilot categories and exact limits."
                                )
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
                    size_kb = (doc.size or 0) / 1024
                    st.write(f"{size_kb:.1f} KB")
                with col3:
                    if st.button("🗑️", key=f"delete_{doc.name}"):
                        import asyncio
                        if doc.id and asyncio.run(self.document_repository.delete_document(doc.id)):
                            st.success(f"Deleted {doc.name}")
                            st.rerun()
                        else:
                            st.error(f"Failed to delete {doc.name}")
        else:
            st.info("No documents uploaded yet")

    def _render_system_status(self) -> None:
        """Render system status information."""
        st.header("System Status")

        # Document statistics from MongoDB
        import asyncio
        documents = asyncio.run(self.document_repository.list_documents())
        doc_count = len(documents)
        st.metric("📄 Documents", doc_count)

        # Vector Store Health Panel
        st.subheader("🧮 Vector Store Health Panel")
        try:
            # Get collection statistics
            stats = self.document_processor.get_collection_stats()
            total_chunks = stats["total_chunks"]

            if total_chunks == 0:
                st.warning("⚠️ **Vector store is empty** - No documents indexed yet")
            else:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("📦 Total Chunks", total_chunks)

                with col2:
                    # Count distinct documents from stats
                    try:
                        distinct_doc_count = stats["processed_documents"]
                        st.metric("📋 Distinct Documents", distinct_doc_count)
                    except Exception:
                        st.metric("📋 Distinct Documents", "Unknown")

                with col3:
                    # Get embedding model info
                    embedding_model = getattr(self.document_processor, 'embedding_model', None)
                    if embedding_model and hasattr(embedding_model, 'get_sentence_transformer'):
                        model_name = embedding_model.get_sentence_transformer().get_model_card().model_name or "multi-qa-MiniLM-L6-cos-v1"
                    else:
                        model_name = "multi-qa-MiniLM-L6-cos-v1"
                    st.metric("🧠 Embedding Model", model_name.split('/')[-1])

                # Last index time
                index_file = self.vectors_dir / ".documents_indexed"
                if index_file.exists():
                    index_content = index_file.read_text()
                    st.success("✅ Vector database indexed and ready")
                    st.caption(f"Last indexed: {index_content.split('at ')[-1] if 'at ' in index_content else 'Unknown'}")
                else:
                    st.info("📊 Vector database operational (no index timestamp)")

        except Exception as e:
            st.error(f"❌ Error accessing vector store: {str(e)}")

        # Vector database status
        if self.are_documents_indexed():
            # Check if documents have been reindexed with improved chunking
            index_file = self.vectors_dir / ".documents_indexed"
            if index_file.exists():
                index_content = index_file.read_text()
                if "reindexed with improved chunking" in index_content:
                    st.success(
                        "🎯 **Enhanced Chunking Active**: Wind limits queries will return specific pilot categories and exact limits"
                    )
                elif (
                    "improved chunking" in index_content
                    or "Documents indexed at" in index_content
                ):
                    st.info(
                        "📈 **Improved Chunking**: Weather limitations table is preserved as high-priority chunk"
                    )
                else:
                    st.warning(
                        "⚡ **Chunking Available**: Click 'Reindex All Documents' to apply improved wind limits detection"
                    )
        else:
            st.warning("⚠️ No documents indexed yet")

        # OpenAI status
        st.subheader("🤖 AI Service Status")
        if self.settings.openai_api_key and self.settings.openai_api_key != "":
            st.success("✅ OpenAI API configured")
            st.metric("🎯 Model", self.settings.openai_model)
        else:
            st.error("❌ OpenAI API key not configured")

        # Chunking strategy information
        st.subheader("🔧 Document Processing Features")
        st.info(
            "**Enhanced Chunking Strategy:**\n"
            "• Weather limitations tables preserved as single chunks\n"
            "• High priority scoring for regulatory tables\n"
            "• Aviation-specific term extraction\n"
            "• Improved search relevance for wind limits queries"
        )

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
            import asyncio
            documents = asyncio.run(self.document_repository.list_documents())
            if documents:
                for doc in documents:
                    st.write(f"📄 {doc.name}")
            else:
                st.info("No documents available yet")

        # Check if documents are available
        if not self.are_documents_indexed():
            st.warning(
                "⚠️ No documents have been processed yet. Please contact your administrator to upload documents."
            )
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
                        ieee_refs = getattr(response, "ieee_references", {})

                        for ref in response.source_references:
                            # Get IEEE reference number
                            ref_num = ieee_refs.get(
                                ref.document_name, len(ieee_refs) + 1
                            )

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
            # Ensure there is at least one indexed document
            stats = self.document_processor.get_collection_stats()
            if stats["total_chunks"] == 0:
                return ChatResponse(
                    message="No indexed documents found. Please upload and process documents first (Admin > Document Management).",
                    sources=[],
                    source_references=[],
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                )

            # Use semantic search against existing vector index
            try:
                # Use higher top_k for comprehensive context, especially for wind limits queries
                top_k = (
                    8
                    if any(
                        term in query.lower()
                        for term in ["wind", "limit", "cadet", "gs", "pilot"]
                    )
                    else 5
                )
                relevant_docs = await self.document_processor.search_documents(
                    query, top_k=top_k
                )
                context_documents = relevant_docs[:3] if relevant_docs else []
                print(
                    f"Search found {len(relevant_docs)} relevant documents for query: '{query}' (top_k={top_k})"
                )
            except Exception as e:
                print(f"Error in semantic search, falling back to all documents: {e}")
                context_documents = []

            # Create a simple chat message for the LLM service
            messages = [
                ChatMessage(
                    role=MessageRole.USER, content=query, timestamp=datetime.now(UTC)
                )
            ]

            # Use the chat service to generate a proper response
            response = await self.chat_service.generate_response(
                messages, context_documents
            )

            return response

        except Exception as e:
            return ChatResponse(
                message=f"I encountered an error while processing your question: {str(e)}. Please try again or contact the administrator.",
                sources=[],
                source_references=[],
                confidence=0.0,
                processing_time=time.time() - start_time,
            )


def main() -> None:
    """Main entry point."""
    app = VGSChatbot()
    app.run()


if __name__ == "__main__":
    main()
