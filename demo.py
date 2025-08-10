#!/usr/bin/env python3
"""Demo script to showcase VGS Chatbot functionality."""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add vgs_chatbot to Python path
sys.path.insert(0, str(Path(__file__).parent))

from vgs_chatbot.models.user import User
from vgs_chatbot.models.document import Document, ProcessedDocument
from vgs_chatbot.models.chat import ChatMessage, MessageRole, ChatResponse
from vgs_chatbot.utils.config import get_settings


class ChatbotDemo:
    """Demo class for VGS Chatbot."""
    
    def __init__(self):
        """Initialize demo."""
        self.settings = get_settings()
    
    def demo_models(self):
        """Demonstrate data models."""
        print("📋 Data Models Demo")
        print("=" * 30)
        
        # User model
        user = User(
            id=1,
            username="pilot_jones",
            email="jones@vgs.com", 
            password_hash="$2b$12$hashed_password",
            is_active=True
        )
        print(f"User: {user.username} ({user.email})")
        
        # Document model
        document = Document(
            id="doc_001",
            name="VGS Safety Manual.pdf",
            url="https://sharepoint.com/safety.pdf",
            file_type="application/pdf",
            size=2048576,
            modified_date=datetime.utcnow(),
            directory_path="/documents/safety"
        )
        print(f"Document: {document.name} ({document.size} bytes)")
        
        # Chat message
        message = ChatMessage(
            role=MessageRole.USER,
            content="What are the pre-flight safety checks?",
            timestamp=datetime.utcnow(),
            user_id=1
        )
        print(f"Chat: {message.role.value} - {message.content[:50]}...")
        print()
    
    def demo_architecture(self):
        """Demonstrate SOLID architecture principles."""
        print("🏗️  SOLID Architecture Demo")
        print("=" * 30)
        
        print("Single Responsibility Principle:")
        print("  ✅ AuthenticationService - only handles user auth")
        print("  ✅ SharePointConnector - only handles SharePoint access")
        print("  ✅ DocumentProcessor - only handles document processing")
        print("  ✅ ChatService - only handles chat generation")
        
        print("\nOpen/Closed Principle:")
        print("  ✅ New document sources via DocumentConnectorInterface")
        print("  ✅ New LLM providers via ChatServiceInterface")
        print("  ✅ New authentication methods via AuthenticationInterface")
        
        print("\nLiskov Substitution Principle:")
        print("  ✅ Any DocumentConnector can replace SharePointConnector")
        print("  ✅ Any ChatService can replace LLMChatService")
        
        print("\nInterface Segregation Principle:")
        print("  ✅ Focused interfaces (auth, documents, chat, processing)")
        print("  ✅ No unnecessary method dependencies")
        
        print("\nDependency Inversion Principle:")
        print("  ✅ Services depend on interfaces, not concrete classes")
        print("  ✅ Easy to mock and test individual components")
        print()
    
    def demo_sharepoint_integration(self):
        """Demonstrate SharePoint integration concepts."""
        print("📁 SharePoint Integration Demo")
        print("=" * 30)
        
        print("Configuration:")
        print(f"  Site URL: {self.settings.sharepoint_site_url or 'Not configured'}")
        print(f"  Directories: {len(self.settings.sharepoint_directory_urls)} configured")
        
        print("\nSupported Features:")
        print("  ✅ User authentication (no admin rights required)")
        print("  ✅ Multiple SharePoint sites")
        print("  ✅ Recursive directory traversal")
        print("  ✅ File type detection (PDF, DOCX, XLSX)")
        print("  ✅ Metadata extraction (size, modified date)")
        
        print("\nExample Directory URLs:")
        example_urls = [
            "https://yourdomain.sharepoint.com/sites/vgs/Shared Documents/2FTS",
            "https://yourdomain.sharepoint.com/sites/vgs/Shared Documents/Manuals",
            "https://yourdomain.sharepoint.com/sites/vgs/Shared Documents/Procedures"
        ]
        for url in example_urls:
            print(f"  📂 {url}")
        print()
    
    def demo_rag_pipeline(self):
        """Demonstrate RAG pipeline concepts."""
        print("🧠 RAG Pipeline Demo")
        print("=" * 30)
        
        print("Document Processing Steps:")
        print("  1️⃣  Document Download (SharePoint → Bytes)")
        print("  2️⃣  Text Extraction (PDF/DOCX → Plain Text)")  
        print("  3️⃣  Text Chunking (Long Text → Manageable Pieces)")
        print("  4️⃣  Embedding Generation (Text → Vectors)")
        print("  5️⃣  Vector Storage (ChromaDB)")
        
        print("\nSearch & Retrieval:")
        print("  🔍 Query → Embedding → Similarity Search → Relevant Chunks")
        
        print("\nSupported File Types:")
        file_types = {
            "PDF": "application/pdf",
            "Word": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "Excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "Plain Text": "text/plain"
        }
        for name, mime in file_types.items():
            print(f"  📄 {name} ({mime})")
        print()
    
    def demo_chat_interface(self):
        """Demonstrate chat interface concepts."""
        print("💬 Chat Interface Demo")
        print("=" * 30)
        
        print("Streamlit Features:")
        print("  🎨 Clean, responsive web interface")
        print("  🔐 User registration and login")
        print("  📊 SharePoint connection status")
        print("  💭 Real-time chat with typing indicators")
        print("  📚 Source document references")
        print("  📈 Confidence scores and processing times")
        
        print("\nChat Flow:")
        print("  1. User logs in → Authentication Service")
        print("  2. Connects to SharePoint → Document Connector")
        print("  3. Asks question → Search Documents → Document Processor")
        print("  4. Generate response → Chat Service → LLM")
        print("  5. Display with sources → User Interface")
        
        print(f"\nAccess: http://localhost:8501")
        print(f"Title: {self.settings.app_title}")
        print()
    
    def demo_docker_deployment(self):
        """Demonstrate Docker deployment."""
        print("🐳 Docker Deployment Demo")
        print("=" * 30)
        
        print("Services:")
        print("  📦 app - VGS Chatbot (Streamlit)")
        print("  🗄️  db - PostgreSQL database")
        print("  🔧 pgadmin - Database management")
        
        print("\nCommands:")
        print("  docker-compose up -d      # Start all services")
        print("  docker-compose logs app   # View application logs")
        print("  docker-compose down       # Stop all services")
        
        print("\nPorts:")
        print("  8501 - Chatbot web interface")
        print("  5432 - PostgreSQL database")
        print("  5050 - PgAdmin web interface")
        print()
    
    def demo_testing(self):
        """Demonstrate testing approach."""
        print("🧪 Testing Demo")
        print("=" * 30)
        
        print("Test Structure:")
        print("  📁 tests/")
        print("    ├── test_auth_service.py      # Unit tests")
        print("    ├── test_document_processor.py # Unit tests")  
        print("    ├── test_sharepoint_connector.py # Unit tests")
        print("    └── test_integration.py       # Integration tests")
        
        print("\nTest Commands:")
        print("  poetry run pytest                    # Run all tests")
        print("  poetry run pytest --cov=src         # With coverage")
        print("  poetry run pytest -v tests/test_auth_service.py # Specific test")
        
        print("\nCode Quality:")
        print("  poetry run ruff check src/ tests/   # Linting")
        print("  poetry run mypy src/                # Type checking")
        print("  poetry run isort src/ tests/        # Import sorting")
        print()
    
    async def run_demo(self):
        """Run complete demo."""
        print("🚀 VGS Chatbot Architecture Demo")
        print("=" * 50)
        print()
        
        self.demo_models()
        self.demo_architecture()
        self.demo_sharepoint_integration()
        self.demo_rag_pipeline()
        self.demo_chat_interface()
        self.demo_docker_deployment()
        self.demo_testing()
        
        print("🎯 Quick Start Guide")
        print("=" * 30)
        print("1. Configure: python configure.py")
        print("2. Validate: python test_setup.py")
        print("3. Run: docker-compose up -d")
        print("4. Visit: http://localhost:8501")
        print()
        
        print("📚 Key Files")
        print("=" * 30)
        print("  SETUP_GUIDE.md      - Detailed setup instructions")
        print("  configure.py        - Interactive configuration")
        print("  test_setup.py       - Setup validation")
        print("  demo.py            - This demonstration")
        print("  CLAUDE.md          - Project requirements")
        print()
        
        print("✨ Ready to start your VGS Chatbot journey!")


async def main():
    """Main demo function."""
    demo = ChatbotDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())