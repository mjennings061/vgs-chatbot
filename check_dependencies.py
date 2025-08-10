#!/usr/bin/env python3
"""Check if all dependencies are properly installed."""

import sys
from pathlib import Path

# Add vgs_chatbot to Python path
sys.path.insert(0, str(Path(__file__).parent))

def check_dependencies():
    """Check all required dependencies."""
    print("🔍 Dependency Check")
    print("=" * 30)
    
    # Core dependencies from pyproject.toml
    dependencies = [
        ("streamlit", "Streamlit web framework"),
        ("sqlalchemy", "Database ORM"),
        ("pydantic", "Data validation"),
        ("jwt", "JWT tokens (PyJWT)"),
        ("bcrypt", "Password hashing"),
        ("langchain", "LLM framework"),
        ("chromadb", "Vector database"),
        ("office365", "SharePoint integration (office365-rest-python-client)"),
        ("sentence_transformers", "ML embeddings"),
        ("pypdf", "PDF processing"),
        ("docx", "Word document processing (python-docx)"),
        ("openpyxl", "Excel processing"),
        ("requests", "HTTP requests"),
        ("dotenv", "Environment variables (python-dotenv)"),
        ("psycopg2", "PostgreSQL adapter"),
        ("asyncpg", "Async PostgreSQL driver"),
    ]
    
    missing = []
    available = []
    
    for package, description in dependencies:
        try:
            __import__(package)
            available.append((package, description))
            print(f"✅ {package:<20} - {description}")
        except ImportError:
            missing.append((package, description))
            print(f"❌ {package:<20} - {description}")
    
    print("\n" + "=" * 30)
    print(f"✅ Available: {len(available)}")
    print(f"❌ Missing: {len(missing)}")
    
    if missing:
        print("\n❌ Missing Dependencies:")
        for package, description in missing:
            print(f"   • {package} - {description}")
        
        print("\n🔧 To fix:")
        print("   poetry install")
        print("   # or")
        print("   pip install -e .")
        return False
    
    print("\n✅ All dependencies available!")
    return True


def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Python Version Check")
    print("=" * 30)
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version >= (3, 11):
        print("✅ Python version compatible")
        return True
    else:
        print("❌ Python 3.11+ required")
        return False


def check_vgs_chatbot_imports():
    """Check if VGS Chatbot modules can be imported."""
    print("📦 VGS Chatbot Module Check")
    print("=" * 30)
    
    modules = [
        "vgs_chatbot",
        "vgs_chatbot.models.user",
        "vgs_chatbot.services.auth_service",
        "vgs_chatbot.services.sharepoint_connector",
        "vgs_chatbot.gui.app",
        "vgs_chatbot.utils.config",
    ]
    
    failed = []
    
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module} - {e}")
            failed.append(module)
    
    if failed:
        print(f"\n❌ {len(failed)} module(s) failed to import")
        return False
    
    print(f"\n✅ All {len(modules)} modules imported successfully")
    return True


def main():
    """Main check function."""
    print("🚀 VGS Chatbot Dependency Check")
    print("=" * 50)
    print()
    
    python_ok = check_python_version()
    print()
    
    deps_ok = check_dependencies()
    print()
    
    modules_ok = False
    if deps_ok:
        modules_ok = check_vgs_chatbot_imports()
        print()
    
    print("=" * 50)
    if python_ok and deps_ok and modules_ok:
        print("🎉 All checks passed! VGS Chatbot is ready to run.")
        print("\nTo start the application:")
        print("   streamlit run vgs_chatbot/gui/app.py")
        print("\nTo test SharePoint:")
        print("   python test_sharepoint.py")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        
        if not python_ok:
            print("\n• Python version issue")
        if not deps_ok:
            print("• Missing dependencies - run: poetry install")
        if not modules_ok and deps_ok:
            print("• Module import issues - check the error messages")
    
    return python_ok and deps_ok and modules_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)