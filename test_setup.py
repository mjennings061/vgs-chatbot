#!/usr/bin/env python3
"""Setup validation script for VGS Chatbot."""

import asyncio
import os
import sys
from pathlib import Path

# Add vgs_chatbot to Python path
sys.path.insert(0, str(Path(__file__).parent))

import requests
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from vgs_chatbot.utils.config import get_settings
from vgs_chatbot.repositories.database import DatabaseManager
from vgs_chatbot.services.auth_service import AuthenticationService
from vgs_chatbot.repositories.user_repository import UserRepository


class SetupValidator:
    """Validate VGS Chatbot setup."""
    
    def __init__(self):
        """Initialize validator."""
        self.settings = get_settings()
        self.results = {}
    
    def check_environment_variables(self) -> bool:
        """Check required environment variables."""
        print("🔍 Checking environment variables...")
        
        required_vars = [
            "DATABASE_URL",
            "JWT_SECRET", 
            "OPENAI_API_KEY"
        ]
        
        missing_vars = []
        for var in required_vars:
            value = getattr(self.settings, var.lower(), None)
            if not value:
                missing_vars.append(var)
        
        if missing_vars:
            print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
            print("   Create .env file with required variables (see .env.example)")
            return False
        
        print("✅ All required environment variables found")
        return True
    
    def check_openai_connection(self) -> bool:
        """Test OpenAI API connection."""
        print("🔍 Testing OpenAI API connection...")
        
        try:
            headers = {
                "Authorization": f"Bearer {self.settings.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                models = response.json()
                model_names = [m["id"] for m in models.get("data", [])]
                
                if self.settings.openai_model in model_names:
                    print(f"✅ OpenAI API connected, model '{self.settings.openai_model}' available")
                else:
                    print(f"⚠️  OpenAI API connected, but model '{self.settings.openai_model}' not found")
                    print(f"   Available models include: {', '.join(model_names[:5])}...")
                
                return True
            else:
                print(f"❌ OpenAI API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to connect to OpenAI API: {e}")
            return False
    
    async def check_database_connection(self) -> bool:
        """Test database connection."""
        print("🔍 Testing database connection...")
        
        try:
            engine = create_async_engine(self.settings.database_url)
            
            async with engine.begin() as conn:
                result = await conn.execute(text("SELECT 1 as test"))
                row = result.fetchone()
                
                if row and row.test == 1:
                    print("✅ Database connection successful")
                    
                    # Test table creation
                    db_manager = DatabaseManager(self.settings.database_url)
                    await db_manager.create_tables()
                    print("✅ Database tables created/verified")
                    
                    await db_manager.close()
                    await engine.dispose()
                    return True
                    
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            print("   Make sure PostgreSQL is running and DATABASE_URL is correct")
            return False
        
        return False
    
    async def test_auth_service(self) -> bool:
        """Test authentication service."""
        print("🔍 Testing authentication service...")
        
        try:
            db_manager = DatabaseManager(self.settings.database_url)
            
            async for session in db_manager.get_session():
                user_repo = UserRepository(session)
                auth_service = AuthenticationService(user_repo, self.settings.jwt_secret)
                
                # Test user creation
                test_user = await auth_service.create_user(
                    username="testuser_setup",
                    password="testpass123",
                    email="test@example.com"
                )
                print("✅ User creation successful")
                
                # Test authentication
                authenticated_user = await auth_service.authenticate(
                    "testuser_setup", 
                    "testpass123"
                )
                
                if authenticated_user:
                    print("✅ User authentication successful")
                    
                    # Test token generation/validation
                    token = auth_service.generate_token(authenticated_user)
                    validated_user = await auth_service.validate_token(token)
                    
                    if validated_user and validated_user.username == "testuser_setup":
                        print("✅ JWT token generation/validation successful")
                        await db_manager.close()
                        return True
                
            await db_manager.close()
            
        except Exception as e:
            print(f"❌ Authentication service test failed: {e}")
            return False
        
        return False
    
    def check_dependencies(self) -> bool:
        """Check Python dependencies."""
        print("🔍 Checking Python dependencies...")
        
        required_packages = [
            "streamlit",
            "sqlalchemy", 
            "psycopg2",
            "pydantic",
            "bcrypt",
            "jwt",
            "requests",
            "sentence_transformers",
            "chromadb",
            "langchain",
            "openai"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing packages: {', '.join(missing_packages)}")
            print("   Run: poetry install")
            return False
        
        print("✅ All required packages installed")
        return True
    
    def check_file_structure(self) -> bool:
        """Check project file structure."""
        print("🔍 Checking project structure...")
        
        required_files = [
            "vgs_chatbot/__init__.py",
            "vgs_chatbot/gui/app.py",
            "vgs_chatbot/services/auth_service.py",
            "vgs_chatbot/services/sharepoint_connector.py",
            "vgs_chatbot/services/document_processor.py",
            "vgs_chatbot/services/chat_service.py",
            "pyproject.toml",
            "docker-compose.yml",
            "Dockerfile",
            ".env.example"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            print(f"❌ Missing files: {', '.join(missing_files)}")
            return False
        
        print("✅ All required files present")
        return True
    
    async def run_all_checks(self) -> bool:
        """Run all validation checks."""
        print("🚀 Starting VGS Chatbot setup validation...\n")
        
        checks = [
            ("File Structure", self.check_file_structure()),
            ("Dependencies", self.check_dependencies()),
            ("Environment Variables", self.check_environment_variables()),
            ("OpenAI Connection", self.check_openai_connection()),
            ("Database Connection", await self.check_database_connection()),
            ("Authentication Service", await self.test_auth_service())
        ]
        
        all_passed = True
        
        print("\n" + "="*50)
        print("VALIDATION RESULTS")
        print("="*50)
        
        for check_name, result in checks:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{check_name:<25} {status}")
            if not result:
                all_passed = False
        
        print("="*50)
        
        if all_passed:
            print("🎉 All checks passed! Your VGS Chatbot is ready to use.")
            print("\nNext steps:")
            print("1. Run: poetry run streamlit run vgs_chatbot/gui/app.py")
            print("2. Or: docker-compose up -d")
            print("3. Visit: http://localhost:8501")
        else:
            print("❌ Some checks failed. Please fix the issues above before running the application.")
        
        return all_passed


async def main():
    """Main validation function."""
    validator = SetupValidator()
    success = await validator.run_all_checks()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())