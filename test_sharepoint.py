#!/usr/bin/env python3
"""Test SharePoint connection specifically."""

import asyncio
import sys
from pathlib import Path

# Add vgs_chatbot to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from vgs_chatbot.services.sharepoint_connector import SharePointConnector
    from vgs_chatbot.utils.config import get_settings
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Run 'poetry install' to install dependencies")
    sys.exit(1)


async def test_sharepoint_connection():
    """Test SharePoint connection with user credentials."""
    print("🔍 SharePoint Connection Test")
    print("=" * 40)
    
    settings = get_settings()
    
    # Display configuration
    print(f"SharePoint Site URL: {settings.sharepoint_site_url or 'Not configured'}")
    print(f"Directory URLs: {len(settings.sharepoint_directory_urls)} configured")
    print()
    
    if not settings.sharepoint_site_url:
        print("⚠️  SharePoint site URL not configured in .env")
        print("Add SHAREPOINT_SITE_URL=https://yourcompany.sharepoint.com/sites/yoursite")
        return False
    
    # Get user credentials
    print("Enter your SharePoint credentials:")
    username = input("Username (email): ").strip()
    if not username:
        print("❌ Username is required")
        return False
    
    import getpass
    password = getpass.getpass("Password: ")
    if not password:
        print("❌ Password is required")
        return False
    
    print("\n🔄 Testing SharePoint connection...")
    
    connector = SharePointConnector()
    
    try:
        # Test connection
        connected = await connector.connect(username, password)
        
        if connected:
            print("✅ SharePoint connection successful!")
            
            # Test document listing if directories are configured
            if settings.sharepoint_directory_urls:
                print(f"\n🔄 Testing document listing from {len(settings.sharepoint_directory_urls)} directories...")
                
                try:
                    documents = await connector.list_documents(settings.sharepoint_directory_urls)
                    print(f"✅ Found {len(documents)} documents")
                    
                    # Show first few documents
                    for i, doc in enumerate(documents[:5]):
                        print(f"  📄 {doc.name} ({doc.file_type})")
                        if i >= 4 and len(documents) > 5:
                            print(f"  ... and {len(documents) - 5} more documents")
                            break
                            
                except Exception as e:
                    print(f"⚠️  Document listing failed: {e}")
                    print("This might be due to permissions or incorrect directory URLs")
            
            await connector.disconnect()
            print("\n✅ SharePoint test completed successfully!")
            return True
            
        else:
            print("❌ SharePoint connection failed")
            print("Possible issues:")
            print("  - Incorrect username/password")
            print("  - MFA enabled (try app password)")
            print("  - Incorrect SharePoint site URL")
            return False
            
    except Exception as e:
        print(f"❌ SharePoint connection error: {e}")
        print("\nTroubleshooting tips:")
        print("  - Verify your SharePoint site URL is correct")
        print("  - Check if you can access SharePoint in browser")
        print("  - If MFA is enabled, you may need an app password")
        print("  - Ensure your account has access to the site")
        return False


def test_sharepoint_dependencies():
    """Test if SharePoint dependencies are available."""
    print("🔍 Testing SharePoint Dependencies")
    print("=" * 40)
    
    missing_packages = []
    
    try:
        from office365.runtime.auth.authentication_context import AuthenticationContext
        print("✅ office365-rest-python-client available")
    except ImportError:
        missing_packages.append("office365-rest-python-client")
        print("❌ office365-rest-python-client missing")
    
    try:
        from pydantic import HttpUrl
        print("✅ pydantic available")
    except ImportError:
        missing_packages.append("pydantic")
        print("❌ pydantic missing")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: poetry install")
        return False
    
    print("\n✅ All SharePoint dependencies available")
    return True


async def main():
    """Main test function."""
    print("🚀 SharePoint Connection Test Suite")
    print("=" * 50)
    print()
    
    # Test dependencies first
    deps_ok = test_sharepoint_dependencies()
    print()
    
    if not deps_ok:
        return False
    
    # Test SharePoint connection
    connection_ok = await test_sharepoint_connection()
    
    print("\n" + "=" * 50)
    if connection_ok:
        print("🎉 SharePoint test completed successfully!")
        print("\nNext steps:")
        print("1. Your SharePoint connection is working")
        print("2. You can now run the full application")
        print("3. Use: streamlit run vgs_chatbot/gui/app.py")
    else:
        print("❌ SharePoint test failed")
        print("\nNext steps:")
        print("1. Check your SharePoint configuration")
        print("2. Verify your credentials")
        print("3. See troubleshooting tips above")
    
    return connection_ok


if __name__ == "__main__":
    asyncio.run(main())