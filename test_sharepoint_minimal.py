#!/usr/bin/env python3
"""Minimal SharePoint connection test - tests only SharePoint dependencies."""

import sys
import asyncio
from typing import List


class MinimalSharePointTest:
    """Minimal SharePoint connection test without full app dependencies."""
    
    def __init__(self):
        """Initialize test."""
        self.client_context = None
        self.is_connected = False
    
    def check_sharepoint_dependencies(self) -> bool:
        """Check if SharePoint specific dependencies are available."""
        print("🔍 Checking SharePoint Dependencies")
        print("=" * 40)
        
        try:
            from office365.runtime.auth.authentication_context import AuthenticationContext
            from office365.sharepoint.client_context import ClientContext
            print("✅ office365-rest-python-client installed")
            return True
        except ImportError as e:
            print(f"❌ SharePoint dependencies missing: {e}")
            print("\nTo install SharePoint dependencies:")
            print("  poetry add office365-rest-python-client")
            print("  # or")
            print("  pip install office365-rest-python-client")
            return False
    
    async def test_connection(self, site_url: str, username: str, password: str) -> bool:
        """Test SharePoint connection."""
        try:
            from office365.runtime.auth.authentication_context import AuthenticationContext
            from office365.sharepoint.client_context import ClientContext
            
            print(f"🔄 Connecting to: {site_url}")
            
            auth_context = AuthenticationContext(site_url)
            if auth_context.acquire_token_for_user(username, password):
                self.client_context = ClientContext(site_url, auth_context)
                self.is_connected = True
                
                # Test basic connection by getting site info
                site = self.client_context.site
                self.client_context.load(site)
                self.client_context.execute_query()
                
                print("✅ SharePoint connection successful!")
                print(f"   Site ID: {site.properties.get('Id', 'N/A')}")
                return True
            else:
                print("❌ Authentication failed")
                return False
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    async def test_document_access(self, directory_url: str) -> bool:
        """Test document access in a specific directory."""
        if not self.is_connected:
            print("❌ Not connected to SharePoint")
            return False
        
        try:
            from urllib.parse import urlparse
            
            parsed_url = urlparse(directory_url)
            relative_url = parsed_url.path
            
            print(f"🔄 Testing document access: {relative_url}")
            
            # Get folder
            folder = self.client_context.web.get_folder_by_server_relative_url(relative_url)
            self.client_context.load(folder)
            self.client_context.execute_query()
            
            # Get files in folder
            files = folder.files
            self.client_context.load(files)
            self.client_context.execute_query()
            
            print(f"✅ Found {len(files)} files in directory")
            
            # List first few files
            for i, file in enumerate(files):
                if i >= 5:  # Limit to first 5 files
                    break
                self.client_context.load(file)
                self.client_context.execute_query()
                print(f"   📄 {file.properties.get('Name', 'Unknown')}")
            
            if len(files) > 5:
                print(f"   ... and {len(files) - 5} more files")
            
            return True
            
        except Exception as e:
            print(f"❌ Document access error: {e}")
            print("   This might be due to:")
            print("   • Incorrect directory URL")
            print("   • Insufficient permissions")
            print("   • Directory doesn't exist")
            return False
    
    def disconnect(self):
        """Disconnect from SharePoint."""
        self.client_context = None
        self.is_connected = False
        print("🔌 Disconnected from SharePoint")


async def main():
    """Main test function."""
    print("🚀 Minimal SharePoint Connection Test")
    print("=" * 50)
    print()
    
    tester = MinimalSharePointTest()
    
    # Check dependencies
    if not tester.check_sharepoint_dependencies():
        return False
    
    print()
    
    # Get SharePoint configuration
    print("SharePoint Configuration:")
    site_url = input("SharePoint Site URL: ").strip()
    if not site_url:
        print("❌ Site URL is required")
        return False
    
    if not site_url.startswith("https://"):
        print("❌ Site URL must start with https://")
        return False
    
    print("\nCredentials:")
    username = input("Username (email): ").strip()
    if not username:
        print("❌ Username is required")
        return False
    
    import getpass
    password = getpass.getpass("Password: ")
    if not password:
        print("❌ Password is required")
        return False
    
    print()
    
    # Test connection
    success = await tester.test_connection(site_url, username, password)
    
    if success:
        print()
        
        # Test document access if user wants to
        test_docs = input("Test document access? (y/N): ").strip().lower()
        if test_docs in ('y', 'yes'):
            directory_url = input("Directory URL (e.g., /sites/yoursite/Shared Documents): ").strip()
            if directory_url:
                full_directory_url = f"{site_url.rstrip('/')}{directory_url}"
                await tester.test_document_access(full_directory_url)
        
        tester.disconnect()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 SharePoint test completed successfully!")
        print("\nYour SharePoint connection is working. Now you can:")
        print("1. Install all dependencies: poetry install")
        print("2. Configure .env with your SharePoint settings")
        print("3. Run the full application: streamlit run vgs_chatbot/gui/app.py")
    else:
        print("❌ SharePoint test failed")
        print("\nTroubleshooting tips:")
        print("• Verify SharePoint site URL is correct")
        print("• Check username/password")
        print("• Ensure you can access SharePoint in browser")
        print("• If MFA is enabled, you may need an app password")
    
    return success


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n❌ Test cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)