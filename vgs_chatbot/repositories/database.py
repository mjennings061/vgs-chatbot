"""Database models and configuration for MongoDB."""


from pymongo import MongoClient
from pymongo.database import Database


class DatabaseManager:
    """MongoDB connection manager."""

    def __init__(self, mongo_uri: str, database_name: str = "chatbot") -> None:
        """Initialize database manager.

        Args:
            mongo_uri: MongoDB connection URI
            database_name: Name of the database to use
        """
        self.mongo_uri = mongo_uri
        self.database_name = database_name
        self.client: MongoClient | None = None
        self.database: Database | None = None

    def connect(self) -> Database:
        """Connect to MongoDB and return database instance.

        Returns:
            MongoDB database instance
        """
        if self.client is None:
            self.client = MongoClient(self.mongo_uri)
            self.database = self.client[self.database_name]
        return self.database

    def get_collection(self, collection_name: str):
        """Get MongoDB collection.

        Args:
            collection_name: Name of the collection

        Returns:
            MongoDB collection instance
        """
        if self.database is None:
            self.connect()
        return self.database[collection_name]

    async def create_indexes(self) -> None:
        """Create database indexes for better performance."""
        db = self.connect()

        # Create unique index for users collection
        users_collection = db.users
        users_collection.create_index("email", unique=True)

        # Create indexes for documents collection (vector storage)
        documents_collection = db.documents
        # Compound index for document queries
        documents_collection.create_index([
            ("document_id", 1),
            ("chunk_index", 1)
        ])
        # Index for metadata queries
        documents_collection.create_index("metadata.document_name")
        # Text search index for content
        documents_collection.create_index([("content", "text")])

        # Create indexes for uploads collection
        uploads_collection = db.uploads
        # Unique index for document names
        uploads_collection.create_index("name", unique=True)
        # Index for document ID
        uploads_collection.create_index("id")
        # Index for file type queries
        uploads_collection.create_index("file_type")
        # Index for upload date queries
        uploads_collection.create_index("uploaded_at")

    def close(self) -> None:
        """Close database connection."""
        if self.client is not None:
            self.client.close()
            self.client = None
            self.database = None
