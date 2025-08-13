"""Document repository for managing uploaded documents in MongoDB."""

import uuid
from datetime import UTC, datetime
from typing import Any

from pymongo.collection import Collection

from vgs_chatbot.models.document import Document


class DocumentRepository:
    """Repository for managing documents in MongoDB uploads collection."""

    def __init__(self, collection: Collection) -> None:
        """Initialize document repository.

        Args:
            collection: MongoDB collection for storing uploaded documents
        """
        self.collection = collection

    async def save_document(self, document: Document) -> Document:
        """Save document to MongoDB uploads collection.

        Args:
            document: Document to save

        Returns:
            Saved document with ID
        """
        # Set ID and upload timestamp if not present
        if not document.id:
            document.id = str(uuid.uuid4())
        if not document.uploaded_at:
            document.uploaded_at = datetime.now(UTC)

        # Convert to dict for MongoDB storage
        document_dict = document.model_dump()

        # Insert into collection
        result = self.collection.insert_one(document_dict)

        # Update document with MongoDB ObjectId if needed
        if not document_dict.get("_id"):
            document_dict["_id"] = result.inserted_id

        return document

    async def get_document_by_id(self, document_id: str) -> Document | None:
        """Get document by ID.

        Args:
            document_id: Document ID

        Returns:
            Document if found, None otherwise
        """
        document_dict = self.collection.find_one({"id": document_id})
        if document_dict:
            # Remove MongoDB ObjectId before creating model
            document_dict.pop("_id", None)
            return Document(**document_dict)
        return None

    async def get_document_by_name(self, name: str) -> Document | None:
        """Get document by name.

        Args:
            name: Document name

        Returns:
            Document if found, None otherwise
        """
        document_dict = self.collection.find_one({"name": name})
        if document_dict:
            # Remove MongoDB ObjectId before creating model
            document_dict.pop("_id", None)
            return Document(**document_dict)
        return None

    async def list_documents(self) -> list[Document]:
        """List all uploaded documents.

        Returns:
            List of all documents
        """
        documents = []
        for document_dict in self.collection.find():
            # Remove MongoDB ObjectId before creating model
            document_dict.pop("_id", None)
            documents.append(Document(**document_dict))
        return documents

    async def delete_document(self, document_id: str) -> bool:
        """Delete document by ID.

        Args:
            document_id: Document ID

        Returns:
            True if deleted, False otherwise
        """
        result = self.collection.delete_one({"id": document_id})
        return result.deleted_count > 0

    async def document_exists(self, name: str) -> bool:
        """Check if document with name already exists.

        Args:
            name: Document name

        Returns:
            True if document exists, False otherwise
        """
        return self.collection.count_documents({"name": name}) > 0

    def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about the uploads collection.

        Returns:
            Dictionary with collection statistics
        """
        total_count = self.collection.count_documents({})

        # Calculate total size if documents have size field
        pipeline = [
            {"$group": {"_id": None, "total_size": {"$sum": "$size"}}}
        ]
        size_result = list(self.collection.aggregate(pipeline))
        total_size = size_result[0]["total_size"] if size_result else 0

        return {
            "total_documents": total_count,
            "total_size_bytes": total_size
        }
