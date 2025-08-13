"""Document manifest system for tracking changes and avoiding redundant processing."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vgs_chatbot.models.document import Document


class DocumentManifest:
    """Manages document manifest for tracking file changes and processing state."""

    def __init__(self, manifest_path: str) -> None:
        """Initialize manifest manager.

        Args:
            manifest_path: Path to manifest JSON file
        """
        self.manifest_path = Path(manifest_path)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest: dict[str, Any] = self._load_manifest()

    def _load_manifest(self) -> dict[str, Any]:
        """Load manifest from disk or create new one."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                # Handle corrupted manifest by creating new one
                pass

        # Create new manifest structure
        return {
            "embedding_model": "multi-qa-MiniLM-L6-cos-v1",
            "last_full_reindex": None,
            "documents": {},
        }

    def _save_manifest(self) -> None:
        """Save manifest to disk."""
        try:
            with open(self.manifest_path, "w") as f:
                json.dump(self._manifest, f, indent=2, default=str)
        except OSError as e:
            print(f"Warning: Could not save manifest: {e}")

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file content.

        Args:
            file_path: Path to file

        Returns:
            SHA256 hash as hex string
        """
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except OSError:
            return ""

    def get_changed_documents(self, documents: list[Document]) -> list[Document]:
        """Get list of documents that have changed since last processing.

        Args:
            documents: List of documents to check

        Returns:
            List of documents that need reprocessing
        """
        changed_documents = []

        for doc in documents:
            # Handle MongoDB-stored documents (no file_path)
            if doc.file_path is None:
                # For MongoDB documents, check if they exist in manifest
                manifest_entry = self._manifest["documents"].get(doc.name)
                if not manifest_entry:
                    # New document not in manifest
                    changed_documents.append(doc)
                elif doc.uploaded_at and manifest_entry.get("uploaded_at"):
                    # Compare upload timestamps if available
                    manifest_upload_time = manifest_entry.get("uploaded_at")
                    current_upload_time = doc.uploaded_at.isoformat() if hasattr(doc.uploaded_at, 'isoformat') else str(doc.uploaded_at)
                    if manifest_upload_time != current_upload_time:
                        changed_documents.append(doc)
                continue

            # Handle file-based documents (legacy)
            file_path = Path(doc.file_path)
            if not file_path.exists():
                continue

            # Calculate current file properties
            current_hash = self._calculate_file_hash(file_path)
            current_size = file_path.stat().st_size
            current_mtime = int(file_path.stat().st_mtime)

            # Check against manifest
            manifest_entry = self._manifest["documents"].get(doc.name)

            if not manifest_entry:
                # New document
                changed_documents.append(doc)
            elif (
                manifest_entry.get("sha256") != current_hash
                or manifest_entry.get("size") != current_size
                or manifest_entry.get("mtime") != current_mtime
            ):
                # Changed document
                changed_documents.append(doc)

        return changed_documents

    def update_document_entry(
        self, doc: Document, chunk_count: int, embedding_model: str
    ) -> None:
        """Update manifest entry for a processed document.

        Args:
            doc: Document that was processed
            chunk_count: Number of chunks generated
            embedding_model: Embedding model used
        """
        # Update manifest
        self._manifest["embedding_model"] = embedding_model

        # Handle MongoDB-stored documents
        if doc.file_path is None:
            # For MongoDB documents, use document ID and upload time
            self._manifest["documents"][doc.name] = {
                "document_id": doc.id,
                "size": doc.size or 0,
                "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at and hasattr(doc.uploaded_at, 'isoformat') else str(doc.uploaded_at) if doc.uploaded_at else None,
                "chunk_count": chunk_count,
                "last_indexed": datetime.now(UTC).isoformat(),
                "storage_type": "mongodb"
            }
        else:
            # Handle file-based documents (legacy)
            file_path = Path(doc.file_path)
            if not file_path.exists():
                return

            # Calculate file properties
            file_hash = self._calculate_file_hash(file_path)
            file_size = file_path.stat().st_size
            file_mtime = int(file_path.stat().st_mtime)

            self._manifest["documents"][doc.name] = {
                "sha256": file_hash,
                "size": file_size,
                "mtime": file_mtime,
                "chunk_count": chunk_count,
                "last_indexed": datetime.now(UTC).isoformat(),
                "storage_type": "filesystem"
            }

        self._save_manifest()

    def mark_full_reindex(self, embedding_model: str) -> None:
        """Mark that a full reindex was completed.

        Args:
            embedding_model: Embedding model used for reindex
        """
        self._manifest["embedding_model"] = embedding_model
        self._manifest["last_full_reindex"] = datetime.now(UTC).isoformat()
        self._save_manifest()

    def should_full_reindex(self, current_embedding_model: str) -> bool:
        """Check if full reindex is needed due to model change.

        Args:
            current_embedding_model: Current embedding model

        Returns:
            True if full reindex is needed
        """
        manifest_model = self._manifest.get("embedding_model", "")
        return manifest_model != current_embedding_model

    def get_stats(self) -> dict[str, Any]:
        """Get manifest statistics.

        Returns:
            Dictionary with manifest statistics
        """
        total_docs = len(self._manifest["documents"])
        total_chunks = sum(
            doc.get("chunk_count", 0) for doc in self._manifest["documents"].values()
        )

        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "embedding_model": self._manifest.get("embedding_model", "Unknown"),
            "last_full_reindex": self._manifest.get("last_full_reindex"),
            "documents": self._manifest["documents"],
        }

    def remove_document(self, document_name: str) -> None:
        """Remove document from manifest.

        Args:
            document_name: Name of document to remove
        """
        if document_name in self._manifest["documents"]:
            del self._manifest["documents"][document_name]
            self._save_manifest()

    def clear(self) -> None:
        """Clear all manifest entries."""
        self._manifest = {
            "embedding_model": self._manifest.get("embedding_model", "multi-qa-MiniLM-L6-cos-v1"),
            "last_full_reindex": None,
            "documents": {},
        }
        self._save_manifest()
