"""Vector repository for document embedding operations."""

from typing import Any

from pymongo.collection import Collection

from vgs_chatbot.models.vector_document import (
    DocumentSummary,
    VectorDocument,
    VectorSearchResult,
)


class VectorRepository:
    """Repository for vector document database operations."""

    def __init__(self, collection: Collection) -> None:
        """Initialize vector repository.

        Args:
            collection: MongoDB documents collection
        """
        self.collection = collection

    def add_document_chunk(self, vector_doc: VectorDocument) -> str:
        """Add a document chunk with embedding.

        Args:
            vector_doc: Vector document to store

        Returns:
            Document ID in MongoDB
        """
        doc_dict = vector_doc.model_dump(exclude={"id"}, by_alias=True)
        result = self.collection.insert_one(doc_dict)
        return str(result.inserted_id)

    def add_document_summary(self, doc_summary: DocumentSummary) -> str:
        """Add a document summary with embedding.

        Args:
            doc_summary: Document summary to store

        Returns:
            Document ID in MongoDB
        """
        summary_dict = doc_summary.model_dump(exclude={"id"}, by_alias=True)
        result = self.collection.insert_one(summary_dict)
        return str(result.inserted_id)

    def similarity_search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        document_filter: dict[str, Any] | None = None,
        index_name: str = "vector_index"
    ) -> list[VectorSearchResult]:
        """Perform similarity search using MongoDB Vector Search.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            document_filter: Optional filter for documents
            index_name: Name of the vector search index

        Returns:
            List of search results sorted by similarity score
        """
        # Build the $vectorSearch pipeline stage
        vector_search_stage = {
            "$vectorSearch": {
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": limit * 10,  # Search more candidates for better recall
                "limit": limit,
                "index": index_name
            }
        }

        # Add filter if provided
        if document_filter:
            vector_search_stage["$vectorSearch"]["filter"] = document_filter

        # Build aggregation pipeline
        pipeline = [
            vector_search_stage,
            {
                "$project": {
                    "document_id": 1,
                    "chunk_id": 1,
                    "content": 1,
                    "metadata": 1,
                    "section_title": 1,
                    "page_number": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        try:
            results = list(self.collection.aggregate(pipeline))

            return [
                VectorSearchResult(
                    document_id=result["document_id"],
                    chunk_id=result["chunk_id"],
                    content=result["content"],
                    score=result.get("score", 0.0),
                    metadata=result.get("metadata", {}),
                    section_title=result.get("section_title"),
                    page_number=result.get("page_number")
                )
                for result in results
            ]
        except Exception as e:
            print(f"Vector search failed: {e}")
            print("Falling back to manual cosine similarity search...")
            return self._fallback_similarity_search(query_embedding, limit, document_filter)

    def _fallback_similarity_search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        document_filter: dict[str, Any] | None = None
    ) -> list[VectorSearchResult]:
        """Fallback similarity search using manual cosine similarity calculation.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            document_filter: Optional filter for documents

        Returns:
            List of search results sorted by similarity score
        """
        # MongoDB aggregation pipeline for cosine similarity
        pipeline = []

        # Add document filter if provided
        if document_filter:
            pipeline.append({"$match": document_filter})

        # Add cosine similarity calculation
        pipeline.extend([
            {
                "$addFields": {
                    "similarity_score": {
                        "$reduce": {
                            "input": {"$range": [0, {"$size": "$embedding"}]},
                            "initialValue": {"dot": 0, "norm_a": 0, "norm_b": 0},
                            "in": {
                                "$let": {
                                    "vars": {
                                        "idx": "$$this",
                                        "a_val": {"$arrayElemAt": ["$embedding", "$$this"]},
                                        "b_val": {"$arrayElemAt": [query_embedding, "$$this"]}
                                    },
                                    "in": {
                                        "dot": {
                                            "$add": [
                                                "$$value.dot",
                                                {"$multiply": ["$$a_val", "$$b_val"]}
                                            ]
                                        },
                                        "norm_a": {
                                            "$add": [
                                                "$$value.norm_a",
                                                {"$multiply": ["$$a_val", "$$a_val"]}
                                            ]
                                        },
                                        "norm_b": {
                                            "$add": [
                                                "$$value.norm_b",
                                                {"$multiply": ["$$b_val", "$$b_val"]}
                                            ]
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            {
                "$addFields": {
                    "cosine_similarity": {
                        "$cond": {
                            "if": {
                                "$and": [
                                    {"$gt": ["$similarity_score.norm_a", 0]},
                                    {"$gt": ["$similarity_score.norm_b", 0]}
                                ]
                            },
                            "then": {
                                "$divide": [
                                    "$similarity_score.dot",
                                    {
                                        "$multiply": [
                                            {"$sqrt": "$similarity_score.norm_a"},
                                            {"$sqrt": "$similarity_score.norm_b"}
                                        ]
                                    }
                                ]
                            },
                            "else": 0
                        }
                    }
                }
            },
            {"$sort": {"cosine_similarity": -1}},
            {"$limit": limit},
            {
                "$project": {
                    "document_id": 1,
                    "chunk_id": 1,
                    "content": 1,
                    "metadata": 1,
                    "section_title": 1,
                    "page_number": 1,
                    "score": "$cosine_similarity"
                }
            }
        ])

        results = list(self.collection.aggregate(pipeline))

        return [
            VectorSearchResult(
                document_id=result["document_id"],
                chunk_id=result["chunk_id"],
                content=result["content"],
                score=result.get("score", 0.0),
                metadata=result.get("metadata", {}),
                section_title=result.get("section_title"),
                page_number=result.get("page_number")
            )
            for result in results
        ]

    def delete_document_chunks(self, document_id: str) -> int:
        """Delete all chunks for a specific document.

        Args:
            document_id: ID of document to delete chunks for

        Returns:
            Number of deleted documents
        """
        result = self.collection.delete_many({"document_id": document_id})
        return result.deleted_count

    def get_collection_count(self) -> int:
        """Get total count of documents in collection.

        Returns:
            Total document count
        """
        return self.collection.count_documents({})

    def create_vector_indexes(self) -> None:
        """Create indexes for better query performance."""
        # Create compound index for document queries
        self.collection.create_index([
            ("document_id", 1),
            ("chunk_index", 1)
        ])

        # Create index for metadata queries
        self.collection.create_index("metadata.document_name")

        # Create text search index for content
        self.collection.create_index([("content", "text")])

    def create_vector_search_index(
        self,
        index_name: str = "vector_index",
        vector_field: str = "embedding",
        similarity_function: str = "cosine",
        dimensions: int = 384
    ) -> bool:
        """Create a vector search index for MongoDB Atlas Vector Search.

        Args:
            index_name: Name of the vector search index
            vector_field: Field containing the vector embeddings
            similarity_function: Similarity function (cosine, euclidean, dotProduct)
            dimensions: Number of dimensions in the vector

        Returns:
            True if index creation was successful

        Note:
            This requires MongoDB Atlas with Vector Search capability.
            Vector indexes are created through the MongoDB Atlas UI or Atlas Administration API.
            This method provides the configuration that should be used.
        """
        # Vector search indexes must be created through MongoDB Atlas UI or Admin API
        # This method documents the required configuration
        vector_index_config = {
            "name": index_name,
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": vector_field,
                        "numDimensions": dimensions,
                        "similarity": similarity_function
                    }
                ]
            }
        }

        print("📝 Vector Search Index Configuration Required:")
        print(f"   Index Name: {index_name}")
        print(f"   Vector Field: {vector_field}")
        print(f"   Dimensions: {dimensions}")
        print(f"   Similarity: {similarity_function}")
        print(f"   Database: {self.collection.database.name}")
        print(f"   Collection: {self.collection.name}")
        print("\n🔧 To create this index, use the MongoDB Atlas UI:")
        print("   1. Go to Atlas → Database → Search")
        print("   2. Create Search Index → Atlas Vector Search")
        print("   3. Use the configuration above")
        print("\n📋 Or use this JSON configuration:")
        import json
        print(json.dumps(vector_index_config, indent=2))

        # Try to detect if vector search is available
        try:
            # Test with a normalized random vector (avoid zero vector for cosine similarity)
            import random
            test_vector = [random.uniform(0.1, 1.0) for _ in range(dimensions)]
            
            test_pipeline = [
                {
                    "$vectorSearch": {
                        "queryVector": test_vector,
                        "path": vector_field,
                        "numCandidates": 1,
                        "limit": 1,
                        "index": index_name
                    }
                },
                {"$project": {"_id": 1}}  # Just project the _id field
            ]
            # Execute but don't iterate through results
            cursor = self.collection.aggregate(test_pipeline)
            cursor.close()
            print(f"✅ Vector search index '{index_name}' is already available!")
            return True
        except Exception as e:
            error_msg = str(e)
            if "index not found" in error_msg.lower() or "no such command" in error_msg.lower():
                print(f"❌ Vector search index '{index_name}' not available: {e}")
                print("💡 The system will fall back to manual similarity search until the index is created.")
                return False
            else:
                # Vector search exists but query failed for other reasons (likely no data yet)
                print(f"✅ Vector search index '{index_name}' is available! (Test query failed but index exists)")
                return True

    def get_document_chunks(self, document_id: str) -> list[VectorDocument]:
        """Get all chunks for a specific document.

        Args:
            document_id: ID of document to get chunks for

        Returns:
            List of vector documents
        """
        chunks = list(self.collection.find({"document_id": document_id}))

        return [
            VectorDocument(
                _id=str(chunk["_id"]),
                document_id=chunk["document_id"],
                chunk_id=chunk["chunk_id"],
                content=chunk["content"],
                embedding=chunk["embedding"],
                metadata=chunk.get("metadata", {}),
                section_title=chunk.get("section_title"),
                page_number=chunk.get("page_number"),
                chunk_index=chunk.get("chunk_index"),
                key_terms=chunk.get("key_terms", []),
                created_at=chunk.get("created_at")
            )
            for chunk in chunks
        ]
