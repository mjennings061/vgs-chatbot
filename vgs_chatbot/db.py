"""MongoDB connectivity helpers and user management."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, Optional

import bcrypt
import gridfs
from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError

from vgs_chatbot import logger
from vgs_chatbot.config import get_settings

try:  # Optional: avoid hard dependency when running outside Streamlit.
    import streamlit as st  # type: ignore
except Exception:  # noqa: BLE001
    st = None

_CLIENT_TTL_SECONDS = 10 * 60  # Refresh the pooled client periodically to avoid leaks.


def _create_client(uri: str) -> MongoClient:
    """Instantiate and validate a MongoDB client."""
    logger.debug("Creating MongoDB client from configured URI.")
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
    except PyMongoError:
        logger.exception("MongoDB connection failed.")
        client.close()
        raise
    logger.info("MongoDB connection established.")
    return client


if st:

    @st.cache_resource(ttl=_CLIENT_TTL_SECONDS)
    def _get_cached_client_streamlit(uri: str) -> MongoClient:
        """Return a cached MongoDB client per Streamlit guidance."""
        logger.debug("Fetching MongoDB client (Streamlit TTL cached).")
        return _create_client(uri)

    _get_cached_client = _get_cached_client_streamlit

else:

    @lru_cache(maxsize=1)
    def _get_cached_client_fallback(uri: str) -> MongoClient:
        """Fallback cache when Streamlit is unavailable."""
        logger.warning("Using lru_cache fallback for MongoDB client caching.")
        return _create_client(uri)

    _get_cached_client = _get_cached_client_fallback


def connect_default() -> MongoClient:
    """Connect to MongoDB Atlas using a cached singleton client.

    Returns:
        MongoClient: Authenticated client instance (cached with TTL).

    Raises:
        PyMongoError: If the connection attempt fails.
    """
    settings = get_settings()
    uri = settings.mongo_uri
    logger.debug("Fetching MongoDB client.")
    return _get_cached_client(uri)


def get_database(client: MongoClient) -> Database:
    """Return the configured database handle.

    Args:
        client: Connected MongoDB client.

    Returns:
        Database: Handle to the configured MongoDB database.
    """
    settings = get_settings()
    return client[settings.mongodb_db]


def get_collections(client: MongoClient) -> Dict[str, Collection]:
    """Return collection handles for the application.

    Args:
        client: Connected MongoDB client.

    Returns:
        dict[str, Collection]: Mapping of logical names to Mongo collections.
    """
    db = get_database(client)
    collections: Dict[str, Collection] = {
        "documents": db["documents"],
        "doc_chunks": db["doc_chunks"],
        "kg_nodes": db["kg_nodes"],
        "kg_edges": db["kg_edges"],
        "users": db["users"],
    }
    return collections


def get_gridfs(client: MongoClient) -> gridfs.GridFS:
    """Return GridFS handle for storing source documents.

    Args:
        client: Connected MongoDB client.

    Returns:
        gridfs.GridFS: GridFS helper bound to the configured database.
    """
    db = get_database(client)
    return gridfs.GridFS(db)


def get_users_collection(client: MongoClient) -> Collection:
    """Return the users collection handle."""
    db = get_database(client)
    return db["users"]


def hash_password(password: str) -> str:
    """Return a salted hash for a plaintext password."""
    password_bytes = password.encode("utf-8")
    hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Check whether a plaintext password matches a stored hash."""
    try:
        password_bytes = password.encode("utf-8")
        hash_bytes = password_hash.encode("utf-8")
        return bcrypt.checkpw(password_bytes, hash_bytes)
    except Exception:
        logger.exception("Password verification failed unexpectedly.")
        return False


def find_user_by_email(client: MongoClient, email: str) -> Optional[Dict[str, Any]]:
    """Look up a user document by email."""
    users = get_users_collection(client)
    return users.find_one({"email": email.strip().lower()})


def create_user(
    client: MongoClient,
    email: str,
    password: str,
    role: str = "user",
) -> Dict[str, Any]:
    """Create a new user document with hashed password."""
    users = get_users_collection(client)
    now = datetime.now(timezone.utc)
    user_doc: Dict[str, Any] = {
        "email": email.strip().lower(),
        "password_hash": hash_password(password),
        "created_at": now,
        "last_login": None,
        "is_active": True,
        "role": role,
    }
    result = users.insert_one(user_doc)
    user_doc["_id"] = result.inserted_id
    return user_doc


def update_last_login(client: MongoClient, user_id: ObjectId) -> None:
    """Update the last login timestamp for a user."""
    users = get_users_collection(client)
    users.update_one(
        {"_id": user_id},
        {"$set": {"last_login": datetime.now(timezone.utc)}},
    )
