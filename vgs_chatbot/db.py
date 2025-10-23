"""MongoDB connectivity helpers."""

from __future__ import annotations

import logging
from typing import Dict

import gridfs
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError

from vgs_chatbot.config import get_settings

logger = logging.getLogger(__name__)


def connect_with_user(username: str, password: str) -> MongoClient:
    """Connect to MongoDB Atlas using supplied demo credentials.

    Args:
        username: Database username used for authentication.
        password: Database password corresponding to the username.

    Returns:
        MongoClient: Authenticated client instance.

    Raises:
        PyMongoError: If the connection attempt fails.
    """
    settings = get_settings()
    uri = settings.build_srv_uri(username, password)
    logger.debug("Creating MongoDB client for host '%s'.", settings.mongodb_host)
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
    except PyMongoError:
        logger.exception("MongoDB connection failed for user '%s'.", username)
        client.close()
        raise
    logger.info("MongoDB connection established for user '%s'.", username)
    return client


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
