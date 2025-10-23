"""MongoDB connectivity helpers."""

from __future__ import annotations

from typing import Dict

import gridfs
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError

from vgs_chatbot.config import get_settings


def connect_with_user(username: str, password: str) -> MongoClient:
    """Connect to MongoDB Atlas using supplied demo credentials."""
    settings = get_settings()
    uri = settings.build_srv_uri(username, password)
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
    except PyMongoError:
        client.close()
        raise
    return client


def get_database(client: MongoClient) -> Database:
    """Return the configured database handle."""
    settings = get_settings()
    return client[settings.mongodb_db]


def get_collections(client: MongoClient) -> Dict[str, Collection]:
    """Return collection handles for the application."""
    db = get_database(client)
    collections: Dict[str, Collection] = {
        "documents": db["documents"],
        "doc_chunks": db["doc_chunks"],
        "kg_nodes": db["kg_nodes"],
        "kg_edges": db["kg_edges"],
    }
    return collections


def get_gridfs(client: MongoClient) -> gridfs.GridFS:
    """Return GridFS handle for storing source documents."""
    db = get_database(client)
    return gridfs.GridFS(db)
