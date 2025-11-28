"""Admin interface for uploading and managing documents."""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st
from bson import ObjectId
from gridfs import GridFS
from pymongo.collection import Collection

from vgs_chatbot import logger
from vgs_chatbot.db import get_collections, get_gridfs
from vgs_chatbot.embeddings import get_embedder
from vgs_chatbot.ingest import ingest_file
from vgs_chatbot.utils_text import clean_title

CollectionsMap = Dict[str, Collection]


def _require_login() -> None:
    """Ensure the admin view is only accessible to signed-in users.

    Returns:
        None: Uses Streamlit control flow to stop unauthenticated sessions.
    """
    if not st.session_state.get("logged_in"):
        logger.warning("Unauthenticated access attempt to admin page.")
        st.error("Please sign in on the Home page before using the admin tools.")
        st.stop()
    if st.session_state.get("user_role") != "admin":
        logger.warning(
            "Access denied for user '%s' attempting admin page.",
            st.session_state.get("user_email"),
        )
        st.error("You need admin rights to view this page.")
        st.stop()
    if "mongo_client" not in st.session_state:
        logger.error("Streamlit session missing MongoDB client for admin page.")
        st.error("Connection not found. Please sign in again.")
        st.stop()


def _sign_out() -> None:
    """Clear session data and return to the login screen."""
    st.session_state.clear()
    try:
        st.switch_page("pages/0_Login.py")
    except Exception:
        st.rerun()


def _delete_document(
    collections: CollectionsMap,
    fs: GridFS,
    doc_id: ObjectId | None,
    gridfs_id: ObjectId,
) -> None:
    """Remove a document, related chunks, and the stored GridFS binary.

    Args:
        collections: MongoDB collections used by the application.
        fs: GridFS bucket storing source documents.
        doc_id: Identifier of the document metadata entry.
        gridfs_id: GridFS identifier for the binary data.

    Returns:
        None: Side effects occur through MongoDB operations.
    """
    doc = None
    if doc_id:
        doc = collections["documents"].find_one({"_id": doc_id})
    if not doc:
        doc = collections["documents"].find_one({"gridfs_id": gridfs_id})
        if doc:
            doc_id = doc["_id"]

    if doc:
        chunk_ids = [
            chunk["_id"]
            for chunk in collections["doc_chunks"].find({"doc_id": doc_id}, {"_id": 1})
        ]
        if chunk_ids:
            collections["doc_chunks"].delete_many({"_id": {"$in": chunk_ids}})
            collections["kg_edges"].update_many(
                {"chunk_ids": {"$in": chunk_ids}},
                {"$pull": {"chunk_ids": {"$in": chunk_ids}}},
            )
            collections["kg_edges"].delete_many({"chunk_ids": {"$size": 0}})
        collections["documents"].delete_one({"_id": doc_id})

    try:
        logger.info("Deleting GridFS file '%s'.", gridfs_id)
        fs.delete(gridfs_id)
    except Exception:  # noqa: BLE001 - GridFS delete best effort
        st.warning("Failed to remove GridFS file; please check the database.")
        logger.exception("Failed to delete GridFS file '%s'.", gridfs_id)


def _list_documents(collections: CollectionsMap) -> List[Dict[str, Any]]:
    """Build a list of documents enriched with metadata and chunk counts.

    Args:
        collections: MongoDB collection map used for lookups.

    Returns:
        list[dict[str, Any]]: Documents with presentation-ready fields.
    """
    files_collection: Collection = collections["documents"].database["fs.files"]
    files: List[Dict[str, Any]] = list(files_collection.find().sort("uploadDate", -1))
    if not files:
        return []

    grid_ids = [file["_id"] for file in files]
    meta_cursor = collections["documents"].find({"gridfs_id": {"$in": grid_ids}})
    meta_by_grid = {doc["gridfs_id"]: doc for doc in meta_cursor}
    doc_ids = [doc["_id"] for doc in meta_by_grid.values()]

    chunk_counts: Dict[ObjectId, int] = {}
    if doc_ids:
        chunk_counts_cursor = collections["doc_chunks"].aggregate(
            [
                {"$match": {"doc_id": {"$in": doc_ids}}},
                {"$group": {"_id": "$doc_id", "count": {"$sum": 1}}},
            ]
        )
        chunk_counts = {item["_id"]: item["count"] for item in chunk_counts_cursor}

    documents: List[Dict[str, Any]] = []
    for file_doc in files:
        meta = meta_by_grid.get(file_doc["_id"])
        doc_id = meta["_id"] if meta else None
        chunk_count = chunk_counts.get(doc_id, 0) if doc_id else 0
        documents.append(
            {
                "doc_id": doc_id,
                "gridfs_id": file_doc["_id"],
                "title": (meta.get("title") if meta else None)
                or file_doc.get("filename")
                or "Untitled",
                "filename": file_doc.get("filename") or "Unknown file",
                "doc_type": (meta.get("doc_type") if meta else None)
                or file_doc.get("contentType", "unknown"),
                "uploaded_by": (
                    meta.get("uploaded_by", "unknown") if meta else "unknown"
                ),
                "uploaded_at": (
                    meta.get("uploaded_at") if meta else file_doc.get("uploadDate")
                ),
                "chunk_count": chunk_count,
            }
        )

    return documents


def main() -> None:
    """Render admin tools for document management.

    Returns:
        None: Streamlit renders and controls navigation.
    """
    _require_login()
    st.title("Document Admin")
    st.caption("Upload or remove Viking training documents for retrieval.")

    if st.button("Sign out"):
        _sign_out()
        st.stop()

    client = st.session_state["mongo_client"]
    collections: CollectionsMap = get_collections(client)
    fs: GridFS = get_gridfs(client)
    embedder = get_embedder()

    st.subheader("Upload")
    uploaded_files = st.file_uploader(
        "Select one or more PDF or Word documents",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )
    uploader = st.session_state.get("user_email") or "admin"

    if st.button("Ingest documents", disabled=not uploaded_files):
        if not uploaded_files:
            st.warning("Please choose at least one file to upload.")
        else:
            with st.spinner("Processing documents…"):
                status_placeholder = st.empty()
                progress_bar = st.progress(0)
                all_success = True

                for uploaded in uploaded_files:
                    progress_bar.progress(0)
                    last_stage: Dict[str, str | None] = {"name": None}

                    def report(
                        stage: str, current: int | None = None, total: int | None = None
                    ) -> None:
                        label = f"{uploaded.name}: {stage}"
                        if stage != last_stage["name"]:
                            logger.debug("Ingestion stage -> %s (%s)", stage, uploaded.name)
                            if total and total > 0:
                                progress_bar.progress(0)
                            last_stage["name"] = stage
                        if total and current is not None and total > 0:
                            percent = min(max(int(current * 100 / total), 0), 100)
                            progress_bar.progress(percent)
                            status_placeholder.markdown(
                                f"**{label}**: {current} of {total}"
                            )
                        else:
                            status_placeholder.markdown(f"**{label}**")

                    try:
                        result = ingest_file(
                            fs=fs,
                            documents=collections["documents"],
                            doc_chunks=collections["doc_chunks"],
                            kg_nodes=collections["kg_nodes"],
                            kg_edges=collections["kg_edges"],
                            embedder=embedder,
                            file_bytes=uploaded.getvalue(),
                            filename=uploaded.name,
                            content_type=uploaded.type or "",
                            uploaded_by=uploader,
                            progress_callback=report,
                        )
                    except Exception as exc:  # noqa: BLE001 - surface ingestion error
                        logger.exception("Ingestion failed for '%s'.", uploaded.name)
                        st.error(
                            f"Ingestion failed for {uploaded.name}. Please review the logs and try again."
                        )
                        st.caption(f"Reason: {exc}")
                        all_success = False
                        break
                    else:
                        logger.info(
                            "Ingestion completed for '%s': chunks=%s, pages=%s",
                            uploaded.name,
                            result.chunk_count,
                            result.page_count,
                        )
                        st.success(
                            f"{uploaded.name}: {result.chunk_count} chunks across {result.page_count} pages."
                        )

                if all_success:
                    st.rerun()

    st.subheader("Library")
    documents = _list_documents(collections)
    if not documents:
        st.info("No documents ingested yet.")
        return

    for index, doc in enumerate(documents):
        with st.container():
            raw_title = (
                doc.get("title")
                or doc.get("doc_title")
                or doc.get("filename")
                or "Untitled"
            )
            title = clean_title(raw_title)
            st.markdown(f"### {title}")
            st.write(
                f"File: `{doc['filename']}` · Type: {doc.get('doc_type', 'unknown')}"
            )
            st.write(f"Chunks: {doc.get('chunk_count', 0)}")
            col1, col2 = st.columns([1, 2])
            with col1:
                delete_key = doc.get("doc_id") or doc["gridfs_id"]
                if st.button("Delete", key=f"delete-{delete_key}"):
                    logger.info("Deleting document '%s'.", delete_key)
                    _delete_document(
                        collections, fs, doc.get("doc_id"), doc["gridfs_id"]
                    )
                    st.rerun()
            with col2:
                st.caption(
                    f"Uploaded by {doc.get('uploaded_by', 'unknown')} on {doc.get('uploaded_at')}"
                )
        if index < len(documents) - 1:
            st.divider()


if __name__ == "__main__":
    main()
