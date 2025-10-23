# VGS Chatbot

A Streamlit application that lets RAF 2FTS instructors explore Viking training material through a knowledge-graph assisted RAG pipeline. Admins can upload documents, the chatbot cites its answers, and MongoDB Atlas stores the full corpus.

## Features
- Streamlit login screen backed by MongoDB Atlas demo credentials stored in `.env`.
- Document ingestion for PDF and DOCX files with progress reporting; sources live in GridFS.
- Automatic section detection, ~900 character chunking, and FastEmbed embeddings.
- Lightweight knowledge graph that links keyphrases to chunk candidates.
- Retrieval that fuses Atlas Vector Search, Atlas Search (BM25), and graph priors before answering.
- Optional OpenAI `gpt-4o-mini` generation with an extractive fallback when the API key is absent.

## Prerequisites
- Python 3.13 (tested with CPython 3.13.9)
- [`uv`](https://docs.astral.sh/uv/) for dependency and virtualenv management
- MongoDB Atlas cluster with Search and Vector Search enabled

Install `uv` if you do not already have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup
```bash
uv venv
uv sync
cp .env.example .env
```

Edit `.env` with the Atlas host, demo credentials, and optional OpenAI API key. The application reads:

- `MONGODB_HOST` – Atlas SRV host (without the `mongodb+srv://` prefix)
- `MONGODB_DB` – database name for documents (`vgs` by default)
- `MONGODB_VECTOR_INDEX` – Atlas Vector Search index on `doc_chunks.embedding`
- `MONGODB_SEARCH_INDEX` – Atlas Search (text) index on `doc_chunks`
- `APP_LOGIN_USER` / `APP_LOGIN_PASS` – demo credentials shown on the login form
- `EMBEDDING_MODEL_NAME` – FastEmbed model name (`snowflake/snowflake-arctic-embed-xs`)
- `OPENAI_API_KEY` – optional; enables generative answers instead of the extractive fallback

## Prepare MongoDB Atlas
Create a database (for example `vgs`) and let the app create the collections on first use:
- `documents` – metadata for each uploaded file
- `doc_chunks` – embedded text chunks with section and page references
- `kg_nodes` / `kg_edges` – knowledge graph nodes and associations
- GridFS buckets `fs.files` / `fs.chunks` – original documents

Configure the indexes before querying:

Vector Search (`doc_chunks`, name `vgs_vector`)

```json
{
  "name": "vgs_vector",
  "type": "vectorSearch",
  "definition": {
    "fields": [
      { "type": "vector", "path": "embedding", "numDimensions": 384, "similarity": "cosine" },
      { "type": "filter", "path": "_id" },
      { "type": "filter", "path": "doc_id" },
      { "type": "filter", "path": "section_id" },
      { "type": "filter", "path": "page_start" }
    ]
  }
}
```

Atlas Search (`doc_chunks`, name `vgs_text`)

```json
{
  "name": "vgs_text",
  "mappings": {
    "dynamic": false,
    "fields": {
      "text": { "type": "string" },
      "section_title": { "type": "string" },
      "doc_title": { "type": "string" }
    }
  }
}
```

## Running the app
```bash
uv run streamlit run streamlit_app.py
```

1. Open the displayed local URL, sign in with the credentials from `.env`, and the home page will confirm the Atlas host in use.
2. Use the sidebar to open **Chat** or **Admin**.

### Admin workflow

- Upload a PDF or DOCX file and trigger **Ingest document** to push it to GridFS, create chunks, embed them, and update the knowledge graph.
- The **Library** section lists stored documents with chunk counts and lets you delete an item (removing GridFS blobs, metadata, and related chunks).
- Progress bars report ingestion stages, and errors surface friendly messages.

### Chat workflow

- Ask a question such as “What are the canopy checks before launch?”.
- Retrieval expands the query to graph-linked chunks, runs Vector Search and text search, fuses the scores, and shows cited answers (`Document · Section · Page`).
- When `OPENAI_API_KEY` is unset, responses fall back to the highest scoring chunk extract.

## Development tasks

- Format and lint: `uv run ruff format .` then `uv run ruff check .`
- Imports: `uv run isort .`
- Run the local hooks: `uv run pre-commit run --all-files`

The repo ships stub typings for third-party libraries under `typings/` to keep Pyright quiet when desired.

## Security notes

- Demo credentials are intentionally low-privilege. Restrict them to the chatbot database and rotate them frequently.
- Do not reuse production secrets in `.env`. Use Atlas network rules to limit inbound connections.
