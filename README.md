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

- Python 3.13
- MongoDB Atlas cluster with Search and Vector Search enabled

## Setup

Create and activate a virtual environment, install dependencies, and prepare env vars.

macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.streamlit/secrets.toml` with the Atlas host and optional OpenAI API key. The application reads:

- `MONGODB_HOST` – Atlas SRV host (without the `mongodb+srv://` prefix)
- `OPENAI_API_KEY` – optional; enables generative answers instead of the extractive fallback
- `LOG_LEVEL` – optional; adjusts app logging (`INFO` by default)

Local development convenience (not required in production):

- `APP_LOGIN_USER` / `APP_LOGIN_PASS` – shown on the login form if present

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
streamlit run streamlit_app.py
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

## Development notes

- Optional: use tools like Ruff, isort, and pre-commit locally if you wish.

### Enable OpenAI (optional)

By default, the app runs without an LLM and falls back to extractive answers. To enable OpenAI responses:

- Install the SDK: `pip install openai` (or add it to `requirements.txt`).
- Set `OPENAI_API_KEY` in `.env`.

## Security notes

- Demo credentials are intentionally low-privilege. Restrict them to the chatbot database and rotate them frequently.
- Do not reuse production secrets in `.env`. Use Atlas network rules to limit inbound connections.
