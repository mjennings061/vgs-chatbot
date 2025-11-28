# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VGS Chatbot is a Streamlit application that enables RAF 2FTS instructors to explore Viking training material through a knowledge-graph assisted RAG (Retrieval-Augmented Generation) pipeline. The system uses MongoDB Atlas for storage, FastEmbed for embeddings, and optionally OpenAI's GPT-4o-mini for answer generation.

## Architecture

### Core Pipeline Flow

1. **Document Ingestion** ([vgs_chatbot/ingest.py](vgs_chatbot/ingest.py))
   - PDF/DOCX files uploaded via Admin page → stored in GridFS
   - Documents are parsed into pages, then sections are detected
   - Text is chunked (~900 characters) and embedded using FastEmbed (snowflake-arctic-embed-xs, 384 dimensions)
   - Keyphrases extracted via YAKE and stored in lightweight knowledge graph

2. **Knowledge Graph** ([vgs_chatbot/kg.py](vgs_chatbot/kg.py))
   - Two-collection graph: `kg_nodes` (concepts/keyphrases) and `kg_edges` (relationships)
   - Nodes track phrase labels and aliases; edges link chunks via "mentions" and "associated_with" relationships
   - Graph expansion uses 1-hop traversal to find candidate chunks related to query keyphrases

3. **Retrieval** ([vgs_chatbot/retrieve.py](vgs_chatbot/retrieve.py))
   - **GraphRAG-lite approach**: Extract keyphrases from query → expand to candidate chunk IDs via KG
   - **Parallel search**:
     - Atlas Vector Search (semantic, cosine similarity)
     - Atlas Search (BM25 full-text)
     - Optional keyword heuristic for day-based limits (e.g., "flights per day")
   - **Query rewriting**: Domain-specific synonym expansion (student→trainee, launch→flight, GS→Gliding Scholarship)
   - **Score fusion**: Weighted combination (vector: 0.7, text: 0.3, keyword: 0.7, KG bonus: 0.1)
   - **Lexical fallback**: Regex-based AND search when Atlas Search returns no results

4. **Answer Generation** ([vgs_chatbot/llm.py](vgs_chatbot/llm.py))
   - If `OPENAI_API_KEY` is set: Use GPT-4o-mini with retrieved chunks as context
   - Otherwise: Extractive fallback returns highest-scoring chunk text

### MongoDB Collections

- `documents`: Document metadata (title, filename, GridFS ID, uploader)
- `doc_chunks`: Text chunks with embeddings, section/page references
- `kg_nodes`: Knowledge graph nodes (keyphrases/concepts)
- `kg_edges`: Knowledge graph relationships (mentions, associations)
- `fs.files` / `fs.chunks`: GridFS buckets for original document storage

### Required Atlas Indexes

**Vector Search** (`doc_chunks`, name `vgs_vector`):

```json
{
  "type": "vectorSearch",
  "fields": [
    { "type": "vector", "path": "embedding", "numDimensions": 384, "similarity": "cosine" },
    { "type": "filter", "path": "_id" },
    { "type": "filter", "path": "doc_id" }
  ]
}
```

**Atlas Search** (`doc_chunks`, name `vgs_text`):

```json
{
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

## Development Commands

### Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Start Streamlit app (opens on http://localhost:8501)
streamlit run streamlit_app.py
```

### Configuration

Environment variables (set in `.env` or `.streamlit/secrets.toml`):

- `MONGO_URI`: Full Atlas connection string (e.g. `mongodb+srv://user:pass@host/?retryWrites=true&w=majority&appName=vgs-chatbot`) - **required**
- `OPENAI_API_KEY`: Optional; enables generative answers instead of extractive fallback
- `LOG_LEVEL`: Optional; defaults to `DEBUG` (see [vgs_chatbot/config.py:60](vgs_chatbot/config.py#L60))

### Code Quality Tools

Pre-commit hooks are configured in [.pre-commit-config.yaml](.pre-commit-config.yaml):

```bash
# Install pre-commit hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

Tools used:

- **Ruff**: Linting and formatting (replaces black, isort, flake8)
- **isort**: Import sorting (configured separately)

Type checking:

```bash
# Pyright is configured in pyrightconfig.json
# Run type checking with:
pyright
```

## Key Implementation Notes

### Session State Management

Streamlit session state ([streamlit_app.py](streamlit_app.py)) tracks:

- `logged_in`: Authentication status
- `user_email` / `user_role` / `user_id`: Current user identifiers
- `mongo_client`: Authenticated MongoDB client (closed on logout)
- `chat_history`: List of Q&A turns with citations
- `redirected_after_login`: Flag to auto-navigate to Chat page after login

### Embedder Singleton

[vgs_chatbot/embeddings.py](vgs_chatbot/embeddings.py) provides a cached `get_embedder()` that returns a singleton FastEmbed instance. Avoid creating multiple embedders as model loading is expensive.

### Progress Reporting

Ingestion pipeline accepts optional `progress_callback(stage, current, total)` to report stages like "Chunking document" (3/50). Admin page hooks this to update Streamlit progress bars.

### Domain-Specific Rewrites

The retrieval module ([retrieve.py:484-499](vgs_chatbot/retrieve.py#L484-L499)) applies targeted query rewrites to align with Viking training terminology. When adding new synonyms or domain terms, update `_expand_query_for_text()` and `_rewrite_query_for_vector()`.

### Text Processing Utilities

[vgs_chatbot/utils_text.py](vgs_chatbot/utils_text.py) contains:

- `read_pdf(bytes)`: Extracts pages using pdfplumber
- `read_docx(bytes)`: Extracts pages using python-docx
- `detect_sections(text)`: Regex-based section header detection
- `chunk_text(text, max_chars=900)`: Sentence-boundary-aware chunking

## Common Issues

### Empty Library Despite Documents in Database

If the Admin page shows no documents but `fs.files` exists in MongoDB:

- The app expects a matching `documents` collection entry with `gridfs_id` linking to `fs.files._id`
- Check that ingestion completed successfully and didn't fail after GridFS upload
- See [pages/2_Admin.py:87-141](pages/2_Admin.py#L87-L141) for the join logic

### Atlas Search Returns No Results

Short queries (e.g., "wind limits") may miss in Atlas Search. The system has a lexical fallback ([retrieve.py:549-601](vgs_chatbot/retrieve.py#L549-L601)) using regex AND search. If results are still poor, consider:

- Verifying the `vgs_text` index is built and active
- Checking if query terms need domain rewrites (add to `_expand_query_for_text`)

### Knowledge Graph Not Improving Retrieval

KG provides a small score bonus (0.1) and candidate expansion (max 300 chunks). If it's not helping:

- Verify `kg_nodes` and `kg_edges` are populated during ingestion
- Check keyphrase extraction in [kg.py:18-46](vgs_chatbot/kg.py#L18-L46) - YAKE parameters may need tuning for your domain
- Graph expansion is logged at DEBUG level; inspect logs to verify candidate expansion

## Future Work

See [TODO](TODO) for tracked tasks.
