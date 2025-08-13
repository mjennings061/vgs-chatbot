# VGS Chatbot

A modern, locally-hosted chatbot that connects to SharePoint to access 2FTS documentation using SOLID design principles and a RAG (Retrieval-Augmented Generation) architecture.

## Features

- **RAG Pipeline**: Advanced document processing and semantic search
- **User Authentication**: PostgreSQL-backed user management
- **Modern Architecture**: SOLID principles with dependency injection
- **Web Interface**: Streamlit-based chat interface
- **Docker Support**: Containerized deployment

## Architecture

The application follows SOLID design principles with clear separation of concerns:

```text
vgs_chatbot/
├── interfaces/         # Abstract interfaces for dependency inversion
├── services/          # Business logic implementations
├── models/           # Data models and schemas
├── repositories/     # Data access layer
├── gui/             # Streamlit web interface
└── utils/           # Utility functions
```

### Key Components

1. **Authentication Service**: PostgreSQL-backed user login system
2. **Document Processor**: RAG pipeline for document indexing and search
3. **Chat Service**: LLM-powered question answering
4. **Web Interface**: Streamlit-based chatbot application

## Setup

### Prerequisites

- Python 3.11+
- Poetry
- PostgreSQL (or use Docker Compose)
- OpenAI API key

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd vgs-chatbot
   ```

2. **Install dependencies**

   ```bash
   poetry install
   ```

3. **Configure environment**

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Initialize database**

   ```bash
   # Using Docker Compose (recommended)
   docker-compose up -d db

   # Or set up PostgreSQL manually and update DATABASE_URL in .env
   ```

5. **Run the application**

   ```bash
   poetry run streamlit run vgs_chatbot/gui/app.py
   ```

### Docker Deployment

For production deployment:

```bash
# Using Docker Compose
docker-compose up -d

# Or build and run manually
docker build -t vgs-chatbot .
docker run -p 8501:8501 vgs-chatbot
```

## Configuration

Key environment variables in `.env`:

- `DATABASE_URL`: PostgreSQL connection string
- `JWT_SECRET`: Secret key for JWT tokens
- `OPENAI_API_KEY`: OpenAI API key
- `SHAREPOINT_SITE_URL`: Your SharePoint site URL
- `SHAREPOINT_DIRECTORY_URLS`: Comma-separated list of document directories

## Usage

1. **Register/Login**: Create an account or log in
2. **Connect to SharePoint**: Provide SharePoint credentials
3. **Chat**: Ask questions about your documentation
4. **View Sources**: See which documents were used to generate responses

## Reprocessing & Reindexing Documents (CLI)

After changing the embedding model, chunking logic, or wanting a clean rebuild of the vector store, you can regenerate all embeddings via the built‑in CLI in `vgs_chatbot/services/document_processor.py`.

This script:

- Deletes the existing Chroma persistent index directory you point it at (if it exists)
- Scans a documents directory for supported files (PDF, DOCX, XLSX, TXT, MD, etc.)
- Extracts text, chunks content, generates embeddings with the selected model
- Rebuilds the Chroma collection from scratch

### When to run it

- After changing the default embedding model (e.g. switching to `multi-qa-MiniLM-L6-cos-v1`)
- After modifying chunking / extraction logic
- If the index becomes corrupted or stale
- Before benchmarking retrieval performance

### Basic usage

```bash
poetry run python vgs_chatbot/services/document_processor.py \
   --documents-dir data/documents \
   --persist-dir data/vectors/chroma \
   --embedding-model multi-qa-MiniLM-L6-cos-v1
```

### Arguments

- `--documents-dir`  Path containing source documents (default: `data/documents`)
- `--persist-dir`    Path for Chroma persistent store (default: `data/vectors/chroma`)
- `--embedding-model` SentenceTransformer model name (default: `multi-qa-MiniLM-L6-cos-v1`)

### Notes & Safety

- The script removes the existing directory at `--persist-dir` before rebuilding. Backup if needed.
- Non-supported file types are skipped gracefully.
- Large Excel sheets are truncated after 100 data rows per sheet to control index size.
- Progress / warnings are printed to stdout; consider redirecting to a log file for automation.

### Example (fresh rebuild after model change)

```bash
poetry run python vgs_chatbot/services/document_processor.py \
   --embedding-model all-mpnet-base-v2 \
   --documents-dir /path/to/new_docs \
   --persist-dir data/vectors/chroma
```

### Automating (Makefile snippet)

Add to a `Makefile` if desired:

```makefile
reindex:
   poetry run python vgs_chatbot/services/document_processor.py \
      --documents-dir data/documents \
      --persist-dir data/vectors/chroma \
      --embedding-model multi-qa-MiniLM-L6-cos-v1
```

Then run:

```bash
make reindex
```

If you use Docker, you can execute inside the container (ensure volumes are mounted):

```bash
docker compose exec app poetry run python vgs_chatbot/services/document_processor.py \
   --documents-dir /app/data/documents \
   --persist-dir /app/data/vectors/chroma
```

## Development

### Code Quality

The project uses strict code quality standards:

```bash
# Run all quality checks
poetry run isort src/ tests/
poetry run ruff check src/ tests/
poetry run mypy src/

# Fix linting issues
poetry run ruff check --fix src/ tests/
```

### Testing

```bash
# Run tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src
```

### Standards

- **Line length**: 88 characters
- **Type hints**: Required for all functions
- **Docstrings**: Google style for public functions
- **Naming**: snake_case for variables/functions, PascalCase for classes

## SharePoint Integration

The connector supports:

- User-based authentication (no admin rights required)
- Multiple SharePoint sites and document libraries
- Recursive directory traversal
- Support for PDF, DOCX, XLSX files
- Proper error handling and logging

## License

MIT License - see LICENSE file for details

## Contributing

1. Follow the existing code style and architecture
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure all quality checks pass
