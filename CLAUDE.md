# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VGS-Chatbot is a modern, locally-hosted chatbot that accesses 2FTS documentation uploaded by administrators. The architecture uses SOLID design principles and modern Python frameworks with GPT-4o-mini for intelligent responses.

## Current Implementation Status

✅ **Completed Features:**

- Document upload and management system for administrators
- Modern Python architecture following SOLID principles
- GUI dashboard application using Streamlit
- GPT-4o-mini integration for intelligent responses
- Local file-based document storage with RAG pipeline
- Source references with section titles and page numbers
- MOD email domain authentication
- Admin/user role separation

🔄 **In Progress:**

- Document search and retrieval optimization
- Vector database improvements for better context matching

❌ **Not Yet Implemented:**

- Local PostgreSQL database for user authentication (currently simplified)
- Docker containerisation capability

## Architecture Requirements

**SOLID Design Principles:**

- **S**ingle Responsibility: Each class has one reason to change
- **O**pen/Closed: Open for extension, closed for modification
- **L**iskov Substitution: Subtypes must be substitutable for base types
- **I**nterface Segregation: Clients shouldn't depend on interfaces they don't use
- **D**ependency Inversion: Depend on abstractions, not concretions

**Key Components:**

1. **Authentication Service**: MOD email domain validation with admin credentials
2. **Document Management**: Admin-uploaded document storage in local filesystem
3. **Document Processor**: RAG pipeline with ChromaDB for document indexing
4. **Chat Service**: GPT-4o-mini powered question answering with source references
5. **Web Interface**: Streamlit-based chatbot application

**Current Technical Stack:**

- **Frontend**: Streamlit web application
- **Backend**: Python with async support
- **LLM**: GPT-4o-mini via LangChain
- **Vector Database**: ChromaDB for document embeddings
- **Document Processing**: PyPDF, python-docx for text extraction
- **Authentication**: Simple credential-based (no database yet)

## Development Commands

**Environment Setup:**

```bash
# Install Poetry (if not available)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
eval "$(poetry env activate)"

```

**Code Quality Tools:**

```bash

# Sort imports with isort
poetry run isort src/ tests/

# Lint with Ruff (modern, fast linter)
poetry run ruff check src/ tests/

# Fix linting issues
poetry run ruff check --fix src/ tests/

# Type checking with mypy
poetry run mypy src/

# Run all quality checks (update paths for current structure)
poetry run isort vgs_chatbot/ tests/ && poetry run ruff check vgs_chatbot/ tests/ && poetry run mypy vgs_chatbot/
```

**Testing:**

```bash
# Run tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=src

# Run specific test file
poetry run pytest tests/test_specific.py
```

## Required Dependencies

**Development Tools:**

- `isort` - Import sorter
- `ruff` - Fast linter and formatter
- `mypy` - Type checker
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting

## Code Quality Standards

**PEP 8 Compliance:**

- Line length: 88 characters
- Use type hints for all functions and methods
- Docstrings for all public functions using Google style
- Snake_case for variables and functions
- PascalCase for classes

**File Structure:**

- `vgs_chatbot/` - Main application code
  - `gui/` - User interface
  - `services/` - External service integrations (document storage, database)
  - `models/` - Data models and schemas
  - `utils/` - Utility functions
- `tests/` - Unit and integration tests

## Docker Configuration

**Basic Dockerfile structure (when needed):**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-dev
COPY src/ ./src/
CMD ["poetry", "run", "streamlit", "run", "vgs_chatbot/gui/app.py"]
```

## Document Management Notes

- Administrators can upload documents through the admin interface
- Support multiple document formats (PDF, Word, text files)
- Documents are stored locally and indexed for search
- Implement proper error handling for file upload failures
- All uploaded documents and their content are searchable

## Testing Strategy

- Minimal unit testing, for function only
- Integration tests for document upload and storage
- End-to-end tests for chat functionality
- Mock external dependencies in tests

## Running the Application

**Start the chatbot:**
```bash
poetry run streamlit run vgs_chatbot/gui/app.py
```

**Access the application:**
- Admin login: Use credentials from `.env` file (`admin_username`/`admin_password`)
- User login: Any email ending with `@mod.gov.uk` or `@mod.uk`

**Admin Functions:**
- Upload PDF, DOCX, XLSX, PPTX, TXT files
- View document management dashboard
- Monitor system status

**User Functions:**
- Chat with uploaded documents
- Receive AI-generated answers with source references
- View available documents in sidebar

## Known Issues

See TODO.md for current issues and planned improvements.
