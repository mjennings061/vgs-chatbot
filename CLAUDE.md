# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VGS-Chatbot is being completely refactored to create a modern, locally-hosted chatbot that connects to SharePoint to access 2FTS documentation. The new architecture will use SOLID design principles and modern Python frameworks.

## Refactoring Goals

**Complete rewrite** of the existing codebase to implement:

- SharePoint integration for document access via user login (no admin/domain requirements)
- Modern Python architecture following SOLID principles
- Local PostgreSQL database for user authentication
- GUI dashboard application using a simple UI framework
- Docker containerisation capability
- Professional code quality standards

## Architecture Requirements

**SOLID Design Principles:**

- **S**ingle Responsibility: Each class has one reason to change
- **O**pen/Closed: Open for extension, closed for modification
- **L**iskov Substitution: Subtypes must be substitutable for base types
- **I**nterface Segregation: Clients shouldn't depend on interfaces they don't use
- **D**ependency Inversion: Depend on abstractions, not concretions

**Key Components:**

1. **Authentication Service**: PostgreSQL-backed user login system
2. **SharePoint Connector**: Document access via configurable URL list
3. **Document Processor**: RAG pipeline for document indexing and search
4. **Chat Service**: LLM-powered question answering
5. **Web Interface**: Chatbot application using simple UI framework

## Development Commands

**Environment Setup:**

```bash
# Install Poetry (if not available)
curl -sSL https://install.python-poetry.org | python3 -

# Initialize new Poetry project
poetry init

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
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

# Run all quality checks
poetry run isort src/ tests/ && poetry run ruff check src/ tests/ && poetry run mypy src/
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
  - `services/` - External service integrations (SharePoint, database)
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
CMD ["poetry", "run", "streamlit", "run", "src/vgs_chatbot/dashboard/app.py"]
```

## SharePoint Integration Notes

- No admin/domain requirements, user login only for authentication
- Handle document directory URLs as configurable list
- Support multiple SharePoint sites/document libraries
- Implement proper error handling for authentication failures
- Use all documents and subdirectories in the specified SharePoint URLs

## Testing Strategy

- Minimal unit testing, for function only
- Integration tests for SharePoint connectivity
- End-to-end tests for chat functionality
- Mock external dependencies in tests
