# VGS Chatbot

A modern, locally-hosted chatbot that connects to SharePoint to access 2FTS documentation using SOLID design principles and a RAG (Retrieval-Augmented Generation) architecture.

## Features

- **SharePoint Integration**: Connect to SharePoint sites using user credentials
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
2. **SharePoint Connector**: Document access via configurable URL list
3. **Document Processor**: RAG pipeline for document indexing and search
4. **Chat Service**: LLM-powered question answering
5. **Web Interface**: Streamlit-based chatbot application

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
