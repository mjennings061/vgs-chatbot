# VGS Chatbot - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Option 1: Docker (Recommended)

```bash
# 1. Configure environment
python configure.py

# 2. Start with Docker
docker-compose up -d

# 3. Visit the app
open http://localhost:8501
```

### Option 2: Manual Setup

```bash
# 1. Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python3 -

# 2. Install dependencies
poetry install

# 3. Configure environment
python configure.py

# 4. Start database
docker-compose up -d db

# 5. Validate setup
python test_setup.py

# 6. Run the application
poetry run streamlit run vgs_chatbot/gui/app.py
```

## 📋 Configuration Checklist

### Required Information

1. **OpenAI API Key**
   - Get from: https://platform.openai.com/api-keys
   - Format: `sk-...`

2. **SharePoint Details**
   - Site URL: `https://yourcompany.sharepoint.com/sites/yoursite`
   - Document URLs: Where your 2FTS documents are stored

3. **Database** (automatic with Docker)
   - PostgreSQL will be configured automatically

### Environment File (.env)

The `configure.py` script will create this for you:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/vgs_chatbot

# Security
JWT_SECRET=automatically-generated-secure-key

# OpenAI
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-3.5-turbo

# SharePoint
SHAREPOINT_SITE_URL=https://yourcompany.sharepoint.com/sites/yoursite
SHAREPOINT_DIRECTORY_URLS=https://yourcompany.sharepoint.com/sites/yoursite/Shared Documents/2FTS

# App
APP_TITLE=VGS Chatbot
DEBUG=false
```

## 🧪 Testing & Validation

### Validate Your Setup

```bash
# Run the setup validator
python test_setup.py
```

This checks:
- ✅ File structure
- ✅ Dependencies installed  
- ✅ Environment variables
- ✅ OpenAI API connection
- ✅ Database connection
- ✅ Authentication system

### Run Unit Tests

```bash
# Install dependencies first
poetry install

# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src --cov-report=html

# Run specific tests
poetry run pytest tests/test_auth_service.py -v
```

### Code Quality Checks

```bash
# Run all quality checks (per CLAUDE.md)
poetry run isort src/ tests/ && poetry run ruff check src/ tests/ && poetry run mypy src/

# Fix linting issues
poetry run ruff check --fix src/ tests/
```

## 🎯 Usage Workflow

### 1. Access the Application
- Navigate to http://localhost:8501
- Register a new account or login

### 2. Connect to SharePoint
- In the sidebar, enter your SharePoint credentials
- Click "Connect to SharePoint"
- Wait for "✅ Connected to SharePoint" confirmation

### 3. Start Chatting
- Ask questions about your 2FTS documentation
- Examples:
  - "What are the pre-flight safety checks?"
  - "How do I perform engine maintenance?"
  - "What are the emergency procedures?"

### 4. View Sources
- Each response shows which documents were used
- Click on document names to see sources
- Responses include confidence scores

## 🛠️ Development Commands

### Project Structure
```
vgs_chatbot/
├── interfaces/         # Abstract interfaces (SOLID principles)
├── services/          # Business logic implementations
├── models/           # Data models (User, Document, Chat)
├── repositories/     # Database access layer
├── gui/             # Streamlit web interface
└── utils/           # Configuration utilities
```

### Useful Commands

```bash
# Development server with auto-reload
poetry run streamlit run vgs_chatbot/gui/app.py --server.runOnSave true

# Database management
docker-compose exec db psql -U postgres -d vgs_chatbot

# View logs
docker-compose logs -f app
docker-compose logs -f db

# Reset everything
docker-compose down -v
rm -f .env
```

## 🔧 Troubleshooting

### Common Issues

#### "Module not found" errors
```bash
# Ensure Poetry environment is activated
poetry shell
# Or run commands with poetry run
poetry run python script.py
```

#### Database connection failed
```bash
# Check if PostgreSQL is running
docker-compose ps

# Restart database
docker-compose restart db

# Check logs
docker-compose logs db
```

#### OpenAI API errors
```bash
# Verify API key
curl -H "Authorization: Bearer sk-your-key" https://api.openai.com/v1/models
```

#### SharePoint authentication fails
- Use your regular SharePoint login credentials
- Check if your organization requires app passwords
- Verify SharePoint URLs are accessible

### Get Help

1. **Check logs**: `docker-compose logs -f app`
2. **Validate setup**: `python test_setup.py`
3. **Run tests**: `poetry run pytest -v`
4. **Check configuration**: Review `.env` file

## 🚀 Production Deployment

### Security Checklist

```bash
# Generate secure JWT secret
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Use strong database passwords
# Enable HTTPS
# Set DEBUG=false
# Use environment-specific .env files
```

### Deploy with Docker

```bash
# Production compose file
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Or build custom image
docker build -t vgs-chatbot:latest .
docker run -d -p 8501:8501 --env-file .env vgs-chatbot:latest
```

---

**Next Steps:**
1. Run `python configure.py` to get started
2. See `SETUP_GUIDE.md` for detailed instructions  
3. Run `python demo.py` for architecture overview