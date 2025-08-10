# VGS Chatbot Setup & Testing Guide

## Quick Start (Docker - Recommended)

### 1. Environment Configuration

```bash
# Copy the example environment file
cp .env.example .env
```

Edit `.env` with your settings:
```bash
# Required - Get from OpenAI
OPENAI_API_KEY=sk-your-openai-api-key-here

# Required - Your SharePoint site
SHAREPOINT_SITE_URL=https://yourdomain.sharepoint.com/sites/yoursite
SHAREPOINT_DIRECTORY_URLS=https://yourdomain.sharepoint.com/sites/yoursite/Shared Documents/2FTS,https://yourdomain.sharepoint.com/sites/yoursite/Shared Documents/Manuals

# Optional - Change in production
JWT_SECRET=your-super-secret-jwt-key-change-in-production
DATABASE_URL=postgresql+asyncpg://postgres:password@db:5432/vgs_chatbot
```

### 2. Start with Docker Compose

```bash
# Start all services (app, database, pgadmin)
docker-compose up -d

# View logs
docker-compose logs -f app

# Access the application
# Web UI: http://localhost:8501
# PgAdmin: http://localhost:5050 (admin@example.com / admin)
```

## Manual Setup (Development)

### 1. Install Dependencies

```bash
# Install Poetry if not available
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install

# Activate virtual environment
poetry shell
```

### 2. Database Setup

Option A - Use Docker for database only:
```bash
docker-compose up -d db
```

Option B - Install PostgreSQL manually:
```bash
# Ubuntu/Debian
sudo apt install postgresql postgresql-contrib

# macOS
brew install postgresql

# Start PostgreSQL service and create database
sudo -u postgres createdb vgs_chatbot
```

### 3. Run the Application

```bash
# Run the Streamlit app
poetry run streamlit run vgs_chatbot/gui/app.py

# Or run with specific port
poetry run streamlit run vgs_chatbot/gui/app.py --server.port 8501
```

## Testing the Application

### 1. Unit Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage report
poetry run pytest --cov=src --cov-report=html

# Run specific test file
poetry run pytest tests/test_auth_service.py -v
```

### 2. Code Quality Checks

```bash
# Run all quality checks (from CLAUDE.md)
poetry run isort src/ tests/ && poetry run ruff check src/ tests/ && poetry run mypy src/

# Fix linting issues
poetry run ruff check --fix src/ tests/

# Individual tools
poetry run isort src/ tests/        # Sort imports
poetry run ruff check src/ tests/   # Lint code
poetry run mypy src/               # Type checking
```

### 3. Manual Testing Steps

#### A. Test Authentication
1. Navigate to http://localhost:8501
2. Click "Register" tab
3. Create account: username `testuser`, email `test@example.com`, password `testpass123`
4. Login with created credentials
5. Verify you see the chat interface

#### B. Test SharePoint Connection
1. In sidebar, enter SharePoint credentials
2. Click "Connect to SharePoint"
3. Should see "✅ Connected to SharePoint" status

#### C. Test Chat Interface
1. Type a question like: "What are the safety procedures?"
2. Should receive response with sources listed
3. Test follow-up questions to verify chat history

## Configuration Options

### Environment Variables

```bash
# Database (required for auth)
DATABASE_URL=postgresql+asyncpg://user:pass@host:port/dbname

# JWT Security (required)
JWT_SECRET=your-secret-key-minimum-32-characters-long

# OpenAI (required for chat)
OPENAI_API_KEY=sk-your-api-key
OPENAI_MODEL=gpt-3.5-turbo  # or gpt-4

# SharePoint (required for documents)
SHAREPOINT_SITE_URL=https://your-site.sharepoint.com/sites/site
SHAREPOINT_DIRECTORY_URLS=url1,url2,url3

# App Settings (optional)
APP_TITLE=VGS Chatbot
DEBUG=false
```

### SharePoint URL Configuration

Examples of proper SharePoint directory URLs:
```bash
# Document library root
https://yourcompany.sharepoint.com/sites/yoursite/Shared Documents

# Specific folders
https://yourcompany.sharepoint.com/sites/yoursite/Shared Documents/2FTS
https://yourcompany.sharepoint.com/sites/yoursite/Shared Documents/Manuals
https://yourcompany.sharepoint.com/sites/yoursite/Shared Documents/Procedures

# Multiple sites (comma-separated)
SHAREPOINT_DIRECTORY_URLS=https://site1.sharepoint.com/docs,https://site2.sharepoint.com/files
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Errors
```bash
# Check if PostgreSQL is running
docker-compose ps
# or
sudo systemctl status postgresql

# Reset database
docker-compose down -v
docker-compose up -d db
```

#### 2. SharePoint Authentication Fails
- Ensure you're using your actual SharePoint login credentials
- Check if your organization requires MFA (may need app passwords)
- Verify SharePoint URLs are accessible from your network

#### 3. OpenAI API Errors
```bash
# Test your API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

#### 4. Import/Module Errors
```bash
# Reinstall dependencies
poetry install --no-cache

# Check Python path
poetry run python -c "import sys; print(sys.path)"
```

### Logs and Debugging

```bash
# Enable debug mode
export DEBUG=true

# View application logs
docker-compose logs -f app

# View database logs
docker-compose logs -f db

# Check Streamlit logs
poetry run streamlit run vgs_chatbot/gui/app.py --logger.level=debug
```

## Production Deployment

### Security Checklist

1. **Change default secrets**:
   ```bash
   # Generate secure JWT secret
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. **Use environment-specific .env**:
   ```bash
   # .env.production
   DEBUG=false
   JWT_SECRET=your-production-secret
   DATABASE_URL=postgresql+asyncpg://user:secure_password@prod-db:5432/vgs_chatbot
   ```

3. **Configure reverse proxy** (nginx example):
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://localhost:8501;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

### Performance Optimization

```bash
# Increase Streamlit performance
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
export STREAMLIT_SERVER_MAX_MESSAGE_SIZE=200

# Database connection pooling
export DATABASE_POOL_SIZE=20
export DATABASE_MAX_OVERFLOW=30
```

## Next Steps

1. **Custom Document Sources**: Extend `DocumentConnectorInterface` for other sources
2. **Different LLM Providers**: Implement `ChatServiceInterface` for local models
3. **Enhanced UI**: Replace Streamlit with React/Vue.js using the same backend
4. **Analytics**: Add usage tracking and performance monitoring
5. **Caching**: Implement Redis for document and embedding caching