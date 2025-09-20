# Enhanced Legal Document Demystifier - Production Setup Guide

## Quick Start Guide

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Google Gemini API Key
- 4GB+ RAM
- 10GB+ disk space

### Installation

1. **Clone and Setup**
```bash
git clone <repository-url>
cd enhanced-legal-demystifier
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Environment Configuration**
```bash
# Create .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env
echo "ENVIRONMENT=production" >> .env
echo "LOG_LEVEL=INFO" >> .env
```

3. **Install System Dependencies**
```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils tesseract-ocr tesseract-ocr-hin tesseract-ocr-tam

# macOS
brew install poppler tesseract tesseract-lang

# Windows - Download binaries from official sites
```

4. **Run Tests**
```bash
pytest tests/ -v --cov=src --cov-report=html
```

5. **Start Application**
```bash
# Development
uvicorn src.web.app:app --reload --host 0.0.0.0 --port 8000

# Production with Docker
docker-compose up -d
```

### Production Deployment

#### Docker Deployment
```bash
# Build and deploy
docker build -t legal-demystifier:latest .
docker-compose -f docker-compose.prod.yml up -d
```

#### Kubernetes Deployment
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: legal-demystifier
spec:
  replicas: 3
  selector:
    matchLabels:
      app: legal-demystifier
  template:
    metadata:
      labels:
        app: legal-demystifier
    spec:
      containers:
      - name: app
        image: legal-demystifier:latest
        ports:
        - containerPort: 8000
        env:
        - name: GEMINI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: gemini-api-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi" 
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: legal-demystifier-service
spec:
  selector:
    app: legal-demystifier
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### Cloud Run Deployment (Google Cloud)
```bash
# Build and deploy to Cloud Run
gcloud builds submit --tag gcr.io/PROJECT_ID/legal-demystifier
gcloud run deploy --image gcr.io/PROJECT_ID/legal-demystifier --platform managed --region us-central1 --memory 4Gi
```

### Configuration Management

#### Environment Variables
```bash
# Required
GEMINI_API_KEY=your_gemini_api_key
ENVIRONMENT=production|development|testing

# Optional
LOG_LEVEL=INFO|DEBUG|WARNING|ERROR
MAX_FILE_SIZE=26214400  # 25MB in bytes
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@localhost:5432/legaldb
SENTRY_DSN=your_sentry_dsn_for_error_tracking
```

#### Configuration Files
```yaml
# config/settings.yml
app:
  name: "Legal Document Demystifier"
  version: "2.0.0"
  debug: false

security:
  max_file_size: 26214400
  allowed_file_types: [".pdf"]
  session_timeout: 3600
  
privacy:
  data_retention_days: 30
  encryption_enabled: true
  audit_logging: true

ai:
  models:
    primary: "gemini-1.5-flash"
    fallback: "gemini-1.5-flash"
  cache_enabled: true
  cache_ttl: 3600
  max_retries: 3
```

### Monitoring & Observability

#### Logging Configuration
```python
# logging_config.py
import structlog
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "structured": {
            "()": structlog.stdlib.ProcessorFormatter,
            "processor": structlog.dev.ConsoleRenderer(colors=False),
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "structured",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "structured",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"],
    },
}
```

#### Health Checks
```python
# health_checks.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

@router.get("/health/detailed")
async def detailed_health_check():
    return {
        "status": "healthy",
        "components": {
            "database": "healthy",
            "ai_service": "healthy", 
            "file_storage": "healthy"
        },
        "metrics": {
            "active_sessions": len(sessions),
            "memory_usage": get_memory_usage(),
            "cpu_usage": get_cpu_usage()
        }
    }
```

### Security Best Practices

#### API Security
```python
# security_middleware.py
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer
import jwt

security = HTTPBearer()

async def verify_token(request: Request):
    # Implement JWT token verification
    # Rate limiting
    # CORS validation
    pass

# Add to FastAPI app
app.add_middleware(SecurityMiddleware)
```

#### Data Protection
```python
# GDPR/Privacy compliance
class PrivacyCompliance:
    def __init__(self):
        self.data_retention = 30  # days
        
    async def anonymize_data(self, data):
        # Remove PII
        # Hash identifiers
        pass
        
    async def delete_expired_data(self):
        # Clean up old sessions
        # Remove temporary files
        pass
```

### Performance Optimization

#### Caching Strategy
```python
# caching.py
import redis
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(expiry=3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try cache first
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            redis_client.setex(cache_key, expiry, json.dumps(result))
            
            return result
        return wrapper
    return decorator
```

#### Background Tasks
```python
# background_tasks.py
from celery import Celery

celery_app = Celery('legal_demystifier')

@celery_app.task
def process_document_async(file_path, session_id):
    # Heavy document processing
    # Risk analysis
    # AI summarization
    pass

@celery_app.task
def cleanup_old_sessions():
    # Remove expired sessions
    # Clean temporary files
    pass
```

### API Documentation

#### OpenAPI Specification
```python
# Custom OpenAPI schema
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Legal Document Demystifier API",
        version="2.0.0",
        description="AI-powered legal document analysis with risk assessment",
        routes=app.routes,
    )
    
    # Add custom security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

### Testing Strategy

#### Test Categories
1. **Unit Tests** - Individual component testing
2. **Integration Tests** - Component interaction testing  
3. **API Tests** - Endpoint testing
4. **Security Tests** - Vulnerability testing
5. **Performance Tests** - Load and stress testing
6. **E2E Tests** - Full user workflow testing

#### Test Execution
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/security/ -v

# Performance testing
locust -f tests/performance/locustfile.py --host=http://localhost:8000
```

### Backup & Disaster Recovery

#### Data Backup
```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/legal_demystifier"

# Database backup
pg_dump legal_demystifier > "$BACKUP_DIR/db_$DATE.sql"

# Configuration backup
tar -czf "$BACKUP_DIR/config_$DATE.tar.gz" config/

# Log retention
find /logs -name "*.log" -mtime +30 -delete
```

#### Recovery Procedures
```bash
# Database recovery
psql legal_demystifier < /backups/legal_demystifier/db_20241201_120000.sql

# Configuration recovery
tar -xzf /backups/legal_demystifier/config_20241201_120000.tar.gz -C /
```

### Troubleshooting

#### Common Issues

1. **Memory Issues**
```bash
# Check memory usage
docker stats
kubectl top pods

# Increase memory limits
docker run -m 4g legal-demystifier
```

2. **API Rate Limits**
```python
# Monitor API usage
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Implement rate limiting logic
    pass
```

3. **File Processing Errors**
```bash
# Check OCR dependencies
tesseract --version
poppler-utils --version

# Verify file permissions
ls -la uploads/
```

### Feature Roadmap

#### Version 2.1 (Q1 2024)
- Advanced ML-based risk scoring
- Integration with legal databases
- Multi-document comparison
- Workflow automation

#### Version 2.2 (Q2 2024)  
- Real-time collaboration
- Advanced analytics dashboard
- Custom risk patterns
- API marketplace integration

#### Version 3.0 (Q3 2024)
- Blockchain document verification
- Advanced AI legal reasoning
- Regulatory compliance modules
- Enterprise SSO integration

---

## Support & Maintenance

### Support Channels
- Documentation: `/docs`
- Issues: GitHub Issues
- Security: security@company.com
- General: support@company.com

### Maintenance Schedule
- Security patches: Weekly
- Feature updates: Monthly
- Major releases: Quarterly

### License
MIT License - See LICENSE file for details

---

*Generated by Enhanced Legal Document Demystifier v2.0*