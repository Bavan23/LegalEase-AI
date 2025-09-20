# 🏛️ LegalEase AI

**LegalEase AI** is a FastAPI-based platform that simplifies complex legal documents into clear, accessible insights using AI-powered document processing, risk analysis, and compliance checks.  
It includes a web interface (Jinja2 templates) and modular Python backend components.

---

## 📜 Features

- 📑 **Document Processor** – parses contracts and legal docs  
- ⚠️ **Risk Analyzer** – flags risky clauses using regex patterns (`config/risk_patterns.yml`)  
- 🧠 **AI Client** – integrates with AI models (`config/models.yml`)  
- 🌍 **Jinja2 Web UI** – upload documents and view risks interactively  
- 🔒 **Compliance Checks** – Indian law compliance patterns  
- 🧪 **Unit Tested** – built-in tests with **pytest**

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Bavan23/LegalEase-AI.git
cd LegalEase-AI
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv

# On Linux/Mac
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4.🔑 Environment Variables
Secrets (like API keys) are stored in a .env file.
Do not commit .env to Git.

**Create .env from template:**
```bash
cp .env.template .env
```
### 5. Running the App
**Run with Uvicorn (local dev):**
```bash
uvicorn src.web.app:app --reload
```

### 6. 🐳 Run with Docker
**Build and Run:**
```bash
docker-compose up --build
```

### 7. 🧪 Running Tests
```bash
pytest tests/
```
---
### 🤝 Contributing
**1. Fork the repo**
**2. Create a feature branch:**
```bash
git checkout -b feature/my-feature
```
**3. Commit changes:**
```bash
git commit -m "feat: add my feature"
```
**4. Push and create a Pull Request**


