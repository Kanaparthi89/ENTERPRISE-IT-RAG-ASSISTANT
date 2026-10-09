
from pathlib import Path

# Project root: current working directory
ROOT = Path.cwd()

# Directories to create
DIRECTORIES = [
    "app/api",
    "app/core",
    "app/rag",
    "app/services",
    "data",
    "templates",
    "static/css",
    "static/js",
    "static/images",
    "uploads",
    "tests",
]

# Files to create with optional starter content
FILES = {
    "app/__init__.py": "",
    "app/main.py": '''"""FastAPI application entry point."""

from fastapi import FastAPI

app = FastAPI(
    title="Enterprise IT Support Agentic RAG Copilot",
    description="AI-powered internal IT support assistant.",
    version="1.0.0",
)


@app.get("/health", tags=["Health"])
def health_check():
    """Return application health status."""
    return {"status": "healthy"}
''',
    "app/api/__init__.py": "",
    "app/core/__init__.py": "",
    "app/rag/__init__.py": "",
    "app/services/__init__.py": "",
    "tests/__init__.py": "",
    "ingest_sample_kb.py": '"""Ingest sample knowledge-base documents."""\n',
    "run.py": '''"""Run the FastAPI development server."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
''',
    "requirements.txt": """fastapi
uvicorn[standard]
python-dotenv
pydantic-settings
langgraph
langchain
langchain-community
langchain-groq
langchain-pinecone
pinecone
tavily-python
pytest
httpx
""",
    ".env.example": """# Copy to .env and add your own credentials.
GROQ_API_KEY=
PINECONE_API_KEY=
TAVILY_API_KEY=
""",
    ".gitignore": """# Secrets
.env
!.env.example

# Python
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/

# IDE and OS
.vscode/
.idea/
.DS_Store

# Generated data
uploads/*
!uploads/.gitkeep
data/*
!data/.gitkeep
""",
    "uploads/.gitkeep": "",
    "data/.gitkeep": "",
    "tests/test_health.py": '''"""Tests for the health endpoint."""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
''',
}


def create_project_structure():
    """Create missing directories and starter files without overwriting."""

    created_directories = 0
    created_files = 0
    skipped_files = 0

    for directory in DIRECTORIES:
        path = ROOT / directory
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created_directories += 1

    for filename, content in FILES.items():
        path = ROOT / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            print(f"Skipped existing file: {filename}")
            skipped_files += 1
            continue

        path.write_text(content, encoding="utf-8")
        print(f"Created: {filename}")
        created_files += 1

    print("\nProject structure setup complete.")
    print(f"Root: {ROOT}")
    print(f"Directories created: {created_directories}")
    print(f"Files created: {created_files}")
    print(f"Existing files preserved: {skipped_files}")


if __name__ == "__main__":
    create_project_structure()
