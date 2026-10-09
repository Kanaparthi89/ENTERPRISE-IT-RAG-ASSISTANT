"""FastAPI application entry point."""

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
