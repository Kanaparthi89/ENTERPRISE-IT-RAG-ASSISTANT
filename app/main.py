"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router
from app.core.config import BASE_DIR, get_settings
from app.core.logging import configure_logging
from app.services.audit import init_db


configure_logging()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    description="AI-powered internal IT support assistant.",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health", tags=["Health"])
def health_check():
    """Return application health status."""
    return {"status": "healthy"}


# Register /api/health, /api/chat, and /api/ingest.
app.include_router(router)

# Serve CSS, JavaScript, and other static assets.
app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)


@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
def home(request: Request):
    """Render the Enterprise IT Support Copilot interface."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "app_name": settings.app_name,
        },
    )
