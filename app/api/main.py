import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router
from app.core.config import get_settings, BASE_DIR
from app.core.logging import configure_logging
from app.services.audit import init_db


# Configure application logging before initializing services.
configure_logging()
logger = logging.getLogger(__name__)

settings = get_settings()

STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize application resources during startup."""
    logger.info("Starting application: %s", settings.app_name)

    try:
        init_db()
        logger.info("Audit database initialized successfully.")

        yield

    except Exception:
        logger.exception("Application startup or runtime lifecycle failed.")
        raise

    finally:
        logger.info("Application shutdown: %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
)


# Register API endpoints.
app.include_router(router)


# Serve static assets only when the directory exists.
if STATIC_DIR.is_dir():
    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static",
    )
else:
    logger.warning("Static directory not found: %s", STATIC_DIR)


# Configure HTML templates.
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    """Return a consistent response for invalid request data."""
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "message": "Please check the request data and try again.",
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request,
    exc: Exception,
):
    """Log unexpected errors without exposing internal details."""
    logger.exception(
        "Unhandled application error on %s %s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again.",
        },
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Render the Enterprise IT Support Copilot homepage."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_name": settings.app_name,
        },
    )
