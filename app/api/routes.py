
import asyncio
import logging
import secrets
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from fastapi.routing import APIRoute
from pydantic import BaseModel, BeforeValidator, StringConstraints

from app.api import streaming
from app.core.config import get_settings
from app.rag.workflow import ask
from app.rag.vectorstore import add_documents
from app.services.ingestion import load_file, chunk_documents, SUPPORTED
from app.services.audit import write_audit

router = APIRouter(prefix="/api")
settings = get_settings()

logger = logging.getLogger(__name__)

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB


def _strip_whitespace(value):
    """Strip with Python's str.strip(), matching the handler and workflow.

    Pydantic's built-in strip_whitespace keeps U+001C to U+001F, which
    str.strip() removes; a question made only of those characters would pass
    validation and then become empty downstream.
    """
    return value.strip() if isinstance(value, str) else value


# Leading and trailing whitespace is stripped before the length limits are
# checked, so whitespace-only questions are rejected with HTTP 422.
# Internal whitespace, such as line breaks in pasted logs, is preserved.
# BeforeValidator is listed last so it wraps the constrained string schema:
# it still runs first, and the limits keep Pydantic's native
# string_too_short/string_too_long errors.
Question = Annotated[
    str,
    StringConstraints(min_length=2, max_length=3000),
    BeforeValidator(_strip_whitespace),
]


class ChatRequest(BaseModel):
    question: Question


@router.get("/health")
def health():
    """Return basic application health information."""
    return {
        "status": "ok",
        "service": settings.app_name,
    }


def _audit_status(source_used: str) -> str:
    """Audit status for a successfully answered question."""
    return "insufficient_evidence" if source_used == "insufficient_evidence" else "success"


def _chat_response(result: dict, request_id: str, question: str) -> dict:
    """The /api/chat response body; also the payload of the stream's ``done`` event."""
    return {
        "request_id": request_id,
        "answer": result.get("answer", ""),
        "source_used": result.get("source_used", "unknown"),
        "trace": result.get("trace", []),
        "citations": result.get("citations", []),
        "rewritten_query": result.get("current_query", question),
    }


@router.post("/chat")
def chat(payload: ChatRequest):
    """Answer a question using the RAG workflow and record an audit event."""
    request_id = str(uuid4())
    question = payload.question.strip()

    try:
        result = ask(question)
        response = _chat_response(result, request_id, question)

        if not response["answer"]:
            raise RuntimeError("The RAG workflow returned an empty answer.")

        # Audit failures should be visible in application logs.
        # They should not silently go unnoticed.
        try:
            write_audit(
                question=question,
                source_used=response["source_used"],
                trace=response["trace"],
                citations=response["citations"],
                request_id=request_id,
                status=_audit_status(response["source_used"]),
            )
        except Exception:
            logger.exception(
                "Audit write failed for request_id=%s",
                request_id,
            )

        return response

    except Exception as exc:
        logger.exception(
            "Chat request failed for request_id=%s",
            request_id,
        )

        try:
            write_audit(
                question=question,
                source_used="error",
                trace=[],
                request_id=request_id,
                status="failed",
                error_type=type(exc).__name__,
            )
        except Exception:
            logger.exception(
                "Failed to audit chat error for request_id=%s",
                request_id,
            )

        # Avoid exposing internal exceptions, credentials, or infrastructure
        # details in the API response.
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Unable to process the question.",
                "request_id": request_id,
            },
        ) from exc


def require_streaming_enabled() -> None:
    """Raise 404 while the streaming endpoint is disabled."""
    if not settings.stream_enabled:
        raise HTTPException(status_code=404, detail="Not Found")


class StreamingFlagRoute(APIRoute):
    """Checks the streaming flag before FastAPI reads or parses the body.

    A disabled endpoint answers 404 for any request body, including invalid
    JSON. When enabled, the request goes to FastAPI's normal handler, so
    body parsing and validation errors are unchanged.
    """

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def route_handler(request):
            require_streaming_enabled()
            return await handler(request)

        return route_handler


async def chat_stream(payload: ChatRequest):
    """Answer a question as Server-Sent Events (controlled prototype, off by default)."""
    request_id = str(uuid4())
    question = payload.question

    run = streaming.get_stream_service().start_run(
        question, request_id, asyncio.get_running_loop()
    )
    if run is None:
        return streaming.busy_response(settings.stream_retry_after_seconds)

    return streaming.SSEResponse(
        run,
        build_done=lambda final: _chat_response(final, request_id, question),
        heartbeat_seconds=settings.stream_heartbeat_seconds,
        retry_after_seconds=settings.stream_retry_after_seconds,
    )


# Registered with add_api_route: the route-class override is not available
# on the @router.post decorator.
router.add_api_route(
    "/chat/stream",
    chat_stream,
    methods=["POST"],
    route_class_override=StreamingFlagRoute,
)


@router.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    x_admin_key: str = Header(default=""),
):
    """Validate and index an uploaded knowledge-base document."""
    if not settings.admin_api_key:
        logger.error("Document ingestion is disabled: admin key is not configured.")
        raise HTTPException(
            status_code=503,
            detail="Document ingestion is not configured.",
        )

    expected_key = settings.admin_api_key
    if hasattr(expected_key, "get_secret_value"):
        expected_key = expected_key.get_secret_value()

    if not x_admin_key or not secrets.compare_digest(
        x_admin_key,
        str(expected_key),
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid admin key.",
        )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    # Treat the supplied filename as untrusted input.
    filename = Path(file.filename).name
    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=f"Supported file types: {', '.join(sorted(SUPPORTED))}",
        )

    upload_dir = Path(settings.upload_dir).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Read at most MAX_UPLOAD_SIZE + 1 bytes to enforce the limit.
    content = await file.read(MAX_UPLOAD_SIZE + 1)

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 10 MB upload limit.",
        )

    # Use a generated storage name to prevent filename collisions and
    # avoid trusting user-supplied paths.
    stored_name = f"{uuid4().hex}{suffix}"
    dest = upload_dir / stored_name

    try:
        dest.write_bytes(content)

        docs = load_file(dest)
        if not docs:
            raise HTTPException(
                status_code=400,
                detail="No readable content was found in the uploaded file.",
            )

        chunks = chunk_documents(docs)
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No indexable text was found in the uploaded file.",
            )

        # Index only after successful validation and chunking.
        ids = add_documents(chunks)

        logger.info(
            "Document indexed: filename=%s chunks=%s",
            filename,
            len(chunks),
        )

        return {
            "message": "Document indexed successfully.",
            "file": filename,
            "chunks": len(chunks),
            "ids_created": len(ids),
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Document ingestion failed for filename=%s", filename)

        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed. Check the application logs.",
        ) from exc

    finally:
        await file.close()
