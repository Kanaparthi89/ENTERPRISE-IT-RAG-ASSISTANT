
import logging
import secrets
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.rag.workflow import ask
from app.rag.vectorstore import add_documents
from app.services.ingestion import load_file, chunk_documents, SUPPORTED
from app.services.audit import write_audit

router = APIRouter(prefix="/api")
settings = get_settings()

logger = logging.getLogger(__name__)

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=2,
        max_length=3000,
        strip_whitespace=True,
    )


@router.get("/health")
def health():
    """Return basic application health information."""
    return {
        "status": "ok",
        "service": settings.app_name,
    }


@router.post("/chat")
def chat(payload: ChatRequest):
    """Answer a question using the RAG workflow and record an audit event."""
    request_id = str(uuid4())
    question = payload.question.strip()

    try:
        result = ask(question)

        answer = result.get("answer", "")
        source_used = result.get("source_used", "unknown")
        trace = result.get("trace", [])
        citations = result.get("citations", [])

        if not answer:
            raise RuntimeError("The RAG workflow returned an empty answer.")

        # Audit failures should be visible in application logs.
        # They should not silently go unnoticed.
        try:
            write_audit(
                question=question,
                source_used=source_used,
                trace=trace,
                citations=citations,
                request_id=request_id,
                status=(
                    "insufficient_evidence"
                    if source_used == "insufficient_evidence"
                    else "success"
                ),
            )
        except Exception:
            logger.exception(
                "Audit write failed for request_id=%s",
                request_id,
            )

        return {
            "request_id": request_id,
            "answer": answer,
            "source_used": source_used,
            "trace": trace,
            "citations": citations,
            "rewritten_query": result.get("current_query", question),
        }

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
