
import asyncio
import json
import logging
import math
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit
from uuid import uuid4

import anyio
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Header, Request, Response
from fastapi.routing import APIRoute
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, BeforeValidator, StringConstraints

from app.api import streaming
from app.core.config import get_settings
from app.rag import cancellation
from app.rag.workflow import ask, history_window
from app.rag.vectorstore import add_documents
from app.services import sessions as session_service
from app.services import transcription
from app.services.ingestion import load_file, chunk_documents, SUPPORTED
from app.services.audit import write_audit
from starlette.requests import ClientDisconnect

settings = get_settings()

SESSION_STARTUP_CLEANUP_WAIT_SECONDS = 2.0
# Application shutdown (H2): how long running stream workers get to finish.
STREAM_SHUTDOWN_DRAIN_SECONDS = 3.0


async def _close_stream_service() -> None:
    """Stop the stream service and drain its workers (bounded), off the event loop."""
    service = streaming.peek_stream_service()
    if service is None:
        return
    try:
        await anyio.to_thread.run_sync(service.close, STREAM_SHUTDOWN_DRAIN_SECONDS)
    except Exception:  # noqa: BLE001 - shutdown must continue to the session manager
        logger.exception("Stream service shutdown failed.")
    finally:
        streaming.clear_stream_service(service)


@asynccontextmanager
async def _sessions_lifespan(app):
    """Session startup cleanup and bounded shutdown (design §7.5, H2).

    Merged by FastAPI into the lifespan of every app that includes this
    router, so the protected entry points stay unchanged. Startup runs one
    bounded cleanup pass only if a sessions database already exists (even when
    sessions are disabled, so leftover memory still expires). Shutdown first
    stops the stream service and drains its workers (at most
    ``STREAM_SHUTDOWN_DRAIN_SECONDS``), then runs the lease manager's bounded
    shutdown if sessions were used. Blocking work runs in worker threads,
    never on the event loop.

    After the drain, a shutdown signal (H1-3a) stops streams and
    ``/api/chat`` questions (with or without a session) that are still
    running before their next external call; a call already in progress is
    not interrupted.
    """
    cancellation.reset_shutdown()
    store = session_service.SessionStore.from_settings(settings)
    await anyio.to_thread.run_sync(store.startup_cleanup, SESSION_STARTUP_CLEANUP_WAIT_SECONDS)
    try:
        yield
    finally:
        try:
            await _close_stream_service()
        finally:
            cancellation.signal_shutdown()
            runtime = session_service.peek_session_runtime()
            if runtime is not None:
                try:
                    await anyio.to_thread.run_sync(runtime.manager.shutdown)
                finally:
                    session_service.clear_session_runtime(runtime)


router = APIRouter(prefix="/api", lifespan=_sessions_lifespan)

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


# -------------------------------------------------------------------
# Conversation sessions (A2-5; design revision 8 §3, §10, §14)
# -------------------------------------------------------------------

# Documents the header in OpenAPI as an API-key scheme; the value itself is
# read below so that a present-but-empty header is not mistaken for "absent".
session_header = APIKeyHeader(
    name="X-Session-ID",
    auto_error=False,
    description="Conversation session token: a bearer credential. Never put it in a URL.",
)

SESSION_ERRORS = {
    "session_expired": (410, "This conversation has expired. Please start a new chat."),
    "session_busy": (409, "A previous question in this conversation is still being answered."),
    "session_unavailable": (503, "Conversation memory is temporarily unavailable."),
    "origin_not_allowed": (403, "Request origin not allowed."),
    "unsupported_media_type": (415, "Use application/json."),
    "invalid_session_id": (400, "Invalid session ID header."),
}


def _session_error(code: str, headers: dict[str, str] | None = None) -> HTTPException:
    """A generic session error; never includes the token or exception text."""
    status_code, message = SESSION_ERRORS[code]
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
        headers=headers,
    )


def _session_token(
    request: Request, _documented: str | None = Depends(session_header)
) -> str | None:
    """The raw X-Session-ID value: None when absent, possibly empty when present."""
    return request.headers.get("x-session-id")


def _host_and_port(netloc: str, default_port: int) -> tuple[str, int] | None:
    try:
        parts = urlsplit(f"//{netloc}")
        if not parts.hostname:
            return None
        return parts.hostname.lower(), parts.port or default_port
    except ValueError:
        return None


def _origin_matches_host(origin: str, host: str) -> bool:
    """Same host and port; the scheme is not compared (a TLS proxy may change it)."""
    origin = origin.strip()
    if not origin or origin.lower() == "null":
        return False
    try:
        parts = urlsplit(origin)
    except ValueError:
        return False
    if parts.scheme not in ("http", "https"):
        return False
    default_port = 443 if parts.scheme == "https" else 80
    origin_host = _host_and_port(parts.netloc, default_port)
    request_host = _host_and_port(host.strip(), default_port)
    return origin_host is not None and origin_host == request_host


# An empty body or ``{}`` is all session creation accepts; a longer body is
# refused without being read in full.
MAX_SESSION_CREATE_BODY_BYTES = 1024


async def _read_session_create_body(request: Request) -> bytes | None:
    """The request body, or None if it is longer than the limit."""
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_SESSION_CREATE_BODY_BYTES:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _is_empty_json_body(body: bytes | None) -> bool:
    """True only for an empty body or the empty JSON object ``{}``."""
    if body is None:
        return False
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return False
    if not text.strip():
        return True
    try:
        value = json.loads(text)
    except ValueError:
        return False
    return isinstance(value, dict) and not value


async def _check_session_creation_request(request: Request) -> None:
    """Block cross-origin forms and scripts from creating sessions (§14.3).

    A form can send a simple POST cross-origin without a preflight; CORS only
    hides the response. In design order: content type and body (415), then
    Sec-Fetch-Site and Origin (403). All checked before any database access.
    """
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        raise _session_error("unsupported_media_type")
    if not _is_empty_json_body(await _read_session_create_body(request)):
        raise _session_error("unsupported_media_type")
    fetch_site = request.headers.get("sec-fetch-site")
    if fetch_site is not None and fetch_site.strip().lower() not in ("same-origin", "none"):
        raise _session_error("origin_not_allowed")
    origin = request.headers.get("origin")
    if origin is not None and not _origin_matches_host(origin, request.headers.get("host", "")):
        raise _session_error("origin_not_allowed")


def _iso_utc(timestamp: float) -> str:
    return (
        datetime.fromtimestamp(timestamp, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _acquire_session_lease(token: str, request_id: str):
    """Take the session's lease, or raise the matching 410 / 409 / 503."""
    if not settings.sessions_enabled or not session_service.is_well_formed_token(token):
        raise _session_error("session_expired")
    runtime = session_service.get_session_runtime(settings)
    try:
        return runtime, runtime.manager.acquire(token, request_id)
    except session_service.SessionExpired as exc:
        raise _session_error("session_expired") from exc
    except session_service.SessionBusy as exc:
        retry_after = str(max(1, math.ceil(exc.retry_after)))
        raise _session_error("session_busy", headers={"Retry-After": retry_after}) from exc
    except session_service.SessionStoreUnavailable as exc:
        raise _session_error("session_unavailable") from exc


# H1-3a: audit error types for an /api/chat question (stateless or session) stopped early.
_STOP_ERROR_TYPES = {
    cancellation.SESSION_TURN_EXPIRED: "SessionTurnExpired",
    cancellation.SESSION_LEASE_LOST: "SessionLeaseLost",
    cancellation.SERVER_SHUTDOWN: "ServerShutdown",
    cancellation.CHAT_DEADLINE: "TimeoutError",  # H1-5 (B), as a stream's deadline is recorded
}


def _chat_error_type(exc: Exception) -> str:
    if isinstance(exc, cancellation.WorkflowStopped):
        return _STOP_ERROR_TYPES.get(exc.reason, type(exc).__name__)
    return type(exc).__name__


def _log_chat_failure(exc: Exception, request_id: str) -> None:
    """A cooperative stop is expected: one warning, no traceback. Others: full traceback."""
    if isinstance(exc, cancellation.WorkflowStopped):
        logger.warning(
            "Chat request stopped for request_id=%s (%s)", request_id, _chat_error_type(exc)
        )
    else:
        logger.exception("Chat request failed for request_id=%s", request_id)


def _no_deadline() -> str | None:
    """Stateless /api/chat without CHAT_DEADLINE_SECONDS: only the shutdown signal stops it."""
    return None


def _stateless_stop_check():
    """The stop check for a stateless /api/chat question (H1-5 B).

    Unset ``CHAT_DEADLINE_SECONDS`` (the default) keeps the previous
    behavior: no deadline. When set, the deadline counts from now (the
    handler's start) and stops the question before its next external call.
    """
    seconds = settings.chat_deadline_seconds
    if seconds is None:
        return _no_deadline
    return cancellation.deadline_check(seconds, cancellation.CHAT_DEADLINE)


def _chat_with_session(question: str, request_id: str, token: str) -> dict:
    """Session-aware /api/chat (design §10).

    Order: take the lease (history is read with it) → ask() → build the
    response → save the turn (fenced) → set turn_saved → audit with the
    internal session UUID → release the lease. The lease is released in
    ``finally`` on every path. A 409 is returned before anything runs, so a
    concurrent duplicate never executes.
    """
    runtime, lease = _acquire_session_lease(token, request_id)
    history = [{"question": turn.question, "answer": turn.answer} for turn in lease.history]
    turns_used = len(history_window(history))
    finished = False

    def stop_reason() -> str | None:
        """H1-3a: after the turn limit or a lost lease, start no further external calls."""
        return runtime.manager.turn_stop_reason(lease)

    try:
        with cancellation.stop_checks(stop_reason):
            result = ask(question, history=history)
        response = _chat_response(result, request_id, question)

        if not response["answer"]:
            raise RuntimeError("The RAG workflow returned an empty answer.")

        # True only if the fenced save actually committed.
        turn_saved = runtime.manager.finish(
            lease, (question, response["answer"], response["source_used"])
        )
        finished = True
        response["session"] = {"history_turns_used": turns_used, "turn_saved": bool(turn_saved)}

        try:
            write_audit(
                question=question,
                source_used=response["source_used"],
                trace=response["trace"],
                citations=response["citations"],
                request_id=request_id,
                status=_audit_status(response["source_used"]),
                session_id=lease.session_id,
            )
        except Exception:
            logger.exception("Audit write failed for request_id=%s", request_id)

        return response

    except Exception as exc:
        _log_chat_failure(exc, request_id)

        try:
            write_audit(
                question=question,
                source_used="error",
                trace=[],
                request_id=request_id,
                status="failed",
                error_type=_chat_error_type(exc),
                session_id=lease.session_id,
            )
        except Exception:
            logger.exception("Failed to audit chat error for request_id=%s", request_id)

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Unable to process the question.",
                "request_id": request_id,
            },
        ) from exc

    finally:
        if not finished:
            runtime.manager.finish(lease)  # release without saving


@router.post("/chat")
def chat(payload: ChatRequest, session_token: str | None = Depends(_session_token)):
    """Answer a question using the RAG workflow and record an audit event.

    Without ``X-Session-ID`` the behaviour is exactly as before (stateless).
    """
    request_id = str(uuid4())
    question = payload.question.strip()

    if session_token is not None:
        return _chat_with_session(question, request_id, session_token)

    try:
        # The optional CHAT_DEADLINE_SECONDS (H1-5 B, off by default) and the
        # shutdown signal (H1-3a) apply before each external call.
        with cancellation.stop_checks(_stateless_stop_check()):
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
        _log_chat_failure(exc, request_id)

        try:
            write_audit(
                question=question,
                source_used="error",
                trace=[],
                request_id=request_id,
                status="failed",
                error_type=_chat_error_type(exc),
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


async def _chat_stream_with_session(question: str, request_id: str, token: str):
    """Session-aware streaming (design revision 8 §11).

    Admission order: stream capacity first (503 ``stream_busy``, the session
    is never touched), then the session lease (410 / 409 / 503, with the
    capacity slot released exactly once), then the run starts. All of these
    errors are returned before any SSE bytes. SQLite runs in worker threads.
    """
    service = streaming.get_stream_service()
    reservation = service.reserve()
    if reservation is None:
        return streaming.busy_response(settings.stream_retry_after_seconds, code="stream_busy")

    try:
        # Shielded: once a lease is taken in the worker thread, this request
        # must not be cancelled before handing it to the run (or releasing it).
        with anyio.CancelScope(shield=True):
            runtime, lease = await anyio.to_thread.run_sync(
                _acquire_session_lease, token, request_id
            )
    except BaseException:
        service.cancel(reservation)
        raise

    history = [{"question": turn.question, "answer": turn.answer} for turn in lease.history]
    turns_used = len(history_window(history))
    session = streaming.StreamSession(
        manager=runtime.manager, lease=lease, history=history, history_turns_used=turns_used
    )
    run = service.start(reservation, question, request_id, asyncio.get_running_loop(), session=session)
    if run is None:
        with anyio.CancelScope(shield=True):
            await anyio.to_thread.run_sync(runtime.manager.finish, lease)
        return streaming.busy_response(settings.stream_retry_after_seconds, code="stream_busy")

    def build_done(final):
        done = _chat_response(final, request_id, question)
        done["session"] = {"history_turns_used": turns_used}  # turn_saved follows in ``session``
        return done

    return streaming.SSEResponse(
        run,
        build_done=build_done,
        heartbeat_seconds=settings.stream_heartbeat_seconds,
        retry_after_seconds=settings.stream_retry_after_seconds,
    )


async def chat_stream(payload: ChatRequest, session_token: str | None = Depends(_session_token)):
    """Answer a question as Server-Sent Events (controlled prototype, off by default).

    Without ``X-Session-ID`` the behaviour is exactly as before (stateless).
    """
    request_id = str(uuid4())
    question = payload.question

    if session_token is not None:
        return await _chat_stream_with_session(question, request_id, session_token)

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


@router.post("/sessions", status_code=201)
async def create_session(request: Request, response: Response):
    """Start a conversation session (design §10, §14.3).

    404 while sessions are disabled; 415 unless the body is empty or ``{}``
    with ``Content-Type: application/json``; 403 for cross-origin requests
    (all checked before any database access); 503 if storage is unavailable.
    The token is returned once, never stored or logged. The database work
    runs in a worker thread, never on the event loop.
    """
    if not settings.sessions_enabled:
        raise HTTPException(status_code=404, detail="Not Found")
    await _check_session_creation_request(request)

    runtime = session_service.get_session_runtime(settings)
    try:
        created = await anyio.to_thread.run_sync(runtime.store.create)
    except session_service.SessionStoreUnavailable as exc:
        raise _session_error("session_unavailable") from exc

    response.headers["Cache-Control"] = "no-store"
    idle = created.idle_timeout_seconds
    return {
        "session_id": created.token,
        "idle_timeout_seconds": int(idle) if float(idle).is_integer() else idle,
        "expires_at": _iso_utc(created.absolute_expires_at),
    }


@router.delete("/sessions", status_code=204)
def delete_session(session_token: str | None = Depends(_session_token)):
    """Clear conversation memory (design §10.1).

    204 for a deleted, unknown, expired or already-deleted session, and while
    sessions are disabled; 400 for a missing or malformed header; 503 if
    deletion can't be confirmed. Audit records are not affected.
    """
    if session_token is None or not session_service.is_well_formed_token(session_token):
        raise _session_error("invalid_session_id")
    runtime = session_service.get_session_runtime(settings)
    try:
        runtime.store.delete(session_token)
    except session_service.SessionStoreUnavailable as exc:
        raise _session_error("session_unavailable") from exc
    return Response(status_code=204)


@router.get("/voice/status")
async def voice_status(response: Response):
    """Whether spoken answers may be offered (Voice V1b).

    404 (the same body as an unknown route) while ``VOICE_ENABLED`` is off,
    the default. When on: ``{"tts": true}``, meaning the page may offer
    browser speech synthesis with on-device voices. Reads one setting and
    nothing else: no speech provider, credentials, storage or audit.
    """
    if not settings.voice_enabled:
        raise HTTPException(status_code=404, detail="Not Found")
    response.headers["Cache-Control"] = "no-store"  # the flag can change on restart
    return {"tts": True}


# Voice V2: speech-to-text. Every error is generic and never includes the
# audio, the transcript, the filename or provider details.
VOICE_ERRORS = {
    "origin_not_allowed": (403, "Request origin not allowed."),
    "unsupported_media_type": (415, "Send the audio as multipart/form-data."),
    "audio_too_large": (413, "Audio must be 2 MiB or smaller."),
    "upload_timeout": (408, "The audio upload took too long."),
    "invalid_upload": (400, 'Send exactly one audio file in the "audio" field.'),
    "unsupported_audio": (415, "Unsupported audio. Use WebM, Ogg, MP4/M4A, MP3 or WAV."),
    "audio_not_transcribable": (422, "The audio could not be transcribed."),
    "transcription_timeout": (504, "Transcription took too long. Please try again."),
    "transcription_unavailable": (503, "Transcription is temporarily unavailable."),
    "transcription_failed": (500, "Transcription failed."),
}
NO_STORE = {"Cache-Control": "no-store"}


def _voice_error(code: str) -> HTTPException:
    status_code, message = VOICE_ERRORS[code]
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
        headers=NO_STORE,
    )


def _check_voice_origin(request: Request) -> None:
    """Refuse cross-site browser requests (a form can POST multipart without a preflight).

    A browser's ``Sec-Fetch-Site`` must be ``same-origin``; an ``Origin``
    header must match the host. Clients that send neither (curl, scripts) are
    not browsers and are allowed: this is CSRF protection, not authentication.
    """
    fetch_site = request.headers.get("sec-fetch-site")
    if fetch_site is not None and fetch_site.strip().lower() != "same-origin":
        raise _voice_error("origin_not_allowed")
    origin = request.headers.get("origin")
    if origin is not None and not _origin_matches_host(origin, request.headers.get("host", "")):
        raise _voice_error("origin_not_allowed")


@router.post("/voice/transcribe")
async def voice_transcribe(request: Request, response: Response):
    """Transcribe one short audio recording to text (Voice V2).

    404 (the same body as an unknown route) while
    ``VOICE_TRANSCRIPTION_ENABLED`` is off, the default. The request is
    ``multipart/form-data`` with one file part named ``audio`` (WebM, Ogg,
    MP4/M4A, MP3 or WAV; at most 2 MiB). Returns ``{"text": "..."}``. The
    transcript is only returned: it is not stored, audited or sent to the
    chat. In order, before any of the body is read: the flag (404), the
    origin (403), the content type (415), the declared length (413) and the
    provider key (503). Then the bounded read (408, 413), validation (400,
    413, 415) and the provider call (422, 503, 504).

    The handler takes only ``request``, so FastAPI never parses or buffers
    the body itself (``File(...)`` would spool the whole upload first).
    """
    if not settings.voice_transcription_enabled:
        raise HTTPException(status_code=404, detail="Not Found")
    _check_voice_origin(request)

    content_type = request.headers.get("content-type")
    try:
        transcription.multipart_boundary(content_type)
        transcription.check_declared_length(request.headers.get("content-length"))
        api_key = settings.openai_api_key.get_secret_value()
        if not api_key:
            logger.error("Voice transcription is enabled but OPENAI_API_KEY is not set.")
            raise transcription.VoiceUploadError("transcription_unavailable")

        body = await transcription.read_bounded_body(request.stream())
        upload = transcription.parse_audio_upload(body, content_type)
        del body
        text = await transcription.transcribe(
            upload,
            api_key=api_key,
            model=settings.voice_transcription_model,
            timeout_seconds=settings.voice_transcription_timeout_seconds,
            max_retries=settings.voice_transcription_max_retries,
        )
    except transcription.VoiceUploadError as exc:
        raise _voice_error(exc.code) from None
    except ClientDisconnect:
        raise _voice_error("invalid_upload") from None
    except Exception as exc:  # noqa: BLE001 - never echo details; log the class only
        logger.error("Voice transcription failed unexpectedly (%s).", type(exc).__name__)
        raise _voice_error("transcription_failed") from None

    response.headers["Cache-Control"] = "no-store"
    return {"text": text}


@router.get("/voice/transcription/status")
async def voice_transcription_status(response: Response):
    """Whether dictation may be offered (Voice V3b).

    404 (the same body as an unknown route) while
    ``VOICE_TRANSCRIPTION_ENABLED`` is off, the default. When on:
    ``{"stt": true}``, meaning the page may offer dictation through
    ``POST /api/voice/transcribe``. Separate from ``GET /api/voice/status``
    (``VOICE_ENABLED``), whose contract is unchanged. Reads one setting and
    nothing else: no provider, credentials, storage or audit.
    """
    if not settings.voice_transcription_enabled:
        raise HTTPException(status_code=404, detail="Not Found")
    response.headers["Cache-Control"] = "no-store"  # the flag can change on restart
    return {"stt": True}


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
