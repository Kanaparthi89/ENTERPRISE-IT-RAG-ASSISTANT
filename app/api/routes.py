
import asyncio
import json
import logging
import math
import secrets
import threading
from contextlib import asynccontextmanager, contextmanager
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
from app.services import access as access_service
from app.services import access_policy
from app.services import quota as quota_service
from app.services import quota_runtime
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
    """Session creation accepts only an empty body or ``{}`` as JSON (§14.3): else 415.

    Checked before any database access. Access and the origin (S1-5) were
    already checked by the route class, before this reads the body.
    """
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        raise _session_error("unsupported_media_type")
    if not _is_empty_json_body(await _read_session_create_body(request)):
        raise _session_error("unsupported_media_type")


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


# ---------------------------------------------------------------------------
# Security S1-4b: chat quotas
#
# Every chat route admits a question with the process-wide QuotaEngine before
# any provider call, after request validation and session checks (so a
# validation error, a session conflict or an expired session never uses
# quota). A refusal calls no provider. The ticket's concurrency slot is
# released only when the question's provider work has stopped: after ask()
# returns (or raises) for /api/chat, and when both the stream worker and its
# consumer are done for /api/chat/stream. Disabled by default (QUOTA_ENABLED).
# ---------------------------------------------------------------------------

QUOTA_ERRORS = {
    quota_service.HOURLY_LIMIT: ("quota_hourly_limit", {
        quota_service.CODE_SCOPE: "You've reached the hourly question limit. Please try again later.",
        quota_service.GLOBAL_SCOPE: "The service has reached its hourly question limit. Please try again later.",
    }),
    quota_service.DAILY_LIMIT: ("quota_daily_limit", {
        quota_service.CODE_SCOPE: "You've reached today's question limit. Please try again tomorrow.",
        quota_service.GLOBAL_SCOPE: "The service has reached today's question limit. Please try again tomorrow.",
    }),
    quota_service.BUSY: ("quota_busy", {
        quota_service.CODE_SCOPE: "Your previous question is still being answered. Please wait for it to finish.",
        quota_service.GLOBAL_SCOPE: "The service is busy answering other questions. Please try again shortly.",
    }),
    quota_service.CAPACITY: ("quota_capacity", {
        quota_service.GLOBAL_SCOPE: "The service is at capacity. Please try again later.",
    }),
}
QUOTA_UNAVAILABLE_MESSAGE = "Questions are temporarily unavailable. Please try again shortly."


def _quota_identity(request: Request | None) -> tuple[str, str | None]:
    """(identity, log pseudonym) from the request's verified access decision (S1-5).

    The route class decided access before the body was read and stored the
    decision on ``request.state.access``; the cookie is never verified a
    second time here. There is no anonymous identity for a paid route: a
    missing, refused or malformed decision fails closed (503).
    """
    state = getattr(request, "state", None) if request is not None else None
    decision = getattr(state, "access", None) if state is not None else None
    if not isinstance(decision, access_policy.AccessDecision) or not decision.granted:
        logger.error("Paid request without a verified access decision: refused.")
        raise _paid_access_error("access_unavailable")
    identity = decision.grant.code_id
    if (
        not isinstance(identity, str)
        or identity == quota_service.ANONYMOUS
        or not access_service.CODE_ID_PATTERN.fullmatch(identity)
    ):
        logger.error("Paid request with an invalid access identity: refused.")
        raise _paid_access_error("access_unavailable")
    return identity, decision.code_ref


def _require_paid_configuration() -> None:
    """Defence in depth while quotas are off: paid work needs the full configuration (503)."""
    problem = access_policy.configuration_problem(settings)
    if problem is not None:
        logger.error("Paid AI request refused: read-only mode (%s).", problem)
        raise _paid_access_error("access_unavailable")


def _recheck_paid_request(request: Request | None) -> None:
    """Defence in depth (S1-5): recheck before a handler allocates anything (503).

    The route class already decided access before the body was read; this
    re-validates, without verifying the cookie again, what may have changed
    since: the configuration (access, quotas and PUBLIC_ORIGINS, all enabled
    and valid) and the verified identity on ``request.state.access``.
    """
    _require_paid_configuration()
    _quota_identity(request)


def _quota_error(status_code: int, code: str, message: str, retry_after: int) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
        headers={"Retry-After": str(max(1, int(retry_after))), **NO_STORE},
    )


def _admit_chat(request: Request | None, request_id: str) -> "quota_service.Ticket | None":
    """Admit one chat question (None while quotas are disabled); raises 429 or 503.

    Call it before any provider call. A refusal reserves nothing.
    """
    if not settings.quota_enabled:
        _require_paid_configuration()  # quotas off: always refused outside the test shim
        return None
    identity, code_ref = _quota_identity(request)  # fails closed without a verified grant
    busy = settings.quota_busy_retry_after_seconds
    try:
        result = quota_runtime.get_quota_engine(settings).admit(quota_service.CHAT, identity)
    except Exception as exc:  # noqa: BLE001 - fail closed; logged by class only
        logger.error("Chat quota unavailable for request_id=%s (%s).", request_id, type(exc).__name__)
        raise _quota_error(503, "quota_unavailable", QUOTA_UNAVAILABLE_MESSAGE, busy) from None
    if isinstance(result, quota_service.Ticket):
        return result
    known = QUOTA_ERRORS.get(result.reason)
    if known is None:  # unavailable (or anything unexpected): fail closed
        logger.error("Chat quota unavailable for request_id=%s (%s).", request_id, result.reason)
        raise _quota_error(503, "quota_unavailable", QUOTA_UNAVAILABLE_MESSAGE, result.retry_after or busy)
    code, messages = known
    message = messages.get(result.scope) or messages[quota_service.GLOBAL_SCOPE]
    logger.warning(
        "Chat refused by quota for request_id=%s: reason=%s scope=%s code_ref=%s",
        request_id, result.reason, result.scope, code_ref or "-",
    )
    raise _quota_error(429, code, message, result.retry_after or busy)


def _release_ticket(ticket: "quota_service.Ticket | None") -> None:
    """Give back a question's concurrency slot (idempotent); never raises."""
    if ticket is None:
        return
    try:
        ticket.release()
    except Exception as exc:  # noqa: BLE001
        logger.error("Chat quota release failed (%s).", type(exc).__name__)


def _chat_with_session(question: str, request_id: str, token: str, request: Request | None = None) -> dict:
    """Session-aware /api/chat (design §10).

    Order: take the lease (history is read with it) → admit the question
    (S1-4b) → ask() → build the response → save the turn (fenced) → set
    turn_saved → audit with the internal session UUID → release the lease.
    A 409 is returned before anything runs, so a concurrent duplicate never
    executes. Once the lease is taken, everything runs inside one cleanup
    scope: on every path the lease is released (without saving unless the
    turn was saved) and the quota ticket, if any, is released once.
    """
    runtime, lease = _acquire_session_lease(token, request_id)
    ticket = None
    finished = False
    try:
        # S1-4b: after the session checks (a conflict or an expired session
        # never uses quota), before any provider call. A refusal (429/503)
        # propagates as is; the cleanup below releases the lease unsaved.
        ticket = _admit_chat(request, request_id)
        try:
            history = [{"question": turn.question, "answer": turn.answer} for turn in lease.history]
            turns_used = len(history_window(history))

            def stop_reason() -> str | None:
                """H1-3a: after the turn limit or a lost lease, start no further external calls."""
                return runtime.manager.turn_stop_reason(lease)

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
        try:
            if not finished:
                runtime.manager.finish(lease)  # release without saving
        finally:
            _release_ticket(ticket)  # ask() has returned (or never ran): no provider work is left


# ---------------------------------------------------------------------------
# Security S1-5: verified access on every paid AI route
#
# Each paid route uses a subclass of AccessEnforcedRoute, which runs before
# FastAPI reads or parses the body (FastAPI parses it before dependencies,
# so a dependency would be too late). In order: the route's feature flag
# (404), access (503 read-only mode / 401), the same-origin check (403) and,
# for JSON routes, the content type (415). Only then does the handler run:
# body validation (422), session and stream capacity checks, the quota and
# the provider. The decision is stored on ``request.state.access`` for the
# quota identity. There is no bypass of any kind (address, Host, headers,
# APP_ENV); see app/services/access_policy.py.
# ---------------------------------------------------------------------------

PAID_ACCESS_ERRORS = {
    "access_required": (401, "Access required. Enter your access code to use this feature."),
    "access_unavailable": (503, "This feature is not available right now."),
    "origin_not_allowed": (403, "Request origin not allowed."),
    "unsupported_media_type": (415, "Use application/json."),
}
ACCESS_UNAVAILABLE_RETRY_AFTER_SECONDS = 60


def _paid_access_error(code: str) -> HTTPException:
    status_code, message = PAID_ACCESS_ERRORS[code]
    headers = dict(NO_STORE)
    if code == "access_required":
        headers["WWW-Authenticate"] = 'AccessCode realm="BluePeak AI Desk"'
    elif code == "access_unavailable":
        headers["Retry-After"] = str(ACCESS_UNAVAILABLE_RETRY_AFTER_SECONDS)
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
        headers=headers,
    )


def _authorize_paid_request(request: Request, *, json_body: bool) -> None:
    """Access, then origin, then (for JSON routes) content type; raises 503/401/403/415."""
    decision = access_policy.evaluate(request, settings)
    request.state.access = decision
    if decision.outcome == access_policy.UNAVAILABLE:
        logger.error("Paid AI request refused: read-only mode (%s).", decision.reason)
        raise _paid_access_error("access_unavailable")
    if not decision.granted:
        logger.info("Paid AI request refused: access required (%s).", decision.reason)
        raise _paid_access_error("access_required")
    try:
        origins = access_policy.public_origins(settings)
    except access_policy.PublicOriginsError:
        logger.error("Paid AI request refused: PUBLIC_ORIGINS is invalid.")
        raise _paid_access_error("access_unavailable") from None
    if not access_policy.origin_allowed(request, origins):
        logger.warning("Paid AI request refused: origin not allowed (code_ref=%s).", decision.code_ref)
        raise _paid_access_error("origin_not_allowed")
    if json_body and not access_policy.is_json_content_type(request):
        raise _paid_access_error("unsupported_media_type")


class AccessEnforcedRoute(APIRoute):
    """A paid AI route: flag, access, origin and content type before the body is read.

    Subclasses set ``feature_flag`` (a settings attribute; None if the route
    has none) and ``json_body`` (True to require ``application/json``).
    """

    feature_flag: str | None = None
    json_body: bool = False

    def get_route_handler(self):
        handler = super().get_route_handler()
        feature_flag, json_body = self.feature_flag, self.json_body

        async def route_handler(request):
            if feature_flag is not None and getattr(settings, feature_flag) is not True:
                raise HTTPException(status_code=404, detail="Not Found")
            _authorize_paid_request(request, json_body=json_body)
            return await handler(request)

        return route_handler


class ChatRoute(AccessEnforcedRoute):
    json_body = True


class SessionCreateRoute(AccessEnforcedRoute):
    feature_flag = "sessions_enabled"  # its body rule (empty or {}) is checked by the handler


class SessionDeleteRoute(AccessEnforcedRoute):
    pass  # no flag (204 while sessions are off, as before) and no body


class TranscriptionRoute(AccessEnforcedRoute):
    feature_flag = "voice_transcription_enabled"  # multipart, checked by the handler


def chat(
    payload: ChatRequest,
    session_token: str | None = Depends(_session_token),
    request: Request = None,
):
    """Answer a question using the RAG workflow and record an audit event.

    Without ``X-Session-ID`` the behaviour is exactly as before (stateless).
    With quotas enabled (S1-4b), the question is admitted first (429/503).
    """
    request_id = str(uuid4())
    question = payload.question.strip()

    # S1-5: on both paths, before the session lease, quota admission or provider.
    _recheck_paid_request(request)

    if session_token is not None:
        return _chat_with_session(question, request_id, session_token, request)

    ticket = _admit_chat(request, request_id)
    try:
        return _chat_stateless(question, request_id)
    finally:
        _release_ticket(ticket)  # ask() has returned (or raised): no provider work is left


def _chat_stateless(question: str, request_id: str) -> dict:
    """Stateless /api/chat: ask() → response → audit (unchanged by S1-4b)."""
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


# Registered with add_api_route: the route-class override is not available
# on the @router.post decorator.
router.add_api_route("/chat", chat, methods=["POST"], route_class_override=ChatRoute)


def require_streaming_enabled() -> None:
    """Raise 404 while the streaming endpoint is disabled."""
    if not settings.stream_enabled:
        raise HTTPException(status_code=404, detail="Not Found")


class StreamingFlagRoute(ChatRoute):
    """Checks the streaming flag before FastAPI reads or parses the body.

    A disabled endpoint answers 404 for any request body, including invalid
    JSON. When enabled, access, the origin and the content type are checked
    (S1-5, AccessEnforcedRoute), then the request goes to FastAPI's normal
    handler, so body parsing and validation errors are unchanged.
    """

    feature_flag = "stream_enabled"


async def _chat_stream_with_session(question: str, request_id: str, token: str, request: Request | None = None):
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

    # From here until service.start() takes the reservation, every failure
    # releases what this request owns: the session lease (unsaved), then the
    # capacity slot, whose release also releases an attached quota ticket.
    lease = runtime = ticket = None
    attached = handed_over = False
    try:
        # Shielded: once a lease is taken in the worker thread, this request
        # must not be cancelled before handing it to the run (or releasing it).
        with anyio.CancelScope(shield=True):
            runtime, lease = await anyio.to_thread.run_sync(
                _acquire_session_lease, token, request_id
            )
        # S1-4b: after the stream capacity and the session checks, before the
        # worker starts. Released with the stream's slot, only once both the
        # worker and the consumer are done.
        ticket = _admit_chat(request, request_id)
        if ticket is not None:
            reservation.lease.add_release_callback(ticket.release)
            attached = True
        history = [{"question": turn.question, "answer": turn.answer} for turn in lease.history]
        turns_used = len(history_window(history))
        session = streaming.StreamSession(
            manager=runtime.manager, lease=lease, history=history, history_turns_used=turns_used
        )
        handed_over = True  # start() owns the reservation now: it releases it if it can't start
        run = service.start(reservation, question, request_id, asyncio.get_running_loop(), session=session)
    except BaseException:
        if not handed_over:
            try:
                if lease is not None:
                    with anyio.CancelScope(shield=True):
                        await anyio.to_thread.run_sync(runtime.manager.finish, lease)  # not saved
            finally:
                service.cancel(reservation)  # even if releasing the lease failed
                if ticket is not None and not attached:
                    _release_ticket(ticket)
        raise
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


async def chat_stream(
    payload: ChatRequest,
    session_token: str | None = Depends(_session_token),
    request: Request = None,
):
    """Answer a question as Server-Sent Events (controlled prototype, off by default).

    Without ``X-Session-ID`` the behaviour is exactly as before (stateless).
    With quotas enabled (S1-4b): stream capacity first (503), then the quota
    (429/503), then the worker starts.
    """
    request_id = str(uuid4())
    question = payload.question

    # S1-5: before any stream capacity, session lease, quota or worker, on both paths.
    _recheck_paid_request(request)

    if session_token is not None:
        return await _chat_stream_with_session(question, request_id, session_token, request)

    if not settings.quota_enabled:
        run = streaming.get_stream_service().start_run(
            question, request_id, asyncio.get_running_loop()
        )
    else:
        service = streaming.get_stream_service()
        reservation = service.reserve()
        if reservation is None:
            return streaming.busy_response(settings.stream_retry_after_seconds)
        ticket = None
        try:
            ticket = _admit_chat(request, request_id)
            # Released with the stream's slot: only after both the worker and
            # the consumer are done (also on a failed start or a cancel).
            reservation.lease.add_release_callback(ticket.release)
            ticket = None  # owned by the stream's slot now
        except BaseException:
            try:
                service.cancel(reservation)
            finally:
                _release_ticket(ticket)  # admitted but never attached
            raise
        run = service.start(reservation, question, request_id, asyncio.get_running_loop())
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


async def create_session(request: Request, response: Response):
    """Start a conversation session (design §10, §14.3).

    404 while sessions are disabled; 415 unless the body is empty or ``{}``
    with ``Content-Type: application/json``; 403 for cross-origin requests
    (all checked before any database access); 503 if storage is unavailable.
    The token is returned once, never stored or logged. The database work
    runs in a worker thread, never on the event loop.
    """
    if not settings.sessions_enabled:  # also checked first by the route class
        raise HTTPException(status_code=404, detail="Not Found")
    _recheck_paid_request(request)  # S1-5: before the body is read or a session created
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


def delete_session(session_token: str | None = Depends(_session_token), request: Request = None):
    """Clear conversation memory (design §10.1).

    204 for a deleted, unknown, expired or already-deleted session, and while
    sessions are disabled; 400 for a missing or malformed header; 503 if
    deletion can't be confirmed. Audit records are not affected.
    """
    _recheck_paid_request(request)  # S1-5: before anything is deleted
    if session_token is None or not session_service.is_well_formed_token(session_token):
        raise _session_error("invalid_session_id")
    runtime = session_service.get_session_runtime(settings)
    try:
        runtime.store.delete(session_token)
    except session_service.SessionStoreUnavailable as exc:
        raise _session_error("session_unavailable") from exc
    return Response(status_code=204)


router.add_api_route(
    "/sessions", create_session, methods=["POST"], status_code=201, route_class_override=SessionCreateRoute
)
router.add_api_route(
    "/sessions", delete_session, methods=["DELETE"], status_code=204, route_class_override=SessionDeleteRoute
)


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
    "unsupported_audio": (415, "Unsupported audio. Send a 16 kHz mono 16-bit PCM WAV recording."),
    "invalid_audio": (400, "The audio is not a valid 16 kHz mono 16-bit PCM WAV recording."),
    "audio_too_long": (413, "Recordings must be 60 seconds or shorter."),
    "voice_busy": (503, "Too many recordings are being sent right now. Please try again shortly."),
    "audio_not_transcribable": (422, "The audio could not be transcribed."),
    "transcription_timeout": (504, "Transcription took too long. Please try again."),
    "transcription_unavailable": (503, "Transcription is temporarily unavailable."),
    "transcription_failed": (500, "Transcription failed."),
}
NO_STORE = {"Cache-Control": "no-store"}


VOICE_BUSY_RETRY_AFTER_SECONDS = 5


def _voice_error(code: str) -> HTTPException:
    status_code, message = VOICE_ERRORS[code]
    headers = dict(NO_STORE)
    if code == "voice_busy":
        headers["Retry-After"] = str(VOICE_BUSY_RETRY_AFTER_SECONDS)
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
        headers=headers,
    )


# ---------------------------------------------------------------------------
# Security S1-4c: transcription quotas and provider-task ownership
#
# A recording's exact duration is measured from the canonical WAV before
# anything else (app/services/audio_duration.py); only then is it admitted,
# reserving ceil(duration) seconds. The provider call runs as its own task,
# which owns the quota ticket once it and its completion callback are
# registered: the ticket is released only when the task is done (completed,
# failed, timed out or cancelled), never while our provider request is open.
# A client disconnect doesn't cancel the handler; a cancelled handler (for
# example at shutdown) stops waiting, but the task keeps its ticket until it
# ends, bounded by a hard timeout. Admitted quota is never refunded.
# ---------------------------------------------------------------------------

VOICE_UPLOAD_SLOTS = 2  # uploads read, checked or transcribed at once (each up to 2 MiB in memory)
PROVIDER_HARD_TIMEOUT_MARGIN_SECONDS = 5.0
# One upload slot per identity (a verified access code, S1-5), and VOICE_UPLOAD_SLOTS in all: no single client or
# bucket can hold every slot, for example with stalled uploads that never
# reach the quota. The identity is resolved before any of the body is read.
_voice_upload_identities: set = set()
_voice_uploads_lock = threading.Lock()
_transcription_tasks: set = set()  # strong references: a running provider task is never garbage-collected

TRANSCRIPTION_QUOTA_ERRORS = {
    quota_service.DAILY_LIMIT: ("quota_daily_limit", {
        quota_service.CODE_SCOPE: "You've reached today's recording limit. Please try again tomorrow.",
        quota_service.GLOBAL_SCOPE: "The service has reached today's recording limit. Please try again tomorrow.",
    }),
    quota_service.MINUTES_LIMIT: ("quota_minutes_limit", {
        quota_service.CODE_SCOPE: "You've used today's recording time. Please try again tomorrow.",
        quota_service.GLOBAL_SCOPE: "The service has used today's recording time. Please try again tomorrow.",
    }),
    quota_service.BUSY: ("quota_busy", {
        quota_service.CODE_SCOPE: "Your previous recording is still being transcribed. Please wait for it to finish.",
        quota_service.GLOBAL_SCOPE: "Another recording is being transcribed. Please try again shortly.",
    }),
    quota_service.CAPACITY: ("quota_capacity", {
        quota_service.GLOBAL_SCOPE: "The service is at capacity. Please try again later.",
    }),
}


@contextmanager
def _voice_upload_slot(identity: str):
    """This identity's single upload slot, or ``voice_busy`` at once (never waits).

    Refused if the identity already has an upload in progress, or if every
    slot is taken. Acquired and released under one lock; released on every
    exit (success, error, timeout or cancellation).
    """
    with _voice_uploads_lock:
        if identity in _voice_upload_identities or len(_voice_upload_identities) >= VOICE_UPLOAD_SLOTS:
            raise transcription.VoiceUploadError("voice_busy")
        _voice_upload_identities.add(identity)
    try:
        yield
    finally:
        with _voice_uploads_lock:
            _voice_upload_identities.discard(identity)


def _check_transcription_deployment() -> None:
    """Checked per request too (fails closed): production needs quotas and access codes."""
    try:
        quota_runtime.check_transcription_deployment(settings)
    except quota_service.QuotaConfigError:
        logger.error("Voice transcription refused: not available in this deployment (production guard).")
        raise transcription.VoiceUploadError("transcription_unavailable") from None


def _admit_transcription(
    identity: str, code_ref: str | None, request_id: str, seconds: int
) -> "quota_service.Ticket | None":
    """Admit one measured recording (None while quotas are disabled); raises 413, 429 or 503.

    ``identity`` is the one resolved before the upload was read (a verified
    access code; S1-5), so the slot and the quota belong to the same identity.
    """
    if not settings.quota_enabled:
        _require_paid_configuration()  # quotas off: always refused outside the test shim
        return None
    busy = settings.quota_busy_retry_after_seconds
    try:
        result = quota_runtime.get_quota_engine(settings).admit(
            quota_service.TRANSCRIPTION, identity, seconds=seconds
        )
    except Exception as exc:  # noqa: BLE001 - fail closed; logged by class only
        logger.error("Transcription quota unavailable for request_id=%s (%s).", request_id, type(exc).__name__)
        raise _quota_error(503, "quota_unavailable", QUOTA_UNAVAILABLE_MESSAGE, busy) from None
    if isinstance(result, quota_service.Ticket):
        return result
    if result.reason == quota_service.REQUEST_TOO_LONG:
        raise _voice_error("audio_too_long")
    known = TRANSCRIPTION_QUOTA_ERRORS.get(result.reason)
    if known is None:  # unavailable (or anything unexpected): fail closed
        logger.error("Transcription quota unavailable for request_id=%s (%s).", request_id, result.reason)
        raise _quota_error(503, "quota_unavailable", QUOTA_UNAVAILABLE_MESSAGE, result.retry_after or busy)
    code, messages = known
    message = messages.get(result.scope) or messages[quota_service.GLOBAL_SCOPE]
    logger.warning(
        "Transcription refused by quota for request_id=%s: reason=%s scope=%s code_ref=%s",
        request_id, result.reason, result.scope, code_ref or "-",
    )
    raise _quota_error(429, code, message, result.retry_after or busy)


async def _provider_transcription(upload, api_key: str, ticket) -> str:
    """The provider call, run as its own task: bounded by a hard timeout, then settled."""
    timeout = settings.voice_transcription_timeout_seconds
    try:
        async with asyncio.timeout(timeout + PROVIDER_HARD_TIMEOUT_MARGIN_SECONDS):
            result = await transcription.transcribe(
                upload,
                api_key=api_key,
                model=settings.voice_transcription_model,
                timeout_seconds=timeout,
                max_retries=settings.voice_transcription_max_retries,
            )
    except TimeoutError:
        logger.warning("Voice transcription stopped at its hard timeout.")
        raise transcription.VoiceUploadError("transcription_timeout") from None
    if ticket is not None:
        # Missing: the exact local measurement (adds nothing). Anything else
        # goes to the engine as is: longer is charged, implausible or
        # malformed is charged its cap; nothing is ever refunded.
        reported = upload.seconds if result.duration is None else result.duration
        try:
            ticket.settle(reported)
        except Exception as exc:  # noqa: BLE001 - the release below still happens
            logger.error("Transcription quota settlement failed (%s).", type(exc).__name__)
    return result.text


def _transcription_finished(ticket):
    def finished(task) -> None:
        _transcription_tasks.discard(task)
        _release_ticket(ticket)  # the provider task is done: nothing of ours is in flight
        if not task.cancelled():
            task.exception()  # retrieved, so an unawaited failure is never reported as lost
    return finished


def _start_provider_task(upload, api_key: str, ticket) -> "asyncio.Task":
    """Start the provider task and hand it the ticket (registered before it can run)."""
    task = asyncio.get_running_loop().create_task(_provider_transcription(upload, api_key, ticket))
    try:
        _transcription_tasks.add(task)
        task.add_done_callback(_transcription_finished(ticket))
    except BaseException:
        # Nothing has awaited since the task was created, so it hasn't run a
        # single step: cancelling it now means no provider call ever starts,
        # and the caller (who still owns the ticket) releases it.
        _transcription_tasks.discard(task)
        task.cancel()
        raise
    return task


async def voice_transcribe(request: Request, response: Response):
    """Transcribe one short recording to text (Voice V2; Security S1-4c).

    404 (the same body as an unknown route) while
    ``VOICE_TRANSCRIPTION_ENABLED`` is off, the default. The request is
    ``multipart/form-data`` with one file part named ``audio``: a canonical
    16 kHz mono 16-bit PCM WAV of at most 60 seconds (and 2 MiB). Returns
    ``{"text": "..."}``; the transcript is only returned, never stored,
    audited or sent to the chat. In order, before any of the body is read:
    the flag (404), verified access (503/401) and the origin (403), both
    checked by the route class (S1-5), the production requirements (503),
    the content type (415), the declared length (413), the provider key
    (503) and an upload slot: one per identity, two in all (503
    ``voice_busy``). Then the bounded read
    (408, 413), validation and the measured duration (400, 413, 415), the
    quota (413, 429, 503) and the provider call (422, 503, 504).

    The handler takes only ``request``, so FastAPI never parses or buffers
    the body itself (``File(...)`` would spool the whole upload first).
    """
    if not settings.voice_transcription_enabled:  # also checked first by the route class
        raise HTTPException(status_code=404, detail="Not Found")
    _recheck_paid_request(request)  # S1-5: before the upload slot and any body read

    request_id = str(uuid4())
    content_type = request.headers.get("content-type")
    ticket = None  # owned by this handler until the provider task takes it
    try:
        _check_transcription_deployment()
        transcription.multipart_boundary(content_type)
        transcription.check_declared_length(request.headers.get("content-length"))
        api_key = settings.openai_api_key.get_secret_value()
        if not api_key:
            logger.error("Voice transcription is enabled but OPENAI_API_KEY is not set.")
            raise transcription.VoiceUploadError("transcription_unavailable")

        # The verified access decision (S1-5): nothing else the client sends is trusted.
        identity, code_ref = _quota_identity(request)
        with _voice_upload_slot(identity):
            body = await transcription.read_bounded_body(request.stream())
            upload = transcription.parse_audio_upload(body, content_type)  # measures the duration
            del body
            ticket = _admit_transcription(identity, code_ref, request_id, upload.reserved_seconds)
            task = _start_provider_task(upload, api_key, ticket)
            ticket = None  # handed over: released by the task's completion callback only
            text = await asyncio.shield(task)  # a cancelled handler stops waiting; the task goes on
    except HTTPException:
        raise
    except transcription.VoiceUploadError as exc:
        raise _voice_error(exc.code) from None
    except ClientDisconnect:
        raise _voice_error("invalid_upload") from None
    except Exception as exc:  # noqa: BLE001 - never echo details; log the class only
        logger.error("Voice transcription failed unexpectedly (%s).", type(exc).__name__)
        raise _voice_error("transcription_failed") from None
    finally:
        _release_ticket(ticket)  # only a ticket never handed to a provider task

    response.headers["Cache-Control"] = "no-store"
    return {"text": text}


router.add_api_route(
    "/voice/transcribe", voice_transcribe, methods=["POST"], route_class_override=TranscriptionRoute
)


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


# ---------------------------------------------------------------------------
# Security S1-3: access codes (Architecture C)
#
# Redeems an access code for a signed, HttpOnly, SameSite=Strict cookie and
# reports or ends that access. Nothing else requires access yet. All three
# routes are 404 (like an unknown route) while ACCESS_ENABLED is off, the
# default. Errors are generic, and codes, cookies, secrets and client
# addresses are never logged (see app/services/access.py).
# ---------------------------------------------------------------------------

ACCESS_ERRORS = {
    "access_denied": (401, "Invalid or expired access code."),
    "access_throttled": (429, "Too many attempts. Please try again later."),
    "access_unavailable": (503, "Access is temporarily unavailable."),
    "origin_not_allowed": (403, "Request origin not allowed."),
    "unsupported_media_type": (415, "Use application/json."),
    "invalid_request": (400, 'Send {"code": "<access code>"}.'),
}


def _access_error(code: str, headers: dict[str, str] | None = None) -> HTTPException:
    status_code, message = ACCESS_ERRORS[code]
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
        headers={**NO_STORE, **(headers or {})},
    )


def _access_config() -> access_service.AccessConfig:
    """Validated per request too: settings changed at runtime must not weaken it (fails closed)."""
    try:
        return access_service.AccessConfig.from_settings(settings)
    except access_service.AccessConfigError:
        logger.error("Access codes are enabled but their settings are invalid.")
        raise _access_error("access_unavailable") from None


def _check_access_origin(request: Request) -> None:
    """Refuse cross-site browser requests; a form can POST without a preflight."""
    fetch_site = request.headers.get("sec-fetch-site")
    if fetch_site is not None and fetch_site.strip().lower() != "same-origin":
        raise _access_error("origin_not_allowed")
    origin = request.headers.get("origin")
    if origin is not None and not _origin_matches_host(origin, request.headers.get("host", "")):
        raise _access_error("origin_not_allowed")


async def _read_body_up_to(request: Request, limit: int) -> bytes | None:
    """The request body, or None if it is longer than ``limit`` (never read further)."""
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _redeemed_code(body: bytes | None) -> str | None:
    """The code from exactly ``{"code": "<string>"}``, or None."""
    if body is None:
        return None
    try:
        data = json.loads(body)
    except ValueError:
        return None
    if not isinstance(data, dict) or set(data) != {"code"} or not isinstance(data["code"], str):
        return None
    return data["code"].strip()


@router.post("/access")
async def redeem_access(request: Request, response: Response):
    """Exchange an access code for a signed access cookie (Security S1-3).

    In order: the flag (404), the settings (503), the origin (403), a
    trustworthy client address (503), the failed-attempt throttle (429 with
    Retry-After), then the content type (415), the bounded body (400) and the
    code (401). Every attempt that gets past the throttle counts as a failure
    unless it succeeds. Success: 200 with ``{"authenticated": true,
    "expires_at"}`` and the cookie; the code itself is never returned.
    """
    if not settings.access_enabled:
        raise HTTPException(status_code=404, detail="Not Found")
    config = _access_config()
    _check_access_origin(request)

    client = access_service.client_key(
        request.client.host if request.client else None, request.headers, config.client_ip_header
    )
    if client is None:
        logger.warning("Access code redemption refused: no trustworthy client address.")
        raise _access_error("access_unavailable")

    throttle = access_service.get_throttle(settings)
    retry_after, stamp = throttle.attempt(client)
    if retry_after is not None:
        logger.warning("Access code rejected (throttled).")
        raise _access_error("access_throttled", {"Retry-After": str(math.ceil(retry_after))})

    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        raise _access_error("unsupported_media_type")
    candidate = _redeemed_code(await _read_body_up_to(request, access_service.MAX_REDEEM_BODY_BYTES))
    if candidate is None:
        raise _access_error("invalid_request")

    now = access_service.wall_clock()
    code = access_service.match_code(config, candidate, now)
    if code is None:
        logger.warning("Access code rejected (invalid or expired).")
        raise _access_error("access_denied")

    throttle.succeeded(client, stamp)
    value, expires_at = access_service.issue_cookie(config, code, now)
    response.set_cookie(
        config.cookie_name,  # __Host-bp_access when Secure (no Domain attribute is ever set)
        value,
        max_age=max(0, expires_at - int(now)),
        expires=datetime.fromtimestamp(expires_at, tz=timezone.utc),
        path="/",
        secure=config.secure,
        httponly=True,
        samesite="strict",
    )
    response.headers["Cache-Control"] = "no-store"
    logger.info("Access granted: code_ref=%s", access_service.code_reference(config, code.code_id))
    return {"authenticated": True, "expires_at": _iso_utc(expires_at)}


@router.get("/access/status")
async def access_status(request: Request, response: Response):
    """Whether this browser holds a valid access cookie (never 401)."""
    if not settings.access_enabled:
        raise HTTPException(status_code=404, detail="Not Found")
    config = _access_config()
    response.headers["Cache-Control"] = "no-store"
    grant = access_service.verify_cookie(
        config, request.cookies.get(config.cookie_name), access_service.wall_clock()
    )
    if grant is None:
        return {"authenticated": False}
    return {"authenticated": True, "expires_at": _iso_utc(grant.expires_at)}


@router.post("/access/logout", status_code=204)
async def access_logout(request: Request):
    """End access in this browser: the cookie is deleted (idempotent)."""
    if not settings.access_enabled:
        raise HTTPException(status_code=404, detail="Not Found")
    _check_access_origin(request)
    response = Response(status_code=204, headers=NO_STORE)
    # Both names, each with the attributes a browser requires to replace it.
    for secure in (True, False):
        response.delete_cookie(
            access_service.cookie_name(secure),
            path="/",
            secure=secure,
            httponly=True,
            samesite="strict",
        )
    return response


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
