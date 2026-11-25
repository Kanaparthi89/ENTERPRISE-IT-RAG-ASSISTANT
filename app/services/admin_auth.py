"""Admin authentication and upload limits for POST /api/ingest (Security S1-5 Phase C).

Used by ``routes.AdminIngestRoute`` before FastAPI reads or parses the body:

* ``key_status(settings)``: the configured ADMIN_API_KEY is ``missing``,
  ``too_short`` (fewer than MIN_ADMIN_KEY_LENGTH characters) or ``ok``.
  Ingestion is refused (503) unless it is ``ok``; the rest of the
  application is unaffected. The key is never logged or returned.
* ``presented_key(request)`` and ``keys_match()``: exactly one
  ``X-Admin-Key`` of printable ASCII, compared as SHA-256 digests with
  ``hmac.compare_digest`` (equal lengths, bytes: no length leak, no
  TypeError on unusual input).
* ``client_identity()`` and ``get_admin_throttle()``: a failed-attempt
  throttle separate from the access-code one. The client is identified as
  for access codes (a verified platform header, or the socket peer; never
  forwarded headers); a request that can't be identified shares one bucket.
  In memory, per process: it resets on restart and is not shared between
  workers (one uvicorn worker today).
* ``bounded_receive()``: caps the bytes read from the client and the wait
  for each chunk (413 / 408).
* ``try_claim_ingest_slot()``: one ingestion at a time, claimed atomically
  without waiting. The slot stays held until the ingestion's worker thread
  has finished, even if the request is cancelled or the client disconnects.
"""

import asyncio
import concurrent.futures
import hashlib
import hmac
import threading

import anyio
from fastapi import HTTPException

from app.services import access

MIN_ADMIN_KEY_LENGTH = 32
MAX_ADMIN_KEY_HEADER_LENGTH = 256
MULTIPART_OVERHEAD_BYTES = 64 * 1024  # boundaries and part headers around the file
UNIDENTIFIED_CLIENT = "*unidentified"  # one shared bucket when a client can't be identified

MISSING = "missing"
TOO_SHORT = "too_short"
OK = "ok"


def key_status(settings) -> str:
    value = settings.admin_api_key
    secret = value.get_secret_value() if hasattr(value, "get_secret_value") else (value or "")
    if not secret:
        return MISSING
    if len(secret) < MIN_ADMIN_KEY_LENGTH:
        return TOO_SHORT
    return OK


def expected_key(settings) -> str:
    value = settings.admin_api_key
    return value.get_secret_value() if hasattr(value, "get_secret_value") else str(value)


def presented_key(request) -> str | None:
    """The single X-Admin-Key header, or None if absent, repeated or malformed."""
    values = request.headers.getlist("x-admin-key")
    if len(values) != 1:
        return None
    value = values[0]
    if not 0 < len(value) <= MAX_ADMIN_KEY_HEADER_LENGTH:
        return None
    if not all(0x21 <= ord(character) <= 0x7E for character in value):
        return None
    return value


def keys_match(presented: str, expected: str) -> bool:
    return hmac.compare_digest(
        hashlib.sha256(presented.encode("utf-8")).digest(),
        hashlib.sha256(expected.encode("utf-8")).digest(),
    )


def client_identity(request, settings) -> str:
    """Who is trying, for the throttle only. Forwarded headers are never trusted."""
    try:
        header = access.check_client_ip_header(settings.access_client_ip_header)
    except access.AccessConfigError:
        return UNIDENTIFIED_CLIENT
    peer = request.client.host if request.client else None
    return access.client_key(peer, request.headers, header) or UNIDENTIFIED_CLIENT


# ---------------------------------------------------------------------------
# Failed-attempt throttle
# ---------------------------------------------------------------------------

_throttle: access.FailureThrottle | None = None
_throttle_limits: tuple | None = None
_throttle_lock = threading.Lock()


def get_admin_throttle(settings) -> access.FailureThrottle:
    """The process-wide admin throttle, rebuilt only if its limits changed."""
    global _throttle, _throttle_limits
    limits = (
        settings.admin_max_failed_attempts,
        settings.admin_failure_window_seconds,
        settings.admin_max_global_failed_attempts,
    )
    with _throttle_lock:
        if _throttle is None or _throttle_limits != limits:
            _throttle = access.FailureThrottle(limits[0], limits[1], limits[2])
            _throttle_limits = limits
        return _throttle


def reset_admin_throttle() -> None:
    """Forget the throttle. Tests only."""
    global _throttle, _throttle_limits
    with _throttle_lock:
        _throttle, _throttle_limits = None, None


# ---------------------------------------------------------------------------
# Bounded body
# ---------------------------------------------------------------------------

def bounded_receive(receive, limit: int, idle_timeout: float, too_large, timed_out):
    """Wrap an ASGI ``receive``: at most ``limit`` body bytes, each chunk within ``idle_timeout``.

    ``too_large()`` and ``timed_out()`` return the HTTPException to raise
    (FastAPI passes an HTTPException from body parsing through unchanged).
    """
    received = 0

    async def wrapped():
        nonlocal received
        message = None
        with anyio.move_on_after(idle_timeout):
            message = await receive()
        if message is None:
            raise timed_out()
        if message.get("type") == "http.request":
            received += len(message.get("body", b""))
            if received > limit:
                raise too_large()
        return message

    return wrapped


# ---------------------------------------------------------------------------
# One ingestion at a time
# ---------------------------------------------------------------------------

_ingest_slot = threading.Lock()
_worker_threads: set = set()
_worker_threads_lock = threading.Lock()


class SlotClaim:
    """This request's hold on the ingest slot, until released by it or by its worker thread.

    Owned by the request until ``run_in_thread`` hands it to the worker
    thread, which releases it when the work finishes (on every path). A
    request that never handed it over releases it with ``release_if_unclaimed``.
    """

    def __init__(self):
        self._guard = threading.Lock()
        self._owner = "request"

    def release_if_unclaimed(self) -> None:
        with self._guard:
            if self._owner != "request":
                return
            self._owner = "released"
        _ingest_slot.release()

    async def run_in_thread(self, work):
        """Run ``work()`` on a worker thread that owns the slot; return its result.

        If this coroutine is cancelled, the thread keeps running (and keeps
        the slot) until ``work()`` returns or raises.
        """
        future: concurrent.futures.Future = concurrent.futures.Future()

        def run():
            try:
                result = work()
            except BaseException as exc:  # noqa: BLE001 - handed to the waiting request
                future.set_exception(exc)
            else:
                future.set_result(result)
            finally:
                with self._guard:
                    self._owner = "released"
                _ingest_slot.release()
                with _worker_threads_lock:
                    _worker_threads.discard(threading.current_thread())

        with self._guard:
            if self._owner != "request":
                raise RuntimeError("The ingest slot is no longer held by this request.")
            self._owner = "thread"
        thread = threading.Thread(target=run, name="ingest-worker", daemon=True)
        with _worker_threads_lock:
            _worker_threads.add(thread)
        try:
            thread.start()
        except BaseException:
            with _worker_threads_lock:
                _worker_threads.discard(thread)
            with self._guard:
                self._owner = "request"  # never started: the request still releases it
            raise
        waiter = asyncio.wrap_future(future)
        # A cancelled request stops waiting; retrieve the outcome so it is never reported as lost.
        waiter.add_done_callback(lambda done: done.cancelled() or done.exception())
        return await asyncio.shield(waiter)


def try_claim_ingest_slot() -> SlotClaim | None:
    """Claim the single ingest slot atomically, without waiting; None if it is taken."""
    if not _ingest_slot.acquire(blocking=False):
        return None
    return SlotClaim()


def ingest_slot_taken() -> bool:
    return _ingest_slot.locked()


def worker_threads() -> list:
    with _worker_threads_lock:
        return list(_worker_threads)


def http_error(status_code: int, code: str, message: str, headers: dict | None = None) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
        headers={"Cache-Control": "no-store", **(headers or {})},
    )
