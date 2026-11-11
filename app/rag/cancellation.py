"""Cooperative stop checks before external calls (H1-3a).

A run that may need to stop early (a streamed question, or a session
``/api/chat`` question) installs a stop check with ``stop_checks()``. Code
that is about to start an external call (OpenAI, Pinecone, Tavily) calls
``raise_if_stopped()`` immediately before it, which raises
``WorkflowStopped`` once the run should stop, so no new external call
starts.

This never interrupts a call that is already running: a synchronous call
in progress finishes (with its own timeouts and retries) before the next
check runs. Outside a ``stop_checks()`` scope (stateless ``/api/chat``,
ingestion) every check does nothing.

The check is kept in a ``ContextVar``. LangGraph and LangChain copy the
context into the threads that run graph nodes, so nodes running in
background threads see the run's check.
"""

import logging
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator

logger = logging.getLogger(__name__)

# Stop reasons set here; streams add their own (for example "timeout" and
# "disconnected", see app/api/streaming.py).
SERVER_SHUTDOWN = "server_shutdown"
SESSION_TURN_EXPIRED = "session_turn_expired"
SESSION_LEASE_LOST = "session_lease_lost"
CHAT_DEADLINE = "chat_deadline"  # H1-5 (B): stateless /api/chat past CHAT_DEADLINE_SECONDS

StopCheck = Callable[[], "str | None"]


def monotonic() -> float:
    """The clock deadline checks read (a seam for tests)."""
    return time.monotonic()


def deadline_check(seconds: float, reason: str) -> StopCheck:
    """A stop check that returns ``reason`` once ``seconds`` have passed from now.

    Cooperative like every check here: it stops the run before its next
    external call, never during one.
    """
    deadline = monotonic() + seconds

    def check() -> str | None:
        return reason if monotonic() >= deadline else None

    return check


_check: ContextVar[StopCheck | None] = ContextVar("workflow_stop_check", default=None)
_shutdown = threading.Event()


class WorkflowStopped(Exception):
    """Raised before an external call once the run should stop."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@contextmanager
def stop_checks(check: StopCheck) -> Iterator[None]:
    """Install ``check`` for the current run; it returns a stop reason or None."""
    token = _check.set(check)
    try:
        yield
    finally:
        _check.reset(token)


def current_check() -> StopCheck | None:
    """The check installed for the current context, if any."""
    return _check.get()


def signal_shutdown() -> None:
    """Application shutdown: runs with a stop check start no further external calls."""
    _shutdown.set()


def reset_shutdown() -> None:
    """Application startup: clear a shutdown signal from an earlier lifespan."""
    _shutdown.clear()


def shutdown_signalled() -> bool:
    return _shutdown.is_set()


def raise_if_stopped(operation: str) -> None:
    """Call immediately before an external call; raises ``WorkflowStopped`` to skip it."""
    check = _check.get()
    if check is None:
        return
    reason = SERVER_SHUTDOWN if _shutdown.is_set() else check()
    if reason is not None:
        logger.info("Skipping %s: the run is stopping (%s).", operation, reason)
        raise WorkflowStopped(reason)
