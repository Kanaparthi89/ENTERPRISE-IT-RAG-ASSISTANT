"""Streaming infrastructure for the chat SSE endpoint (Phase A1).

Connects the synchronous LangGraph workflow, run on a dedicated worker pool,
to an asynchronous consumer on the event loop. The HTTP/SSE layer that uses
it (``SSEResponse`` and ``POST /api/chat/stream``) is added separately.

Components:

* ``EventChannel``: bounded hand-off between worker threads and the event
  loop: up to 256 normal events plus one reserved terminal slot, with
  backpressure, thread-safe wake-ups, and abort/finish semantics.
* ``StreamOutcome``: the first claimed outcome wins and is final.
* ``StreamCapacity`` / ``StreamLease``: admission control; a slot is released
  exactly once, after both the worker and the consumer are done.
* ``StreamRun``: one streamed request: worker lifecycle, deadline and grace
  periods, stop signalling, and exactly-once audit finalization.
* ``StreamService``: the dedicated worker pool and active-run registry.

Synchronization rules:

* ``EventChannel`` has one lock (``_cond``). It is never held while calling
  out to the event loop. State changes (``_mark_aborted_locked``) and
  wake-ups (``_schedule_wake``) never call each other.
* ``asyncio.Event`` methods run only on the event-loop thread; worker threads
  wake the consumer with ``loop.call_soon_threadsafe``.
* Lock order is channel → lease; no code holds a lease lock while taking the
  channel lock.
"""

import asyncio
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, InvalidStateError, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import anyio
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse, StreamingResponse

from app.core.config import get_settings
from app.rag.cancellation import (
    SERVER_SHUTDOWN,
    SESSION_LEASE_LOST,
    SESSION_TURN_EXPIRED,
    WorkflowStopped,
    stop_checks,
)
from app.rag.workflow import stream_ask
from app.services.audit import write_audit

logger = logging.getLogger(__name__)

CHANNEL_MAX_EVENTS = 256
POLL_S = 0.25
TIMEOUT_ERROR_GRACE = 2.0
END_OF_BODY_GRACE = 2.0
ATTACH_GRACE = 2.0
FINALIZE_GRACE = 5.0  # TIMEOUT_ERROR_GRACE + END_OF_BODY_GRACE + 1 s margin

WORKER_THREAD_PREFIX = "bp-stream"

# Session streams (A2-6): after ``done`` the response waits at most this long,
# outside the deadline scope, for the worker's save result before sending the
# ``session`` event (design revision 8 §11.2).
SESSION_SAVE_REPORT_WAIT = 2.0


class _Sentinel:
    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return self._name


END = _Sentinel("END")
ABORTED = _Sentinel("ABORTED")


class StreamStopped(Exception):
    """Raised on a producer thread when the stream can no longer accept output."""


# H1-3a: outcomes for a cooperative stop (WorkflowStopped) that nobody else
# has claimed, by stop reason. A stop requested by the consumer, a backstop,
# or the stream shutdown is normally claimed by the requester first and is
# never reclaimed here. The deadline reasons cover a race: the check saw a
# due backstop, but it no longer applied when the worker tried it (for
# example, the response attached in between). The outcome then matches the
# deadline instead of a generic one.
WORKFLOW_STOP_OUTCOMES = {
    SERVER_SHUTDOWN: ("cancelled", "ServerShutdown", "server shutdown"),
    SESSION_TURN_EXPIRED: ("timeout", "SessionTurnExpired", "session turn limit reached"),
    SESSION_LEASE_LOST: ("failed", "SessionLeaseLost", "session lease lost"),
    "timeout": ("timeout", "TimeoutError", "deadline"),
    "abandon_due": ("timeout", "TimeoutError", "deadline before the response attached"),
    "consumer_unresponsive": ("timeout", "TimeoutError", "consumer unresponsive at deadline"),
    "loop_closed": ("cancelled", "LoopClosed", "event loop closed"),
}


# -------------------------------------------------------------------
# Event channel
# -------------------------------------------------------------------

class EventChannel:
    """Bounded hand-off from producer threads to the event-loop consumer.

    Producers (the worker thread, and the model thread via the answer-chunk
    forwarder) call ``publish``, ``set_terminal``, ``finish`` and ``abort``.
    The consumer coroutine calls ``receive_batch`` on the event-loop thread
    that constructed the channel.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        capacity: int = CHANNEL_MAX_EVENTS,
        should_stop: Callable[[], str | None] = lambda: None,
        poll_seconds: float = POLL_S,
    ):
        self._loop = loop
        self._loop_thread_id = threading.get_ident()
        self._capacity = capacity
        self._should_stop = should_stop
        self._poll_seconds = poll_seconds

        self._cond = threading.Condition()
        self._items: deque = deque()
        self._terminal: Any = None
        self._terminal_set = False
        self._terminal_delivered = False
        self._state = "open"  # open → finished | aborted
        self.abort_reason: str | None = None

        self._wake = asyncio.Event()  # event-loop thread only
        self._wake_pending = False

        # Test hook: runs on the loop thread between the locked check and the
        # wait in receive_batch, to make wake-up races deterministic.
        self._before_wait: Callable[[], None] | None = None

    # -- state inspection (tests and diagnostics) ---------------------

    @property
    def state(self) -> str:
        with self._cond:
            return self._state

    def __len__(self) -> int:
        with self._cond:
            return len(self._items)

    # -- internal primitives -----------------------------------------

    def _mark_aborted_locked(self, reason: str) -> bool:
        """Caller holds ``_cond``. State change only: never wakes the consumer."""
        first = self._state != "aborted"
        if first:
            self._state = "aborted"
            self.abort_reason = reason
            self._items.clear()
            self._terminal = None
        self._cond.notify_all()  # always release blocked publishers
        return first

    def _schedule_wake(self) -> bool:
        """Any thread, without ``_cond`` held. Wake-up only: never changes state.

        Returns True if a wake callback is pending or was scheduled on a live
        loop, False if the loop is closed (no wake is possible or needed).
        """
        if self._loop.is_closed():  # before the coalescing shortcut
            return False
        with self._cond:
            if self._wake_pending:
                return True
            self._wake_pending = True
        try:
            self._loop.call_soon_threadsafe(self._wake_callback)
            return True
        except RuntimeError:  # the loop closed after the check
            with self._cond:
                self._wake_pending = False
            return False

    def _wake_callback(self) -> None:
        """Runs on the event-loop thread only."""
        with self._cond:
            self._wake_pending = False
        self._wake.set()

    def _notify(self) -> bool:
        """After a producer's state change. False means the channel is unusable."""
        if not self._schedule_wake():
            with self._cond:
                self._mark_aborted_locked("loop_closed")
            return False
        with self._cond:
            return self._state != "aborted"

    # -- producer API --------------------------------------------------

    def publish(self, item) -> bool:
        """Append a normal event, blocking while the channel is full.

        True: enqueued and the channel was usable when the call returned.
        False: stop producing (not enqueued, or the channel became unusable).
        """
        aborted_here = False
        with self._cond:
            while len(self._items) >= self._capacity and self._state == "open":
                if threading.get_ident() == self._loop_thread_id:
                    # Waiting here would block the consumer that frees space.
                    raise RuntimeError(
                        "publish would block the event loop; call it from a producer thread."
                    )
                self._cond.wait(self._poll_seconds)
                if self._loop.is_closed():
                    self._mark_aborted_locked("loop_closed")
                elif (reason := self._should_stop()) is not None:
                    aborted_here = self._mark_aborted_locked(reason)
            if self._state != "open":
                enqueued = False
            else:
                self._items.append(item)
                enqueued = True
        if not enqueued:
            if aborted_here:
                self._schedule_wake()
            return False
        return self._notify()

    def set_terminal(self, item) -> bool:
        """Store the single terminal event; never blocks; ignores the limit."""
        with self._cond:
            if self._state != "open" or self._terminal_set:
                return False
            self._terminal = item
            self._terminal_set = True
        return self._notify()

    def finish(self) -> bool:
        """Producer side complete; queued and terminal events stay deliverable."""
        with self._cond:
            if self._state != "open" or not self._terminal_set:
                return False
            self._state = "finished"
        return self._notify()

    def abort(self, reason: str) -> bool:
        """Discard everything and wake both sides. Idempotent; never raises."""
        with self._cond:
            first = self._mark_aborted_locked(reason)
        self._schedule_wake()
        return first

    # -- consumer API (event-loop thread) -----------------------------

    async def receive_batch(self, max_wait: float):
        """Return queued normal events, then the terminal event, then END.

        Returns ABORTED once aborted, or ``[]`` when ``max_wait`` elapses.
        """
        if threading.get_ident() != self._loop_thread_id:
            raise RuntimeError("receive_batch must run on the event-loop thread.")

        while True:
            self._wake.clear()
            with self._cond:
                if self._state == "aborted":
                    return ABORTED
                if self._items:
                    batch = list(self._items)
                    self._items.clear()
                    self._cond.notify_all()  # space for blocked publishers
                    return batch
                if self._terminal_set and not self._terminal_delivered:
                    self._terminal_delivered = True
                    return [self._terminal]
                if self._state == "finished":
                    return END
            if self._before_wait is not None:
                self._before_wait()
            try:
                await asyncio.wait_for(self._wake.wait(), max_wait)
            except asyncio.TimeoutError:
                return []


# -------------------------------------------------------------------
# Outcome
# -------------------------------------------------------------------

@dataclass(frozen=True)
class Decision:
    status: str  # completed | failed | timeout | cancelled
    error_type: str | None = None
    marker: str | None = None


class StreamOutcome:
    """The first successful claim decides the outcome; it never changes."""

    STATUSES = frozenset({"completed", "failed", "timeout", "cancelled"})

    def __init__(self):
        self._lock = threading.Lock()
        self._decision: Decision | None = None
        self.decided = threading.Event()

    def claim(self, status: str, error_type: str | None = None, marker: str | None = None) -> bool:
        if status not in self.STATUSES:
            raise ValueError(f"Unknown outcome status: {status!r}")
        with self._lock:
            if self._decision is not None:
                return False
            self._decision = Decision(status, error_type, marker)
        self.decided.set()
        return True

    @property
    def decision(self) -> Decision | None:
        with self._lock:
            return self._decision


# -------------------------------------------------------------------
# Capacity and leases
# -------------------------------------------------------------------

class StreamCapacity:
    """Non-blocking admission control backed by a bounded semaphore."""

    def __init__(self, limit: int):
        self.limit = limit
        self._semaphore = threading.BoundedSemaphore(limit)
        self._lock = threading.Lock()
        self._in_use = 0
        self.acquired_total = 0
        self.released_total = 0

    @property
    def in_use(self) -> int:
        with self._lock:
            return self._in_use

    def try_acquire(self, on_release: Callable[[], None] | None = None) -> "StreamLease | None":
        if not self._semaphore.acquire(blocking=False):
            return None
        with self._lock:
            self._in_use += 1
            self.acquired_total += 1
        return StreamLease(self, on_release)

    def _release(self) -> None:
        with self._lock:
            self._in_use -= 1
            self.released_total += 1
        self._semaphore.release()  # BoundedSemaphore raises on over-release


class StreamLease:
    """One admitted stream. Released exactly once, after both parties finish.

    Also tracks whether a consumer attached (``pending`` → ``attached`` or
    ``abandoned``); ``attach`` and ``abandon`` are mutually exclusive.
    """

    def __init__(self, capacity: StreamCapacity, on_release: Callable[[], None] | None):
        self._capacity = capacity
        self._on_release = on_release
        self._lock = threading.Lock()
        self.worker_done = False
        self.consumer_done = False
        self.released = False
        self._attach_state = "pending"

    @property
    def attach_state(self) -> str:
        with self._lock:
            return self._attach_state

    def attach(self) -> bool:
        with self._lock:
            if self._attach_state != "pending":
                return False
            self._attach_state = "attached"
            return True

    def abandon(self) -> bool:
        with self._lock:
            if self._attach_state != "pending":
                return False
            self._attach_state = "abandoned"
            return True

    def mark_worker_done(self) -> bool:
        return self._mark("worker_done")

    def mark_consumer_done(self) -> bool:
        return self._mark("consumer_done")

    def _mark(self, flag: str) -> bool:
        with self._lock:
            setattr(self, flag, True)
            release = self.worker_done and self.consumer_done and not self.released
            if release:
                self.released = True
        if release:  # outside the lease lock
            self._capacity._release()
            if self._on_release is not None:
                self._on_release()
        return release


# -------------------------------------------------------------------
# One streamed request
# -------------------------------------------------------------------

HOOK_NAMES = (
    "trace_published",
    "terminal_set",
    "finished",
    "waiting_for_decision",
    "audit_written",
    "worker_exited",
)

CONSUMER_EXITS = {
    # exit kind: (status, error_type, marker)
    "disconnected": ("cancelled", "ClientDisconnected", "client disconnected"),
    "send_failed": ("cancelled", "SendFailed", "send failed"),
    "cancelled": ("cancelled", "Cancelled", "cancelled"),
    "consumer_error": ("failed", "ConsumerError", "consumer error"),
}


@dataclass
class StreamSession:
    """The conversation-session side of a run (A2-6).

    ``manager`` and ``lease`` come from ``app.services.sessions``; the lease
    was taken before the run started. ``result`` receives ``turn_saved``
    from the worker; the response waits for it (bounded) after ``done``.
    """

    manager: Any
    lease: Any
    history: list[dict[str, str]]
    history_turns_used: int
    result: Future = field(default_factory=Future)

    def report(self, turn_saved: bool) -> None:
        try:
            self.result.set_result(turn_saved)
        except InvalidStateError:
            pass  # already reported


class StreamRun:
    """A single streamed chat request.

    The worker thread runs the workflow, publishes events, waits for the
    consumer's decision on success, and writes exactly one audit row. The
    consumer (on the event loop) attaches, receives events, and claims
    ``completed`` only after sending ``done``. Either side can request a stop;
    the outcome is claimed first and the stop requested second.
    """

    def __init__(
        self,
        *,
        question: str,
        request_id: str,
        lease: StreamLease,
        loop: asyncio.AbstractEventLoop,
        timeout_seconds: float,
        clock: Callable[[], float],
        poll_seconds: float,
        workflow_stream: Callable[..., Iterator[tuple[str, Any]]],
        audit_writer: Callable[..., Any],
        session: StreamSession | None = None,
    ):
        self.question = question
        self.request_id = request_id
        self.lease = lease
        self.session = session
        self._loop = loop
        self._clock = clock
        self._poll_seconds = poll_seconds
        self._workflow_stream = workflow_stream
        self._audit_writer = audit_writer

        self.created_at = clock()
        self.deadline = self.created_at + timeout_seconds

        self.outcome = StreamOutcome()
        self._stop_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._stop_reason: str | None = None

        self.channel = EventChannel(
            loop, should_stop=self.stop_reason, poll_seconds=poll_seconds
        )

        self.response_started = False
        self.trace: list[str] = []
        self.final_state: dict | None = None
        self.stage = "starting"
        self.audit_writes = 0
        self.worker_started = False
        self._future = None  # set by StreamService.start_run
        self.defensive_claim_fired = False
        self.hooks = {name: threading.Event() for name in HOOK_NAMES}

    # -- stop signal -------------------------------------------------

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def request_stop(self, reason: str) -> None:
        """Ask the worker to stop. Callers claim the outcome first."""
        with self._stop_lock:
            if self._stop_reason is None:
                self._stop_reason = reason
        self._stop_event.set()
        self.channel.abort(reason)

    def stop_reason(self) -> str | None:
        """Why producers should stop, or None. Pure: never claims or aborts."""
        if self._stop_event.is_set():
            return self._stop_reason
        if self._loop.is_closed():
            return "loop_closed"
        attach_state = self.lease.attach_state
        now = self._clock()
        if attach_state == "pending" and now >= self.deadline + ATTACH_GRACE:
            return "abandon_due"
        if attach_state == "attached" and now >= self.deadline + FINALIZE_GRACE:
            return "consumer_unresponsive"
        return None

    def _apply_backstops(self) -> bool:
        """Worker-side backstops: claim, then request a stop. True if one applied."""
        if self._loop.is_closed():
            self.outcome.claim("cancelled", "LoopClosed", "event loop closed")
            self.lease.mark_consumer_done()  # the consumer can never run
            self.request_stop("loop_closed")
            return True

        now = self._clock()
        if now >= self.deadline + ATTACH_GRACE and self.lease.abandon():
            self.outcome.claim("timeout", "TimeoutError", "abandoned: response never attached")
            self.lease.mark_consumer_done()  # no consumer will ever run
            self.request_stop("abandoned")
            return True

        if self.lease.attach_state == "attached" and now >= self.deadline + FINALIZE_GRACE:
            self.outcome.claim("timeout", "TimeoutError", "consumer unresponsive at deadline")
            self.request_stop("consumer_unresponsive")
            return True

        return False

    # -- worker --------------------------------------------------------

    def _forward_chunk(self, text: str) -> None:
        """Answer-chunk sink; runs on the model's (or worker's) thread."""
        if not self.channel.publish(("delta", {"text": text})):
            raise StreamStopped()

    def external_call_stop_reason(self) -> str | None:
        """Checked immediately before each external call (H1-3a).

        Any stop reason of the run (a requested stop such as ``timeout``,
        ``disconnected`` or ``server_shutdown``, or a due backstop), or for a
        session the turn limit or a lost lease (in-memory state; no database
        access). Never claims anything: the worker does that when the stop
        reaches ``_complete``.
        """
        reason = self.stop_reason()
        if reason is None and self.session is not None:
            return self.session.manager.turn_stop_reason(self.session.lease)
        return reason

    def _run_worker(self) -> None:
        self.worker_started = True
        self.stage = "running"
        final = None
        error = None
        try:
            # Nodes see this check through the context, also in LangGraph's
            # threads; a stop raises WorkflowStopped before the next call.
            with stop_checks(self.external_call_stop_reason):
                if self.session is None:
                    stream = self._workflow_stream(self.question, self._forward_chunk)
                else:
                    stream = self._workflow_stream(
                        self.question, self._forward_chunk, history=self.session.history
                    )
                try:
                    for kind, payload in stream:
                        if self.stop_reason() is not None:
                            self._apply_backstops()
                            break
                        if kind == "trace":
                            self.trace.append(payload)
                            event = ("trace", {"index": len(self.trace) - 1, "step": payload})
                            if not self.channel.publish(event):
                                break
                            self.hooks["trace_published"].set()
                        elif kind == "final":
                            final = payload
                finally:
                    stream.close()  # waits only for the in-flight node
        except StreamStopped:
            pass
        except Exception as exc:  # noqa: BLE001 - recorded as the outcome
            error = exc

        try:
            self._complete(final, error)
        finally:
            self._finalize()

    def _complete(self, final: dict | None, error: BaseException | None) -> None:
        if not self.stop_requested:
            self._apply_backstops()
        if self.stop_requested:
            return  # the requester has already claimed the outcome

        if isinstance(error, WorkflowStopped):
            # Stopped before an external call for a reason nobody claimed (the
            # shutdown signal, a session limit, or a backstop that lost a
            # race). The most specific outcome for the reason; never a success.
            status, error_type, marker = WORKFLOW_STOP_OUTCOMES.get(
                error.reason,
                CONSUMER_EXITS.get(error.reason, ("cancelled", "WorkflowStopped", "workflow stopped")),
            )
            logger.warning(
                "Stream workflow stopped for request_id=%s (%s)", self.request_id, error.reason
            )
            if self.outcome.claim(status, error_type, marker):
                self._publish_error("timeout" if status == "timeout" else "internal_error")
            return

        if error is not None:
            logger.error(
                "Stream workflow failed for request_id=%s (%s)",
                self.request_id,
                type(error).__name__,
            )
            if self.outcome.claim("failed", type(error).__name__, "workflow error"):
                self._publish_error()
            return

        if final is None or not final.get("answer"):
            if self.outcome.claim("failed", "EmptyAnswer", "empty answer"):
                self._publish_error()
            return

        self.final_state = final
        if self.channel.set_terminal(("final", final)):
            self.hooks["terminal_set"].set()
            if self.channel.finish():
                self.hooks["finished"].set()
                self._wait_for_decision()
                return
        self._apply_backstops()  # e.g. the loop closed while handing over

    def _publish_error(self, code: str = "internal_error") -> None:
        if self.channel.set_terminal(("error", {"code": code})):
            self.hooks["terminal_set"].set()
            if self.channel.finish():
                self.hooks["finished"].set()

    def _wait_for_decision(self) -> None:
        """The worker never claims success; it waits for the consumer."""
        self.stage = "waiting_for_decision"
        self.hooks["waiting_for_decision"].set()
        while not self.outcome.decided.wait(self._poll_seconds):
            if self._apply_backstops():
                break

    def _finalize(self) -> None:
        self.stage = "finalizing"
        try:
            if self.stop_requested:
                self.channel.abort(self._stop_reason or "stopped")
            if not self.outcome.decided.is_set():
                self.defensive_claim_fired = True
                logger.error("Stream worker exited undecided for request_id=%s", self.request_id)
                self.outcome.claim("cancelled", "WorkerExit", "worker exited undecided")
            self._finish_session()
            self._write_audit()
        finally:
            self.stage = "done"
            self.lease.mark_worker_done()
            self.hooks["worker_exited"].set()

    def _finish_session(self) -> None:
        """Save the turn (completed only, ownership checked) or release; report turn_saved.

        Runs on the worker thread after the outcome is decided and before the
        audit row. The lease is always released and removed from the keeper.
        """
        session = self.session
        if session is None:
            return
        saved = False
        try:
            final = self.final_state or {}
            if self.outcome.decision.status == "completed" and final.get("answer"):
                saved = bool(session.manager.finish(
                    session.lease,
                    (self.question.strip(), final.get("answer", ""), final.get("source_used")),
                ))
            else:
                session.manager.finish(session.lease)
        except Exception:  # noqa: BLE001 - reported as not saved
            logger.exception("Session turn finish failed for request_id=%s", self.request_id)
            saved = False
        finally:
            session.report(saved)

    def _write_audit(self) -> None:
        decision = self.outcome.decision
        if decision.status == "completed":
            final = self.final_state or {}
            source_used = final.get("source_used") or "unknown"
            status = "insufficient_evidence" if source_used == "insufficient_evidence" else "success"
            fields = dict(
                source_used=source_used,
                trace=list(final.get("trace", self.trace)),
                citations=list(final.get("citations", [])),
                status=status,
                error_type=None,
            )
        else:
            fields = dict(
                source_used="cancelled" if decision.status == "cancelled" else "error",
                trace=[*self.trace, f"Stream → {decision.marker}"],
                citations=[],
                status="cancelled" if decision.status == "cancelled" else "failed",
                error_type=decision.error_type,
            )

        if self.session is not None:
            fields["session_id"] = self.session.lease.session_id  # internal UUID, never the token

        self.audit_writes += 1
        try:
            self._audit_writer(
                question=self.question.strip(),
                request_id=self.request_id,
                **fields,
            )
        except Exception:  # noqa: BLE001 - audit failures are logged, not raised
            logger.exception("Stream audit write failed for request_id=%s", self.request_id)
        finally:
            self.hooks["audit_written"].set()

    def _on_worker_future_done(self, future) -> None:
        """Done-callback for the submitted worker.

        A future can only be cancelled before it starts, in which case
        ``_run_worker`` (and its cleanup) never runs. Release the run here
        instead. A started worker cannot be cancelled, so this never
        finalizes a run twice.
        """
        if future.cancelled():
            self._finalize_unstarted()

    def _finalize_unstarted(self) -> None:
        """Clean up a run whose worker never started; writes no audit row."""
        self.stage = "done"
        self.outcome.claim("cancelled", "WorkerNotStarted", "worker never started")
        if self.lease.abandon():  # no consumer attached; a late one is refused
            self.lease.mark_consumer_done()
        self.request_stop("worker_not_started")  # an attached consumer sees ABORTED
        if self.session is not None:
            session = self.session
            session.report(False)
            # May run on any thread, including the event loop: release off it.
            threading.Thread(
                target=session.manager.finish, args=(session.lease,),
                name="session-release", daemon=True,
            ).start()
        self.lease.mark_worker_done()
        self.hooks["worker_exited"].set()

    # -- consumer API (event-loop thread) -----------------------------

    def attach(self) -> bool:
        """Called first by the response. False once the run was abandoned."""
        return self.lease.attach()

    def mark_response_started(self) -> None:
        """Call after ``http.response.start`` has been sent successfully."""
        self.response_started = True

    def claim_completed(self) -> bool:
        """Claim success after ``done`` was sent. Requires a started response."""
        if not self.response_started:
            raise RuntimeError("Cannot claim completion before the response started.")
        return self.outcome.claim("completed")

    def claim_timeout(self) -> bool:
        """The consumer's deadline expired. Claims, then requests a stop."""
        marker = (
            "deadline during response"
            if self.response_started
            else "deadline before response start"
        )
        claimed = self.outcome.claim("timeout", "TimeoutError", marker)
        if claimed:
            self.request_stop("timeout")
        return claimed

    def consumer_finished(self, exit_kind: str | None = None) -> None:
        """The consumer's exit handler. ``None`` means a normal exit.

        Normal exits never request a stop. Exceptional exits claim their
        outcome first-wins and request a stop unless the outcome is
        ``completed``. ``consumer_done`` is marked on every exit.
        """
        try:
            if exit_kind is not None:
                status, error_type, marker = CONSUMER_EXITS[exit_kind]
                self.outcome.claim(status, error_type, marker)
                decision = self.outcome.decision
                if decision is None or decision.status != "completed":
                    self.request_stop(exit_kind)
        finally:
            self.lease.mark_consumer_done()

    def remaining(self) -> float:
        return self.deadline - self._clock()


# -------------------------------------------------------------------
# Service: worker pool, admission, registry
# -------------------------------------------------------------------

@dataclass
class Reservation:
    """A capacity slot taken by ``StreamService.reserve()`` and not yet used."""

    lease: StreamLease
    run_key: list[int]
    used: bool = False


class StreamService:
    """Owns the dedicated worker pool, capacity, and active-run registry."""

    def __init__(
        self,
        *,
        max_concurrent: int,
        timeout_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        poll_seconds: float = POLL_S,
        workflow_stream: Callable[..., Iterator[tuple[str, Any]]] = stream_ask,
        audit_writer: Callable[..., Any] = write_audit,
    ):
        self.capacity = StreamCapacity(max_concurrent)
        self.timeout_seconds = timeout_seconds
        self.clock = clock
        self.poll_seconds = poll_seconds
        self.workflow_stream = workflow_stream
        self.audit_writer = audit_writer
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent, thread_name_prefix=WORKER_THREAD_PREFIX
        )
        self._registry_lock = threading.Lock()
        self._active: dict[int, StreamRun] = {}

    def start_run(
        self, question: str, request_id: str, loop: asyncio.AbstractEventLoop
    ) -> StreamRun | None:
        """Admit and start a run, or return None when no capacity is available."""
        reservation = self.reserve()
        if reservation is None:
            return None
        return self.start(reservation, question, request_id, loop)

    def reserve(self) -> "Reservation | None":
        """Take a capacity slot without starting a worker (A2-6: capacity first).

        Returns None when capacity is full. A reservation must be passed to
        ``start()`` or ``cancel()`` exactly once.
        """
        run_key: list[int] = []

        def on_release():
            with self._registry_lock:
                if run_key:
                    self._active.pop(run_key[0], None)

        lease = self.capacity.try_acquire(on_release)
        if lease is None:
            self._log_saturation()
            return None
        return Reservation(lease=lease, run_key=run_key)

    def cancel(self, reservation: "Reservation") -> None:
        """Give an unused reservation's slot back (exactly once; no worker, no audit)."""
        if reservation.used:
            return
        reservation.used = True
        reservation.lease.mark_worker_done()
        reservation.lease.mark_consumer_done()

    def start(
        self,
        reservation: "Reservation",
        question: str,
        request_id: str,
        loop: asyncio.AbstractEventLoop,
        session: StreamSession | None = None,
    ) -> StreamRun | None:
        """Start a run on a reserved slot; None if the worker can't be submitted.

        On failure the slot is released here; a session lease is the caller's
        to release.
        """
        if reservation.used:
            raise RuntimeError("This reservation has already been used.")
        reservation.used = True
        lease = reservation.lease
        run = StreamRun(
            question=question,
            request_id=request_id,
            lease=lease,
            loop=loop,
            timeout_seconds=self.timeout_seconds,
            clock=self.clock,
            poll_seconds=self.poll_seconds,
            workflow_stream=self.workflow_stream,
            audit_writer=self.audit_writer,
            session=session,
        )
        reservation.run_key.append(id(run))
        with self._registry_lock:
            self._active[id(run)] = run

        try:
            run._future = self._executor.submit(run._run_worker)
            run._future.add_done_callback(run._on_worker_future_done)
        except RuntimeError:
            logger.exception("Could not start stream worker for request_id=%s", request_id)
            lease.mark_worker_done()
            lease.mark_consumer_done()
            return None
        return run

    def active_runs(self) -> list[StreamRun]:
        with self._registry_lock:
            return list(self._active.values())

    def _log_saturation(self) -> None:
        active = self.active_runs()
        if active:
            oldest = min(active, key=lambda run: run.created_at)
            logger.warning(
                "Stream capacity exhausted: %d/%d in use; oldest run %.1fs old, stage=%s",
                self.capacity.in_use,
                self.capacity.limit,
                self.clock() - oldest.created_at,
                oldest.stage,
            )
        else:
            logger.warning(
                "Stream capacity exhausted: %d/%d in use.",
                self.capacity.in_use,
                self.capacity.limit,
            )

    def shutdown(self, wait: bool = True) -> None:
        """Stop accepting runs. Already-submitted workers are never cancelled.

        Queued workers still start and complete their lifecycle (one audit
        row, one lease release). With ``wait=False`` this returns at once;
        with ``wait=True`` it waits for every worker to finish.
        """
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def close(self, drain_seconds: float) -> dict[str, int]:
        """Application shutdown (H2). Blocks; call it off the event loop.

        Stops accepting runs, then waits at most ``drain_seconds`` for running
        workers to finish, so their turns are saved and their audit rows are
        written. A run that is still undecided afterwards is claimed
        ``cancelled`` and asked to stop; a decided outcome, ``completed``
        included, is never changed. Workers are never killed: one blocked in
        an external call finishes when that call returns.
        """
        self.shutdown(wait=False)
        deadline = time.monotonic() + max(0.0, drain_seconds)
        for run in self.active_runs():
            run.hooks["worker_exited"].wait(max(0.0, deadline - time.monotonic()))

        running = [run for run in self.active_runs() if not run.hooks["worker_exited"].is_set()]
        cancelled = 0
        for run in running:
            if run.outcome.claim("cancelled", "ServerShutdown", "server shutdown"):
                run.request_stop("server_shutdown")
                # A response that never attached will never mark the consumer
                # done; abandon it so the lease (and its capacity slot) is
                # released when the worker exits. An attached one is unaffected.
                if run.lease.abandon():
                    run.lease.mark_consumer_done()
                cancelled += 1
        if running:
            logger.warning(
                "Stream shutdown: %d worker(s) still running after %.1fs; %d undecided run(s) cancelled.",
                len(running), drain_seconds, cancelled,
            )
        return {"still_running": len(running), "cancelled": cancelled}


# -------------------------------------------------------------------
# Service singleton
# -------------------------------------------------------------------

_service: StreamService | None = None
_service_lock = threading.Lock()


def get_stream_service() -> StreamService:
    """Return the process-wide stream service, created from settings on first use."""
    global _service
    with _service_lock:
        if _service is None:
            settings = get_settings()
            _service = StreamService(
                max_concurrent=settings.max_concurrent_streams,
                timeout_seconds=settings.stream_timeout_seconds,
            )
        return _service


def peek_stream_service() -> StreamService | None:
    """The service if one was created; never creates one."""
    with _service_lock:
        return _service


def clear_stream_service(service: StreamService | None = None) -> None:
    """Forget the service (after shutdown), so a restarted app gets a fresh one."""
    global _service
    with _service_lock:
        if service is None or _service is service:
            _service = None


# -------------------------------------------------------------------
# Server-Sent Events
# -------------------------------------------------------------------

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
}
SSE_MEDIA_TYPE = "text/event-stream"
KEEPALIVE = b": keepalive\n\n"
STREAM_VERSION = 1

ERROR_MESSAGES = {
    "internal_error": "Unable to process the question.",
    "timeout": "The request took too long. Please try again.",
}
BUSY_MESSAGE = "The server is busy. Please retry shortly."


def format_sse(event: str, data: Any) -> bytes:
    """One SSE frame: ``event:`` line, single-line JSON ``data:`` line, blank line."""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


def busy_response(retry_after_seconds: int, code: str | None = None) -> JSONResponse:
    """503 when stream capacity is full.

    ``code`` (``"stream_busy"``) is added only for session requests, so the
    stateless response body stays exactly as before.
    """
    detail = {"message": BUSY_MESSAGE}
    if code is not None:
        detail = {"code": code, **detail}
    return JSONResponse(
        status_code=503,
        content={"detail": detail},
        headers={"Retry-After": str(retry_after_seconds)},
    )


async def _no_body():  # StreamingResponse requires an iterator; __call__ replaces it
    return
    yield  # pragma: no cover


class SSEResponse(StreamingResponse):
    """Streams one ``StreamRun`` to the client as Server-Sent Events.

    Lifecycle (the consumer side of ``StreamRun``):

    * Attach first; if the run was already abandoned, answer 503.
    * Everything sent, including ``http.response.start``, runs inside one
      deadline scope ending at the run's deadline. The deadline is also
      checked explicitly before the headers, on every loop iteration and
      before ``done``.
    * ``done`` is sent, the task yields once so an already-delivered
      disconnect is processed, and only then is ``completed`` claimed.
    * When the deadline expires, ``timeout`` is claimed and, if headers were
      sent, an ``error`` event is attempted within a short grace period,
      outside the expired scope, unless ``done`` or ``error`` was already
      sent: a stream never carries two terminal events.
    * The exit handler classifies the exit (normal, disconnected, send
      failed, cancelled, consumer error) and always marks the consumer done.
    """

    def __init__(
        self,
        run: StreamRun,
        *,
        build_done: Callable[[dict], dict],
        heartbeat_seconds: float,
        retry_after_seconds: int,
    ):
        super().__init__(_no_body(), media_type=SSE_MEDIA_TYPE, headers=SSE_HEADERS)
        self.run = run
        self._build_done = build_done
        self._heartbeat = heartbeat_seconds
        self._retry_after = retry_after_seconds
        self._disconnect_received = False
        self._send_failed = False
        self._consumer_error: Exception | None = None
        self._terminal_sent = False  # "done" or "error" delivered
        self._report_session = False  # completed with a session: send the ``session`` event

    async def __call__(self, scope, receive, send) -> None:
        run = self.run
        exit_kind = None
        try:
            if not run.attach():
                await busy_response(self._retry_after)(scope, receive, send)
                return

            spec_version = tuple(
                map(int, scope.get("asgi", {}).get("spec_version", "2.0").split("."))
            )
            if spec_version >= (2, 4):
                # Starlette's 2.4 behaviour: a disconnect surfaces as OSError on send.
                await self._guarded_stream(send)
            else:
                async with anyio.create_task_group() as task_group:

                    async def listen_for_disconnect():
                        while True:
                            message = await receive()
                            if message["type"] == "http.disconnect":
                                self._disconnect_received = True
                                task_group.cancel_scope.cancel()
                                return

                    task_group.start_soon(listen_for_disconnect)
                    try:
                        await self._guarded_stream(send)
                    finally:
                        task_group.cancel_scope.cancel()

            if self._consumer_error is not None:
                exit_kind = "consumer_error"
                raise self._consumer_error
            if self._send_failed:
                exit_kind = "send_failed"
            elif self._disconnect_received:
                exit_kind = "disconnected"
        except anyio.get_cancelled_exc_class():
            exit_kind = "disconnected" if self._disconnect_received else "cancelled"
            raise
        finally:
            run.consumer_finished(exit_kind)

    async def _guarded_stream(self, send) -> None:
        """Run ``_stream`` and record failures instead of raising them.

        anyio task groups wrap exceptions from their body in an
        ``ExceptionGroup``; recording them here keeps the exit classification
        exact. Cancellation is not caught and propagates normally.
        """
        try:
            await self._stream(send)
        except (OSError, ClientDisconnect):
            self._send_failed = True  # the client is gone; nothing left to send
        except Exception as exc:  # noqa: BLE001 - re-raised after the task group
            logger.exception("Stream response failed for request_id=%s", self.run.request_id)
            self._consumer_error = exc

    async def _send_body(self, send, chunk: bytes) -> None:
        await send({"type": "http.response.body", "body": chunk, "more_body": True})

    async def _end_body(self, send) -> None:
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def _check_deadline(self, scope: anyio.CancelScope) -> None:
        """Enforce the run's deadline even if nothing has suspended yet.

        An expired scope deadline only takes effect when the task suspends,
        and uvicorn's ``send()`` and ``receive_batch()`` often return without
        suspending. Without this check a run past its deadline could still
        send headers or ``done``. Cancelling here hands over to the timeout
        protocol inside the scope.
        """
        if self.run.remaining() <= 0:
            scope.cancel()
            await anyio.lowlevel.checkpoint()

    def _error_frame(self, code: str) -> bytes:
        return format_sse(
            "error",
            {"request_id": self.run.request_id, "code": code, "message": ERROR_MESSAGES[code]},
        )

    async def _stream(self, send) -> None:
        run = self.run
        # The run's deadline is on time.monotonic; the event loop's clock is too.
        deadline = anyio.current_time() + run.remaining()

        with anyio.CancelScope(deadline=deadline) as scope:
            await self._check_deadline(scope)
            await send({"type": "http.response.start", "status": 200, "headers": self.raw_headers})
            run.mark_response_started()
            start = {"request_id": run.request_id, "version": STREAM_VERSION}
            if run.session is not None:
                start["session"] = {"history_turns_used": run.session.history_turns_used}
            await self._send_body(send, format_sse("start", start))

            while True:
                await self._check_deadline(scope)
                batch = await run.channel.receive_batch(self._heartbeat)
                if batch is END or batch is ABORTED:
                    break  # stopped elsewhere: end without a terminal event
                if batch == []:
                    await self._send_body(send, KEEPALIVE)
                    continue

                kind, payload = batch[-1]
                if kind == "final":
                    await self._check_deadline(scope)
                    await self._send_body(send, format_sse("done", self._build_done(payload)))
                    self._terminal_sent = True
                    await anyio.lowlevel.checkpoint()  # process an already-delivered disconnect
                    if not self._disconnect_received:
                        completed = run.claim_completed()
                        self._report_session = completed and run.session is not None
                    break
                if kind == "error":
                    await self._send_body(send, self._error_frame(payload.get("code", "internal_error")))
                    self._terminal_sent = True
                    break

                await self._send_body(
                    send, b"".join(format_sse(event, data) for event, data in batch)
                )

        if scope.cancelled_caught:
            if run.claim_timeout() and run.response_started:
                with anyio.move_on_after(TIMEOUT_ERROR_GRACE):  # outside the expired scope
                    if not self._terminal_sent:  # never a second terminal event
                        await self._send_body(send, self._error_frame("timeout"))
                    await self._end_body(send)
            return  # headers never sent, or the grace expired: leave it to the server

        if self._report_session:
            await self._send_session_event(send)

        # Transport cleanup only: a failure here never changes a claimed outcome.
        with anyio.move_on_after(END_OF_BODY_GRACE):
            await self._end_body(send)

    async def _send_session_event(self, send) -> None:
        """After ``done``: wait (bounded) for the worker's save, then send ``session``.

        Outside the deadline scope; ``completed`` is already claimed, so
        nothing here can change the outcome. ``turn_saved`` is None when the
        save result didn't arrive in time (the client treats that as not saved).
        """
        session = self.run.session
        turn_saved = None
        with anyio.move_on_after(SESSION_SAVE_REPORT_WAIT):
            # shield: giving up on the wait must not cancel the worker's result.
            turn_saved = await asyncio.shield(asyncio.wrap_future(session.result))
        with anyio.move_on_after(END_OF_BODY_GRACE):
            await self._send_body(
                send,
                format_sse(
                    "session",
                    {
                        "request_id": self.run.request_id,
                        "turn_saved": turn_saved,
                        "history_turns_used": session.history_turns_used,
                    },
                ),
            )
