"""H2: graceful shutdown through a real Uvicorn server, and the Docker budget.

The loopback tests run ``app.main.app`` with ``lifespan="on"`` on a real
Uvicorn server (``LoopbackServer``), on runtime A (asyncio + h11) and runtime
P (uvloop + httptools, what production selects; H3). Shutdown is started
the way a signal starts it, by setting ``should_exit``, so Uvicorn's own order
applies: stop accepting, wait for open connections (cancelling them after
``timeout_graceful_shutdown``), then run the lifespan shutdown (stream drain,
then session shutdown). The real ``close()`` and ``manager.shutdown()`` are
wrapped only to record when they run.
"""

import json
import re
import shlex
import threading
import time
from pathlib import Path

import httpx
import pytest

import app.main
from app.api import routes, streaming
from app.api.streaming import SESSION_SAVE_REPORT_WAIT
from app.core.config import Settings
from app.services import sessions as session_service
from tests.streaming_support import (
    SESSION_STREAM_EVENTS,
    LoopbackServer,
    assert_run_cleaned_up,
    parse_sse,
    read_sse_until,
)
from tests.test_session_stream import HistoryWorkflow, answer_steps, row_for, turn_count
from tests.test_stream_run import Gate

GUARD = 10.0
QUESTION = "How do I connect to the VPN?"
REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Real Uvicorn shutdown
# ---------------------------------------------------------------------------

@pytest.fixture
def shutdown_app(stream_app, settings, monkeypatch):
    """Sessions on (small shutdown budgets); a test stream service the lifespan will close."""
    monkeypatch.setattr(settings, "sessions_enabled", True)
    monkeypatch.setattr(settings, "session_shutdown_grace_seconds", 0.2)
    monkeypatch.setattr(settings, "session_keeper_join_seconds", 0.3)
    monkeypatch.setattr(settings, "session_release_budget_seconds", 0.5)
    monkeypatch.setattr(settings, "session_db_busy_timeout_seconds", 0.5)

    def build(workflow):
        service = stream_app(workflow)
        # stream_app patches get_stream_service; the lifespan reads the singleton.
        monkeypatch.setattr(streaming, "_service", service)
        return service

    return build


class ShutdownRecorder:
    """Records when the real stream close and session shutdown run."""

    def __init__(self, server, service):
        self.server = server
        self.events = []
        self.close_started = threading.Event()
        self.summary = None
        self.drain_seconds = None
        real_close = service.close

        def close(drain_seconds):
            started = time.monotonic()
            self.events.append(("stream.close", self.open_connections()))
            self.close_started.set()
            self.summary = real_close(drain_seconds)
            self.drain_seconds = time.monotonic() - started
            self.events.append(("stream.closed", self.open_connections()))
            return self.summary

        service.close = close

    def open_connections(self):
        return len(self.server.server.server_state.connections)

    def watch_session_shutdown(self):
        manager = session_service.peek_session_runtime().manager
        real_shutdown = manager.shutdown

        def shutdown():
            self.events.append(("session.shutdown", self.open_connections()))
            return real_shutdown()

        manager.shutdown = shutdown


def client(server):
    return httpx.Client(base_url=server.base_url, trust_env=False, timeout=GUARD)


def new_session(http):
    response = http.post("/api/sessions", json={})
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


def wait_for(event, timeout=GUARD):
    assert event.wait(timeout), "event not set in time"


def test_shutdown_cancels_an_open_session_stream_then_shuts_sessions_down(shutdown_app, settings, loopback_runtime):
    gate = Gate()
    service = shutdown_app(HistoryWorkflow(answer_steps(gate=gate)))
    buffer = bytearray()

    with LoopbackServer(app.main.app, runtime=loopback_runtime, lifespan="on",
                        graceful_shutdown_timeout=0.5) as server, client(server) as http:
        recorder = ShutdownRecorder(server, service)
        token = new_session(http)
        recorder.watch_session_shutdown()

        with http.stream("POST", "/api/chat/stream", json={"question": QUESTION},
                         headers={"X-Session-ID": token}) as response:
            assert response.status_code == 200
            chunks = response.iter_raw()
            read_sse_until(chunks, buffer, b"event: trace")  # the worker is now held at the gate
            [run] = service.started

            server.server.should_exit = True  # what SIGTERM does
            try:
                for chunk in chunks:  # Uvicorn waits 0.5 s, then cancels the response
                    buffer += chunk
            except httpx.HTTPError:
                pass  # the connection may close mid-body

        wait_for(run.outcome.decided)
        gate.release.set()  # the worker finishes its step, then sees the stop
        wait_for(run.hooks["worker_exited"])
        server.thread.join(GUARD)
        assert not server.thread.is_alive()

    decision = run.outcome.decision
    assert (decision.status, decision.error_type) == ("cancelled", "Cancelled")
    assert b"event: done" not in buffer and b"event: session" not in buffer

    session_row = row_for(settings, token)
    [audit] = service.audit.rows
    assert (audit["status"], audit["error_type"]) == ("cancelled", "Cancelled")
    assert audit["session_id"] == session_row["id"]  # the internal ID, never the token
    assert session_row["lease_owner"] is None  # released by the worker
    assert turn_count(settings, token) == 0  # a cancelled turn is not saved

    # Order: Uvicorn cancels the response, then the lifespan drains streams, then
    # shuts sessions down. After the graceful timeout Uvicorn cancels the task
    # and starts the lifespan at once, so the cancelled connection may still be
    # closing when the drain starts; it is closed before sessions shut down.
    assert [name for name, _ in recorder.events] == ["stream.close", "stream.closed", "session.shutdown"]
    assert recorder.events[0][1] <= 1
    assert recorder.events[-1] == ("session.shutdown", 0)
    assert recorder.summary == {"still_running": 0, "cancelled": 0}  # already decided
    assert streaming.peek_stream_service() is None
    assert session_service.peek_session_runtime() is None
    assert_run_cleaned_up(service, run, server)


def test_a_slow_save_finishes_within_the_drain_after_the_response_ended(
    shutdown_app, settings, monkeypatch, loopback_runtime
):
    service = shutdown_app(HistoryWorkflow(answer_steps()))
    save_entered, save_release = threading.Event(), threading.Event()
    save_times = {}
    real_save = session_service.SessionStore.save_turn

    def slow_save(self, *args, **kwargs):
        save_times["entered"] = time.monotonic()
        save_entered.set()
        assert save_release.wait(GUARD), "the save was never released"
        try:
            return real_save(self, *args, **kwargs)
        finally:
            save_times["finished"] = time.monotonic()

    monkeypatch.setattr(session_service.SessionStore, "save_turn", slow_save)

    with LoopbackServer(app.main.app, runtime=loopback_runtime, lifespan="on",
                        graceful_shutdown_timeout=0.5) as server, client(server) as http:
        recorder = ShutdownRecorder(server, service)
        token = new_session(http)
        recorder.watch_session_shutdown()

        response = http.post("/api/chat/stream", json={"question": QUESTION},
                             headers={"X-Session-ID": token})

        # The response gave up waiting for the save after the report wait.
        parsed = parse_sse(response.content, allowed=SESSION_STREAM_EVENTS)
        assert parsed.names[-2:] == ["done", "session"]
        assert parsed.events[-1].data["turn_saved"] is None
        [run] = service.started
        assert save_entered.is_set() and not run.hooks["worker_exited"].is_set()

        server.server.should_exit = True
        wait_for(recorder.close_started)  # the lifespan is draining stream workers
        save_release.set()  # the save completes well within the drain
        server.thread.join(GUARD)
        assert not server.thread.is_alive()

    save_seconds = save_times["finished"] - save_times["entered"]
    assert save_seconds > SESSION_SAVE_REPORT_WAIT  # slower than the report wait
    assert recorder.drain_seconds < routes.STREAM_SHUTDOWN_DRAIN_SECONDS  # but within the drain
    assert recorder.summary == {"still_running": 0, "cancelled": 0}

    assert turn_count(settings, token) == 1  # saved after the response had ended
    assert row_for(settings, token)["lease_owner"] is None  # released by the save
    [audit] = service.audit.rows
    assert audit["status"] == "success" and audit["session_id"] == row_for(settings, token)["id"]
    assert run.outcome.decision.status == "completed"
    assert recorder.events == [
        ("stream.close", 0),
        ("stream.closed", 0),
        ("session.shutdown", 0),
    ]
    assert_run_cleaned_up(service, run, server)


def test_shutdown_never_interrupts_the_call_in_flight_and_starts_no_later_call(
    stream_app, fake_rag, monkeypatch, loopback_runtime, record_property
):
    """S3 (H1-3a under H3): a provider call in flight when the server stops.

    The real workflow runs with fake providers; the embedding call is held
    open across the whole shutdown. It is not interrupted: it returns only
    when the test releases it, after the server has stopped. The run then
    starts no further external call (no Pinecone query, no grading), and its
    slot and lease are held until its worker exits.
    """
    from tests.test_cancellation import GatedRetriever, gated_rag, provider_calls

    finished = threading.Event()

    class InFlightRetriever(GatedRetriever):
        def invoke(self, query):
            self.gate.reached.set()
            assert self.gate.release.wait(GUARD), "the in-flight call was never released"
            finished.set()  # the call in flight completed normally
            self.timed.query(vector=[0.1], top_k=4)  # the next external call (checked)
            return []

    gate, llm, _ = gated_rag(fake_rag)
    retriever = InFlightRetriever(gate)
    fake_rag(llm=llm, retriever=retriever)
    service = stream_app(streaming.stream_ask)
    monkeypatch.setattr(streaming, "_service", service)  # the lifespan closes the singleton
    monkeypatch.setattr(routes, "STREAM_SHUTDOWN_DRAIN_SECONDS", 0.2)

    with LoopbackServer(app.main.app, runtime=loopback_runtime, lifespan="on",
                        graceful_shutdown_timeout=0.5) as server, client(server) as http:
        with http.stream("POST", "/api/chat/stream", json={"question": QUESTION}) as response:
            assert response.status_code == 200
            chunks = response.iter_raw()
            read_sse_until(chunks, bytearray(), b"event: trace")  # routing done
            wait_for(gate.reached)  # the embedding call is in flight
            [run] = service.started
            server.server.should_exit = True  # what SIGTERM does
            try:
                for _ in chunks:  # Uvicorn waits 0.5 s, then cancels the response
                    pass
            except httpx.HTTPError:
                pass
        server.thread.join(GUARD)
        assert not server.thread.is_alive()  # the server stopped with the call still in flight

    # Shutdown completed (drain, shutdown signal, sessions) without the worker.
    from app.rag import cancellation

    assert cancellation.shutdown_signalled()  # set by the real lifespan after the drain
    assert not finished.is_set() and not run.hooks["worker_exited"].is_set()
    assert not run.lease.released and service.capacity.in_use == 1  # held while the call runs
    decision = run.outcome.decision
    assert decision.status == "cancelled"
    record_property("observed_error_type", decision.error_type)

    gate.release.set()  # the in-flight call returns on its own
    wait_for(finished)
    wait_for(run.hooks["worker_exited"])

    assert provider_calls(llm, retriever) == ["RouteDecision"]  # no query, no grading after it
    [row] = service.audit.rows
    assert row["status"] == "cancelled"
    assert_run_cleaned_up(service, run, server)


# ---------------------------------------------------------------------------
# Docker shutdown budget (static: reads the Dockerfile and compose.yaml)
# ---------------------------------------------------------------------------

UVICORN_GRACEFUL_TIMEOUT = 8
MARGIN_SECONDS = 2


def dockerfile_command() -> list[str]:
    """The arguments of the final top-level ``CMD ["sh", "-c", "..."]``."""
    commands = re.findall(r"^CMD (\[.*\])\s*$", (REPO / "Dockerfile").read_text(), re.MULTILINE)
    assert commands, "no top-level CMD in the Dockerfile"
    shell, flag, script = json.loads(commands[-1])
    assert (shell, flag) == ("sh", "-c")
    return shlex.split(script)


def option(arguments: list[str], name: str) -> str:
    assert arguments.count(name) == 1, f"{name} must appear exactly once"
    return arguments[arguments.index(name) + 1]


def compose_stop_grace_seconds() -> float:
    text = (REPO / "compose.yaml").read_text()
    [value] = re.findall(r"^\s*stop_grace_period:\s*(\d+)s\s*$", text, re.MULTILINE)
    return float(value)


def test_dockerfile_sets_the_uvicorn_graceful_shutdown_timeout():
    arguments = dockerfile_command()

    assert arguments[:2] == ["exec", "uvicorn"]
    assert option(arguments, "--timeout-graceful-shutdown") == str(UVICORN_GRACEFUL_TIMEOUT)
    # The other arguments are unchanged.
    assert [a for a in arguments if a not in ("--timeout-graceful-shutdown", "8")] == [
        "exec", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "${PORT:-8080}", "--no-proxy-headers",
    ]


def test_shutdown_budgets_fit_within_the_compose_stop_grace_period():
    fields = Settings.model_fields
    session_budget = (
        fields["session_shutdown_grace_seconds"].default
        + fields["session_keeper_join_seconds"].default
        + fields["session_release_budget_seconds"].default
    )
    graceful = float(option(dockerfile_command(), "--timeout-graceful-shutdown"))
    total = graceful + routes.STREAM_SHUTDOWN_DRAIN_SECONDS + session_budget

    assert (graceful, routes.STREAM_SHUTDOWN_DRAIN_SECONDS, session_budget) == (8.0, 3.0, 7.0)
    assert compose_stop_grace_seconds() == 20.0  # compose.yaml is not changed by H2
    assert total + MARGIN_SECONDS <= compose_stop_grace_seconds()
