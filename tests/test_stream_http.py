"""A1-4 loopback tests: POST /api/chat/stream over real TCP through uvicorn.

``LoopbackServer`` runs uvicorn on 127.0.0.1 with ``loop="asyncio"`` and
``http="h11"`` in a background thread; the client is httpx with
``trust_env=False``. Ordering uses workflow gates and the injectable clock,
not sleeps. A blocked header send (scenarios 12a/12b) is not reproduced
here: socket buffers make it non-deterministic; the ASGI harness covers it.
"""

import threading
import time

import httpx
import pytest

import app.main
from tests.streaming_support import LoopbackServer, FakeClock, parse_sse, read_sse_until
from tests.test_stream_run import Gate, ScriptedWorkflow, final_state

GUARD = 5.0
TIMEOUT = 120.0
QUESTION = "How do I connect to the VPN?"


def wait_for(event: threading.Event, timeout=GUARD):
    assert event.wait(timeout), "event not set in time"


def client(server):
    return httpx.Client(base_url=server.base_url, trust_env=False, timeout=GUARD)


def stream_request(http):
    return http.stream("POST", "/api/chat/stream", json={"question": QUESTION})


def rows(service):
    return [(row["status"], row["error_type"]) for row in service.audit.rows]


def test_deltas_arrive_before_the_answer_is_finished(stream_app):
    gate = Gate()
    service = stream_app(ScriptedWorkflow([("chunk", "first "), ("gate", gate), ("chunk", "second"), ("final", final_state(answer="first second"))]))

    with LoopbackServer(app.main.app) as server, client(server) as http:
        with stream_request(http) as response:
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
            chunks = response.iter_raw()
            buffer = bytearray()
            read_sse_until(chunks, buffer, b"event: delta")
            assert not gate.release.is_set()  # received over TCP while the graph is held
            gate.release.set()
            buffer += b"".join(chunks)

    [run] = service.started
    wait_for(run.hooks["worker_exited"])
    parsed = parse_sse(bytes(buffer))
    assert parsed.names == ["start", "delta", "delta", "done"]
    assert rows(service) == [("success", None)]


def test_client_disconnect_is_recorded_as_cancelled(stream_app):
    gate = Gate()
    service = stream_app(ScriptedWorkflow([("chunk", "A"), ("gate", gate), ("final", final_state())]))

    with LoopbackServer(app.main.app) as server, client(server) as http:
        with stream_request(http) as response:
            read_sse_until(response.iter_raw(), bytearray(), b"event: delta")
        # Leaving the block closes the connection mid-stream.
        [run] = service.started
        wait_for(run.outcome.decided)

    gate.release.set()
    wait_for(run.hooks["worker_exited"])
    decision = run.outcome.decision
    assert (decision.status, decision.error_type) == ("cancelled", "ClientDisconnected")
    assert rows(service) == [("cancelled", "ClientDisconnected")]


def test_deadline_before_headers_gives_uvicorns_plain_500(stream_app):
    clock = FakeClock()
    service = stream_app(ScriptedWorkflow([("final", final_state())]), clock=clock)
    start_run = service.start_run

    def start_run_then_expire(*args, **kwargs):
        run = start_run(*args, **kwargs)
        clock.advance(TIMEOUT + 0.5)  # the response attaches in (D, D+G]
        return run

    service.start_run = start_run_then_expire

    with LoopbackServer(app.main.app) as server, client(server) as http:
        response = http.post("/api/chat/stream", json={"question": QUESTION})

    [run] = service.started
    wait_for(run.hooks["worker_exited"])
    assert response.status_code == 500
    assert response.headers["content-type"].startswith("text/plain")
    assert response.headers["connection"] == "close"
    assert response.text == "Internal Server Error"
    assert b"event:" not in response.content
    assert run.outcome.decision.marker == "deadline before response start"
    assert rows(service) == [("failed", "TimeoutError")]


def test_deadline_after_headers_ends_with_timeout_error_or_close(stream_app):
    gate = Gate()
    service = stream_app(ScriptedWorkflow([("chunk", "A"), ("gate", gate), ("final", final_state())]), timeout=0.5)

    with LoopbackServer(app.main.app) as server, client(server) as http:
        with stream_request(http) as response:
            assert response.status_code == 200
            body = b"".join(response.iter_raw())

    gate.release.set()
    [run] = service.started
    wait_for(run.hooks["worker_exited"])
    parsed = parse_sse(body)
    # Either is consistent with uvicorn's behaviour (F20); "done" never is.
    assert parsed.names in (["start", "delta", "error"], ["start", "delta"])
    if parsed.names[-1] == "error":
        assert parsed.events[-1].data["code"] == "timeout"
    assert run.outcome.decision.marker == "deadline during response"
    assert rows(service) == [("failed", "TimeoutError")]
