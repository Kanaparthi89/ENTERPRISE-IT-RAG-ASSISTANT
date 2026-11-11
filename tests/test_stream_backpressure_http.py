"""H3-A: slow-reader backpressure through a real Uvicorn server.

The in-process tests (``tests/test_stream_channel.py``,
``tests/test_stream_run.py``, ``tests/test_stream_lifecycle.py``) simulate a
slow client with a send gate: an artificial ``await`` in place of Uvicorn's
``send()``. These tests use a real server (``LoopbackServer``) on runtime A
(asyncio + h11) and runtime P (uvloop + httptools) and a real TCP client that
stops reading, so the whole chain is exercised:

1. the client stops reading; its receive buffer and the server's send buffer
   fill;
2. the transport's write buffer passes its high-water mark and the transport
   calls ``protocol.pause_writing()``, so Uvicorn's ``FlowControl`` sets
   ``write_paused``;
3. Uvicorn's ``send()`` waits in ``flow.drain()``; ``SSEResponse`` stops
   draining the ``EventChannel``, which fills (``CHANNEL_MAX_EVENTS``);
4. the worker blocks in ``EventChannel.publish()``, holding the model.

The evidence that step 2 happened on the real transport, not in a
simulation, is the server connection's ``flow.write_paused``, read on the
server's own event loop (``LoopbackServer.connection_flow()``).

Then one of three ways out: the client reads again (``resume_writing()``),
the stream deadline passes, or the client disconnects.

Buffers: the server's SO_SNDBUF is set small (``sndbuf``) and the client's
SO_RCVBUF too, so the pause comes after a few hundred KB rather than after
the OS's auto-tuned maximum (several MB). The answer is about 2 MB of
deltas, well above what the buffers hold, and the tests wait on the
observed ``write_paused`` flag, not on byte counts.

Limits: loopback TCP, no reverse proxy or TLS, and timing that depends on the
OS's socket buffers (see the README's H3 row).
"""

import socket
import struct
import threading
import time

import httpx
import pytest

import app.main
from app.api.streaming import CHANNEL_MAX_EVENTS, FINALIZE_GRACE, TIMEOUT_ERROR_GRACE
from tests.streaming_support import LoopbackServer, assert_run_cleaned_up, parse_sse
from tests.test_stream_run import final_state

GUARD = 10.0
SNDBUF = 32 * 1024
RCVBUF = 4 * 1024
CHUNKS = 20_000
CHUNK_SIZE = 100  # characters per delta; about 2 MB of deltas in all
HOLD_SECONDS = 0.5  # how long the held producer is watched for progress
DEADLINE = 3.0  # stream deadline for the past-the-deadline test
QUESTION = "How do I connect to the VPN?"


class VolumeWorkflow:
    """Stands in for ``stream_ask``: streams ``count`` deltas, then the answer.

    ``produced`` counts the deltas the forwarder accepted; while the worker
    is blocked in ``publish()`` it does not move.
    """

    def __init__(self, count=CHUNKS, size=CHUNK_SIZE):
        self.count = count
        self.texts = [f"{i:08d}".ljust(size, "x") for i in range(count)]
        self.produced = 0
        self.closed = threading.Event()

    def __call__(self, question, on_chunk):
        def run():
            try:
                for text in self.texts:
                    on_chunk(text)  # raises once the stream is stopped
                    self.produced += 1
                yield ("final", final_state(answer="".join(self.texts)))
            finally:
                self.closed.set()

        return run()


def slow_client(server):
    """An HTTP client with a small receive buffer; closing it resets the connection."""
    transport = httpx.HTTPTransport(
        socket_options=[
            (socket.SOL_SOCKET, socket.SO_RCVBUF, RCVBUF),
            (socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)),  # close sends RST
        ]
    )
    return httpx.Client(base_url=server.base_url, transport=transport, trust_env=False, timeout=GUARD)


def open_stream(http):
    request = http.build_request("POST", "/api/chat/stream", json={"question": QUESTION})
    response = http.send(request, stream=True)  # returns after the headers; the body is not read
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    return response


def wait_until(condition, message, timeout=GUARD):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(message)
        time.sleep(0.01)


def wait_for(event: threading.Event, timeout=GUARD):
    assert event.wait(timeout), "event not set in time"


def paused_state(server):
    """The paused connection's state from one snapshot, or None."""
    return next((state for state in server.connection_flow() if state["write_paused"]), None)


def paused(server) -> bool:
    return paused_state(server) is not None


def wait_for_backpressure(server, run, workflow, record_property):
    """Steps 2-4: the real transport paused writing and the channel is full.

    Waits for one snapshot in which both hold, then for the producer to stay
    still for ``HOLD_SECONDS`` while paused (under CPU load the transport can
    briefly resume and pause again before the buffers settle). Returns the
    producer count at the end of that quiet period.
    """

    def settled():
        state = paused_state(server)
        return state is not None and len(run.channel) == CHANNEL_MAX_EVENTS and state

    deadline = time.monotonic() + GUARD
    while True:
        wait_until(settled, "the server never paused writing with a full channel",
                   timeout=max(0.01, deadline - time.monotonic()))
        held_at = workflow.produced
        time.sleep(HOLD_SECONDS)
        state = settled()
        if state and workflow.produced == held_at:
            break
        if time.monotonic() > deadline:
            raise AssertionError("the producer never stayed still while the server was paused")

    record_property("server_connection", repr(state))  # actual SO_SNDBUF and write buffer
    assert state["write_buffer"] > 0  # bytes waiting in the transport, not the kernel
    return held_at


def read_what_is_left(response) -> bytes:
    """Read until the server closes; a stream cut short is not an error here."""
    body = bytearray()
    try:
        for chunk in response.iter_raw():
            body += chunk
    except (httpx.RemoteProtocolError, httpx.ReadError):
        pass  # the server closed the connection before the end of the body
    return bytes(body)


def complete_frames(body: bytes) -> bytes:
    """The body up to its last complete SSE frame."""
    end = body.rfind(b"\n\n")
    return body[: end + 2] if end >= 0 else b""


def rows(service):
    return [(row["status"], row["error_type"]) for row in service.audit.rows]


def test_slow_reader_pauses_the_server_and_resumes_without_loss(stream_app, loopback_runtime, record_property):
    workflow = VolumeWorkflow()
    service = stream_app(workflow)

    with LoopbackServer(app.main.app, runtime=loopback_runtime, sndbuf=SNDBUF) as server, slow_client(server) as http:
        response = open_stream(http)
        try:
            [run] = service.started
            wait_for_backpressure(server, run, workflow, record_property)

            # The producer is held: no delta is accepted while the client does not read.
            held_at = workflow.produced
            time.sleep(HOLD_SECONDS)
            assert workflow.produced == held_at
            assert held_at < workflow.count
            assert not run.hooks["worker_exited"].is_set()
            record_property("produced_while_held", held_at)

            # The event loop is not blocked: another request is served meanwhile.
            with httpx.Client(base_url=server.base_url, trust_env=False, timeout=GUARD) as other:
                started = time.monotonic()
                health = other.get("/health")
                elapsed = time.monotonic() - started
            assert health.status_code == 200
            assert elapsed < 2.0
            assert paused(server)  # the slow connection is still paused

            # The client reads again: writing resumes and everything arrives.
            body = b"".join(response.iter_raw())
        finally:
            response.close()

    wait_for(run.hooks["worker_exited"])
    parsed = parse_sse(body)
    assert parsed.names[0] == "start"
    assert parsed.names[-1] == "done"
    assert [event.data["text"] for event in parsed.events if event.event == "delta"] == workflow.texts
    assert parsed.events[-1].data["answer"] == "".join(workflow.texts)
    assert workflow.produced == workflow.count
    assert rows(service) == [("success", None)]
    assert_run_cleaned_up(service, run, server)


def test_slow_reader_past_the_deadline_releases_the_run(stream_app, loopback_runtime, record_property):
    workflow = VolumeWorkflow()
    service = stream_app(workflow, timeout=DEADLINE)

    with LoopbackServer(app.main.app, runtime=loopback_runtime, sndbuf=SNDBUF) as server, slow_client(server) as http:
        response = open_stream(http)
        try:
            [run] = service.started
            wait_for_backpressure(server, run, workflow, record_property)
            assert run.remaining() > 0  # paused before the deadline, so the deadline ends it

            # The client never reads. The deadline alone frees the worker,
            # the lease, and the capacity slot, within the finalize grace.
            wait_for(run.hooks["worker_exited"], timeout=DEADLINE + FINALIZE_GRACE + GUARD)
            wait_until(lambda: run.lease.released, "the lease was not released",
                       timeout=TIMEOUT_ERROR_GRACE + GUARD)
            assert service.capacity.in_use == 0
            assert workflow.produced < workflow.count  # the model was stopped, not run to the end

            body = read_what_is_left(response)
        finally:
            response.close()

    parsed = parse_sse(complete_frames(body))
    record_property("observed_ending", ",".join(parsed.names[-2:]))
    assert parsed.names[0] == "start"
    assert "done" not in parsed.names  # never a success after the deadline
    if parsed.names[-1] == "error":
        assert parsed.events[-1].data["code"] == "timeout"
    record_property("outcome_marker", run.outcome.decision.marker)
    assert rows(service) == [("failed", "TimeoutError")]
    assert_run_cleaned_up(service, run, server)


def test_slow_reader_disconnect_stops_the_blocked_run(stream_app, loopback_runtime, record_property):
    workflow = VolumeWorkflow()
    service = stream_app(workflow)

    with LoopbackServer(app.main.app, runtime=loopback_runtime, sndbuf=SNDBUF) as server:
        http = slow_client(server)
        response = open_stream(http)
        try:
            [run] = service.started
            wait_for_backpressure(server, run, workflow, record_property)
            assert paused(server)
            held_at = workflow.produced  # immediately before the disconnect
        finally:
            response.close()  # unread data and SO_LINGER 0: the connection is reset
            http.close()

        # The disconnect wakes the paused send; the blocked worker stops.
        wait_for(run.hooks["worker_exited"])
        wait_until(lambda: run.lease.released, "the lease was not released")
        record_property("produced_before_disconnect", held_at)
        record_property("produced_after_disconnect", workflow.produced)
        assert workflow.produced <= held_at + 1  # at most the delta that was being handed over
        assert workflow.produced < workflow.count

    decision = run.outcome.decision
    assert (decision.status, decision.error_type) == ("cancelled", "ClientDisconnected")
    assert rows(service) == [("cancelled", "ClientDisconnected")]
    assert_run_cleaned_up(service, run, server)
