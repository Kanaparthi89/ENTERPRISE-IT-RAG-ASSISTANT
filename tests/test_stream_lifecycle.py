"""A1-4: stream lifecycle through the real SSE response (ASGI harness).

``test_stream_run.py`` covers the same protocol with a minimal consumer. Here
every scenario runs through ``SSEResponse`` (and usually the route), so the
consumer's deadline handling, exit classification and terminal events are
part of what is checked. Ordering is controlled with send gates, workflow
gates and the injectable clock; real time is only used where the deadline
must expire while the consumer is suspended (12a/12b, timeout grace).

The ``stream_app`` fixture (conftest) checks after every test: no worker
threads left, capacity released, one lease release and one audit write per
started worker, and no defensive worker-exit claim.
"""

import asyncio
import threading
import time
import tracemalloc

import pytest

import app.main
from app.api import routes, streaming
from app.api.streaming import ATTACH_GRACE, CHANNEL_MAX_EVENTS, CONSUMER_EXITS, FINALIZE_GRACE
from app.rag import workflow
from app.rag.state import EvidenceGrade, RouteDecision
from tests.conftest import FakeLLM, FakeRetriever, FakeWebSearch, kb_doc
from tests.streaming_support import (
    ASGIHarness,
    ChunkScript,
    FakeClock,
    assert_stream_grammar,
    body_contains,
    is_end_of_body,
    is_response_start,
    parse_sse,
)
from tests.test_stream_run import Gate, ScriptedWorkflow, final_state

GUARD = 5.0
TIMEOUT = 120.0
QUESTION = "How do I connect to the VPN?"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

async def wait_event(event, timeout=GUARD):
    if isinstance(event, asyncio.Event):
        await asyncio.wait_for(event.wait(), timeout)
    elif not await asyncio.to_thread(event.wait, timeout):
        raise AssertionError("event not set in time")


async def wait_until(predicate, timeout=GUARD):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.002)


def post(harness, question=QUESTION):
    return asyncio.ensure_future(
        harness.request("POST", "/api/chat/stream", json_body={"question": question})
    )


def respond(run, *, spec_version="2.4", **harness_kwargs):
    """Call SSEResponse directly for a run created with ``service.start_run``."""
    response = streaming.SSEResponse(
        run,
        build_done=lambda final: routes._chat_response(final, run.request_id, run.question),
        heartbeat_seconds=15.0,
        retry_after_seconds=5,
    )
    return ASGIHarness(response, spec_version=spec_version, **harness_kwargs)


def deltas(parsed):
    return [event.data["text"] for event in parsed.events if event.event == "delta"]


def rows(service):
    return [(row["status"], row["error_type"]) for row in service.audit.rows]


@pytest.fixture
def stop_calls(monkeypatch):
    """Records every request_stop(reason)."""
    calls = []
    real = streaming.StreamRun.request_stop

    def spy(self, reason):
        calls.append(reason)
        return real(self, reason)

    monkeypatch.setattr(streaming.StreamRun, "request_stop", spy)
    return calls


@pytest.fixture
def completed_claimers(monkeypatch):
    """Thread names of every claim('completed')."""
    names = []
    real = streaming.StreamOutcome.claim

    def spy(self, status, *args, **kwargs):
        if status == "completed":
            names.append(threading.current_thread().name)
        return real(self, status, *args, **kwargs)

    monkeypatch.setattr(streaming.StreamOutcome, "claim", spy)
    return names


@pytest.fixture
def received_batches(monkeypatch):
    """Every value returned by EventChannel.receive_batch, in order."""
    batches = []
    real = streaming.EventChannel.receive_batch

    async def spy(self, max_wait):
        result = await real(self, max_wait)
        batches.append(result)
        return result

    monkeypatch.setattr(streaming.EventChannel, "receive_batch", spy)
    return batches


def kb_llm(script):
    return FakeLLM(
        structured={RouteDecision: [{"route": "kb"}], EvidenceGrade: [{"grade": "good"}]},
        text=[script],
    )


# -------------------------------------------------------------------
# Success sequence
# -------------------------------------------------------------------

@pytest.mark.anyio
async def test_success_sequence_order(stream_app, stop_calls, completed_claimers, received_batches):
    service = stream_app(ScriptedWorkflow([("chunk", "A"), ("final", final_state())]))
    harness = ASGIHarness(app.main.app)
    done_gate = harness.gate(body_contains(b"event: done"))

    request = post(harness)
    await wait_event(done_gate.reached)
    [run] = service.started

    # While "done" is held, the worker sets the terminal, finishes and waits
    # for a decision (its hooks may trail the consumer's receive slightly).
    for hook in ("terminal_set", "finished", "waiting_for_decision"):
        await wait_event(run.hooks[hook])
    assert not run.outcome.decided.is_set()
    assert not run.hooks["audit_written"].is_set() and service.audit.rows == []

    done_gate.release.set()
    result = await request
    await wait_event(run.hooks["worker_exited"])

    assert run.outcome.decision.status == "completed"
    assert rows(service) == [("success", None)]
    assert result.complete
    assert stop_calls == []
    assert completed_claimers and not any(n.startswith("bp-stream") for n in completed_claimers)
    assert received_batches[-1][-1][0] == "final"  # no receive after "done"
    assert run.lease.consumer_done and run.lease.released


@pytest.mark.anyio
@pytest.mark.parametrize("count", [CHANNEL_MAX_EVENTS, 600])
async def test_slow_consumer_with_a_full_channel(stream_app, stop_calls, count):
    texts = [f"t{i} " for i in range(count)]
    final = final_state(answer="".join(texts).strip())
    service = stream_app(ScriptedWorkflow([("chunks", texts), ("final", final)]))
    harness = ASGIHarness(app.main.app)
    header_gate = harness.gate(is_response_start)

    request = post(harness)
    await wait_event(header_gate.reached)
    [run] = service.started
    await wait_until(lambda: len(run.channel) == CHANNEL_MAX_EVENTS)

    if count == CHANNEL_MAX_EVENTS:
        # Exactly full: the terminal slot ignores the limit, the worker waits for a decision.
        await wait_event(run.hooks["waiting_for_decision"])
        assert run.hooks["finished"].is_set()
    else:
        await asyncio.sleep(0.05)
        assert not run.hooks["terminal_set"].is_set()  # blocked publishing at 256
        assert len(run.channel) == CHANNEL_MAX_EVENTS
    assert service.audit.rows == []

    header_gate.release.set()
    result = await request
    await wait_event(run.hooks["worker_exited"])

    parsed = parse_sse(result.body)
    assert_stream_grammar(parsed, connected=True)
    assert deltas(parsed) == texts
    assert parsed.names[-1] == "done"
    assert "".join(deltas(parsed)).strip() == parsed.events[-1].data["answer"]
    assert rows(service) == [("success", None)]
    assert stop_calls == []


@pytest.mark.anyio
async def test_insufficient_evidence_streams_one_delta_and_no_rewrites(stream_app, fake_rag):
    fake_rag(
        llm=FakeLLM(
            structured={RouteDecision: [{"route": "kb"}], EvidenceGrade: [{"grade": "weak"}]},
            text=["rewrite one", "rewrite two"],
            stream=True,  # rewrites would stream if they were not tagged
        ),
        retriever=FakeRetriever([kb_doc("Unrelated.")]),
        web=FakeWebSearch({"results": []}),
    )
    service = stream_app(streaming.stream_ask)

    result = await post(ASGIHarness(app.main.app))
    [run] = service.started
    await wait_event(run.hooks["worker_exited"])

    parsed = parse_sse(result.body)
    done = parsed.events[-1].data
    assert parsed.names[-1] == "done" and done["source_used"] == "insufficient_evidence"
    assert deltas(parsed) == [done["answer"]]
    assert not any("rewrite" in text for text in deltas(parsed))
    assert rows(service) == [("insufficient_evidence", None)]


@pytest.mark.anyio
async def test_end_of_body_failure_after_completed_keeps_success(stream_app, stop_calls):
    service = stream_app(ScriptedWorkflow([("chunk", "A"), ("final", final_state())]))
    harness = ASGIHarness(app.main.app, spec_version="2.4")
    harness.fail_send(is_end_of_body)

    result = await post(harness)
    [run] = service.started
    await wait_event(run.hooks["worker_exited"])

    assert parse_sse(result.body).names[-1] == "done"
    assert run.outcome.decision.status == "completed"
    assert rows(service) == [("success", None)]
    assert stop_calls == []


@pytest.mark.anyio
async def test_non_ascii_and_newlines_survive_framing(stream_app):
    texts = ["Größe ✓ ", "日本語\n", "line \"two\"\r\n", "tab\tend"]
    final = final_state(answer="".join(texts).strip())
    service = stream_app(ScriptedWorkflow([("chunks", texts), ("final", final)]))

    result = await post(ASGIHarness(app.main.app))
    await wait_event(service.started[0].hooks["worker_exited"])

    parsed = parse_sse(result.body)
    assert_stream_grammar(parsed, connected=True)
    assert deltas(parsed) == texts
    assert "".join(deltas(parsed)).strip() == parsed.events[-1].data["answer"]
    for frame in result.body.split(b"\n\n"):
        assert frame.count(b"data: ") <= 1  # one single-line data field per event


# -------------------------------------------------------------------
# Consumer exit classification (section 7.4)
# -------------------------------------------------------------------

def consumer_stops(calls):
    return [reason for reason in calls if reason in CONSUMER_EXITS]


@pytest.mark.anyio
async def test_error_delivered_is_a_normal_exit(stream_app, stop_calls):
    service = stream_app(ScriptedWorkflow([("chunk", "A"), ("raise", RuntimeError("boom"))]))

    result = await post(ASGIHarness(app.main.app))
    [run] = service.started
    await wait_event(run.hooks["worker_exited"])

    assert parse_sse(result.body).names == ["start", "delta", "error"]
    assert consumer_stops(stop_calls) == []
    assert run.lease.consumer_done


@pytest.mark.anyio
async def test_timeout_protocol_requests_one_stop(stream_app, stop_calls):
    clock = FakeClock()
    gate = Gate()
    service = stream_app(ScriptedWorkflow([("chunk", "A"), ("gate", gate), ("final", final_state())]), clock=clock)
    harness = ASGIHarness(app.main.app)

    def expire_after_delta(message):
        if body_contains(b"event: delta")(message):
            clock.advance(TIMEOUT + 1)
        return False

    harness.disconnect_on(expire_after_delta)
    result = await post(harness)
    [run] = service.started
    gate.release.set()
    await wait_event(run.hooks["worker_exited"])

    assert parse_sse(result.body).names == ["start", "delta", "error"]
    assert stop_calls == ["timeout"]
    assert rows(service) == [("failed", "TimeoutError")]


@pytest.mark.anyio
@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
async def test_server_cancelling_the_response_is_classified_cancelled(stream_app, stop_calls, spec_version):
    gate = Gate()
    service = stream_app(ScriptedWorkflow([("chunk", "A"), ("gate", gate), ("final", final_state())]))
    harness = ASGIHarness(app.main.app, spec_version=spec_version)

    request = post(harness)
    await harness.wait_for_send(body_contains(b"event: delta"))
    [run] = service.started
    await asyncio.sleep(0)  # let the consumer return to receive_batch
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    gate.release.set()
    await wait_event(run.hooks["worker_exited"])

    decision = run.outcome.decision
    assert (decision.status, decision.error_type) == ("cancelled", "Cancelled")
    assert stop_calls == ["cancelled"]
    assert rows(service) == [("cancelled", "Cancelled")]


@pytest.mark.anyio
async def test_disconnect_seen_before_done_with_queued_events_is_cancelled(stream_app):
    texts = [f"t{i} " for i in range(50)]
    service = stream_app(ScriptedWorkflow([("chunks", texts), ("final", final_state())]))
    harness = ASGIHarness(app.main.app, spec_version="2.3", uvicorn_faithful=True)
    header_gate = harness.gate(is_response_start)
    harness.disconnect_on(body_contains(b"event: start"))

    request = post(harness)
    await wait_event(header_gate.reached)
    [run] = service.started
    await wait_event(run.hooks["waiting_for_decision"])  # 50 events and the final state queued
    header_gate.release.set()
    await request
    await wait_event(run.hooks["worker_exited"])

    decision = run.outcome.decision
    assert (decision.status, decision.error_type) == ("cancelled", "ClientDisconnected")
    assert rows(service) == [("cancelled", "ClientDisconnected")]


# -------------------------------------------------------------------
# Outcome and finalisation
# -------------------------------------------------------------------

@pytest.mark.anyio
async def test_header_send_timeout_when_the_graph_finishes_first(stream_app):  # 12a
    service = stream_app(ScriptedWorkflow([("chunk", "A"), ("final", final_state())]), timeout=1.0)
    harness = ASGIHarness(app.main.app)
    harness.gate(is_response_start)  # never released

    request = post(harness)
    await wait_until(lambda: service.started)
    [run] = service.started
    await wait_event(run.hooks["waiting_for_decision"])
    result = await request  # the real deadline cancels the blocked header send
    await wait_event(run.hooks["worker_exited"])

    assert result.delivered == []
    assert [m["type"] for m in harness.attempted] == ["http.response.start"]
    decision = run.outcome.decision
    assert (decision.status, decision.marker) == ("timeout", "deadline before response start")
    assert rows(service) == [("failed", "TimeoutError")]


@pytest.mark.anyio
async def test_header_send_timeout_when_the_deadline_comes_first(stream_app):  # 12b
    gate = Gate()
    service = stream_app(ScriptedWorkflow([("gate", gate), ("final", final_state())]), timeout=0.3)
    harness = ASGIHarness(app.main.app)
    harness.gate(is_response_start)

    result = await post(harness)
    [run] = service.started
    assert not run.hooks["terminal_set"].is_set()
    gate.release.set()  # the final state arrives after the decision and is discarded
    await wait_event(run.hooks["worker_exited"])

    assert result.delivered == []
    assert [m["type"] for m in harness.attempted] == ["http.response.start"]
    assert run.outcome.decision.marker == "deadline before response start"
    assert rows(service) == [("failed", "TimeoutError")]


@pytest.mark.anyio
@pytest.mark.parametrize("count", [CHANNEL_MAX_EVENTS, 300])
async def test_graph_failure_with_a_full_channel_sends_one_error(stream_app, count):
    texts = [f"t{i} " for i in range(count)]
    service = stream_app(ScriptedWorkflow([("chunks", texts), ("raise", RuntimeError("boom"))]))
    harness = ASGIHarness(app.main.app)
    header_gate = harness.gate(is_response_start)

    request = post(harness)
    await wait_event(header_gate.reached)
    [run] = service.started
    await wait_until(lambda: len(run.channel) == CHANNEL_MAX_EVENTS)
    if count == CHANNEL_MAX_EVENTS:
        await wait_event(run.hooks["worker_exited"])  # failed and audited while the response is held
    header_gate.release.set()
    result = await request
    await wait_event(run.hooks["worker_exited"])

    parsed = parse_sse(result.body)
    assert_stream_grammar(parsed, connected=True)
    assert deltas(parsed) == texts
    assert parsed.names.count("error") == 1 and parsed.names[-1] == "error"
    assert rows(service) == [("failed", "RuntimeError")]


@pytest.mark.anyio
async def test_empty_answer_is_one_generic_error(stream_app):
    service = stream_app(ScriptedWorkflow([("final", final_state(answer=""))]))

    result = await post(ASGIHarness(app.main.app))
    [run] = service.started
    await wait_event(run.hooks["worker_exited"])

    parsed = parse_sse(result.body)
    assert parsed.names == ["start", "error"]
    assert parsed.events[-1].data["code"] == "internal_error"
    assert rows(service) == [("failed", "EmptyAnswer")]


@pytest.mark.anyio
async def test_blocked_timeout_error_send_is_abandoned_after_the_grace(stream_app, monkeypatch):
    monkeypatch.setattr(streaming, "TIMEOUT_ERROR_GRACE", 0.2)
    gate = Gate()
    service = stream_app(ScriptedWorkflow([("gate", gate), ("final", final_state())]), timeout=0.3)
    harness = ASGIHarness(app.main.app)
    harness.gate(body_contains(b"event: error"))  # never released

    started = time.monotonic()
    result = await post(harness)
    elapsed = time.monotonic() - started
    [run] = service.started
    gate.release.set()
    await wait_event(run.hooks["worker_exited"])

    assert elapsed < 0.3 + 0.2 + 1.0
    assert parse_sse(result.body).names == ["start"]  # error attempted, never delivered
    assert not result.complete
    assert run.outcome.decision.marker == "deadline during response"
    assert rows(service) == [("failed", "TimeoutError")]


@pytest.mark.anyio
async def test_unresponsive_consumer_backstop_through_the_response(stream_app):  # scenario 18
    clock = FakeClock()
    texts = [f"t{i} " for i in range(4 * CHANNEL_MAX_EVENTS)]  # refills after the first batch
    service = stream_app(ScriptedWorkflow([("chunks", texts), ("final", final_state())]), clock=clock)
    harness = ASGIHarness(app.main.app)
    delta_gate = harness.gate(body_contains(b"event: delta"))

    request = post(harness)
    await wait_event(delta_gate.reached)
    [run] = service.started
    await wait_until(lambda: len(run.channel) == CHANNEL_MAX_EVENTS)  # the worker is blocked publishing
    clock.advance(TIMEOUT + FINALIZE_GRACE + 1)
    await wait_event(run.hooks["worker_exited"])  # the worker's backstop decided and audited
    delta_gate.release.set()
    result = await request

    decision = run.outcome.decision
    assert (decision.status, decision.marker) == ("timeout", "consumer unresponsive at deadline")
    assert "done" not in parse_sse(result.body).names
    assert rows(service) == [("failed", "TimeoutError")]
    assert run.lease.consumer_done and run.lease.released


# -------------------------------------------------------------------
# Attach window
# -------------------------------------------------------------------

@pytest.mark.anyio
async def test_worker_ahead_of_attach_streams_everything_before_the_deadline(stream_app):
    texts = [f"t{i} " for i in range(300)]
    final = final_state(answer="".join(texts).strip())
    service = stream_app(ScriptedWorkflow([("chunks", texts), ("final", final)]), clock=FakeClock())
    run = service.start_run(QUESTION, "req-ahead", asyncio.get_running_loop())
    await wait_until(lambda: len(run.channel) == CHANNEL_MAX_EVENTS)  # unattached, blocked at 256

    result = await respond(run).request("POST", "/api/chat/stream")
    await wait_event(run.hooks["worker_exited"])

    parsed = parse_sse(result.body)
    assert deltas(parsed) == texts and parsed.names[-1] == "done"
    assert rows(service) == [("success", None)]


@pytest.mark.anyio
async def test_attach_past_the_grace_by_a_blocked_worker_is_a_consumer_timeout(stream_app):
    clock = FakeClock()
    gate = Gate()
    service = stream_app(ScriptedWorkflow([("gate", gate), ("final", final_state())]), clock=clock)
    run = service.start_run(QUESTION, "req-blocked", asyncio.get_running_loop())
    await wait_event(gate.reached)
    clock.advance(TIMEOUT + ATTACH_GRACE + 1)  # the worker can't check for abandonment while gated

    result = await respond(run).request("POST", "/api/chat/stream")
    gate.release.set()
    await wait_event(run.hooks["worker_exited"])

    assert result.status != 503 and result.delivered == []
    assert run.outcome.decision.marker == "deadline before response start"  # not abandoned
    assert rows(service) == [("failed", "TimeoutError")]


# -------------------------------------------------------------------
# Two-party lease
# -------------------------------------------------------------------

@pytest.mark.anyio
@pytest.mark.parametrize("variant", ["failure", "success"])
async def test_slot_stays_held_while_the_first_response_drains(stream_app, variant):
    if variant == "failure":
        steps = [("chunks", ["a ", "b "]), ("raise", RuntimeError("boom"))]
        hold = body_contains(b"event: delta")
    else:
        steps = [("chunk", "A"), ("final", final_state())]
        hold = is_end_of_body
    service = stream_app(ScriptedWorkflow(steps), max_concurrent=1)
    first = ASGIHarness(app.main.app)
    drain_gate = first.gate(hold)

    first_request = post(first)
    await wait_event(drain_gate.reached)
    [run] = service.started
    await wait_event(run.hooks["worker_exited"])  # worker done and audited
    assert len(service.audit.rows) == 1

    second = await post(ASGIHarness(app.main.app))
    assert second.status == 503 and second.headers["retry-after"] == "5"

    drain_gate.release.set()
    await first_request
    assert run.lease.released and service.capacity.in_use == 0

    third = await post(ASGIHarness(app.main.app))
    assert third.status == 200
    await wait_event(service.started[-1].hooks["worker_exited"])


# -------------------------------------------------------------------
# Backpressure and memory (real stream_ask)
# -------------------------------------------------------------------

@pytest.mark.anyio
async def test_backpressure_holds_the_model_and_delivers_everything(stream_app, fake_rag):
    script = ChunkScript(count=20_000)
    fake_rag(llm=kb_llm(script), retriever=FakeRetriever([kb_doc("VPN uses GlobalProtect.")]))
    service = stream_app(streaming.stream_ask)
    harness = ASGIHarness(app.main.app)
    header_gate = harness.gate(is_response_start)

    request = post(harness)
    await wait_event(header_gate.reached)
    [run] = service.started
    await wait_until(lambda: len(run.channel) == CHANNEL_MAX_EVENTS)
    await asyncio.sleep(0.05)
    assert script.produced <= CHANNEL_MAX_EVENTS + 1
    assert len(run.channel) == CHANNEL_MAX_EVENTS

    header_gate.release.set()
    result = await request
    await wait_event(run.hooks["worker_exited"])

    parsed = parse_sse(result.body)
    assert deltas(parsed) == [script.text(i) for i in range(script.count)]
    assert "".join(deltas(parsed)).strip() == parsed.events[-1].data["answer"]
    assert rows(service) == [("success", None)]


@pytest.mark.anyio
async def test_backpressure_bounds_memory_for_a_very_long_answer(stream_app, fake_rag):
    script = ChunkScript(count=200_000)
    fake_rag(llm=kb_llm(script), retriever=FakeRetriever([kb_doc("VPN uses GlobalProtect.")]))
    service = stream_app(streaming.stream_ask)
    harness = ASGIHarness(app.main.app, spec_version="2.3")
    header_gate = harness.gate(is_response_start)

    tracemalloc.start()
    try:
        request = post(harness)
        await wait_event(header_gate.reached)
        [run] = service.started
        await wait_until(lambda: len(run.channel) == CHANNEL_MAX_EVENTS)
        baseline = tracemalloc.take_snapshot()
        await asyncio.sleep(0.2)  # a producer that ignored backpressure would keep growing here
        growth = sum(stat.size_diff for stat in tracemalloc.take_snapshot().compare_to(baseline, "filename"))
    finally:
        tracemalloc.stop()

    assert script.produced <= CHANNEL_MAX_EVENTS + 1
    assert growth < 5 * 1024 * 1024

    harness.disconnect()  # releases the held header send; the run is cancelled
    await request
    await wait_event(run.hooks["worker_exited"])
    assert script.produced <= CHANNEL_MAX_EVENTS + 2
    assert rows(service) == [("cancelled", "ClientDisconnected")]


def test_raw_messages_mode_does_not_apply_backpressure(fake_rag):
    """Characterisation: without the forwarder the model runs to completion unread."""
    script = ChunkScript(count=2_000)
    fake_rag(llm=kb_llm(script), retriever=FakeRetriever([kb_doc("VPN uses GlobalProtect.")]))
    stream = workflow.agent_graph.stream(workflow._initial_state(QUESTION), stream_mode="messages")
    try:
        next(stream)  # read one item, then stop reading
        deadline = time.monotonic() + GUARD
        while script.produced < script.count and time.monotonic() < deadline:
            time.sleep(0.005)
        assert script.produced == script.count
    finally:
        stream.close()


# -------------------------------------------------------------------
# Concurrency
# -------------------------------------------------------------------

class EchoWorkflow:
    """Each run answers with its own question, after a shared gate."""

    def __init__(self, gate_count):
        self.gates = [Gate() for _ in range(gate_count)]
        self._next = 0
        self._lock = threading.Lock()

    def __call__(self, question, on_chunk):
        with self._lock:
            gate = self.gates[self._next]
            self._next += 1

        def run():
            gate.reached.set()
            if not gate.release.wait(GUARD):
                raise TimeoutError("gate never released")
            on_chunk(f"answer to {question}")
            yield ("final", final_state(answer=f"answer to {question}"))

        return run()


@pytest.mark.anyio
async def test_concurrent_streams_do_not_cross_talk(stream_app):
    echo = EchoWorkflow(5)
    service = stream_app(echo, max_concurrent=5)
    questions = [f"Question number {i}" for i in range(5)]

    requests = [post(ASGIHarness(app.main.app), question) for question in questions]
    for gate in echo.gates:
        await wait_event(gate.reached)
    health = await ASGIHarness(app.main.app).request("GET", "/api/health")
    assert health.status == 200

    for gate in echo.gates:
        gate.release.set()
    results = await asyncio.gather(*requests)
    for run in service.started:
        await wait_event(run.hooks["worker_exited"])

    for question, result in zip(questions, results):
        parsed = parse_sse(result.body)
        assert_stream_grammar(parsed, connected=True)
        start, done = parsed.events[0].data, parsed.events[-1].data
        assert start["request_id"] == done["request_id"]
        assert deltas(parsed) == [f"answer to {question}"] == [done["answer"]]
    assert len({parse_sse(r.body).events[0].data["request_id"] for r in results}) == 5
    assert sorted(rows(service)) == [("success", None)] * 5
