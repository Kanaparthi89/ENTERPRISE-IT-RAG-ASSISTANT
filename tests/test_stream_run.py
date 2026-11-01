"""Tests for the A1-2 streaming core: outcome, lease, run lifecycle, service.

A scripted workflow stands in for ``stream_ask`` (one test uses the real one),
and a minimal consumer stands in for the SSE response added in A1-3. Every
service created through ``make_service`` is checked after the test: no live
worker threads, capacity fully released exactly once per run, exactly one
audit write per run, and no defensive "worker exited undecided" claim.
"""

import asyncio
import logging
import threading
import time

import pytest

from app.api import streaming
from app.api.streaming import (
    ABORTED,
    ATTACH_GRACE,
    CHANNEL_MAX_EVENTS,
    END,
    FINALIZE_GRACE,
    WORKER_THREAD_PREFIX,
    StreamCapacity,
    StreamOutcome,
    StreamService,
)
from app.rag.state import EvidenceGrade, RouteDecision
from app.services.audit import get_recent_audits, init_db, write_audit
from tests.conftest import TEST_SETTINGS, FakeLLM, FakeRetriever, kb_doc
from tests.streaming_support import ChunkScript, FakeClock, LoopClosingOnSchedule

GUARD = 5.0
TIMEOUT = 120.0


# -------------------------------------------------------------------
# Test doubles
# -------------------------------------------------------------------

class Gate:
    def __init__(self):
        self.reached = threading.Event()
        self.release = threading.Event()


class ScriptedWorkflow:
    """Stands in for stream_ask(question, on_chunk), driven by a list of steps.

    Steps: ("trace", step) | ("chunk", text) | ("chunks", [texts]) |
    ("gate", Gate) | ("raise", exc) | ("final", state)
    """

    def __init__(self, steps):
        self.steps = steps
        self.closed = threading.Event()
        self.stopped_by = None
        self.completed_steps = []

    def __call__(self, question, on_chunk):
        def run():
            try:
                for step in self.steps:
                    kind = step[0]
                    if kind == "trace":
                        yield ("trace", step[1])
                    elif kind == "chunk":
                        on_chunk(step[1])
                    elif kind == "chunks":
                        for text in step[1]:
                            on_chunk(text)
                    elif kind == "gate":
                        step[1].reached.set()
                        if not step[1].release.wait(GUARD):
                            raise TimeoutError("test gate never released")
                    elif kind == "raise":
                        raise step[1]
                    elif kind == "final":
                        yield ("final", step[1])
                    self.completed_steps.append(kind)
            except BaseException as exc:
                self.stopped_by = type(exc).__name__
                raise
            finally:
                self.closed.set()

        return run()


def final_state(answer="Install GlobalProtect.", source_used="private_kb"):
    return {
        "answer": answer,
        "source_used": source_used,
        "trace": ["Router → KB", "Answer generation → PRIVATE KB"],
        "citations": [{"title": "handbook.md", "url": "", "type": "private_kb"}],
    }


class AuditLog:
    """Thread-safe audit writer double that records the outcome at write time."""

    def __init__(self):
        self.rows = []
        self._lock = threading.Lock()

    def __call__(self, **row):
        with self._lock:
            self.rows.append(row)


async def wait_event(event, timeout=GUARD):
    if not await asyncio.to_thread(event.wait, timeout):
        raise AssertionError("event not set in time")


async def wait_until(predicate, timeout=GUARD):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.002)


async def consume(run, *, max_wait=GUARD, stop_after=None, exit_kind=None):
    """Minimal consumer: attach, receive, claim completion after "done".

    ``stop_after=n`` exits after n items with ``exit_kind`` (simulating a
    disconnect or send failure).
    """
    received = []
    assert run.attach()
    run.mark_response_started()
    kind = None
    try:
        while True:
            batch = await run.channel.receive_batch(max_wait)
            if batch is END or batch is ABORTED:
                return received, batch
            if batch == []:
                raise AssertionError("consumer idled: lost wake-up")
            for item in batch:
                received.append(item)
                if stop_after is not None and len(received) >= stop_after:
                    kind = exit_kind
                    return received, "stopped"
                if item[0] == "final":
                    run.claim_completed()  # "done" was sent
                    return received, "final"
                if item[0] == "error":
                    return received, "error"
    finally:
        run.consumer_finished(kind)


def deltas(received):
    return [payload["text"] for kind, payload in received if kind == "delta"]


# -------------------------------------------------------------------
# Service fixture with leak checks
# -------------------------------------------------------------------

def stream_threads():
    return [thread for thread in threading.enumerate() if thread.name.startswith(WORKER_THREAD_PREFIX)]


@pytest.fixture
def make_service():
    created = []

    def make(workflow, *, max_concurrent=2, clock=None, audit=None, timeout=TIMEOUT):
        audit = audit if audit is not None else AuditLog()
        service = StreamService(
            max_concurrent=max_concurrent,
            timeout_seconds=timeout,
            clock=clock or FakeClock(),
            poll_seconds=0.01,
            workflow_stream=workflow,
            audit_writer=audit,
        )
        service.audit = audit
        service.started = []
        real_start = service.start_run

        def start_run(*args, **kwargs):
            run = real_start(*args, **kwargs)
            if run is not None:
                service.started.append(run)
            return run

        service.start_run = start_run
        created.append(service)
        return service

    yield make

    for service in created:
        service.shutdown(wait=False)
    deadline = time.monotonic() + GUARD
    while stream_threads() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert stream_threads() == [], "leaked stream worker threads"
    for service in created:
        assert service.capacity.in_use == 0, "capacity not released"
        assert service.capacity.acquired_total == service.capacity.released_total
        for run in service.started:
            assert run.lease.released, f"lease not released for {run.request_id}"
            # Exactly one audit row for every run whose worker started; none
            # for a run whose worker was cancelled before it started.
            expected = 1 if run.worker_started else 0
            assert run.audit_writes == expected, f"audit writes for {run.request_id}: {run.audit_writes}"
            assert not run.defensive_claim_fired


def start(service, question="How do I connect to the VPN?", request_id="req-1"):
    run = service.start_run(question, request_id, asyncio.get_running_loop())
    assert run is not None
    return run


# -------------------------------------------------------------------
# Outcome
# -------------------------------------------------------------------

def test_first_claim_wins_and_is_final():
    outcome = StreamOutcome()

    assert outcome.claim("timeout", "TimeoutError", "deadline during response")
    assert not outcome.claim("completed")
    assert not outcome.claim("cancelled", "ClientDisconnected")
    assert outcome.decided.is_set()
    assert outcome.decision.status == "timeout"
    with pytest.raises(ValueError):
        outcome.claim("success")


def test_concurrent_claims_have_exactly_one_winner():
    statuses = ["completed", "failed", "timeout", "cancelled"]
    for _ in range(1_000):
        outcome = StreamOutcome()
        barrier = threading.Barrier(len(statuses))
        wins = []

        def claim(status):
            barrier.wait()
            if outcome.claim(status):
                wins.append(status)

        threads = [threading.Thread(target=claim, args=(status,)) for status in statuses]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(wins) == 1
        assert outcome.decision.status == wins[0]


# -------------------------------------------------------------------
# Capacity and leases
# -------------------------------------------------------------------

@pytest.mark.parametrize("order", [("worker", "consumer"), ("consumer", "worker")])
def test_lease_released_exactly_once_after_both_parties(order):
    capacity = StreamCapacity(1)
    releases = []
    lease = capacity.try_acquire(lambda: releases.append(True))

    marks = {"worker": lease.mark_worker_done, "consumer": lease.mark_consumer_done}
    assert marks[order[0]]() is False
    assert capacity.in_use == 1 and capacity.try_acquire() is None
    assert marks[order[1]]() is True
    for mark in marks.values():  # repeated marks are no-ops
        assert mark() is False

    assert releases == [True]
    assert capacity.in_use == 0 and capacity.released_total == 1
    assert capacity.try_acquire() is not None


def test_capacity_over_release_is_impossible():
    capacity = StreamCapacity(1)
    lease = capacity.try_acquire()
    lease.mark_worker_done()
    lease.mark_consumer_done()

    with pytest.raises(ValueError):
        capacity._release()  # BoundedSemaphore guards against double release


def test_attach_and_abandon_are_mutually_exclusive():
    for _ in range(1_000):
        lease = StreamCapacity(1).try_acquire()
        barrier = threading.Barrier(2)
        results = {}

        def attach():
            barrier.wait()
            results["attach"] = lease.attach()

        def abandon():
            barrier.wait()
            results["abandon"] = lease.abandon()

        threads = [threading.Thread(target=attach), threading.Thread(target=abandon)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert results["attach"] != results["abandon"]
        assert lease.attach_state == ("attached" if results["attach"] else "abandoned")


# -------------------------------------------------------------------
# Successful runs
# -------------------------------------------------------------------

@pytest.mark.anyio
async def test_successful_run_streams_completes_and_audits_once(make_service):
    service = make_service(
        ScriptedWorkflow(
            [("trace", "Router → KB"), ("chunks", ["Install ", "GlobalProtect."]), ("final", final_state())]
        )
    )
    run = start(service)
    claimed_by = []
    real_claim = run.outcome.claim

    def recording_claim(status, *args, **kwargs):
        claimed_by.append((status, threading.current_thread().name))
        return real_claim(status, *args, **kwargs)

    run.outcome.claim = recording_claim

    received, end = await consume(run)
    await wait_event(run.hooks["worker_exited"])

    assert end == "final"
    assert received[0] == ("trace", {"index": 0, "step": "Router → KB"})
    assert deltas(received) == ["Install ", "GlobalProtect."]
    assert received[-1] == ("final", final_state())
    assert run.outcome.decision.status == "completed"
    assert claimed_by == [("completed", threading.current_thread().name)]  # never the worker
    assert not run.stop_requested  # normal completion requests no stop
    [row] = service.audit.rows
    assert row["status"] == "success"
    assert row["trace"] == final_state()["trace"]
    assert row["request_id"] == "req-1"


@pytest.mark.anyio
async def test_insufficient_evidence_is_audited_as_such(make_service):
    service = make_service(
        ScriptedWorkflow([("chunk", "Not enough evidence."), ("final", final_state("x", "insufficient_evidence"))])
    )
    run = start(service)

    await consume(run)
    await wait_event(run.hooks["worker_exited"])

    assert service.audit.rows[0]["status"] == "insufficient_evidence"


@pytest.mark.anyio
async def test_worker_waits_for_the_consumer_decision(make_service):
    service = make_service(ScriptedWorkflow([("chunk", "A"), ("final", final_state())]))
    run = start(service)

    await wait_event(run.hooks["waiting_for_decision"])
    await asyncio.sleep(0.05)
    assert not run.hooks["audit_written"].is_set()  # no audit before the decision
    assert run.channel.state == "finished"  # finish() did not end the wait

    await consume(run)
    await wait_event(run.hooks["worker_exited"])
    assert service.audit.rows[0]["status"] == "success"


@pytest.mark.anyio
async def test_claim_completed_requires_a_started_response(make_service):
    service = make_service(ScriptedWorkflow([("final", final_state())]))
    run = start(service)

    with pytest.raises(RuntimeError, match="response started"):
        run.claim_completed()

    await consume(run)
    await wait_event(run.hooks["worker_exited"])


@pytest.mark.anyio
async def test_send_failure_after_completion_keeps_success(make_service):
    service = make_service(ScriptedWorkflow([("chunk", "A"), ("final", final_state())]))
    run = start(service)
    assert run.attach()
    run.mark_response_started()

    while True:
        batch = await run.channel.receive_batch(GUARD)
        if batch and batch[-1][0] == "final":
            break
    run.claim_completed()  # "done" sent
    run.consumer_finished("send_failed")  # the end-of-body send then failed
    await wait_event(run.hooks["worker_exited"])

    assert run.outcome.decision.status == "completed"
    assert not run.stop_requested
    assert service.audit.rows[0]["status"] == "success"


# -------------------------------------------------------------------
# Failures
# -------------------------------------------------------------------

@pytest.mark.anyio
async def test_graph_failure_delivers_exactly_one_error(make_service):
    service = make_service(ScriptedWorkflow([("chunk", "partial"), ("raise", ValueError("boom"))]))
    run = start(service)

    received, end = await consume(run)
    await wait_event(run.hooks["worker_exited"])

    assert end == "error"
    assert deltas(received) == ["partial"]
    assert [kind for kind, _ in received].count("error") == 1
    assert received[-1] == ("error", {"code": "internal_error"})
    assert run.outcome.decision.error_type == "ValueError"
    assert not run.stop_requested
    [row] = service.audit.rows
    assert (row["status"], row["error_type"]) == ("failed", "ValueError")
    assert "boom" not in str(row)


@pytest.mark.anyio
async def test_graph_failure_with_a_full_channel_delivers_one_error(make_service):
    chunks = [f"c{index} " for index in range(CHANNEL_MAX_EVENTS)]
    service = make_service(ScriptedWorkflow([("chunks", chunks), ("raise", RuntimeError("late failure"))]))
    run = start(service)

    # The consumer attaches only after the worker has failed and audited.
    await wait_event(run.hooks["audit_written"])
    assert len(run.channel) == CHANNEL_MAX_EVENTS
    assert run.lease.worker_done and not run.lease.released  # consumer still holds the slot

    received, end = await consume(run)

    assert end == "error"
    assert deltas(received) == chunks
    assert [kind for kind, _ in received].count("error") == 1
    assert service.audit.rows[0]["status"] == "failed"


@pytest.mark.anyio
async def test_empty_answer_fails(make_service):
    service = make_service(ScriptedWorkflow([("final", final_state(answer=""))]))
    run = start(service)

    _, end = await consume(run)
    await wait_event(run.hooks["worker_exited"])

    assert end == "error"
    assert run.outcome.decision.error_type == "EmptyAnswer"


# -------------------------------------------------------------------
# Disconnect, cancellation, send failure (either completion order)
# -------------------------------------------------------------------

@pytest.mark.anyio
@pytest.mark.parametrize(
    "exit_kind, error_type",
    [("disconnected", "ClientDisconnected"), ("send_failed", "SendFailed"), ("cancelled", "Cancelled")],
)
async def test_consumer_exit_mid_stream_cancels_and_stops_worker(make_service, exit_kind, error_type):
    gate = Gate()
    workflow = ScriptedWorkflow([("trace", "Router → KB"), ("gate", gate), ("chunk", "never"), ("final", final_state())])
    service = make_service(workflow)
    run = start(service)
    assert run.attach()
    run.mark_response_started()

    batch = await run.channel.receive_batch(GUARD)
    assert batch == [("trace", {"index": 0, "step": "Router → KB"})]
    await wait_event(gate.reached)  # the worker is inside its in-flight "external call"
    run.consumer_finished(exit_kind)

    assert run.outcome.decision.status == "cancelled"
    assert run.outcome.decision.error_type == error_type
    assert run.stop_requested
    assert run.lease.consumer_done and not run.lease.released  # worker still in its "call"
    assert service.capacity.in_use == 1

    gate.release.set()  # the in-flight external call returns
    await wait_event(run.hooks["worker_exited"])

    assert workflow.stopped_by == "StreamStopped"  # the next chunk was refused
    assert run.lease.released and service.capacity.in_use == 0
    [row] = service.audit.rows
    assert (row["status"], row["error_type"]) == ("cancelled", error_type)
    assert row["trace"][-1].startswith("Stream → ")


@pytest.mark.anyio
async def test_immediate_disconnect_race_stops_the_workflow_either_way(make_service):
    """Disconnecting right after the first event races the worker.

    Three interleavings are valid: the worker's ``publish`` returns False
    after delivering the trace (the workflow is closed at its yield:
    GeneratorExit); the worker resumes and a later chunk is refused
    (StreamStopped); or the worker finishes the whole workflow before the
    disconnect lands (no exception) and waits for a decision. In every case
    the disconnect decides the outcome: cancelled, never success.
    """
    seen = set()
    for attempt in range(30):
        gate = Gate()
        gate.release.set()  # no in-flight pause: maximize the race window
        workflow = ScriptedWorkflow(
            [("trace", "t"), ("gate", gate), ("chunks", ["a", "b", "c"]), ("final", final_state())]
        )
        service = make_service(workflow)
        run = start(service, request_id=f"race-{attempt}")

        await consume(run, stop_after=1, exit_kind="disconnected")
        await wait_event(run.hooks["worker_exited"])

        assert workflow.stopped_by in {None, "GeneratorExit", "StreamStopped"}
        if workflow.stopped_by is not None:
            assert "final" not in workflow.completed_steps  # stopped before finishing
        seen.add(workflow.stopped_by)
        assert run.outcome.decision.error_type == "ClientDisconnected"
        assert [row["status"] for row in service.audit.rows] == ["cancelled"]
        assert run.lease.released
    assert seen  # at least one interleaving observed (which ones varies by machine)


@pytest.mark.anyio
async def test_consumer_error_is_recorded_as_failure(make_service):
    gate = Gate()
    service = make_service(ScriptedWorkflow([("trace", "t"), ("gate", gate), ("final", final_state())]))
    run = start(service)

    await consume(run, stop_after=1, exit_kind="consumer_error")
    gate.release.set()
    await wait_event(run.hooks["worker_exited"])

    assert run.outcome.decision.status == "failed"
    assert service.audit.rows[0]["error_type"] == "ConsumerError"


# -------------------------------------------------------------------
# Deadline, attach grace, finalize grace
# -------------------------------------------------------------------

@pytest.mark.anyio
async def test_late_attach_before_the_deadline_streams_normally(make_service):
    clock = FakeClock()
    chunks = [f"c{index} " for index in range(CHANNEL_MAX_EVENTS + 44)]
    service = make_service(ScriptedWorkflow([("chunks", chunks), ("final", final_state())]), clock=clock)
    run = start(service)

    await wait_until(lambda: len(run.channel) == CHANNEL_MAX_EVENTS)  # worker blocked: no consumer yet
    clock.advance(TIMEOUT - 1)  # still before the deadline

    received, end = await consume(run)
    await wait_event(run.hooks["worker_exited"])

    assert end == "final"
    assert deltas(received) == chunks
    assert service.audit.rows[0]["status"] == "success"


@pytest.mark.anyio
async def test_attach_within_grace_succeeds_and_consumer_decides_timeout(make_service):
    clock = FakeClock()
    service = make_service(ScriptedWorkflow([("final", final_state())]), clock=clock)
    run = start(service)
    await wait_event(run.hooks["waiting_for_decision"])

    clock.advance(TIMEOUT + ATTACH_GRACE / 2)  # past D, inside the attach grace
    assert run.attach()  # a delayed attach is not converted into an abandonment
    assert run.claim_timeout()  # the consumer's own deadline has expired
    run.consumer_finished()
    await wait_event(run.hooks["worker_exited"])

    decision = run.outcome.decision
    assert (decision.status, decision.marker) == ("timeout", "deadline before response start")
    [row] = service.audit.rows
    assert (row["status"], row["error_type"]) == ("failed", "TimeoutError")


@pytest.mark.anyio
async def test_unattached_run_is_abandoned_after_the_attach_grace(make_service):
    clock = FakeClock()
    service = make_service(ScriptedWorkflow([("final", final_state())]), clock=clock)
    run = start(service)
    await wait_event(run.hooks["waiting_for_decision"])

    clock.advance(TIMEOUT + ATTACH_GRACE - 0.001)
    await asyncio.sleep(0.05)
    assert not run.outcome.decided.is_set()  # not before D + ATTACH_GRACE

    clock.advance(0.001)
    await wait_event(run.hooks["worker_exited"])

    decision = run.outcome.decision
    assert (decision.status, decision.marker) == ("timeout", "abandoned: response never attached")
    assert run.lease.consumer_done and run.lease.released  # marked on the consumer's behalf
    assert run.attach() is False  # a late response gets refused (503 in A1-3)
    assert service.audit.rows[0]["error_type"] == "TimeoutError"


@pytest.mark.anyio
async def test_unattached_worker_blocked_on_full_channel_is_abandoned(make_service):
    clock = FakeClock()
    chunks = [f"c{index}" for index in range(CHANNEL_MAX_EVENTS * 2)]
    workflow = ScriptedWorkflow([("chunks", chunks), ("final", final_state())])
    service = make_service(workflow, clock=clock)
    run = start(service)
    await wait_until(lambda: len(run.channel) == CHANNEL_MAX_EVENTS)

    clock.advance(TIMEOUT + ATTACH_GRACE)
    await wait_event(run.hooks["worker_exited"])

    assert workflow.stopped_by == "StreamStopped"
    assert run.outcome.decision.marker == "abandoned: response never attached"
    assert run.lease.released


@pytest.mark.anyio
async def test_unresponsive_attached_consumer_hits_the_finalize_backstop(make_service):
    clock = FakeClock()
    service = make_service(ScriptedWorkflow([("final", final_state())]), clock=clock)
    run = start(service)
    assert run.attach()
    await wait_event(run.hooks["waiting_for_decision"])

    clock.advance(TIMEOUT + FINALIZE_GRACE - 0.001)
    await asyncio.sleep(0.05)
    assert not run.outcome.decided.is_set()

    clock.advance(0.001)
    await wait_event(run.hooks["audit_written"])
    assert run.outcome.decision.marker == "consumer unresponsive at deadline"
    assert run.stop_requested
    assert not run.lease.released  # the consumer has not exited yet

    assert await run.channel.receive_batch(GUARD) is ABORTED
    run.consumer_finished()  # normal exit after ABORTED
    await wait_event(run.hooks["worker_exited"])
    assert run.lease.released


@pytest.mark.anyio
async def test_timeout_during_response_stops_a_blocked_worker(make_service):
    chunks = [f"c{index}" for index in range(CHANNEL_MAX_EVENTS * 4)]
    workflow = ScriptedWorkflow([("chunks", chunks), ("final", final_state())])
    service = make_service(workflow)
    run = start(service)
    assert run.attach()
    run.mark_response_started()
    await wait_until(lambda: len(run.channel) == CHANNEL_MAX_EVENTS)  # worker blocked publishing

    assert run.claim_timeout()  # the consumer's deadline scope fired
    run.consumer_finished()
    await wait_event(run.hooks["worker_exited"])

    assert workflow.stopped_by == "StreamStopped"
    decision = run.outcome.decision
    assert (decision.status, decision.marker) == ("timeout", "deadline during response")
    assert service.audit.rows[0]["status"] == "failed"


# -------------------------------------------------------------------
# Event-loop closure
# -------------------------------------------------------------------

def test_closed_event_loop_finalizes_the_run(make_service):
    gate = Gate()
    service = make_service(ScriptedWorkflow([("trace", "t"), ("gate", gate), ("chunk", "x"), ("final", final_state())]))
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    holder = {}

    def run_loop():
        asyncio.set_event_loop(loop)

        async def create():
            holder["run"] = service.start_run("q", "req-loop", asyncio.get_running_loop())
            ready.set()

        loop.run_until_complete(create())

    thread = threading.Thread(target=run_loop)
    thread.start()
    thread.join(GUARD)
    assert ready.wait(GUARD)
    run = holder["run"]
    assert gate.reached.wait(GUARD)

    loop.close()  # the consumer's loop is gone
    gate.release.set()
    assert run.hooks["worker_exited"].wait(GUARD)

    decision = run.outcome.decision
    assert (decision.status, decision.error_type) == ("cancelled", "LoopClosed")
    assert run.lease.consumer_done and run.lease.released
    assert service.audit.rows[0]["status"] == "cancelled"


# -------------------------------------------------------------------
# Service: admission, saturation, submission failure
# -------------------------------------------------------------------

@pytest.mark.anyio
async def test_capacity_full_refuses_new_runs_and_logs_saturation(make_service, caplog):
    gates = [Gate(), Gate()]
    clock = FakeClock()

    class TwoWorkflows:
        def __init__(self):
            self.count = 0

        def __call__(self, question, on_chunk):
            gate = gates[self.count]
            self.count += 1
            return ScriptedWorkflow([("gate", gate), ("final", final_state())])(question, on_chunk)

    service = make_service(TwoWorkflows(), max_concurrent=2, clock=clock)
    first = start(service, request_id="a")
    second = start(service, request_id="b")
    clock.advance(3.0)

    with caplog.at_level(logging.WARNING, logger="app.api.streaming"):
        assert service.start_run("q", "c", asyncio.get_running_loop()) is None

    assert "2/2 in use" in caplog.text and "3.0s old" in caplog.text
    assert {run.request_id for run in service.active_runs()} == {"a", "b"}

    for gate in gates:
        gate.release.set()
    for run in (first, second):
        await consume(run)
        await wait_event(run.hooks["worker_exited"])
    assert service.active_runs() == []
    assert start(service, request_id="d") is not None
    await consume(service.started[-1])
    await wait_event(service.started[-1].hooks["worker_exited"])


@pytest.mark.anyio
async def test_submission_failure_releases_capacity(make_service):
    service = make_service(ScriptedWorkflow([("final", final_state())]))
    service._executor.shutdown(wait=True)

    assert service.start_run("q", "r", asyncio.get_running_loop()) is None
    assert service.capacity.in_use == 0
    assert service.capacity.acquired_total == service.capacity.released_total == 1
    assert service.audit.rows == []


@pytest.mark.anyio
async def test_worker_pool_is_dedicated_and_bounded(make_service):
    gate = Gate()
    service = make_service(ScriptedWorkflow([("gate", gate), ("final", final_state())]), max_concurrent=3)
    run = start(service)
    assert gate.reached.wait(GUARD)

    assert [thread.name for thread in stream_threads()][0].startswith(WORKER_THREAD_PREFIX)
    assert service._executor._max_workers == 3

    gate.release.set()
    await consume(run)
    await wait_event(run.hooks["worker_exited"])


# -------------------------------------------------------------------
# Integration with the real workflow and audit store
# -------------------------------------------------------------------

@pytest.mark.anyio
async def test_real_stream_ask_and_audit_store(fake_rag, make_service):
    init_db()
    raw = "\n Install GlobalProtect and sign in with MFA. \n"
    fake_rag(
        llm=FakeLLM(
            structured={RouteDecision: [{"route": "kb"}], EvidenceGrade: [{"grade": "good"}]},
            text=[ChunkScript.from_text(raw)],
        ),
        retriever=FakeRetriever([kb_doc("VPN uses GlobalProtect.")]),
    )
    service = make_service(streaming.stream_ask, audit=write_audit)
    run = start(service, request_id="real-1")

    received, end = await consume(run)
    await wait_event(run.hooks["worker_exited"])

    assert end == "final"
    final = received[-1][1]
    assert "".join(deltas(received)) == raw
    assert "".join(deltas(received)).strip() == final["answer"]
    [row] = get_recent_audits(limit=5)
    assert (row["request_id"], row["status"], row["source_used"]) == ("real-1", "success", "private_kb")


def test_cancelled_is_an_allowed_audit_status():
    init_db()
    write_audit(question="q", source_used="cancelled", status="cancelled", error_type="ClientDisconnected")

    [row] = get_recent_audits(limit=1)
    assert (row["status"], row["error_type"]) == ("cancelled", "ClientDisconnected")


def test_stream_settings_and_constants():
    assert TEST_SETTINGS.stream_enabled is False
    assert TEST_SETTINGS.stream_timeout_seconds == 120.0
    assert TEST_SETTINGS.stream_heartbeat_seconds == 15.0
    assert TEST_SETTINGS.max_concurrent_streams == 8
    assert TEST_SETTINGS.stream_retry_after_seconds == 5
    assert (CHANNEL_MAX_EVENTS, ATTACH_GRACE, FINALIZE_GRACE) == (256, 2.0, 5.0)


# -------------------------------------------------------------------
# Shutdown lifecycle: submitted work that has not started yet
# -------------------------------------------------------------------

def occupy_only_worker_thread(service):
    """Keep the pool's single thread busy so the next run stays queued."""
    occupied, release = threading.Event(), threading.Event()

    def blocker():
        occupied.set()
        release.wait(GUARD)

    service._executor.submit(blocker)
    assert occupied.wait(GUARD)
    return release


@pytest.mark.anyio
async def test_shutdown_without_wait_still_runs_a_queued_worker(make_service):
    service = make_service(ScriptedWorkflow([("chunk", "A"), ("final", final_state())]), max_concurrent=1)
    release = occupy_only_worker_thread(service)
    run = start(service)
    await asyncio.sleep(0.05)
    assert not run.worker_started  # queued behind the busy thread

    began = time.monotonic()
    service.shutdown(wait=False)
    assert time.monotonic() - began < 0.5  # non-blocking
    assert not run._future.cancelled()  # submitted work is never cancelled

    release.set()  # the queued worker now starts and completes its lifecycle
    received, end = await consume(run)
    await wait_event(run.hooks["worker_exited"])

    assert end == "final" and run.worker_started
    assert deltas(received) == ["A"]
    assert [row["status"] for row in service.audit.rows] == ["success"]
    assert run.lease.released and service.capacity.in_use == 0


@pytest.mark.anyio
async def test_shutdown_with_wait_completes_queued_work(make_service):
    service = make_service(ScriptedWorkflow([("final", final_state())]), max_concurrent=1)
    release = occupy_only_worker_thread(service)
    run = start(service)

    consumer = asyncio.ensure_future(consume(run))
    release.set()
    await asyncio.to_thread(service.shutdown, True)  # waits for the queued worker
    await consumer

    assert run.hooks["worker_exited"].is_set()
    assert [row["status"] for row in service.audit.rows] == ["success"]


# -------------------------------------------------------------------
# H2: close() at application shutdown
# -------------------------------------------------------------------

@pytest.mark.anyio
async def test_close_drains_a_worker_that_finishes_within_the_bound(make_service):
    gate = Gate()
    service = make_service(ScriptedWorkflow([("gate", gate), ("chunk", "A"), ("final", final_state())]))
    run = start(service)
    consumer = asyncio.ensure_future(consume(run))
    await wait_event(gate.reached)

    closing = asyncio.ensure_future(asyncio.to_thread(service.close, 2.0))  # off the event loop
    await asyncio.sleep(0.05)
    gate.release.set()  # the consumer keeps running on the loop meanwhile
    received, end = await consumer
    summary = await closing

    assert summary == {"still_running": 0, "cancelled": 0}
    assert end == "final" and deltas(received) == ["A"]
    assert run.outcome.decision.status == "completed"
    assert [row["status"] for row in service.audit.rows] == ["success"]


@pytest.mark.anyio
async def test_close_is_bounded_and_cancels_undecided_runs(make_service):
    gate = Gate()
    service = make_service(ScriptedWorkflow([("gate", gate), ("final", final_state())]))
    run = start(service)
    consumer = asyncio.ensure_future(consume(run))
    await wait_event(gate.reached)

    began = time.monotonic()
    summary = await asyncio.to_thread(service.close, 0.2)
    assert time.monotonic() - began < 0.2 + 0.5  # bounded by the drain, not by the worker

    assert summary == {"still_running": 1, "cancelled": 1}
    decision = run.outcome.decision
    assert (decision.status, decision.error_type) == ("cancelled", "ServerShutdown")
    assert run.stop_requested
    _, end = await consumer
    assert end is ABORTED  # the consumer is told to stop

    gate.release.set()  # the worker finishes its step, then sees the stop
    await wait_event(run.hooks["worker_exited"])
    [row] = service.audit.rows
    assert (row["status"], row["error_type"]) == ("cancelled", "ServerShutdown")


@pytest.mark.anyio
async def test_close_never_changes_a_completed_outcome(make_service):
    audit_gate = Gate()

    class BlockingAudit(AuditLog):
        def __call__(self, **row):
            audit_gate.reached.set()
            audit_gate.release.wait(GUARD)
            super().__call__(**row)

    service = make_service(ScriptedWorkflow([("final", final_state())]), audit=BlockingAudit())
    run = start(service)
    _, end = await consume(run)  # claims "completed" after the final event
    assert end == "final"
    await wait_event(audit_gate.reached)  # the worker is still finishing

    summary = await asyncio.to_thread(service.close, 0.2)

    assert summary == {"still_running": 1, "cancelled": 0}
    assert run.outcome.decision.status == "completed"  # not overwritten
    assert not run.stop_requested
    audit_gate.release.set()
    await wait_event(run.hooks["worker_exited"])
    assert [row["status"] for row in service.audit.rows] == ["success"]


@pytest.mark.anyio
async def test_close_stops_new_runs(make_service):
    service = make_service(ScriptedWorkflow([("final", final_state())]))

    assert await asyncio.to_thread(service.close, 0.1) == {"still_running": 0, "cancelled": 0}

    run = service.start_run("q?", "req-after-close", asyncio.get_running_loop())
    assert run is None  # the pool is shut down; the reserved slot is given back
    assert service.capacity.in_use == 0


@pytest.mark.anyio
async def test_close_releases_a_run_whose_response_never_attached(make_service):
    """U6: no consumer will ever mark itself done, so close() abandons it."""
    gate = Gate()
    service = make_service(ScriptedWorkflow([("gate", gate), ("final", final_state())]))
    run = start(service)  # no consumer attaches
    await wait_event(gate.reached)
    assert service.capacity.in_use == 1

    summary = await asyncio.to_thread(service.close, 0.1)

    assert summary == {"still_running": 1, "cancelled": 1}
    assert run.lease.attach_state == "abandoned"
    assert run.attach() is False  # a late response can no longer attach
    assert not run.lease.released  # the worker is still running

    gate.release.set()
    await wait_event(run.hooks["worker_exited"])
    await wait_until(lambda: run.lease.released)
    assert service.capacity.in_use == 0
    assert service.active_runs() == []
    [row] = service.audit.rows
    assert (row["status"], row["error_type"]) == ("cancelled", "ServerShutdown")


@pytest.mark.anyio
async def test_close_leaves_an_attached_consumer_to_finish_itself(make_service):
    """U6 keeps attached responses as they were: the lease waits for the consumer."""
    gate = Gate()
    service = make_service(ScriptedWorkflow([("gate", gate), ("final", final_state())]))
    run = start(service)
    assert run.attach()  # a response is attached but has not finished yet
    run.mark_response_started()
    await wait_event(gate.reached)

    summary = await asyncio.to_thread(service.close, 0.1)

    assert summary == {"still_running": 1, "cancelled": 1}
    assert run.lease.attach_state == "attached"  # not abandoned
    gate.release.set()
    await wait_event(run.hooks["worker_exited"])
    assert not run.lease.released  # still waiting for the attached consumer
    assert service.capacity.in_use == 1

    run.consumer_finished(None)  # the response ends
    assert run.lease.released
    assert service.capacity.in_use == 0


@pytest.mark.anyio
async def test_worker_cancelled_before_start_releases_lease_without_audit(make_service):
    service = make_service(ScriptedWorkflow([("final", final_state())]), max_concurrent=1)
    release = occupy_only_worker_thread(service)
    run = start(service)

    assert run._future.cancel()  # possible only because it has not started

    assert run.hooks["worker_exited"].is_set()
    assert not run.worker_started
    decision = run.outcome.decision
    assert (decision.status, decision.error_type) == ("cancelled", "WorkerNotStarted")
    assert run.lease.released and service.capacity.in_use == 0
    assert service.capacity.released_total == 1
    assert run.channel.state == "aborted"
    assert run.attach() is False  # a late response is refused
    assert service.audit.rows == []  # no worker ran, so no audit row
    assert service.active_runs() == []
    release.set()


@pytest.mark.anyio
async def test_attached_consumer_exits_when_unstarted_worker_is_cancelled(make_service):
    service = make_service(ScriptedWorkflow([("final", final_state())]), max_concurrent=1)
    release = occupy_only_worker_thread(service)
    run = start(service)
    assert run.attach()
    run.mark_response_started()
    waiting = asyncio.ensure_future(run.channel.receive_batch(GUARD))
    await asyncio.sleep(0.01)  # the consumer is waiting

    assert run._future.cancel()

    assert await waiting is ABORTED
    assert not run.lease.released  # the consumer has not exited yet
    run.consumer_finished()  # normal exit after ABORTED
    assert run.lease.released and service.capacity.released_total == 1
    assert service.audit.rows == []
    release.set()


@pytest.mark.anyio
async def test_cancellation_keeps_an_earlier_consumer_decision(make_service):
    service = make_service(ScriptedWorkflow([("final", final_state())]), max_concurrent=1)
    release = occupy_only_worker_thread(service)
    run = start(service)
    assert run.attach()
    run.consumer_finished("disconnected")  # decides first

    assert run._future.cancel()

    assert run.outcome.decision.error_type == "ClientDisconnected"  # first claim wins
    assert run.lease.released and service.capacity.released_total == 1
    release.set()


@pytest.mark.anyio
async def test_started_worker_cannot_be_cancelled_or_finalized_twice(make_service):
    gate = Gate()
    service = make_service(ScriptedWorkflow([("gate", gate), ("final", final_state())]))
    run = start(service)
    await wait_event(gate.reached)

    assert run._future.cancel() is False  # running: cancellation is impossible
    gate.release.set()
    await consume(run)
    await wait_event(run.hooks["worker_exited"])

    assert run.outcome.decision.status == "completed"
    assert run.audit_writes == 1 and service.capacity.released_total == 1


# -------------------------------------------------------------------
# Event loop closing between the closed-check and scheduling
# -------------------------------------------------------------------

@pytest.mark.anyio
async def test_loop_closing_during_wake_scheduling_finalizes_the_run(make_service):
    workflow = ScriptedWorkflow([("trace", "t"), ("chunk", "x"), ("final", final_state())])
    service = make_service(workflow)
    proxy = LoopClosingOnSchedule(asyncio.get_running_loop())
    proxy.arm()  # the first wake-up the worker schedules hits the race

    run = service.start_run("q", "req-race", proxy)
    await wait_event(run.hooks["worker_exited"])

    assert proxy.attempts == 1
    assert run.channel.abort_reason == "loop_closed"
    assert run.channel._wake_pending is False
    decision = run.outcome.decision
    assert (decision.status, decision.error_type) == ("cancelled", "LoopClosed")
    assert run.lease.consumer_done and run.lease.released  # marked for the consumer that can never run
    assert [row["status"] for row in service.audit.rows] == ["cancelled"]
    assert "final" not in workflow.completed_steps
