"""A2-6: session-aware streaming (design revision 8 §11).

Runs go through the real route and SSEResponse via the ASGI harness, with a
test StreamService (``stream_app``) and the real session store and lease
manager under the test's temporary directory.
"""

import asyncio
import json
import logging
import sqlite3
import threading
import time

import pytest
from fastapi.testclient import TestClient

import app.main
from app.api import streaming
from app.rag.state import EvidenceGrade, RouteDecision
from app.services import sessions as session_service
from app.services.audit import get_recent_audits, init_db, write_audit
from tests.conftest import FakeLLM, FakeRetriever, kb_doc, legacy_owner_ref
from tests.session_support import find_token
from tests.stateless_scenarios import SCENARIO_NAMES, load_golden, run_scenario
from tests.streaming_support import (
    SESSION_STREAM_EVENTS,
    ASGIHarness,
    SSEProtocolError,
    assert_stream_grammar,
    body_contains,
    parse_sse,
)
from tests.test_stream_run import Gate, ScriptedWorkflow, final_state

QUESTION = "How do I connect to the VPN?"
ANSWER = "Install GlobalProtect."
GUARD = 5.0


class HistoryWorkflow(ScriptedWorkflow):
    """ScriptedWorkflow that also accepts ``history`` (as stream_ask does) and records it."""

    def __init__(self, steps):
        super().__init__(steps)
        self.histories = []
        self.calls = 0

    def __call__(self, question, on_chunk, history=None):
        self.calls += 1
        self.histories.append(history)
        return super().__call__(question, on_chunk)


def answer_steps(answer=ANSWER, gate=None):
    steps = [("trace", "Router → KB"), ("chunk", answer)]
    if gate is not None:
        steps.insert(1, ("gate", gate))
    return steps + [("final", final_state(answer=answer))]


@pytest.fixture
def sessions_on(monkeypatch, settings):
    monkeypatch.setattr(settings, "sessions_enabled", True)
    monkeypatch.setattr(settings, "session_shutdown_grace_seconds", 0.2)
    monkeypatch.setattr(settings, "session_keeper_join_seconds", 0.3)
    monkeypatch.setattr(settings, "session_release_budget_seconds", 0.5)
    monkeypatch.setattr(settings, "session_db_busy_timeout_seconds", 0.5)
    return settings


async def create_token() -> str:
    result = await ASGIHarness(app.main.app).request("POST", "/api/sessions", json_body={})
    assert result.status == 201, result.body
    return json.loads(result.body)["session_id"]


async def stream(token=None, question=QUESTION, *, harness=None, **harness_kwargs):
    harness = harness or ASGIHarness(app.main.app, **harness_kwargs)
    headers = {"X-Session-ID": token} if token is not None else None
    return await harness.request("POST", "/api/chat/stream", json_body={"question": question}, headers=headers)


async def wait_event(event, timeout=GUARD):
    if not await asyncio.to_thread(event.wait, timeout):
        raise AssertionError("event not set in time")


def events(result):
    return parse_sse(result.body, allowed=SESSION_STREAM_EVENTS)


def data(parsed, name):
    [event] = [e for e in parsed.events if e.event == name]
    return event.data


def session_db(settings, sql, params=()):
    connection = sqlite3.connect(settings.session_db_path)
    try:
        return connection.execute(sql, params).fetchall()
    finally:
        connection.close()


def row_for(settings, token):
    connection = sqlite3.connect(settings.session_db_path)
    connection.row_factory = sqlite3.Row
    try:
        found = connection.execute(
            "SELECT * FROM sessions WHERE token_hash = ?", (session_service.hash_token(token),)).fetchone()
        return dict(found)
    finally:
        connection.close()


def turn_count(settings, token):
    return row_for(settings, token)["turn_count"]


def pre_sse(result, status, code):
    assert result.status == status
    assert result.headers["content-type"].startswith("application/json")
    body = json.loads(result.body)["detail"]
    assert body["code"] == code
    return body


async def runtime_acquire(token, request_id):
    rt = session_service.get_session_runtime(app.main.settings)
    return rt, await asyncio.to_thread(rt.manager.acquire, token, request_id, owner_ref=legacy_owner_ref())


# ===================================================================
# First turn, follow-up, event order
# ===================================================================

@pytest.mark.anyio
async def test_first_turn_streams_saves_and_reports(stream_app, sessions_on):
    workflow = HistoryWorkflow(answer_steps())
    service = stream_app(workflow)
    token = await create_token()

    result = await stream(token)
    [run] = service.started
    await wait_event(run.hooks["worker_exited"])

    parsed = events(result)
    assert_stream_grammar(parsed, connected=True, session=True)
    assert parsed.names == ["start", "trace", "delta", "done", "session"]
    assert data(parsed, "start") == {"request_id": run.request_id, "version": 1,
                                     "session": {"history_turns_used": 0}}
    done = data(parsed, "done")
    assert done["session"] == {"history_turns_used": 0} and "turn_saved" not in done["session"]
    assert data(parsed, "session") == {"request_id": run.request_id, "turn_saved": True,
                                       "history_turns_used": 0}
    assert workflow.histories == [[]]
    assert turn_count(sessions_on, token) == 1
    assert row_for(sessions_on, token)["lease_owner"] is None
    [audit_row] = service.audit.rows
    assert audit_row["session_id"] == row_for(sessions_on, token)["id"]
    assert audit_row["status"] == "success"


@pytest.mark.anyio
async def test_follow_up_receives_history_and_counts_it(stream_app, sessions_on):
    workflow = HistoryWorkflow(answer_steps())
    service = stream_app(workflow)
    token = await create_token()

    await stream(token, "How do I connect to the VPN?")
    await wait_event(service.started[0].hooks["worker_exited"])
    result = await stream(token, "What about on a Mac?")
    await wait_event(service.started[1].hooks["worker_exited"])

    assert workflow.histories[1] == [{"question": "How do I connect to the VPN?", "answer": ANSWER}]
    parsed = events(result)
    assert data(parsed, "start")["session"] == {"history_turns_used": 1}
    assert data(parsed, "session")["turn_saved"] is True
    assert data(parsed, "session")["history_turns_used"] == 1
    assert turn_count(sessions_on, token) == 2


@pytest.mark.anyio
async def test_follow_up_end_to_end_with_the_real_workflow(stream_app, sessions_on, fake_rag):
    llm = FakeLLM(
        structured={RouteDecision: [{"route": "kb"}] * 2, EvidenceGrade: [{"grade": "good"}] * 2},
        text=["Open GlobalProtect.", "How do I connect to the VPN on a Mac?", "Install the macOS client."],
        stream=True,
    )
    fake_rag(llm=llm, retriever=FakeRetriever([kb_doc("GlobalProtect has a macOS client.")]))
    service = stream_app(streaming.stream_ask)
    token = await create_token()

    await stream(token, "How do I connect to the VPN?")
    await wait_event(service.started[0].hooks["worker_exited"])
    result = await stream(token, "What about on a Mac?")
    await wait_event(service.started[1].hooks["worker_exited"])

    parsed = events(result)
    deltas = "".join(e.data["text"] for e in parsed.events if e.event == "delta")
    assert deltas.strip() == data(parsed, "done")["answer"] == "Install the macOS client."
    assert "on a Mac?" not in deltas  # the internal rewrite never streams
    assert "User: How do I connect to the VPN?" in llm.calls[3]["prompt"]
    assert data(parsed, "session") == {"request_id": service.started[1].request_id,
                                       "turn_saved": True, "history_turns_used": 1}


# ===================================================================
# Stateless streaming is unchanged
# ===================================================================

@pytest.mark.anyio
async def test_stateless_stream_has_no_session_parts(stream_app, settings):
    workflow = HistoryWorkflow(answer_steps())
    service = stream_app(workflow)

    result = await stream(None)
    await wait_event(service.started[0].hooks["worker_exited"])

    parsed = parse_sse(result.body)  # the strict A1 parser: a 'session' event would fail
    assert_stream_grammar(parsed, connected=True)
    assert "session" not in parsed.events[0].data and "session" not in data(parsed, "done")
    assert workflow.histories == [None]  # called exactly as before (no history keyword)
    assert "session_id" not in service.audit.rows[0]
    assert session_service.peek_session_runtime() is None
    assert not settings.session_db_path.exists()


GOLDEN = load_golden()


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_stateless_stream_prompts_match_the_pre_a2_golden(client, stream_app, fake_rag, name):
    stream_app(streaming.stream_ask)

    def through_the_stream(question):
        body = client.post("/api/chat/stream", json={"question": question}).content
        done = data(parse_sse(body), "done")
        return {"answer": done["answer"], "source_used": done["source_used"], "trace": done["trace"],
                "citations": done["citations"], "current_query": done["rewritten_query"]}

    assert run_scenario(fake_rag, name, through_the_stream) == GOLDEN[name]


@pytest.mark.anyio
async def test_stateless_capacity_503_body_is_unchanged(stream_app, settings):
    gate = Gate()
    service = stream_app(HistoryWorkflow(answer_steps(gate=gate)), max_concurrent=1)
    first = asyncio.ensure_future(stream(None))
    await wait_event(gate.reached)

    busy = await stream(None)

    assert busy.status == 503
    assert json.loads(busy.body) == {"detail": {"message": "The server is busy. Please retry shortly."}}
    gate.release.set()
    await first
    await wait_event(service.started[0].hooks["worker_exited"])


# ===================================================================
# Pre-SSE errors, capacity-first admission
# ===================================================================

@pytest.mark.anyio
async def test_busy_session_gets_409_before_any_sse_bytes(stream_app, sessions_on):
    workflow = HistoryWorkflow(answer_steps())
    service = stream_app(workflow)
    token = await create_token()
    rt, held = await runtime_acquire(token, "someone-else")

    result = await stream(token)

    pre_sse(result, 409, "session_busy")
    assert int(result.headers["retry-after"]) >= 1
    assert workflow.calls == 0 and service.started == []
    assert service.capacity.in_use == 0  # the reservation was given back
    assert service.capacity.acquired_total == service.capacity.released_total == 1
    await asyncio.to_thread(rt.manager.finish, held)


@pytest.mark.anyio
async def test_concurrent_streams_on_one_session_run_once(stream_app, sessions_on):
    gate = Gate()
    workflow = HistoryWorkflow(answer_steps(gate=gate))
    service = stream_app(workflow)
    token = await create_token()

    first = asyncio.ensure_future(stream(token))
    await wait_event(gate.reached)
    second = await stream(token, "duplicate")
    gate.release.set()
    first_result = await first
    await wait_event(service.started[0].hooks["worker_exited"])

    pre_sse(second, 409, "session_busy")
    assert workflow.calls == 1
    assert data(events(first_result), "session")["turn_saved"] is True


@pytest.mark.anyio
@pytest.mark.parametrize("token_kind", ["valid", "unknown", "expired"])
async def test_capacity_is_checked_before_the_session(stream_app, sessions_on, token_kind):
    gate = Gate()
    service = stream_app(HistoryWorkflow(answer_steps(gate=gate)), max_concurrent=1)
    token = await create_token()
    if token_kind == "unknown":
        token = "never-issued"
    elif token_kind == "expired":
        connection = sqlite3.connect(sessions_on.session_db_path)
        connection.execute("UPDATE sessions SET idle_expires_at = 0")
        connection.commit()
        connection.close()
    occupier = asyncio.ensure_future(stream(None))
    await wait_event(gate.reached)
    before = session_db(sessions_on, "SELECT lease_epoch, lease_owner FROM sessions")

    result = await stream(token)

    pre_sse(result, 503, "stream_busy")  # not 410 or 409: capacity comes first
    assert result.headers["retry-after"] == str(sessions_on.stream_retry_after_seconds)
    assert session_db(sessions_on, "SELECT lease_epoch, lease_owner FROM sessions") == before
    gate.release.set()
    await occupier
    await wait_event(service.started[0].hooks["worker_exited"])


@pytest.mark.anyio
@pytest.mark.parametrize("case", ["unknown", "expired", "too_long", "empty", "disabled"])
async def test_session_problems_get_410_and_never_fall_back(stream_app, sessions_on, monkeypatch, case):
    workflow = HistoryWorkflow(answer_steps())
    service = stream_app(workflow)
    token = await create_token()
    if case == "unknown":
        token = "never-issued"
    elif case == "expired":
        connection = sqlite3.connect(sessions_on.session_db_path)
        connection.execute("UPDATE sessions SET idle_expires_at = 0")
        connection.commit()
        connection.close()
    elif case == "too_long":
        token = "x" * 129
    elif case == "empty":
        token = ""
    else:
        monkeypatch.setattr(sessions_on, "sessions_enabled", False)

    result = await stream(token)

    pre_sse(result, 410, "session_expired")
    assert workflow.calls == 0  # no stateless stream instead
    assert service.capacity.in_use == 0
    assert service.capacity.acquired_total == service.capacity.released_total == 1


@pytest.mark.anyio
async def test_storage_failure_on_the_lease_gets_503_session_unavailable(stream_app, sessions_on, monkeypatch):
    workflow = HistoryWorkflow(answer_steps())
    service = stream_app(workflow)
    token = await create_token()
    rt = session_service.get_session_runtime(sessions_on)

    def broken(*args, **kwargs):
        raise session_service.SessionStoreUnavailable("down")

    monkeypatch.setattr(rt.manager, "acquire", broken)
    result = await stream(token)

    pre_sse(result, 503, "session_unavailable")
    assert workflow.calls == 0 and service.capacity.in_use == 0


@pytest.mark.anyio
async def test_submission_failure_releases_both_lease_and_slot(stream_app, sessions_on):
    service = stream_app(HistoryWorkflow(answer_steps()))
    token = await create_token()
    service._executor.shutdown(wait=True)

    result = await stream(token)

    pre_sse(result, 503, "stream_busy")
    assert row_for(sessions_on, token)["lease_owner"] is None
    assert service.capacity.in_use == 0
    assert session_service.get_session_runtime(sessions_on).manager.active_count() == 0


# ===================================================================
# Cancellation, disconnect, errors, timeouts
# ===================================================================

@pytest.mark.anyio
async def test_disconnect_mid_stream_saves_nothing_and_releases(stream_app, sessions_on):
    gate = Gate()
    service = stream_app(HistoryWorkflow(answer_steps(gate=gate)))
    token = await create_token()
    harness = ASGIHarness(app.main.app, spec_version="2.3")
    harness.disconnect_on(body_contains(b"event: trace"))

    result = await stream(token, harness=harness)
    [run] = service.started
    gate.release.set()
    await wait_event(run.hooks["worker_exited"])

    parsed = events(result)
    assert "session" not in parsed.names and "done" not in parsed.names
    assert run.outcome.decision.status == "cancelled"
    assert turn_count(sessions_on, token) == 0
    assert row_for(sessions_on, token)["lease_owner"] is None
    assert service.audit.rows[0]["status"] == "cancelled"
    assert service.audit.rows[0]["session_id"] == row_for(sessions_on, token)["id"]


@pytest.mark.anyio
async def test_send_failure_on_done_saves_nothing(stream_app, sessions_on):
    service = stream_app(HistoryWorkflow(answer_steps()))
    token = await create_token()
    harness = ASGIHarness(app.main.app, spec_version="2.4")
    harness.fail_send(body_contains(b"event: done"))

    await stream(token, harness=harness)
    [run] = service.started
    await wait_event(run.hooks["worker_exited"])

    assert run.outcome.decision.status == "cancelled"
    assert turn_count(sessions_on, token) == 0
    assert run.session.result.result() is False


@pytest.mark.anyio
async def test_workflow_error_sends_error_without_a_session_event(stream_app, sessions_on):
    service = stream_app(HistoryWorkflow([("chunk", "partial"), ("raise", RuntimeError("secret detail"))]))
    token = await create_token()

    result = await stream(token)
    [run] = service.started
    await wait_event(run.hooks["worker_exited"])

    parsed = events(result)
    assert_stream_grammar(parsed, connected=True, session=True)
    assert parsed.names[-1] == "error" and "session" not in parsed.names
    assert b"secret detail" not in result.body
    assert turn_count(sessions_on, token) == 0
    assert row_for(sessions_on, token)["lease_owner"] is None
    assert service.audit.rows[0]["status"] == "failed"


@pytest.mark.anyio
async def test_timeout_saves_nothing_and_sends_no_session_event(stream_app, sessions_on):
    gate = Gate()
    service = stream_app(HistoryWorkflow(answer_steps(gate=gate)), timeout=0.3)
    token = await create_token()

    result = await stream(token)
    [run] = service.started
    gate.release.set()
    await wait_event(run.hooks["worker_exited"])

    parsed = events(result)
    assert parsed.names[-1] == "error" and data(parsed, "error")["code"] == "timeout"
    assert "session" not in parsed.names
    assert turn_count(sessions_on, token) == 0
    assert row_for(sessions_on, token)["lease_owner"] is None


# ===================================================================
# Ownership, fencing and the save report
# ===================================================================

@pytest.mark.anyio
async def test_lost_lease_reports_turn_saved_false(stream_app, sessions_on):
    gate = Gate()
    service = stream_app(HistoryWorkflow(answer_steps(gate=gate)))
    token = await create_token()

    pending = asyncio.ensure_future(stream(token))
    await wait_event(gate.reached)
    manager = session_service.get_session_runtime(sessions_on).manager
    with manager._lock:
        next(iter(manager._entries.values())).state = session_service.LEASE_LOST
    gate.release.set()
    result = await pending
    await wait_event(service.started[0].hooks["worker_exited"])

    parsed = events(result)
    assert parsed.names[-2:] == ["done", "session"]  # the answer is still delivered
    assert data(parsed, "session")["turn_saved"] is False
    assert turn_count(sessions_on, token) == 0


@pytest.mark.anyio
async def test_session_deleted_mid_stream_reports_turn_saved_false(stream_app, sessions_on):
    gate = Gate()
    service = stream_app(HistoryWorkflow(answer_steps(gate=gate)))
    token = await create_token()

    pending = asyncio.ensure_future(stream(token))
    await wait_event(gate.reached)
    await asyncio.to_thread(
        session_service.get_session_runtime(sessions_on).store.delete, token, owner_ref=legacy_owner_ref()
    )
    gate.release.set()
    result = await pending
    await wait_event(service.started[0].hooks["worker_exited"])

    assert data(events(result), "session")["turn_saved"] is False


@pytest.mark.anyio
async def test_a_slow_save_reports_null_but_the_turn_is_still_saved(stream_app, sessions_on, monkeypatch):
    monkeypatch.setattr(streaming, "SESSION_SAVE_REPORT_WAIT", 0.1)
    service = stream_app(HistoryWorkflow(answer_steps()))
    token = await create_token()
    manager = session_service.get_session_runtime(sessions_on).manager
    real_finish = manager.finish

    def slow_finish(lease, turn=None):
        time.sleep(0.5)
        return real_finish(lease, turn)

    monkeypatch.setattr(manager, "finish", slow_finish)
    result = await stream(token)
    [run] = service.started
    await wait_event(run.hooks["worker_exited"])

    assert data(events(result), "session")["turn_saved"] is None  # unknown in time
    assert run.session.result.result() is True  # the late result was not cancelled
    assert turn_count(sessions_on, token) == 1


@pytest.mark.anyio
async def test_lease_renewal_keeps_a_long_stream_owned(stream_app, sessions_on, monkeypatch):
    """A short lease TTL so the keeper must renew during the stream."""
    monkeypatch.setattr(sessions_on, "session_lease_ttl_seconds", 0.6)
    monkeypatch.setattr(sessions_on, "session_lease_margin_seconds", 0.05)
    gate = Gate()
    service = stream_app(HistoryWorkflow(answer_steps(gate=gate)))
    token = await create_token()

    pending = asyncio.ensure_future(stream(token))
    await wait_event(gate.reached)
    await asyncio.sleep(1.5)  # 2.5 TTLs
    assert row_for(sessions_on, token)["lease_owner"] is not None
    gate.release.set()
    result = await pending
    await wait_event(service.started[0].hooks["worker_exited"])

    assert data(events(result), "session")["turn_saved"] is True


# ===================================================================
# Audit linkage, token leaks, event loop, shutdown
# ===================================================================

@pytest.mark.anyio
async def test_audit_rows_link_the_internal_session_id(stream_app, sessions_on):
    await asyncio.to_thread(init_db)
    service = stream_app(HistoryWorkflow(answer_steps()), audit=write_audit)
    token = await create_token()

    await stream(token)
    await stream(None, "Stateless question")
    for run in service.started:
        await wait_event(run.hooks["worker_exited"])

    records = {r["question"]: r for r in await asyncio.to_thread(get_recent_audits, 5)}
    assert records[QUESTION]["session_id"] == row_for(sessions_on, token)["id"]
    assert records["Stateless question"]["session_id"] is None


@pytest.mark.anyio
async def test_token_never_leaks(stream_app, sessions_on, caplog):
    await asyncio.to_thread(init_db)
    service = stream_app(HistoryWorkflow(answer_steps()), audit=write_audit)
    bodies = []
    with caplog.at_level(logging.DEBUG):
        token = await create_token()
        bodies.append((await stream(token)).body)
        rt, held = await runtime_acquire(token, "holder")
        bodies.append((await stream(token)).body)  # 409
        await asyncio.to_thread(rt.manager.finish, held)
        bodies.append((await stream("x" * 129)).body)  # 410
        for run in service.started:
            await wait_event(run.hooks["worker_exited"])

    hits = find_token(token, records=caplog.records, bodies=bodies,
                      db_paths=[sessions_on.audit_db_path, sessions_on.session_db_path])
    assert not hits, hits.locations


@pytest.mark.anyio
async def test_no_sqlite_on_the_event_loop(stream_app, sessions_on, monkeypatch):
    observed = []
    real_guard = session_service._forbid_event_loop_thread

    def recording_guard():
        try:
            asyncio.get_running_loop()
            observed.append("event-loop")
        except RuntimeError:
            observed.append("worker")
        real_guard()

    monkeypatch.setattr(session_service, "_forbid_event_loop_thread", recording_guard)
    service = stream_app(HistoryWorkflow(answer_steps()))
    token = await create_token()
    await stream(token)
    rt, held = await runtime_acquire(token, "holder")
    await stream(token)  # 409 path
    await asyncio.to_thread(rt.manager.finish, held)
    await wait_event(service.started[0].hooks["worker_exited"])

    assert observed and set(observed) == {"worker"}


@pytest.mark.anyio
async def test_bounded_shutdown_during_a_session_stream(stream_app, sessions_on):
    gate = Gate()
    service = stream_app(HistoryWorkflow(answer_steps(gate=gate)))
    token = await create_token()
    pending = asyncio.ensure_future(stream(token))
    await wait_event(gate.reached)
    manager = session_service.get_session_runtime(sessions_on).manager

    report = await asyncio.to_thread(manager.shutdown)

    assert report.elapsed <= 0.2 + 0.3 + 0.5 + 0.5
    assert report.released == 1
    assert row_for(sessions_on, token)["lease_owner"] is None
    gate.release.set()
    result = await pending
    await wait_event(service.started[0].hooks["worker_exited"])
    assert data(events(result), "session")["turn_saved"] is False  # can't save after release
    assert turn_count(sessions_on, token) == 0


@pytest.mark.anyio
async def test_a_stream_finishing_after_a_rolled_back_release_cannot_save(stream_app, sessions_on):
    """H2: the database still shows the lease, but the manager's fence refuses the save."""
    gate = Gate()
    service = stream_app(HistoryWorkflow(answer_steps(gate=gate)))
    token = await create_token()
    pending = asyncio.ensure_future(stream(token))
    await wait_event(gate.reached)
    manager = session_service.get_session_runtime(sessions_on).manager

    blocker = sqlite3.connect(sessions_on.session_db_path, isolation_level=None)
    blocker.execute("BEGIN EXCLUSIVE")
    try:
        report = await asyncio.to_thread(manager.shutdown)
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    assert report.release_status in ("rolled_back", "unfinished")
    assert row_for(sessions_on, token)["lease_owner"] is not None  # not released in the database

    gate.release.set()
    result = await pending
    await wait_event(service.started[0].hooks["worker_exited"])
    assert data(events(result), "session")["turn_saved"] is False
    assert turn_count(sessions_on, token) == 0


# ===================================================================
# Grammar helper
# ===================================================================

def test_grammar_allows_session_only_after_done_and_last():
    ok = parse_sse(
        b'event: start\ndata: {}\n\nevent: done\ndata: {}\n\nevent: session\ndata: {}\n\n',
        allowed=SESSION_STREAM_EVENTS)
    assert_stream_grammar(ok, connected=True, session=True)
    with pytest.raises(SSEProtocolError):
        assert_stream_grammar(ok, connected=True)  # not allowed for stateless streams
    for bad in (
        b'event: start\ndata: {}\n\nevent: session\ndata: {}\n\nevent: done\ndata: {}\n\n',
        b'event: start\ndata: {}\n\nevent: error\ndata: {}\n\nevent: session\ndata: {}\n\n',
        b'event: start\ndata: {}\n\nevent: done\ndata: {}\n\nevent: session\ndata: {}\n\nevent: session\ndata: {}\n\n',
    ):
        with pytest.raises(SSEProtocolError):
            assert_stream_grammar(parse_sse(bad, allowed=SESSION_STREAM_EVENTS), connected=True, session=True)
