"""H1-3a: cooperative stop checks before external calls (no external network)."""

import logging
import threading

import pytest

from app.rag import cancellation, vectorstore, workflow
from tests.conftest import FakeLLM, FakeRetriever, FakeWebSearch, kb_doc


# ---------------------------------------------------------------------------
# Propagation: the run's check reaches the threads that execute graph nodes
# ---------------------------------------------------------------------------

class ContextRecorder:
    """Records, at each provider call, the thread and the stop check it sees."""

    def __init__(self):
        self.seen = []

    def record(self, label):
        self.seen.append((label, threading.get_ident(), cancellation.current_check()))


def recording_fakes(fake_rag, recorder, *, kb_grade="good"):
    class RecordingLLM(FakeLLM):
        def invoke(self, prompt, config=None):
            recorder.record("llm")
            return super().invoke(prompt, config)

        def with_structured_output(self, schema, **kwargs):
            structured = super().with_structured_output(schema, **kwargs)
            real = structured.invoke

            def invoke(prompt, config=None):
                recorder.record(f"structured:{schema.__name__}")
                return real(prompt, config)

            structured.invoke = invoke
            return structured

    class RecordingRetriever(FakeRetriever):
        def invoke(self, query):
            recorder.record("retriever")
            return super().invoke(query)

    llm = RecordingLLM(
        structured={
            workflow.RouteDecision: [{"route": "kb", "reason": "internal"}],
            workflow.EvidenceGrade: [{"grade": kb_grade, "reason": "ok"}],
        },
        text=["Use GlobalProtect."],
        stream=True,
    )
    return fake_rag(llm=llm, retriever=RecordingRetriever([kb_doc("Use GlobalProtect.")]))


def check_never_stops():
    return None


def test_the_stop_check_reaches_nodes_run_by_stream_ask_in_background_threads(fake_rag):
    recorder = ContextRecorder()
    recording_fakes(fake_rag, recorder)
    caller = threading.get_ident()

    with cancellation.stop_checks(check_never_stops):
        events = list(workflow.stream_ask("How do I connect to the VPN?", lambda text: None))

    assert events[-1][0] == "final"
    labels = [label for label, _, _ in recorder.seen]
    assert labels == ["structured:RouteDecision", "retriever", "structured:EvidenceGrade", "llm"]
    assert all(check is check_never_stops for _, _, check in recorder.seen)
    # The "messages" stream mode runs nodes in LangGraph's background threads,
    # so this proves the check crosses threads, not just the caller's context.
    assert any(thread != caller for _, thread, _ in recorder.seen)


def test_the_stop_check_reaches_nodes_run_by_ask(fake_rag):
    recorder = ContextRecorder()
    recording_fakes(fake_rag, recorder)

    with cancellation.stop_checks(check_never_stops):
        result = workflow.ask("How do I connect to the VPN?")

    assert result["answer"] == "Use GlobalProtect."
    assert recorder.seen and all(check is check_never_stops for _, _, check in recorder.seen)


def test_without_a_scope_nodes_see_no_check(fake_rag):
    recorder = ContextRecorder()
    recording_fakes(fake_rag, recorder)

    workflow.ask("How do I connect to the VPN?")

    assert recorder.seen and all(check is None for _, _, check in recorder.seen)


def test_scopes_do_not_leak_between_threads():
    seen = {}

    def other_thread():
        seen["check"] = cancellation.current_check()

    with cancellation.stop_checks(check_never_stops):
        thread = threading.Thread(target=other_thread)  # a plain thread: fresh context
        thread.start()
        thread.join(5)

    assert seen == {"check": None}
    assert cancellation.current_check() is None


# ---------------------------------------------------------------------------
# The check itself
# ---------------------------------------------------------------------------

def test_outside_a_scope_checks_do_nothing_even_during_shutdown():
    cancellation.raise_if_stopped("anything")
    cancellation.signal_shutdown()
    cancellation.raise_if_stopped("anything")  # stateless /api/chat and ingestion are unaffected


def test_a_scope_raises_with_the_reason_its_check_returns():
    reasons = iter([None, "disconnected"])

    with cancellation.stop_checks(lambda: next(reasons)):
        cancellation.raise_if_stopped("first call")  # not stopped yet
        with pytest.raises(cancellation.WorkflowStopped) as stopped:
            cancellation.raise_if_stopped("second call")

    assert stopped.value.reason == "disconnected"


def test_the_shutdown_signal_stops_every_scoped_run():
    with cancellation.stop_checks(check_never_stops):
        cancellation.signal_shutdown()
        with pytest.raises(cancellation.WorkflowStopped) as stopped:
            cancellation.raise_if_stopped("call")
        cancellation.reset_shutdown()
        cancellation.raise_if_stopped("call")  # cleared again (application startup)

    assert stopped.value.reason == cancellation.SERVER_SHUTDOWN


# ---------------------------------------------------------------------------
# Workflow: no provider call starts after a stop, in any node
# ---------------------------------------------------------------------------

class ProviderLog:
    """Counts provider calls; the check stops the run after ``stop_after`` calls."""

    def __init__(self, stop_after):
        self.calls = []
        self.stop_after = stop_after

    def record(self, label):
        self.calls.append(label)

    def check(self):
        return "disconnected" if len(self.calls) >= self.stop_after else None


def logging_fakes(fake_rag, log, *, route="kb", grades=("good",), texts=("Standalone question?", "Use GlobalProtect.")):
    class LoggingLLM(FakeLLM):
        def invoke(self, prompt, config=None):
            log.record("llm")
            return super().invoke(prompt, config)

        def with_structured_output(self, schema, **kwargs):
            structured = super().with_structured_output(schema, **kwargs)
            real = structured.invoke

            def invoke(prompt, config=None):
                log.record(schema.__name__)
                return real(prompt, config)

            structured.invoke = invoke
            return structured

    class LoggingRetriever(FakeRetriever):
        def invoke(self, query):
            log.record("retriever")
            return super().invoke(query)

    class LoggingWeb(FakeWebSearch):
        def invoke(self, payload):
            log.record("tavily")
            return super().invoke(payload)

    llm = LoggingLLM(
        structured={
            workflow.RouteDecision: [{"route": route}],
            workflow.EvidenceGrade: [{"grade": grade} for grade in grades],
        },
        text=list(texts),
    )
    web = LoggingWeb({"results": [{"title": "CISA", "url": "https://www.cisa.gov/a", "content": "Evidence."}]})
    fake_rag(llm=llm, retriever=LoggingRetriever([kb_doc("Use GlobalProtect.")]), web=web)


HISTORY = [{"question": "What VPN do we use?", "answer": "GlobalProtect."}]
KB_PATH = ["llm", "RouteDecision", "retriever", "EvidenceGrade", "llm"]
WEB_PATH = ["llm", "RouteDecision", "retriever", "EvidenceGrade", "tavily", "EvidenceGrade", "llm"]
# Weak web evidence twice: two query rewrites (MAX_RETRIES=2), then a good grade.
REWRITE_PATH = [
    "llm", "RouteDecision", "retriever", "EvidenceGrade", "tavily", "EvidenceGrade",
    "llm", "tavily", "EvidenceGrade", "llm", "tavily", "EvidenceGrade", "llm",
]
DIRECT_PATH = ["llm", "RouteDecision", "llm"]

PATHS = {
    "kb-path": (KB_PATH, dict(grades=("good",)), "Use GlobalProtect."),
    "web-path": (WEB_PATH, dict(grades=("weak", "good")), "Use GlobalProtect."),
    "rewrite-path": (
        REWRITE_PATH,
        dict(
            grades=("weak", "weak", "weak", "good"),
            texts=("Standalone question?", "VPN client setup", "GlobalProtect setup", "Use GlobalProtect."),
        ),
        "Use GlobalProtect.",
    ),
    "direct-path": (DIRECT_PATH, dict(route="direct", grades=(), texts=("Standalone question?", "Hello!")), "Hello!"),
}


@pytest.mark.parametrize("name", PATHS)
def test_no_provider_call_starts_after_the_stop(fake_rag, settings, name):
    path, fakes, _ = PATHS[name]
    assert settings.max_retries == 2  # REWRITE_PATH depends on it
    # Stop after each possible number of calls; exactly that many happen.
    for stop_after in range(len(path)):
        log = ProviderLog(stop_after)
        logging_fakes(fake_rag, log, **fakes)

        with cancellation.stop_checks(log.check):
            with pytest.raises(cancellation.WorkflowStopped):
                workflow.ask("How do I connect to the VPN?", history=HISTORY)

        assert log.calls == path[:stop_after], f"stop_after={stop_after}"


@pytest.mark.parametrize("name", PATHS)
def test_an_unstopped_run_makes_every_call(fake_rag, name):
    path, fakes, answer = PATHS[name]
    log = ProviderLog(stop_after=len(path) + 1)
    logging_fakes(fake_rag, log, **fakes)

    with cancellation.stop_checks(log.check):
        result = workflow.ask("How do I connect to the VPN?", history=HISTORY)

    assert log.calls == path
    assert result["answer"] == answer


def test_a_stop_before_web_search_is_not_a_failed_search(fake_rag, caplog):
    log = ProviderLog(stop_after=WEB_PATH.index("tavily"))
    logging_fakes(fake_rag, log, grades=("weak", "good"))

    with cancellation.stop_checks(log.check):
        with pytest.raises(cancellation.WorkflowStopped):
            workflow.ask("How do I connect to the VPN?", history=HISTORY)

    assert "External web search failed" not in caplog.text  # not swallowed by search_web


# ---------------------------------------------------------------------------
# Pinecone: setup calls and queries are checked (upserts are not: ingestion)
# ---------------------------------------------------------------------------

class RecordingIndex:
    def __init__(self):
        self.calls = []

    def query(self, *args, **kwargs):
        self.calls.append("query")
        return {"matches": []}

    def upsert(self, *args, **kwargs):
        self.calls.append("upsert")
        return {"upserted_count": 0}


def stopped():
    return "timeout"


def test_a_pinecone_query_is_not_sent_after_a_stop():
    index = RecordingIndex()
    timed = vectorstore._TimeoutIndex(index, 1.0)

    with cancellation.stop_checks(stopped):
        with pytest.raises(cancellation.WorkflowStopped):
            timed.query(vector=[0.1], top_k=1)
        timed.upsert(vectors=[])  # ingestion is unchanged

    assert index.calls == ["upsert"]
    timed.query(vector=[0.1], top_k=1)  # outside a scope: unchanged
    assert index.calls == ["upsert", "query"]


@pytest.mark.parametrize(
    "indexes, stop_after, expected",
    [
        ("existing", 0, []),
        ("existing", 1, ["list_indexes"]),
        ("missing", 1, ["list_indexes"]),  # create_index is not sent
        ("missing", 2, ["list_indexes"]),  # created; the first readiness describe is not sent
    ],
)
def test_pinecone_setup_stops_before_its_next_call(monkeypatch, settings, indexes, stop_after, expected):
    from tests.test_vectorstore import FakePinecone, index_info

    fake = FakePinecone({settings.pinecone_index_name: index_info()} if indexes == "existing" else {})
    monkeypatch.setattr(vectorstore, "Pinecone", fake)
    created_before = len(fake.created)

    def check():
        calls = len(fake.control_calls) + len(fake.created) - created_before
        return "timeout" if calls >= stop_after else None

    with cancellation.stop_checks(check):
        with pytest.raises(cancellation.WorkflowStopped):
            vectorstore.ensure_index()

    assert [name for name, _, _ in fake.control_calls] == expected
    assert len(fake.created) == (1 if (indexes, stop_after) == ("missing", 2) else 0)
    assert fake.opened == []


# ---------------------------------------------------------------------------
# Streams: the real stream_ask, a provider call in flight, then a stop
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402
import time  # noqa: E402

import app.main  # noqa: E402
from app.api import routes, streaming  # noqa: E402
from app.services import sessions as session_service  # noqa: E402
from app.services.audit import get_recent_audits, init_db  # noqa: E402
from tests.streaming_support import ASGIHarness, body_contains  # noqa: E402
from tests.test_stream_run import (  # noqa: E402,F401 - make_service is a fixture
    ABORTED,
    GUARD,
    Gate,
    consume,
    make_service,
    start,
    wait_event,
)


class GatedRetriever:
    """The retrieval step: an embedding call in flight (held by ``gate``),
    then the Pinecone query through the real ``_TimeoutIndex`` (checked)."""

    def __init__(self, gate):
        self.gate = gate
        self.index = RecordingIndex()
        self.timed = vectorstore._TimeoutIndex(self.index, 1.0)

    def invoke(self, query):
        self.gate.reached.set()
        assert self.gate.release.wait(GUARD), "the gate was never released"
        self.timed.query(vector=[0.1], top_k=4)
        return [kb_doc("Use GlobalProtect.")]


def gated_rag(fake_rag):
    gate = Gate()
    llm = FakeLLM(
        structured={workflow.RouteDecision: [{"route": "kb"}], workflow.EvidenceGrade: [{"grade": "good"}]},
        text=["Use GlobalProtect."],
    )
    retriever = GatedRetriever(gate)
    fake_rag(llm=llm, retriever=retriever)
    return gate, llm, retriever


def provider_calls(llm, retriever):
    return [call["schema"].__name__ if call["schema"] else call["kind"] for call in llm.calls] + retriever.index.calls


def assert_held_until_the_worker_exits(service, run):
    # The consumer is done, but the worker is still in its provider call.
    assert not run.hooks["worker_exited"].is_set()
    assert not run.lease.released
    assert service.capacity.in_use == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    "stop, expected",
    [
        ("disconnected", ("cancelled", "ClientDisconnected")),
        ("timeout", ("timeout", "TimeoutError")),
    ],
)
async def test_a_stream_stop_prevents_the_next_provider_call(make_service, fake_rag, stop, expected):
    gate, llm, retriever = gated_rag(fake_rag)
    service = make_service(streaming.stream_ask)
    run = start(service)
    assert run.attach()
    run.mark_response_started()
    await wait_event(gate.reached)  # the embedding call is in flight

    if stop == "disconnected":
        run.consumer_finished("disconnected")
    else:
        assert run.claim_timeout()  # the stream deadline
        run.consumer_finished(None)
    assert_held_until_the_worker_exits(service, run)

    gate.release.set()  # the in-flight call returns
    await wait_event(run.hooks["worker_exited"])

    assert provider_calls(llm, retriever) == ["RouteDecision"]  # no Pinecone query, no grading
    assert (run.outcome.decision.status, run.outcome.decision.error_type) == expected
    assert run.lease.released and service.capacity.in_use == 0
    [row] = service.audit.rows
    assert row["error_type"] == expected[1]


@pytest.mark.anyio
async def test_a_real_disconnect_prevents_the_next_provider_call(stream_app, fake_rag):
    gate, llm, retriever = gated_rag(fake_rag)
    service = stream_app(streaming.stream_ask)
    harness = ASGIHarness(app.main.app, spec_version="2.3")
    harness.disconnect_on(body_contains(b"event: trace"))

    result = await harness.request("POST", "/api/chat/stream", json_body={"question": "VPN?"})
    [run] = service.started
    assert not result.complete
    assert (run.outcome.decision.status, run.outcome.decision.error_type) == ("cancelled", "ClientDisconnected")

    await wait_event(gate.reached)
    assert_held_until_the_worker_exits(service, run)
    gate.release.set()
    await wait_event(run.hooks["worker_exited"])

    assert provider_calls(llm, retriever) == ["RouteDecision"]
    assert [(r["status"], r["error_type"]) for r in service.audit.rows] == [("cancelled", "ClientDisconnected")]


@pytest.mark.anyio
async def test_stream_service_shutdown_prevents_the_next_provider_call(make_service, fake_rag):
    gate, llm, retriever = gated_rag(fake_rag)
    service = make_service(streaming.stream_ask)
    run = start(service)
    consumer = asyncio.ensure_future(consume(run))
    await wait_event(gate.reached)

    summary = await asyncio.to_thread(service.close, 0.1)  # the drain ends with the call in flight
    assert summary == {"still_running": 1, "cancelled": 1}
    _, end = await consumer
    assert end is ABORTED
    assert_held_until_the_worker_exits(service, run)

    gate.release.set()
    await wait_event(run.hooks["worker_exited"])

    assert provider_calls(llm, retriever) == ["RouteDecision"]
    [row] = service.audit.rows
    assert (row["status"], row["error_type"]) == ("cancelled", "ServerShutdown")


@pytest.mark.anyio
async def test_the_shutdown_signal_stops_a_stream_with_one_error_and_no_answer(make_service, fake_rag):
    gate, llm, retriever = gated_rag(fake_rag)
    service = make_service(streaming.stream_ask)
    run = start(service)
    consumer = asyncio.ensure_future(consume(run))
    await wait_event(gate.reached)

    cancellation.signal_shutdown()  # what the lifespan does after the stream drain
    gate.release.set()
    received, end = await consumer
    await wait_event(run.hooks["worker_exited"])

    assert end == "error"  # never "done": no successful partial answer
    assert [kind for kind, _ in received if kind in ("final", "error")] == ["error"]
    assert provider_calls(llm, retriever) == ["RouteDecision"]
    assert (run.outcome.decision.status, run.outcome.decision.error_type) == ("cancelled", "ServerShutdown")
    [row] = service.audit.rows
    assert (row["status"], row["error_type"]) == ("cancelled", "ServerShutdown")


class StubSessionManager:
    def __init__(self):
        self.expired = False
        self.lost = False
        self.finished = []

    def turn_stop_reason(self, lease):  # as SessionLeaseManager.turn_stop_reason
        if self.expired:
            return cancellation.SESSION_TURN_EXPIRED
        if self.lost:
            return cancellation.SESSION_LEASE_LOST
        return None

    def finish(self, lease, turn=None):
        self.finished.append(turn)
        return None if turn is None else True


@pytest.mark.anyio
async def test_the_session_turn_limit_stops_a_stream_without_saving(make_service, fake_rag):
    from types import SimpleNamespace

    gate, llm, retriever = gated_rag(fake_rag)
    service = make_service(streaming.stream_ask)
    manager = StubSessionManager()
    session = streaming.StreamSession(
        manager=manager, lease=SimpleNamespace(session_id="internal-id"), history=[], history_turns_used=0
    )
    run = service.start(service.reserve(), "VPN?", "req-1", asyncio.get_running_loop(), session=session)
    service.started.append(run)  # for the fixture's leak and audit checks
    consumer = asyncio.ensure_future(consume(run))
    await wait_event(gate.reached)

    manager.expired = True  # the turn limit passes during the embedding call
    gate.release.set()
    _, end = await consumer
    await wait_event(run.hooks["worker_exited"])

    assert end == "error"
    assert provider_calls(llm, retriever) == ["RouteDecision"]
    assert (run.outcome.decision.status, run.outcome.decision.error_type) == ("timeout", "SessionTurnExpired")
    assert manager.finished == [None]  # released, never saved
    assert session.result.result(GUARD) is False
    [row] = service.audit.rows
    assert (row["error_type"], row["session_id"]) == ("SessionTurnExpired", "internal-id")


def test_a_stream_reports_its_own_stop_reasons():
    from types import SimpleNamespace

    run = SimpleNamespace(
        stop_reason=lambda: None,
        session=SimpleNamespace(manager=StubSessionManager(), lease=None),
    )
    reason = streaming.StreamRun.external_call_stop_reason
    assert reason(run) is None
    run.session.manager.lost = True
    assert reason(run) == cancellation.SESSION_LEASE_LOST
    run.session.manager.expired = True  # the turn limit is the more specific reason
    assert reason(run) == cancellation.SESSION_TURN_EXPIRED
    run.stop_reason = lambda: "disconnected"  # the run's own reason comes first
    assert reason(run) == "disconnected"


# ---------------------------------------------------------------------------
# /api/chat: the session turn limit (sessions enabled); stateless unchanged
# ---------------------------------------------------------------------------

class ControlledClock:
    """Wall clock plus an offset the test moves forward (database and manager share it)."""

    def __init__(self):
        self.offset = 0.0

    def __call__(self):
        return time.time() + self.offset

    def advance(self, seconds):
        self.offset += seconds


@pytest.fixture
def sessions_enabled(monkeypatch, settings):
    monkeypatch.setattr(settings, "sessions_enabled", True)
    monkeypatch.setattr(settings, "session_shutdown_grace_seconds", 0.2)
    monkeypatch.setattr(settings, "session_keeper_join_seconds", 0.3)
    monkeypatch.setattr(settings, "session_release_budget_seconds", 0.5)
    monkeypatch.setattr(settings, "session_db_busy_timeout_seconds", 0.5)
    clock = ControlledClock()
    real = session_service.SessionStore.from_settings.__func__

    def from_settings(cls, settings, **overrides):
        overrides.setdefault("clock", clock)
        return real(cls, settings, **overrides)

    monkeypatch.setattr(session_service.SessionStore, "from_settings", classmethod(from_settings))
    return clock


def new_session(client):
    response = client.post("/api/sessions", json={})
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


def session_row(settings, token):
    from tests.test_session_stream import row_for

    return row_for(settings, token)


def kb_fakes(fake_rag, log, on_call=None):
    class LoggingLLM(FakeLLM):
        def invoke(self, prompt, config=None):
            log.append("llm")
            if on_call:
                on_call("llm")
            return super().invoke(prompt, config)

        def with_structured_output(self, schema, **kwargs):
            structured = super().with_structured_output(schema, **kwargs)
            real = structured.invoke

            def invoke(prompt, config=None):
                log.append(schema.__name__)
                if on_call:
                    on_call(schema.__name__)
                return real(prompt, config)

            structured.invoke = invoke
            return structured

    class LoggingRetriever(FakeRetriever):
        def invoke(self, query):
            log.append("retriever")
            return super().invoke(query)

    llm = LoggingLLM(
        structured={workflow.RouteDecision: [{"route": "kb"}], workflow.EvidenceGrade: [{"grade": "good"}]},
        text=["Use GlobalProtect."],
    )
    fake_rag(llm=llm, retriever=LoggingRetriever([kb_doc("Use GlobalProtect.")]))


def audits_for(request_id):
    """Every audit row of one request (not only the latest row)."""
    return [row for row in get_recent_audits(limit=500) if row["request_id"] == request_id]


def only_audit(response):
    body = response.json()
    request_id = body["request_id"] if "request_id" in body else body["detail"]["request_id"]
    [row] = audits_for(request_id)  # exactly one outcome
    return row


def test_session_chat_stops_once_the_turn_limit_passes(client, sessions_enabled, fake_rag, settings):
    token = new_session(client)
    log = []

    def slow_route(label):  # the routing call outlasts the turn limit
        if label == "RouteDecision":
            sessions_enabled.advance(settings.session_turn_max_seconds + 1)

    kb_fakes(fake_rag, log, on_call=slow_route)

    response = client.post("/api/chat", json={"question": "VPN?"}, headers={"X-Session-ID": token})

    assert response.status_code == 500
    assert response.json()["detail"]["message"] == "Unable to process the question."  # generic
    assert log == ["RouteDecision"]  # no retrieval, grading, or answer after the limit
    row = session_row(settings, token)
    assert row["turn_count"] == 0 and row["lease_owner"] is None  # not saved; released
    audit = only_audit(response)
    assert (audit["status"], audit["error_type"], audit["session_id"]) == (
        "failed", "SessionTurnExpired", row["id"],
    )


def test_session_chat_never_saves_a_turn_whose_limit_passed_during_the_last_call(
    client, sessions_enabled, fake_rag, settings
):
    token = new_session(client)
    log = []

    def slow_answer(label):  # the final answer call outlasts the turn limit
        if label == "llm":
            sessions_enabled.advance(settings.session_turn_max_seconds + 1)

    kb_fakes(fake_rag, log, on_call=slow_answer)

    response = client.post("/api/chat", json={"question": "VPN?"}, headers={"X-Session-ID": token})

    assert response.status_code == 200
    assert log == ["RouteDecision", "retriever", "EvidenceGrade", "llm"]  # no call left to skip
    assert response.json()["session"]["turn_saved"] is False  # the fenced save refused it
    row = session_row(settings, token)
    assert row["turn_count"] == 0 and row["lease_owner"] is None


def test_session_chat_within_the_limit_is_unchanged(client, sessions_enabled, fake_rag, settings):
    token = new_session(client)
    log = []
    kb_fakes(fake_rag, log)

    response = client.post("/api/chat", json={"question": "VPN?"}, headers={"X-Session-ID": token})

    assert response.status_code == 200
    assert response.json()["answer"] == "Use GlobalProtect."
    assert response.json()["session"]["turn_saved"] is True
    assert session_row(settings, token)["turn_count"] == 1
    assert only_audit(response)["status"] == "success"


def test_session_chat_stops_at_the_shutdown_signal(client, sessions_enabled, fake_rag, settings):
    token = new_session(client)
    log = []
    kb_fakes(fake_rag, log)
    cancellation.signal_shutdown()

    response = client.post("/api/chat", json={"question": "VPN?"}, headers={"X-Session-ID": token})

    assert response.status_code == 500
    assert log == []
    assert session_row(settings, token)["lease_owner"] is None
    audit = only_audit(response)
    assert (audit["status"], audit["error_type"]) == ("failed", "ServerShutdown")


@pytest.mark.parametrize("sessions", [False, True])
def test_stateless_chat_is_unchanged_without_shutdown(client, monkeypatch, settings, fake_rag, sessions):
    monkeypatch.setattr(settings, "sessions_enabled", sessions)
    log = []
    kb_fakes(fake_rag, log)

    response = client.post("/api/chat", json={"question": "VPN?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Use GlobalProtect."
    assert log == ["RouteDecision", "retriever", "EvidenceGrade", "llm"]
    assert only_audit(response)["status"] == "success"


def test_stateless_chat_has_no_deadline(client, fake_rag):
    # A check is installed (so the shutdown signal applies), but on its own
    # it never stops the run: there is no request deadline.
    log, seen = [], []

    def record(label):
        check = cancellation.current_check()
        seen.append(check is not None and check() is None)

    kb_fakes(fake_rag, log, on_call=record)
    response = client.post("/api/chat", json={"question": "VPN?"})

    assert response.status_code == 200
    assert seen == [True, True, True]  # routing, grading, and the answer


@pytest.mark.parametrize("sessions", [False, True])
def test_stateless_chat_stops_at_the_shutdown_signal(client, monkeypatch, settings, fake_rag, sessions, caplog):
    monkeypatch.setattr(settings, "sessions_enabled", sessions)
    log = []

    def shutdown_after_routing(label):  # the signal arrives while routing is in flight
        if label == "RouteDecision":
            cancellation.signal_shutdown()

    kb_fakes(fake_rag, log, on_call=shutdown_after_routing)
    caplog.set_level(logging.INFO, logger=routes.__name__)

    response = client.post("/api/chat", json={"question": "VPN?"})

    assert response.status_code == 500  # never a successful answer after the stop
    assert response.json()["detail"]["message"] == "Unable to process the question."
    assert log == ["RouteDecision"]  # the in-flight call finished; no later call started
    audit = only_audit(response)
    assert (audit["status"], audit["error_type"]) == ("failed", "ServerShutdown")
    # An expected stop is one warning without a traceback.
    [record] = [r for r in caplog.records if r.name == routes.__name__ and "request_id" in r.getMessage()]
    assert record.levelno == logging.WARNING and record.exc_info is None
    assert record.getMessage() == f"Chat request stopped for request_id={audit['request_id']} (ServerShutdown)"


def test_an_unexpected_chat_error_is_still_logged_with_its_traceback(client, fake_rag, caplog):
    log = []

    def fail(label):
        if label == "RouteDecision":
            raise RuntimeError("provider exploded")

    kb_fakes(fake_rag, log, on_call=fail)
    caplog.set_level(logging.INFO, logger=routes.__name__)

    response = client.post("/api/chat", json={"question": "VPN?"})

    assert response.status_code == 500
    audit = only_audit(response)
    assert (audit["status"], audit["error_type"]) == ("failed", "RuntimeError")
    [record] = [r for r in caplog.records if r.getMessage().startswith("Chat request failed")]
    assert record.levelno == logging.ERROR and record.exc_info is not None


# ---------------------------------------------------------------------------
# H1-5 (B): optional CHAT_DEADLINE_SECONDS for stateless /api/chat
# ---------------------------------------------------------------------------

class DeadlineClock:
    """Replaces cancellation.monotonic(); moves only when a test advances it."""

    def __init__(self):
        self.now = 500.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def deadline_clock(monkeypatch):
    clock = DeadlineClock()
    monkeypatch.setattr(cancellation, "monotonic", clock)
    return clock


def test_a_deadline_check_returns_its_reason_once_the_time_has_passed(deadline_clock):
    check = cancellation.deadline_check(30, cancellation.CHAT_DEADLINE)

    assert check() is None
    deadline_clock.advance(29.9)
    assert check() is None
    deadline_clock.advance(0.1)  # exactly at the deadline
    assert check() == cancellation.CHAT_DEADLINE
    deadline_clock.advance(100)
    assert check() == cancellation.CHAT_DEADLINE


def test_a_deadline_check_counts_from_when_it_is_created(deadline_clock):
    deadline_clock.advance(1000)  # time before the check exists does not count
    check = cancellation.deadline_check(30, cancellation.CHAT_DEADLINE)

    assert check() is None


def test_the_chat_deadline_is_off_by_default(settings):
    assert settings.chat_deadline_seconds is None
    assert routes._stateless_stop_check() is routes._no_deadline


@pytest.mark.parametrize("sessions", [False, True])
def test_stateless_chat_stops_at_its_deadline_before_the_next_call(
    client, monkeypatch, settings, fake_rag, deadline_clock, sessions, caplog
):
    monkeypatch.setattr(settings, "sessions_enabled", sessions)  # no X-Session-ID either way
    monkeypatch.setattr(settings, "chat_deadline_seconds", 30.0)
    log = []

    def slow_route(label):  # the routing call outlasts the deadline
        if label == "RouteDecision":
            deadline_clock.advance(31)

    kb_fakes(fake_rag, log, on_call=slow_route)
    caplog.set_level(logging.INFO, logger=routes.__name__)

    response = client.post("/api/chat", json={"question": "VPN?"})

    assert response.status_code == 500
    assert response.json()["detail"]["message"] == "Unable to process the question."  # generic
    assert log == ["RouteDecision"]  # the in-flight call finished; no later call started
    audit = only_audit(response)
    assert (audit["status"], audit["error_type"]) == ("failed", "TimeoutError")
    [record] = [r for r in caplog.records if r.name == routes.__name__ and "request_id" in r.getMessage()]
    assert record.levelno == logging.WARNING and record.exc_info is None
    assert record.getMessage() == f"Chat request stopped for request_id={audit['request_id']} (TimeoutError)"


def test_stateless_chat_within_its_deadline_is_unchanged(client, monkeypatch, settings, fake_rag, deadline_clock):
    monkeypatch.setattr(settings, "chat_deadline_seconds", 30.0)
    log = []

    def steady(label):  # every call takes 7 s: 21 s in all, within 30 s
        deadline_clock.advance(7)

    kb_fakes(fake_rag, log, on_call=steady)

    response = client.post("/api/chat", json={"question": "VPN?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Use GlobalProtect."
    assert log == ["RouteDecision", "retriever", "EvidenceGrade", "llm"]
    assert only_audit(response)["status"] == "success"


def test_a_deadline_passed_during_the_last_call_still_answers(client, monkeypatch, settings, fake_rag, deadline_clock):
    # Cooperative: only calls not yet started are skipped. The answer call
    # was already running when the deadline passed, so its answer is used.
    monkeypatch.setattr(settings, "chat_deadline_seconds", 30.0)
    log = []

    def slow_answer(label):
        if label == "llm":
            deadline_clock.advance(31)

    kb_fakes(fake_rag, log, on_call=slow_answer)

    response = client.post("/api/chat", json={"question": "VPN?"})

    assert response.status_code == 200
    assert log == ["RouteDecision", "retriever", "EvidenceGrade", "llm"]
    assert only_audit(response)["status"] == "success"


def test_each_stateless_question_gets_its_own_deadline(client, monkeypatch, settings, fake_rag, deadline_clock):
    monkeypatch.setattr(settings, "chat_deadline_seconds", 30.0)
    first_log, second_log = [], []
    kb_fakes(fake_rag, first_log)  # scripted fakes answer one question
    first = client.post("/api/chat", json={"question": "VPN?"})

    deadline_clock.advance(1000)  # long after the first question's deadline
    kb_fakes(fake_rag, second_log)
    second = client.post("/api/chat", json={"question": "VPN?"})

    assert first.status_code == second.status_code == 200
    assert second_log == ["RouteDecision", "retriever", "EvidenceGrade", "llm"]  # not stopped


def test_the_shutdown_signal_still_wins_over_the_deadline(client, monkeypatch, settings, fake_rag, deadline_clock):
    monkeypatch.setattr(settings, "chat_deadline_seconds", 30.0)
    log = []

    def both(label):  # shutdown and the deadline arrive during the same call
        if label == "RouteDecision":
            deadline_clock.advance(31)
            cancellation.signal_shutdown()

    kb_fakes(fake_rag, log, on_call=both)

    response = client.post("/api/chat", json={"question": "VPN?"})

    assert response.status_code == 500
    assert log == ["RouteDecision"]
    assert only_audit(response)["error_type"] == "ServerShutdown"


def test_the_chat_deadline_does_not_apply_to_session_questions(
    client, sessions_enabled, monkeypatch, settings, fake_rag, deadline_clock
):
    # Session questions keep their own limit (SESSION_TURN_MAX_SECONDS).
    monkeypatch.setattr(settings, "chat_deadline_seconds", 30.0)
    token = new_session(client)
    log = []

    def slow_route(label):  # past CHAT_DEADLINE_SECONDS, within the turn limit
        if label == "RouteDecision":
            deadline_clock.advance(31)

    kb_fakes(fake_rag, log, on_call=slow_route)

    response = client.post("/api/chat", json={"question": "VPN?"}, headers={"X-Session-ID": token})

    assert response.status_code == 200
    assert log == ["RouteDecision", "retriever", "EvidenceGrade", "llm"]
    assert response.json()["session"]["turn_saved"] is True


# ---------------------------------------------------------------------------
# Lifespan: the signal is set after the stream drain, and cleared at startup
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_lifespan_signals_shutdown_after_the_stream_drain(monkeypatch):
    seen = {}

    async def close_stream_service():
        seen["during_drain"] = cancellation.shutdown_signalled()

    monkeypatch.setattr(routes, "_close_stream_service", close_stream_service)
    cancellation.signal_shutdown()  # left over from an earlier lifespan

    async with routes._sessions_lifespan(None):
        seen["running"] = cancellation.shutdown_signalled()

    assert seen == {"running": False, "during_drain": False}
    assert cancellation.shutdown_signalled()


# ---------------------------------------------------------------------------
# Lost leases (the lease manager's in-memory LEASE_LOST state)
# ---------------------------------------------------------------------------

def mark_lease_lost(manager, request_id=None):
    """What the keeper does when renewal shows the lease is gone (sessions.py: _apply)."""
    with manager._lock:
        if request_id is None:
            [request_id] = list(manager._entries)
        entry = manager._entries[request_id]
        manager._set_state_locked(entry, session_service.LEASE_LOST, "test")


def test_the_manager_reports_turn_expiry_and_lease_loss(sessions_enabled, settings, client):
    token = new_session(client)
    runtime = session_service.get_session_runtime(settings)
    lease = runtime.manager.acquire(token, "req-lost")
    try:
        assert runtime.manager.turn_stop_reason(lease) is None
        mark_lease_lost(runtime.manager, "req-lost")
        assert runtime.manager.lease_lost(lease)
        assert runtime.manager.turn_stop_reason(lease) == cancellation.SESSION_LEASE_LOST
        sessions_enabled.advance(settings.session_turn_max_seconds + 1)
        assert runtime.manager.turn_stop_reason(lease) == cancellation.SESSION_TURN_EXPIRED
    finally:
        assert runtime.manager.finish(lease, ("q", "a", "private_kb")) is False  # never saved
    assert session_row(settings, token)["turn_count"] == 0
    # The manager's literals are the stop reasons the routes and streams map.
    assert (cancellation.SESSION_TURN_EXPIRED, cancellation.SESSION_LEASE_LOST) == (
        "session_turn_expired", "session_lease_lost",
    )


def test_session_chat_stops_after_its_lease_is_lost_during_a_call(client, sessions_enabled, fake_rag, settings):
    token = new_session(client)
    runtime = session_service.get_session_runtime(settings)
    log = []

    def lose_during_routing(label):
        if label == "RouteDecision":
            mark_lease_lost(runtime.manager)

    kb_fakes(fake_rag, log, on_call=lose_during_routing)

    response = client.post("/api/chat", json={"question": "VPN?"}, headers={"X-Session-ID": token})

    assert response.status_code == 500
    assert log == ["RouteDecision"]  # no retrieval, grading, or answer after the loss
    row = session_row(settings, token)
    assert row["turn_count"] == 0 and row["lease_owner"] is None
    audit = only_audit(response)
    assert (audit["status"], audit["error_type"], audit["session_id"]) == ("failed", "SessionLeaseLost", row["id"])


def test_session_chat_lost_before_its_first_call_makes_none(client, sessions_enabled, fake_rag, settings, monkeypatch):
    token = new_session(client)
    runtime = session_service.get_session_runtime(settings)
    real_acquire = runtime.manager.acquire

    def acquire_then_lose(token, request_id):
        lease = real_acquire(token, request_id)
        mark_lease_lost(runtime.manager, request_id)
        return lease

    monkeypatch.setattr(runtime.manager, "acquire", acquire_then_lose)
    log = []
    kb_fakes(fake_rag, log)

    response = client.post("/api/chat", json={"question": "VPN?"}, headers={"X-Session-ID": token})

    assert response.status_code == 500
    assert log == []
    assert only_audit(response)["error_type"] == "SessionLeaseLost"
    assert session_row(settings, token)["turn_count"] == 0


def test_session_chat_lost_during_the_last_call_is_not_saved(client, sessions_enabled, fake_rag, settings):
    token = new_session(client)
    runtime = session_service.get_session_runtime(settings)
    log = []

    def lose_during_answer(label):
        if label == "llm":
            mark_lease_lost(runtime.manager)

    kb_fakes(fake_rag, log, on_call=lose_during_answer)

    response = client.post("/api/chat", json={"question": "VPN?"}, headers={"X-Session-ID": token})

    assert response.status_code == 200  # no call was left to skip
    assert response.json()["session"]["turn_saved"] is False  # the fenced save refused it
    assert session_row(settings, token)["turn_count"] == 0
    assert only_audit(response)["status"] == "success"


@pytest.mark.anyio
async def test_a_session_stream_stops_after_its_lease_is_lost(stream_app, sessions_enabled, fake_rag, settings):
    from tests.test_session_stream import create_token, events

    gate, llm, retriever = gated_rag(fake_rag)
    service = stream_app(streaming.stream_ask)
    token = await create_token()
    request = asyncio.ensure_future(
        ASGIHarness(app.main.app).request(
            "POST", "/api/chat/stream", json_body={"question": "VPN?"}, headers={"X-Session-ID": token}
        )
    )
    await wait_event(gate.reached)  # the embedding call is in flight
    [run] = service.started
    mark_lease_lost(session_service.get_session_runtime(settings).manager, run.session.lease.request_id)
    gate.release.set()
    result = await request
    await wait_event(run.hooks["worker_exited"])

    parsed = events(result)
    assert parsed.names[-1] == "error" and "done" not in parsed.names and "session" not in parsed.names
    assert provider_calls(llm, retriever) == ["RouteDecision"]
    assert (run.outcome.decision.status, run.outcome.decision.error_type) == ("failed", "SessionLeaseLost")
    assert run.session.result.result(GUARD) is False
    row = session_row(settings, token)
    assert row["turn_count"] == 0 and row["lease_owner"] is None
    [audit] = service.audit.rows
    assert (audit["error_type"], audit["session_id"]) == ("SessionLeaseLost", row["id"])


# ---------------------------------------------------------------------------
# Stop-reason race: the most specific outcome, not a generic one
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_a_backstop_that_loses_a_race_still_records_the_deadline(make_service, fake_rag, monkeypatch):
    from tests.test_stream_run import TIMEOUT

    gate, llm, retriever = gated_rag(fake_rag)
    raced = []
    real_reason = streaming.StreamRun.external_call_stop_reason

    def reason_then_attach(self):
        reason = real_reason(self)
        if reason == "abandon_due" and not raced:
            # The response attaches between the check and the backstop, so
            # abandoning (the backstop) no longer applies when the worker tries it.
            raced.append(self.lease.attach())
        return reason

    monkeypatch.setattr(streaming.StreamRun, "external_call_stop_reason", reason_then_attach)
    service = make_service(streaming.stream_ask)
    run = start(service)
    await wait_event(gate.reached)
    run._clock.advance(TIMEOUT + streaming.ATTACH_GRACE + 0.1)  # past the attach grace, unattached
    gate.release.set()
    await wait_event(run.hooks["terminal_set"])

    assert raced == [True]
    received = []
    while not any(kind == "error" for kind, _ in received):  # the attached response reads to the end
        batch = await run.channel.receive_batch(GUARD)
        assert isinstance(batch, list) and batch, f"unexpected end of channel: {batch!r}"
        received += batch
    assert received[-1] == ("error", {"code": "timeout"})
    assert [kind for kind, _ in received if kind in ("final", "error")] == ["error"]
    run.consumer_finished(None)
    await wait_event(run.hooks["worker_exited"])

    assert (run.outcome.decision.status, run.outcome.decision.error_type) == ("timeout", "TimeoutError")
    assert provider_calls(llm, retriever) == ["RouteDecision"]
    [row] = service.audit.rows
    assert (row["status"], row["error_type"]) == ("failed", "TimeoutError")


@pytest.mark.parametrize(
    "reason, expected",
    [
        (cancellation.SERVER_SHUTDOWN, ("cancelled", "ServerShutdown")),
        (cancellation.SESSION_TURN_EXPIRED, ("timeout", "SessionTurnExpired")),
        (cancellation.SESSION_LEASE_LOST, ("failed", "SessionLeaseLost")),
        ("timeout", ("timeout", "TimeoutError")),
        ("disconnected", ("cancelled", "ClientDisconnected")),
        ("abandon_due", ("timeout", "TimeoutError")),
        ("consumer_unresponsive", ("timeout", "TimeoutError")),
        ("loop_closed", ("cancelled", "LoopClosed")),
    ],
)
def test_every_stop_reason_has_a_specific_stream_outcome(reason, expected):
    outcome = streaming.WORKFLOW_STOP_OUTCOMES.get(reason, streaming.CONSUMER_EXITS.get(reason))
    assert outcome is not None and outcome[:2] == expected


# ---------------------------------------------------------------------------
# The real LangChain retriever path
# ---------------------------------------------------------------------------

def test_a_stop_propagates_through_the_real_langchain_retriever():
    from langchain_pinecone import PineconeVectorStore as LangChainPineconeVectorStore

    from tests.test_vectorstore import FakeIndex

    events_seen = []

    class Embeddings:
        def embed_query(self, text):
            events_seen.append("embed")  # the embedding call (in flight)
            return [0.1, 0.2]

        def embed_documents(self, texts):
            return [[0.1, 0.2] for _ in texts]

    index = FakeIndex()
    store = LangChainPineconeVectorStore(
        index=vectorstore._TimeoutIndex(index, 1.0), embedding=Embeddings(), namespace="company-it-kb"
    )
    retriever = store.as_retriever(search_kwargs={"k": 2})

    def stop_after_embedding():
        return "disconnected" if events_seen else None

    with cancellation.stop_checks(stop_after_embedding):
        with pytest.raises(cancellation.WorkflowStopped) as stopped:
            retriever.invoke("VPN?")

    assert stopped.value.reason == "disconnected"  # not wrapped or swallowed by LangChain
    assert events_seen == ["embed"]
    assert index.calls == []  # the Pinecone query was never sent


# ---------------------------------------------------------------------------
# Shutdown through the application lifespan
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_lifespan_shutdown_stops_a_running_stream_before_its_next_call(stream_app, fake_rag, monkeypatch):
    gate, llm, retriever = gated_rag(fake_rag)
    service = stream_app(streaming.stream_ask)
    monkeypatch.setattr(streaming, "_service", service)  # the lifespan closes the singleton
    monkeypatch.setattr(routes, "STREAM_SHUTDOWN_DRAIN_SECONDS", 0.1)

    async with routes._sessions_lifespan(None):
        run = service.start_run("VPN?", "req-1", asyncio.get_running_loop())
        consumer = asyncio.ensure_future(consume(run))
        await wait_event(gate.reached)
    # Lifespan shutdown: the drain ended with the call in flight; the signal is set.
    assert cancellation.shutdown_signalled()
    _, end = await consumer
    assert end is ABORTED
    assert_held_until_the_worker_exits(service, run)

    gate.release.set()
    await wait_event(run.hooks["worker_exited"])

    assert provider_calls(llm, retriever) == ["RouteDecision"]
    [row] = service.audit.rows
    assert (row["status"], row["error_type"]) == ("cancelled", "ServerShutdown")


@pytest.mark.anyio
async def test_lifespan_shutdown_stops_a_running_stateless_chat_before_its_next_call(fake_rag):
    from fastapi import HTTPException

    init_db()
    gate, llm, retriever = gated_rag(fake_rag)
    outcome = {}

    def chat():
        try:
            outcome["response"] = routes.chat(routes.ChatRequest(question="VPN?"), None)
        except HTTPException as exc:
            outcome["error"] = exc

    async with routes._sessions_lifespan(None):
        worker = asyncio.ensure_future(asyncio.to_thread(chat))
        await wait_event(gate.reached)  # the embedding call is in flight
    assert cancellation.shutdown_signalled()
    gate.release.set()
    await asyncio.wait_for(worker, GUARD)

    assert "response" not in outcome and outcome["error"].status_code == 500
    assert provider_calls(llm, retriever) == ["RouteDecision"]
    [row] = audits_for(outcome["error"].detail["request_id"])
    assert (row["status"], row["error_type"]) == ("failed", "ServerShutdown")
