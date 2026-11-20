"""Endpoint tests for POST /api/chat/stream (A1-3), driven through the ASGI harness.

A test ``StreamService`` (scripted workflow or the real ``stream_ask`` with
fakes) replaces the service singleton through the ``stream_app`` fixture in
conftest, which also checks the run invariants after each test.
"""

import asyncio
import json

import pytest

import app.main
from app.api import routes, streaming
from app.rag.state import EvidenceGrade, RouteDecision
from app.services.audit import get_recent_audits, init_db, write_audit
from tests.conftest import FakeLLM, FakeRetriever, kb_doc
from tests.streaming_support import (
    ASGIHarness,
    ChunkScript,
    FakeClock,
    assert_stream_grammar,
    body_contains,
    is_response_start,
    parse_sse,
)
from tests.test_stream_run import Gate, ScriptedWorkflow, final_state

GUARD = 5.0
QUESTION = "How do I connect to the VPN?"


async def post_stream(question=QUESTION, *, harness=None, **harness_kwargs):
    harness = harness or ASGIHarness(app.main.app, **harness_kwargs)
    return await harness.request("POST", "/api/chat/stream", json_body={"question": question})


async def wait_event(event, timeout=GUARD):
    if not await asyncio.to_thread(event.wait, timeout):
        raise AssertionError("event not set in time")


def frames(result):
    return parse_sse(result.body)


# -------------------------------------------------------------------
# Disabled mode and validation
# -------------------------------------------------------------------

@pytest.mark.anyio
@pytest.mark.parametrize("body", [{"question": QUESTION}, {"question": "a"}, {}])
async def test_disabled_endpoint_returns_404(body, settings):
    assert settings.stream_enabled is False
    harness = ASGIHarness(app.main.app)

    result = await harness.request("POST", "/api/chat/stream", json_body=body)

    assert result.status == 404
    assert json.loads(result.body) == {"detail": "Not Found"}
    assert streaming._service is None  # no service or worker created


RAW_BODIES = [
    pytest.param(b'{"question": "How do I', "application/json", id="malformed-json"),
    pytest.param(b"", "application/json", id="empty-body"),
    pytest.param(b"question=VPN", "application/x-www-form-urlencoded", id="not-json"),
]


@pytest.mark.parametrize("body, content_type", RAW_BODIES)
def test_disabled_endpoint_returns_404_before_parsing_the_body(client, settings, body, content_type):
    assert settings.stream_enabled is False

    response = client.post("/api/chat/stream", content=body, headers={"content-type": content_type})

    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}
    assert streaming._service is None


@pytest.mark.parametrize("body, content_type", RAW_BODIES)
def test_enabled_endpoint_rejects_unparsable_bodies_like_api_chat(client, stream_app, body, content_type):
    service = stream_app(ScriptedWorkflow([("final", final_state())]))
    headers = {"content-type": content_type}

    chat = client.post("/api/chat", content=body, headers=headers)
    response = client.post("/api/chat/stream", content=body, headers=headers)

    # S1-5: a body that isn't application/json is refused (415) before parsing.
    expected = 422 if content_type == "application/json" else 415
    assert response.status_code == chat.status_code == expected
    assert response.json() == chat.json()  # same error format on both routes
    assert service.started == []


@pytest.mark.anyio
@pytest.mark.parametrize("question, error_type",[("a", "string_too_short"), ("   ", "string_too_short"), ("x" * 3001, "string_too_long")])
async def test_validation_errors_match_api_chat(stream_app, question, error_type):
    service = stream_app(ScriptedWorkflow([("final", final_state())]))
    chat = await ASGIHarness(app.main.app).request("POST", "/api/chat", json_body={"question": question})

    result = await post_stream(question)

    assert result.status == chat.status == 422
    stream_error = json.loads(result.body)["detail"][0]
    chat_error = json.loads(chat.body)["detail"][0]
    assert (stream_error["type"], stream_error["loc"]) == (error_type, ["body", "question"])
    assert (stream_error["type"], stream_error["loc"]) == (chat_error["type"], chat_error["loc"])
    assert service.started == []


# -------------------------------------------------------------------
# Success and the SSE contract
# -------------------------------------------------------------------

@pytest.mark.anyio
async def test_successful_stream_follows_the_contract(stream_app):
    final = final_state()
    service = stream_app(
        ScriptedWorkflow(
            [("trace", "Router → KB"), ("chunks", ["Install ", "GlobalProtect."]), ("trace", "Answer generation → PRIVATE KB"), ("final", final)]
        )
    )

    result = await post_stream()
    [run] = service.started
    await wait_event(run.hooks["worker_exited"])

    assert result.status == 200 and result.complete
    assert result.headers["content-type"] == "text/event-stream; charset=utf-8"
    assert result.headers["cache-control"] == "no-cache, no-transform"
    assert result.headers["x-accel-buffering"] == "no"
    assert "content-length" not in result.headers

    parsed = frames(result)
    assert_stream_grammar(parsed, connected=True)
    assert parsed.names == ["start", "trace", "delta", "delta", "trace", "done"]
    assert parsed.events[0].data == {"request_id": run.request_id, "version": 1}
    assert parsed.events[1].data == {"index": 0, "step": "Router → KB"}
    assert [e.data["text"] for e in parsed.events if e.event == "delta"] == ["Install ", "GlobalProtect."]
    assert parsed.events[-1].data == routes._chat_response(final, run.request_id, QUESTION)
    assert run.outcome.decision.status == "completed"
    assert [row["status"] for row in service.audit.rows] == ["success"]


@pytest.mark.anyio
async def test_done_payload_equals_api_chat_response(stream_app, fake_rag):
    init_db()

    def install():
        fake_rag(
            llm=FakeLLM(
                structured={RouteDecision: [{"route": "kb"}], EvidenceGrade: [{"grade": "good"}]},
                text=[ChunkScript.from_text("  Install GlobalProtect.\n- Then sign in.  ")],
            ),
            retriever=FakeRetriever([kb_doc("VPN uses GlobalProtect.", source="/kb/policy.pdf", page=0)]),
        )

    install()
    chat = await ASGIHarness(app.main.app).request("POST", "/api/chat", json_body={"question": QUESTION})
    install()
    service = stream_app(streaming.stream_ask, audit=write_audit)
    result = await post_stream()
    await wait_event(service.started[0].hooks["worker_exited"])  # the audit row is written by the worker

    chat_body = json.loads(chat.body)
    parsed = frames(result)
    done = parsed.events[-1].data
    assert parsed.events[-1].event == "done"
    assert {k: v for k, v in done.items() if k != "request_id"} == {
        k: v for k, v in chat_body.items() if k != "request_id"
    }
    assert "".join(e.data["text"] for e in parsed.events if e.event == "delta").strip() == done["answer"]
    assert sorted(row["status"] for row in get_recent_audits(limit=5)) == ["success", "success"]


# -------------------------------------------------------------------
# Admission: capacity and submission failure
# -------------------------------------------------------------------

@pytest.mark.anyio
async def test_capacity_exhaustion_returns_503_with_retry_after(stream_app, settings, monkeypatch):
    monkeypatch.setattr(settings, "stream_retry_after_seconds", 7)
    gate = Gate()
    service = stream_app(ScriptedWorkflow([("gate", gate), ("final", final_state())]), max_concurrent=1)
    first = asyncio.ensure_future(post_stream())
    await wait_event(gate.reached)

    second = await post_stream()

    assert second.status == 503
    assert second.headers["retry-after"] == "7"
    assert json.loads(second.body) == {"detail": {"message": "The server is busy. Please retry shortly."}}
    assert len(service.started) == 1

    gate.release.set()
    first_result = await first
    assert frames(first_result).names[-1] == "done"


@pytest.mark.anyio
async def test_submission_failure_returns_503_and_releases_capacity(stream_app, settings):
    service = stream_app(ScriptedWorkflow([("final", final_state())]))
    service._executor.shutdown(wait=True)

    result = await post_stream()

    assert result.status == 503
    assert result.headers["retry-after"] == str(settings.stream_retry_after_seconds)
    assert service.capacity.in_use == 0 and service.capacity.released_total == 1
    assert service.audit.rows == []


@pytest.mark.anyio
async def test_late_response_for_an_abandoned_run_gets_503(stream_app, settings):
    clock = FakeClock()
    service = stream_app(ScriptedWorkflow([("final", final_state())]), clock=clock)
    run = service.start_run(QUESTION, "req-late", asyncio.get_running_loop())
    await wait_event(run.hooks["waiting_for_decision"])
    clock.advance(120.0 + streaming.ATTACH_GRACE)
    await wait_event(run.hooks["worker_exited"])  # abandoned by the worker

    response = streaming.SSEResponse(
        run, build_done=lambda final: {}, heartbeat_seconds=15.0, retry_after_seconds=settings.stream_retry_after_seconds
    )
    result = await ASGIHarness(response).request("POST", "/api/chat/stream")

    assert result.status == 503
    assert result.headers["retry-after"] == str(settings.stream_retry_after_seconds)
    assert run.outcome.decision.marker == "abandoned: response never attached"
    assert len(service.audit.rows) == 1  # the abandoned run's row; none for the late response


# -------------------------------------------------------------------
# Disconnects and send failures
# -------------------------------------------------------------------

@pytest.mark.anyio
async def test_disconnect_mid_stream_cancels_without_terminal_event(stream_app):
    gate = Gate()
    workflow = ScriptedWorkflow([("trace", "Router → KB"), ("gate", gate), ("chunk", "never"), ("final", final_state())])
    service = stream_app(workflow)
    harness = ASGIHarness(app.main.app, spec_version="2.3")
    harness.disconnect_on(body_contains(b"event: trace"))

    result = await post_stream(harness=harness)
    [run] = service.started

    parsed = frames(result)
    assert_stream_grammar(parsed, connected=False)
    assert parsed.names == ["start", "trace"]
    assert not result.complete
    decision = run.outcome.decision
    assert (decision.status, decision.error_type) == ("cancelled", "ClientDisconnected")
    assert run.stop_requested

    gate.release.set()
    await wait_event(run.hooks["worker_exited"])
    assert [(row["status"], row["error_type"]) for row in service.audit.rows] == [("cancelled", "ClientDisconnected")]


@pytest.mark.anyio
async def test_send_failure_on_done_never_records_success(stream_app):
    service = stream_app(ScriptedWorkflow([("chunk", "A"), ("final", final_state())]))
    harness = ASGIHarness(app.main.app, spec_version="2.4")
    harness.fail_send(body_contains(b"event: done"))

    result = await post_stream(harness=harness)
    [run] = service.started
    await wait_event(run.hooks["worker_exited"])

    assert result.exception is None  # handled inside the response
    decision = run.outcome.decision
    assert (decision.status, decision.error_type) == ("cancelled", "SendFailed")
    assert [row["status"] for row in service.audit.rows] == ["cancelled"]


@pytest.mark.anyio
async def test_send_failure_mid_stream_is_classified(stream_app):
    gate = Gate()
    service = stream_app(ScriptedWorkflow([("trace", "t"), ("gate", gate), ("final", final_state())]))
    harness = ASGIHarness(app.main.app, spec_version="2.4")
    harness.fail_send(body_contains(b"event: trace"))

    await post_stream(harness=harness)
    [run] = service.started
    gate.release.set()
    await wait_event(run.hooks["worker_exited"])

    assert run.outcome.decision.error_type == "SendFailed"


@pytest.mark.anyio
async def test_disconnect_already_seen_before_done_is_not_success(stream_app):
    gate = Gate()
    service = stream_app(ScriptedWorkflow([("gate", gate), ("chunk", "A"), ("final", final_state())]))
    harness = ASGIHarness(app.main.app, spec_version="2.3", uvicorn_faithful=True)
    harness.disconnect_on(body_contains(b"event: start"))

    request = asyncio.ensure_future(post_stream(harness=harness))
    await wait_event(gate.reached)
    gate.release.set()  # later sends return silently, like uvicorn after connection_lost
    await request
    [run] = service.started
    await wait_event(run.hooks["worker_exited"])

    assert run.outcome.decision.status == "cancelled"
    assert [row["status"] for row in service.audit.rows] == ["cancelled"]


# -------------------------------------------------------------------
# Timeouts
# -------------------------------------------------------------------

@pytest.mark.anyio
async def test_timeout_during_response_sends_one_generic_error(stream_app):
    gate = Gate()
    service = stream_app(ScriptedWorkflow([("trace", "t"), ("gate", gate), ("final", final_state())]), timeout=0.3)

    result = await post_stream()
    [run] = service.started

    parsed = frames(result)
    assert_stream_grammar(parsed, connected=True)
    assert parsed.names == ["start", "trace", "error"]
    assert parsed.events[-1].data == {
        "request_id": run.request_id,
        "code": "timeout",
        "message": "The request took too long. Please try again.",
    }
    assert result.complete
    decision = run.outcome.decision
    assert (decision.status, decision.marker) == ("timeout", "deadline during response")

    gate.release.set()
    await wait_event(run.hooks["worker_exited"])
    assert [(row["status"], row["error_type"]) for row in service.audit.rows] == [("failed", "TimeoutError")]


@pytest.mark.anyio
async def test_timeout_before_response_start_sends_nothing(stream_app):
    gate = Gate()
    service = stream_app(ScriptedWorkflow([("gate", gate), ("final", final_state())]), timeout=0.3)
    harness = ASGIHarness(app.main.app)
    harness.gate(is_response_start)  # the header send never completes

    result = await post_stream(harness=harness)
    [run] = service.started

    assert result.delivered == [] and result.status is None  # left to the server (plain 500)
    decision = run.outcome.decision
    assert (decision.status, decision.marker) == ("timeout", "deadline before response start")

    gate.release.set()
    await wait_event(run.hooks["worker_exited"])
    assert service.audit.rows[0]["error_type"] == "TimeoutError"


# -------------------------------------------------------------------
# Errors and heartbeat
# -------------------------------------------------------------------

@pytest.mark.anyio
async def test_workflow_error_is_generic_and_hides_details(stream_app):
    service = stream_app(ScriptedWorkflow([("chunk", "partial"), ("raise", ValueError("db password hunter2"))]))

    result = await post_stream()
    [run] = service.started
    await wait_event(run.hooks["worker_exited"])

    parsed = frames(result)
    assert_stream_grammar(parsed, connected=True)
    assert parsed.names == ["start", "delta", "error"]
    assert parsed.events[-1].data == {
        "request_id": run.request_id,
        "code": "internal_error",
        "message": "Unable to process the question.",
    }
    assert b"hunter2" not in result.body and b"ValueError" not in result.body
    assert [(row["status"], row["error_type"]) for row in service.audit.rows] == [("failed", "ValueError")]


@pytest.mark.anyio
async def test_consumer_error_is_classified_and_reraised(stream_app):
    service = stream_app(ScriptedWorkflow([("final", final_state())]))
    run = service.start_run(QUESTION, "req-bug", asyncio.get_running_loop())

    def broken_build_done(final):
        raise KeyError("bug in payload builder")

    response = streaming.SSEResponse(run, build_done=broken_build_done, heartbeat_seconds=15.0, retry_after_seconds=5)
    result = await ASGIHarness(response).request("POST", "/api/chat/stream")
    await wait_event(run.hooks["worker_exited"])

    assert isinstance(result.exception, KeyError)  # unwrapped, not an ExceptionGroup
    decision = run.outcome.decision
    assert (decision.status, decision.error_type) == ("failed", "ConsumerError")
    assert run.lease.released


@pytest.mark.anyio
async def test_heartbeat_keeps_an_idle_stream_alive(stream_app, settings, monkeypatch):
    monkeypatch.setattr(settings, "stream_heartbeat_seconds", 0.05)
    gate = Gate()
    service = stream_app(ScriptedWorkflow([("gate", gate), ("final", final_state())]))
    harness = ASGIHarness(app.main.app)

    request = asyncio.ensure_future(post_stream(harness=harness))
    await harness.wait_for_send(body_contains(b": keepalive"))
    gate.release.set()
    result = await request

    parsed = frames(result)
    assert parsed.keepalives >= 1
    assert parsed.names == ["start", "done"]
    await wait_event(service.started[0].hooks["worker_exited"])


# -------------------------------------------------------------------
# Wiring and /api/chat compatibility
# -------------------------------------------------------------------

def test_stream_service_is_built_from_settings(settings, monkeypatch):
    monkeypatch.setattr(settings, "max_concurrent_streams", 3)
    monkeypatch.setattr(settings, "stream_timeout_seconds", 42.0)

    service = streaming.get_stream_service()

    assert service is streaming.get_stream_service()  # one per process
    assert service.capacity.limit == 3
    assert service.timeout_seconds == 42.0
    assert service._executor._max_workers == 3


def test_both_chat_routes_are_registered():
    paths = {(route.path, tuple(sorted(route.methods))) for route in app.main.app.routes if hasattr(route, "methods")}

    assert ("/api/chat", ("POST",)) in paths
    assert ("/api/chat/stream", ("POST",)) in paths


# -------------------------------------------------------------------
# A1-4 regression: the run deadline is enforced even without suspension
# -------------------------------------------------------------------

@pytest.mark.anyio
async def test_attach_after_the_deadline_sends_no_bytes(stream_app):
    clock = FakeClock()
    service = stream_app(ScriptedWorkflow([("final", final_state())]), clock=clock)
    run = service.start_run(QUESTION, "req-s14", asyncio.get_running_loop())
    await wait_event(run.hooks["waiting_for_decision"])
    clock.advance(120.0 + streaming.ATTACH_GRACE / 4)  # inside (D, D+G]

    response = streaming.SSEResponse(run, build_done=lambda final: {}, heartbeat_seconds=15.0, retry_after_seconds=5)
    result = await ASGIHarness(response, spec_version="2.4").request("POST", "/api/chat/stream")
    await wait_event(run.hooks["worker_exited"])

    assert result.delivered == []  # no headers, no SSE bytes: the server sends its plain 500
    decision = run.outcome.decision
    assert (decision.status, decision.marker) == ("timeout", "deadline before response start")
    assert [(row["status"], row["error_type"]) for row in service.audit.rows] == [("failed", "TimeoutError")]


@pytest.mark.anyio
async def test_deadline_passing_mid_stream_never_sends_done(stream_app):
    clock = FakeClock()
    service = stream_app(ScriptedWorkflow([("chunk", "A"), ("final", final_state())]), clock=clock)
    run = service.start_run(QUESTION, "req-mid", asyncio.get_running_loop())
    await wait_event(run.hooks["waiting_for_decision"])  # deltas and final already queued
    harness = ASGIHarness(
        streaming.SSEResponse(run, build_done=lambda final: {}, heartbeat_seconds=15.0, retry_after_seconds=5),
        spec_version="2.4",
    )

    def expire_after_start(message):
        if body_contains(b"event: start")(message):
            clock.advance(121.0)  # no send suspends, so only the explicit check sees this
        return False

    harness.disconnect_on(expire_after_start)
    result = await harness.request("POST", "/api/chat/stream")
    await wait_event(run.hooks["worker_exited"])

    parsed = frames(result)
    assert_stream_grammar(parsed, connected=True)
    assert parsed.names == ["start", "error"]
    assert parsed.events[-1].data["code"] == "timeout"
    assert run.outcome.decision.marker == "deadline during response"
    assert service.audit.rows[0]["error_type"] == "TimeoutError"


@pytest.mark.anyio
async def test_no_second_terminal_event_after_done(stream_app, monkeypatch):
    """A deadline cancellation landing after ``done`` was delivered adds no ``error``."""
    import anyio

    scopes = []

    class AnyioProxy:  # records the response's deadline scope; only streaming.py sees it
        def __getattr__(self, name):
            return getattr(anyio, name)

        @staticmethod
        def CancelScope(*args, **kwargs):
            scope = anyio.CancelScope(*args, **kwargs)
            scopes.append(scope)
            return scope

    monkeypatch.setattr(streaming, "anyio", AnyioProxy())
    service = stream_app(ScriptedWorkflow([("chunk", "A"), ("final", final_state())]))
    harness = ASGIHarness(app.main.app, spec_version="2.4")

    def cancel_after_done(message):
        if body_contains(b"event: done")(message):
            scopes[-1].cancel()  # takes effect at the checkpoint after "done"
        return False

    harness.disconnect_on(cancel_after_done)
    result = await post_stream(harness=harness)
    [run] = service.started
    await wait_event(run.hooks["worker_exited"])

    parsed = frames(result)
    assert parsed.names == ["start", "delta", "done"]
    assert result.complete
    assert run.outcome.decision.status == "timeout"  # conservative rule (revision 8, section 17)
