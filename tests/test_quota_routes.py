"""Security S1-4b: chat quotas on every chat route (mocked providers only).

The four chat entry points (stateless and session ``POST /api/chat`` and
``POST /api/chat/stream``) share one QuotaEngine. ``routes.ask`` and the
stream workflow are replaced by fakes, and the suite's network guard and
blocked clients make any real provider call fail, so nothing here can reach
a paid API.
"""

import asyncio
import json
import logging
import threading
import time

import pytest
from fastapi import HTTPException
from pydantic import SecretStr, ValidationError

import app.main
from app.api import routes, streaming
from app.rag import cancellation
from app.services import access, quota, quota_runtime
from app.services import sessions as session_service
from tests.conftest import IsolatedSettings
from tests.streaming_support import ASGIHarness, body_contains
from tests.test_chat_api import workflow_result
from tests.test_stream_run import Gate, ScriptedWorkflow, final_state

QUESTION = "How do I connect to the VPN?"
GUARD = 5.0
CODE_ALICE = "Alice-test-code-0123456789abcdefGHIJ"
CODE_BOB = "Bob-test-code-0123456789abcdefghijKL"
SECRET = "Q7v-test-secret-9fK2pX4mZ8rT1yW6nB3cJ5hL0dSxE4uA"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def quotas_on(monkeypatch, settings):
    """Quotas on, with the deadlines they require; small limits are set per test."""
    for name, value in (
        ("quota_enabled", True),
        ("chat_deadline_seconds", 90.0),
        ("stream_timeout_seconds", 90.0),
        ("session_turn_max_seconds", 105.0),
        ("openai_timeout_seconds", 30.0),
        ("openai_max_retries", 1),
        ("pinecone_retry_after_max_seconds", 5),
    ):
        monkeypatch.setattr(settings, name, value)
    return settings


def limits(monkeypatch, settings, **values):
    for name, value in values.items():
        monkeypatch.setattr(settings, f"quota_chat_{name}", value)


@pytest.fixture
def sessions_on(monkeypatch, settings):
    monkeypatch.setattr(settings, "sessions_enabled", True)
    monkeypatch.setattr(settings, "session_shutdown_grace_seconds", 0.2)
    monkeypatch.setattr(settings, "session_keeper_join_seconds", 0.3)
    monkeypatch.setattr(settings, "session_release_budget_seconds", 0.5)
    monkeypatch.setattr(settings, "session_db_busy_timeout_seconds", 0.5)
    return settings


@pytest.fixture
def fake_ask(monkeypatch):
    """A recording stand-in for ask(); ``behaviour`` can block or raise."""

    class FakeAsk:
        def __init__(self):
            self.calls = 0
            self.behaviour = None
            self.lock = threading.Lock()

        def __call__(self, question, history=None):
            with self.lock:
                self.calls += 1
            if self.behaviour is not None:
                self.behaviour()
            return workflow_result()

    fake = FakeAsk()
    monkeypatch.setattr(routes, "ask", fake)
    return fake


class CountingWorkflow(ScriptedWorkflow):
    """ScriptedWorkflow that accepts ``history`` and counts its calls."""

    def __init__(self, steps):
        super().__init__(steps)
        self.calls = 0

    def __call__(self, question, on_chunk, history=None):
        self.calls += 1
        return super().__call__(question, on_chunk)


def answer_steps(gate=None):
    steps = [("trace", "Router → KB"), ("chunk", "Install GlobalProtect.")]
    if gate is not None:
        steps.insert(1, ("gate", gate))
    return steps + [("final", final_state())]


def engine():
    return quota_runtime.peek_quota_engine()


def usage(identity=None):
    current = engine()
    if current is None:
        return {"daily": {"chat": 0}, "active": {"chat": 0}}
    return current.usage(identity)


def active():
    return usage()["active"]["chat"]


def wait_until(condition, timeout=GUARD):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(0.005)


async def wait_event(event, timeout=GUARD):
    if not await asyncio.to_thread(event.wait, timeout):
        raise AssertionError("event not set in time")


def assert_refused(response_status, headers, body, status, code):
    assert response_status == status
    assert headers["cache-control"] == "no-store"
    assert int(headers["retry-after"]) >= 1
    detail = body["detail"]
    assert detail["code"] == code and set(detail) == {"code", "message"}


def refused(response, status=429, code="quota_hourly_limit"):
    assert_refused(response.status_code, response.headers, response.json(), status, code)


def ask_chat(client, headers=None):
    return client.post("/api/chat", json={"question": QUESTION}, headers=headers or {})


def new_session(client):
    response = client.post("/api/sessions", json={})
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


def session_row(settings, token):
    import sqlite3

    connection = sqlite3.connect(settings.session_db_path)
    connection.row_factory = sqlite3.Row
    try:
        return dict(connection.execute(
            "SELECT * FROM sessions WHERE token_hash = ?", (session_service.hash_token(token),)).fetchone())
    finally:
        connection.close()


async def stream_request(token=None, harness=None, headers=None):
    harness = harness or ASGIHarness(app.main.app, spec_version="2.3")
    merged = dict(headers or {})
    if token is not None:
        merged["X-Session-ID"] = token
    return await harness.request("POST", "/api/chat/stream", json_body={"question": QUESTION}, headers=merged)


async def chat_request(token=None, headers=None):
    merged = dict(headers or {})
    if token is not None:
        merged["X-Session-ID"] = token
    return await ASGIHarness(app.main.app).request("POST", "/api/chat", json_body={"question": QUESTION}, headers=merged)


async def create_token():
    result = await ASGIHarness(app.main.app).request("POST", "/api/sessions", json_body={})
    assert result.status == 201, result.body
    return json.loads(result.body)["session_id"]


def harness_refused(result, status=429, code="quota_hourly_limit"):
    assert result.headers["content-type"].startswith("application/json")  # before any SSE byte
    assert_refused(result.status, result.headers, json.loads(result.body), status, code)


@pytest.fixture
def access_codes(monkeypatch, settings):
    """Access codes on (still not enforced); returns cookie headers per code."""
    entries = ",".join(f"{name}:{access.hash_code(code)}" for name, code in (("alice", CODE_ALICE), ("bob", CODE_BOB)))
    for name, value in (
        ("access_enabled", True),
        ("access_code_hashes", entries),
        ("access_cookie_secret", SecretStr(SECRET)),
        ("access_cookie_previous_secrets", SecretStr("")),
        ("access_cookie_secure", True),
        ("access_client_ip_header", ""),
        ("access_trust_socket_peer", False),
    ):
        monkeypatch.setattr(settings, name, value)
    config = access.AccessConfig.from_settings(settings)

    def cookie_for(code):
        now = access.wall_clock()
        value, _ = access.issue_cookie(config, access.match_code(config, code, now), now)
        return {"cookie": f"{config.cookie_name}={value}"}

    return {"alice": cookie_for(CODE_ALICE), "bob": cookie_for(CODE_BOB), "config": config}


# ---------------------------------------------------------------------------
# Startup configuration
# ---------------------------------------------------------------------------

VALID = dict(quota_enabled=True, chat_deadline_seconds=90, stream_timeout_seconds=90,
             session_turn_max_seconds=105, openai_timeout_seconds=30, openai_max_retries=1,
             pinecone_retry_after_max_seconds=5)


def test_quotas_are_disabled_by_default():
    from app.core import config

    assert config.Settings.model_fields["quota_enabled"].default is False
    assert IsolatedSettings().quota_enabled is False


def test_a_valid_quota_configuration_starts():
    settings = IsolatedSettings(**VALID)
    assert settings.quota_enabled is True
    assert quota_runtime.check_settings(settings).chat.global_concurrent == 2


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(chat_deadline_seconds=None), "CHAT_DEADLINE_SECONDS"),
        (dict(chat_deadline_seconds=90.5), "CHAT_DEADLINE_SECONDS"),
        (dict(stream_timeout_seconds=91, session_turn_max_seconds=106), "STREAM_TIMEOUT_SECONDS"),
        (dict(openai_timeout_seconds=30.5), "OPENAI_TIMEOUT_SECONDS"),
        (dict(openai_max_retries=2), "OPENAI_MAX_RETRIES"),
        (dict(session_turn_max_seconds=106), "SESSION_TURN_MAX_SECONDS"),
        (dict(quota_chat_per_code_hourly=30), "hourly chat limit"),  # above the daily 25
        (dict(quota_chat_per_code_concurrent=3), "per-code chat limit"),  # above the global 2
        (dict(quota_chat_global_daily=0), "greater than or equal to 1"),
        (dict(quota_busy_retry_after_seconds=0), "greater than or equal to 1"),
    ],
)
def test_an_unsafe_quota_configuration_refuses_to_start(overrides, message):
    with pytest.raises(ValidationError, match=message):
        IsolatedSettings(**{**VALID, **overrides})


def test_the_default_deadlines_are_refused_when_quotas_are_enabled():
    with pytest.raises(ValidationError, match="CHAT_DEADLINE_SECONDS"):
        IsolatedSettings(quota_enabled=True)


@pytest.mark.parametrize("value", ["1", "yes", "on", "", 1])
def test_the_quota_flag_accepts_only_true_or_false(value):
    with pytest.raises(ValidationError, match="must be true or false"):
        IsolatedSettings(quota_enabled=value)


def test_session_turns_stay_compatible_with_the_stream_deadline():
    # The existing rule (turn >= stream timeout + 15 s) still applies.
    with pytest.raises(ValidationError, match="session_turn_max_seconds must be at least"):
        IsolatedSettings(**{**VALID, "stream_timeout_seconds": 90, "session_turn_max_seconds": 100})


# ---------------------------------------------------------------------------
# Engine lifecycle
# ---------------------------------------------------------------------------

def test_with_quotas_disabled_no_engine_is_created_and_nothing_changes(client, fake_ask):
    for _ in range(30):  # more than any limit
        assert ask_chat(client).status_code == 200
    assert engine() is None
    assert fake_ask.calls == 30


def test_one_engine_per_process_never_rebuilt(client, quotas_on, fake_ask, monkeypatch):
    assert ask_chat(client).status_code == 200
    first = engine()
    monkeypatch.setattr(quotas_on, "quota_chat_per_code_hourly", 2)  # changed while running
    for _ in range(3):
        ask_chat(client)
    assert engine() is first  # same engine, same counters
    assert first.config.chat.per_code_hourly == 10  # limits apply from the next restart
    assert usage(quota.ANONYMOUS)["daily"]["chat"] == 4


# ---------------------------------------------------------------------------
# Stateless /api/chat
# ---------------------------------------------------------------------------

def test_an_admitted_question_counts_and_releases(client, quotas_on, fake_ask):
    assert ask_chat(client).status_code == 200
    assert usage(quota.ANONYMOUS)["daily"]["chat"] == 1
    assert active() == 0


def test_a_refused_question_calls_no_provider(client, quotas_on, fake_ask, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    limits(monkeypatch, quotas_on, per_code_hourly=2, per_code_daily=3)
    assert ask_chat(client).status_code == 200
    assert ask_chat(client).status_code == 200
    response = ask_chat(client)
    refused(response, 429, "quota_hourly_limit")
    assert response.json()["detail"]["message"] == "You've reached the hourly question limit. Please try again later."
    assert int(response.headers["retry-after"]) <= 3600
    assert fake_ask.calls == 2  # the refused question never reached ask()
    assert usage(quota.ANONYMOUS)["daily"]["chat"] == 2
    assert "Chat refused by quota for request_id=" in caplog.text
    assert QUESTION not in caplog.text


def test_validation_errors_use_no_quota(client, quotas_on, fake_ask):
    for body in ({"question": "x"}, {"question": "  "}, {}, {"question": "y" * 3001}):
        assert client.post("/api/chat", json=body).status_code == 422
    assert fake_ask.calls == 0
    assert usage()["daily"]["chat"] == 0


def test_a_provider_failure_still_counts_and_releases(client, quotas_on, fake_ask):
    def fail():
        raise RuntimeError("provider failed with secret-detail")

    fake_ask.behaviour = fail
    response = ask_chat(client)
    assert response.status_code == 500
    assert "secret-detail" not in response.text
    assert usage(quota.ANONYMOUS)["daily"]["chat"] == 1
    assert active() == 0


def test_the_slot_is_held_while_the_provider_works(client, quotas_on, fake_ask):
    started, finish = threading.Event(), threading.Event()

    def block():
        started.set()
        assert finish.wait(GUARD)

    fake_ask.behaviour = block
    first = {}
    thread = threading.Thread(target=lambda: first.setdefault("response", ask_chat(client)))
    thread.start()
    assert started.wait(GUARD)
    assert active() == 1  # provider work in progress: the slot is held
    fake_ask.behaviour = None
    refused(ask_chat(client), 429, "quota_busy")  # per code (anonymous): 1 at a time
    finish.set()
    thread.join(GUARD)
    assert first["response"].status_code == 200
    assert active() == 0


def test_the_deadline_stops_the_question_and_releases(client, quotas_on, fake_ask, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(cancellation, "monotonic", lambda: now[0])

    def past_deadline():
        cancellation.raise_if_stopped("OpenAI chat")  # before the first call: fine
        now[0] += 91  # the in-flight call ran past the 90 s deadline
        cancellation.raise_if_stopped("OpenAI chat")  # the next call never starts

    fake_ask.behaviour = past_deadline
    response = ask_chat(client)
    assert response.status_code == 500
    assert active() == 0
    assert usage(quota.ANONYMOUS)["daily"]["chat"] == 1


def test_a_shutdown_stop_releases(client, quotas_on, fake_ask):
    def shutdown():
        cancellation.signal_shutdown()
        cancellation.raise_if_stopped("OpenAI chat")

    fake_ask.behaviour = shutdown
    assert ask_chat(client).status_code == 500
    assert active() == 0


@pytest.mark.parametrize("failure", ["raises", "unavailable"])
def test_an_unavailable_engine_fails_closed_with_503(client, quotas_on, fake_ask, monkeypatch, failure):
    if failure == "raises":
        def broken(settings):
            raise RuntimeError("engine exploded")

        monkeypatch.setattr(quota_runtime, "get_quota_engine", broken)
    else:  # the engine's own fail-closed refusal (a broken clock)
        broken_engine = quota.QuotaEngine(monotonic=lambda: float("nan"))
        monkeypatch.setattr(quota_runtime, "get_quota_engine", lambda settings: broken_engine)
    response = ask_chat(client)
    refused(response, 503, "quota_unavailable")
    assert "exploded" not in response.text
    assert fake_ask.calls == 0


def test_50_concurrent_questions_admit_exactly_the_concurrency_limit(quotas_on, fake_ask, monkeypatch):
    limits(monkeypatch, quotas_on, per_code_concurrent=2, global_concurrent=2)
    release = threading.Event()
    fake_ask.behaviour = lambda: release.wait(GUARD)
    barrier = threading.Barrier(50)
    outcomes, lock = [], threading.Lock()

    def one():
        barrier.wait(GUARD)
        try:
            routes.chat(routes.ChatRequest(question=QUESTION), None)
            outcome = "answered"
        except HTTPException as exc:
            outcome = exc.detail["code"] if isinstance(exc.detail, dict) else str(exc.status_code)
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=one) for _ in range(50)]
    for thread in threads:
        thread.start()
    wait_until(lambda: len(outcomes) == 48)  # the refusals come back at once
    assert outcomes.count("quota_busy") == 48
    assert active() == 2 and fake_ask.calls == 2
    release.set()
    for thread in threads:
        thread.join(GUARD)
    assert outcomes.count("answered") == 2
    assert active() == 0


# ---------------------------------------------------------------------------
# Identity: verified codes only
# ---------------------------------------------------------------------------

def test_verified_codes_have_separate_buckets(client, quotas_on, fake_ask, access_codes, monkeypatch):
    limits(monkeypatch, quotas_on, per_code_hourly=2, per_code_daily=2)
    for _ in range(2):
        assert ask_chat(client, access_codes["alice"]).status_code == 200
    refused(ask_chat(client, access_codes["alice"]), 429, "quota_daily_limit")
    assert ask_chat(client, access_codes["bob"]).status_code == 200  # another code
    assert ask_chat(client).status_code == 200  # anonymous: its own bucket
    assert usage("alice")["daily"]["chat"] == 2
    assert usage("bob")["daily"]["chat"] == 1
    assert usage(quota.ANONYMOUS)["daily"]["chat"] == 1


def test_unverified_identities_fall_back_to_the_shared_anonymous_bucket(client, quotas_on, fake_ask, access_codes, monkeypatch):
    alice = access_codes["alice"]["cookie"]
    name, value = alice.split("=", 1)
    payload, signature = value.split(".")
    forged = f"{name}={payload}.{signature[:-2]}AA"
    other_name = f"bp_access={value}"  # the plain-HTTP name is not the configured one
    for headers in (
        {"cookie": forged},
        {"cookie": other_name},
        {"x-forwarded-for": "203.0.113.7"},
        {"x-quota-identity": "alice"},
        {"x-session-id-like": "alice"},
        {},
    ):
        assert ask_chat(client, headers).status_code == 200
    assert usage("alice")["daily"]["chat"] == 0
    assert usage(quota.ANONYMOUS)["daily"]["chat"] == 6


def test_a_revoked_codes_cookie_is_anonymous(client, quotas_on, fake_ask, access_codes, monkeypatch):
    monkeypatch.setattr(quotas_on, "access_code_hashes", f"bob:{access.hash_code(CODE_BOB)}")
    assert ask_chat(client, access_codes["alice"]).status_code == 200
    assert usage("alice")["daily"]["chat"] == 0
    assert usage(quota.ANONYMOUS)["daily"]["chat"] == 1


def test_with_access_codes_disabled_every_request_is_anonymous(client, quotas_on, fake_ask, access_codes, monkeypatch):
    monkeypatch.setattr(quotas_on, "access_enabled", False)
    assert ask_chat(client, access_codes["alice"]).status_code == 200
    assert usage(quota.ANONYMOUS)["daily"]["chat"] == 1 and usage("alice")["daily"]["chat"] == 0


def test_logs_never_contain_codes_cookies_ids_or_questions(client, quotas_on, fake_ask, access_codes, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    limits(monkeypatch, quotas_on, per_code_hourly=1, per_code_daily=1)
    cookie = access_codes["alice"]
    assert ask_chat(client, cookie).status_code == 200
    response = ask_chat(client, cookie)
    refused(response, 429, "quota_daily_limit")
    text = caplog.text + response.text
    for secret in (CODE_ALICE, cookie["cookie"].split("=", 1)[1], SECRET, "alice", QUESTION):
        assert secret not in text, secret[:12]
    assert "code_ref=" in caplog.text


# ---------------------------------------------------------------------------
# Session /api/chat
# ---------------------------------------------------------------------------

def test_session_conflicts_and_expired_sessions_use_no_quota(client, quotas_on, sessions_on, fake_ask):
    token = new_session(client)
    runtime = session_service.get_session_runtime(quotas_on)
    lease = runtime.manager.acquire(token, "holder")
    try:
        assert ask_chat(client, {"X-Session-ID": token}).status_code == 409
    finally:
        runtime.manager.finish(lease)
    assert ask_chat(client, {"X-Session-ID": "not-a-session"}).status_code == 410
    assert fake_ask.calls == 0
    assert usage()["daily"]["chat"] == 0


def test_a_session_question_counts_and_releases_both(client, quotas_on, sessions_on, fake_ask):
    token = new_session(client)
    response = ask_chat(client, {"X-Session-ID": token})
    assert response.status_code == 200 and response.json()["session"]["turn_saved"] is True
    assert usage(quota.ANONYMOUS)["daily"]["chat"] == 1 and active() == 0
    assert session_row(quotas_on, token)["lease_owner"] is None


def test_a_session_quota_refusal_saves_nothing_and_leaves_the_session_usable(client, quotas_on, sessions_on, fake_ask, monkeypatch):
    limits(monkeypatch, quotas_on, per_code_hourly=1, per_code_daily=1)
    token = new_session(client)
    assert ask_chat(client).status_code == 200  # uses the anonymous allowance
    refused(ask_chat(client, {"X-Session-ID": token}), 429, "quota_daily_limit")
    assert fake_ask.calls == 1
    row = session_row(quotas_on, token)
    assert row["turn_count"] == 0 and row["lease_owner"] is None  # nothing saved, lease released

    monkeypatch.setattr(quotas_on, "quota_enabled", False)  # the session itself still works
    response = ask_chat(client, {"X-Session-ID": token})
    assert response.status_code == 200 and response.json()["session"]["turn_saved"] is True


def test_a_session_provider_failure_releases_the_ticket_and_the_lease(client, quotas_on, sessions_on, fake_ask):
    def fail():
        raise RuntimeError("provider failed")

    fake_ask.behaviour = fail
    token = new_session(client)
    assert ask_chat(client, {"X-Session-ID": token}).status_code == 500
    assert active() == 0
    row = session_row(quotas_on, token)
    assert row["lease_owner"] is None and row["turn_count"] == 0


# ---------------------------------------------------------------------------
# /api/chat/stream (stateless and session)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_a_stream_is_admitted_counted_and_released(stream_app, quotas_on):
    service = stream_app(CountingWorkflow(answer_steps()))
    result = await stream_request()
    assert result.status == 200
    [run] = service.started
    await wait_event(run.hooks["worker_exited"])
    wait_until(lambda: active() == 0)
    assert usage(quota.ANONYMOUS)["daily"]["chat"] == 1


@pytest.mark.anyio
async def test_a_refused_stream_sends_json_before_sse_and_starts_nothing(stream_app, quotas_on, monkeypatch):
    limits(monkeypatch, quotas_on, per_code_hourly=1, per_code_daily=1)
    workflow = CountingWorkflow(answer_steps())
    service = stream_app(workflow)
    assert (await stream_request()).status == 200
    await wait_event(service.started[0].hooks["worker_exited"])
    result = await stream_request()
    harness_refused(result, 429, "quota_daily_limit")
    assert workflow.calls == 1 and len(service.started) == 1  # no worker for the refusal
    assert service.capacity.in_use == 0  # its capacity reservation was given back


@pytest.mark.anyio
async def test_stream_capacity_is_checked_before_quota(stream_app, quotas_on):
    gate = Gate()
    service = stream_app(CountingWorkflow(answer_steps(gate=gate)), max_concurrent=1)
    first = asyncio.create_task(stream_request())
    await wait_event(gate.reached)
    result = await stream_request()
    assert result.status == 503 and json.loads(result.body)["detail"] == {"message": streaming.BUSY_MESSAGE}
    assert usage(quota.ANONYMOUS)["daily"]["chat"] == 1  # the busy request used no quota
    gate.release.set()
    await first
    await wait_event(service.started[0].hooks["worker_exited"])
    wait_until(lambda: active() == 0)


@pytest.mark.anyio
async def test_a_disconnect_keeps_the_slot_until_the_worker_finishes(stream_app, quotas_on):
    gate = Gate()
    service = stream_app(CountingWorkflow(answer_steps(gate=gate)))
    harness = ASGIHarness(app.main.app, spec_version="2.3")
    harness.disconnect_on(body_contains(b"event: trace"))
    await stream_request(harness=harness)  # the client is gone
    [run] = service.started
    assert not run.hooks["worker_exited"].is_set()
    assert active() == 1  # provider work still running: never released early
    gate.release.set()
    await wait_event(run.hooks["worker_exited"])
    wait_until(lambda: active() == 0)
    assert run.outcome.decision.status == "cancelled"


@pytest.mark.anyio
async def test_a_stream_timeout_keeps_the_slot_until_the_worker_finishes(stream_app, quotas_on):
    gate = Gate()
    service = stream_app(CountingWorkflow(answer_steps(gate=gate)), timeout=0.2)
    result = await stream_request()
    assert b"timeout" in result.body
    [run] = service.started
    assert active() == 1  # the deadline passed, but the worker is still in its call
    gate.release.set()
    await wait_event(run.hooks["worker_exited"])
    wait_until(lambda: active() == 0)


@pytest.mark.anyio
async def test_a_failed_stream_start_releases_the_ticket(stream_app, quotas_on, monkeypatch):
    service = stream_app(CountingWorkflow(answer_steps()))

    def refuse(*args, **kwargs):
        raise RuntimeError("cannot schedule new futures after shutdown")

    monkeypatch.setattr(service._executor, "submit", refuse)
    result = await stream_request()
    assert result.status == 503
    assert active() == 0  # released with the slot
    assert usage(quota.ANONYMOUS)["daily"]["chat"] == 1  # admitted, so counted
    assert service.capacity.in_use == 0


@pytest.mark.anyio
async def test_shutdown_releases_the_slot_only_when_the_worker_exits(stream_app, quotas_on):
    gate = Gate()
    service = stream_app(CountingWorkflow(answer_steps(gate=gate)))
    harness = ASGIHarness(app.main.app, spec_version="2.3")
    harness.disconnect_on(body_contains(b"event: trace"))
    await stream_request(harness=harness)
    [run] = service.started
    report = await asyncio.to_thread(service.close, 0.1)
    assert report["still_running"] == 1
    assert active() == 1  # workers are never killed; the slot waits for this one
    gate.release.set()
    await wait_event(run.hooks["worker_exited"])
    wait_until(lambda: active() == 0)


@pytest.mark.anyio
async def test_a_session_stream_refusal_saves_nothing_and_releases_everything(stream_app, quotas_on, sessions_on, fake_ask, monkeypatch):
    limits(monkeypatch, quotas_on, per_code_hourly=1, per_code_daily=1)
    workflow = CountingWorkflow(answer_steps())
    service = stream_app(workflow)
    token = await create_token()
    assert (await chat_request()).status == 200  # the anonymous allowance is used
    result = await stream_request(token)
    harness_refused(result, 429, "quota_daily_limit")
    assert workflow.calls == 0 and service.started == []
    assert service.capacity.in_use == 0
    row = session_row(quotas_on, token)
    assert row["turn_count"] == 0 and row["lease_owner"] is None


@pytest.mark.anyio
async def test_a_session_stream_conflict_uses_no_quota(stream_app, quotas_on, sessions_on):
    service = stream_app(CountingWorkflow(answer_steps()))
    token = await create_token()
    runtime = session_service.get_session_runtime(quotas_on)
    lease = await asyncio.to_thread(runtime.manager.acquire, token, "holder")
    try:
        result = await stream_request(token)
        assert result.status == 409
    finally:
        await asyncio.to_thread(runtime.manager.finish, lease)
    assert usage()["daily"]["chat"] == 0 and service.capacity.in_use == 0


@pytest.mark.anyio
async def test_a_session_stream_is_counted_and_released(stream_app, quotas_on, sessions_on):
    service = stream_app(CountingWorkflow(answer_steps()))
    token = await create_token()
    result = await stream_request(token)
    assert result.status == 200
    [run] = service.started
    await wait_event(run.hooks["worker_exited"])
    wait_until(lambda: active() == 0)
    assert usage(quota.ANONYMOUS)["daily"]["chat"] == 1


# ---------------------------------------------------------------------------
# One pool for all four routes
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_all_four_routes_share_one_quota(stream_app, quotas_on, sessions_on, fake_ask, monkeypatch):
    limits(monkeypatch, quotas_on, per_code_hourly=4, per_code_daily=4)
    service = stream_app(CountingWorkflow(answer_steps()))
    token = await create_token()
    assert (await chat_request()).status == 200  # 1: stateless chat
    assert (await chat_request(token)).status == 200  # 2: session chat
    assert (await stream_request()).status == 200  # 3: stateless stream
    await wait_event(service.started[-1].hooks["worker_exited"])
    wait_until(lambda: active() == 0)
    assert (await stream_request(token)).status == 200  # 4: session stream
    await wait_event(service.started[-1].hooks["worker_exited"])
    wait_until(lambda: active() == 0)
    for request in (chat_request(), chat_request(token), stream_request(), stream_request(token)):
        harness_refused(await request, 429, "quota_daily_limit")  # every route is now refused
    assert usage(quota.ANONYMOUS)["daily"]["chat"] == 4


@pytest.mark.anyio
async def test_a_running_stream_holds_the_slot_for_chat_too(stream_app, quotas_on, fake_ask):
    gate = Gate()
    service = stream_app(CountingWorkflow(answer_steps(gate=gate)))
    first = asyncio.create_task(stream_request())
    await wait_event(gate.reached)
    harness_refused(await chat_request(), 429, "quota_busy")  # one pool across routes
    assert fake_ask.calls == 0
    gate.release.set()
    await first
    await wait_event(service.started[0].hooks["worker_exited"])
    wait_until(lambda: active() == 0)
    assert (await chat_request()).status == 200


# ---------------------------------------------------------------------------
# Exactly-once release
# ---------------------------------------------------------------------------

@pytest.fixture
def release_calls(monkeypatch):
    calls = []
    real = quota.Ticket.release

    def counting(self):
        calls.append(id(self))
        return real(self)

    monkeypatch.setattr(quota.Ticket, "release", counting)
    return calls


@pytest.mark.anyio
async def test_every_ticket_is_released_exactly_once_on_every_route(stream_app, quotas_on, sessions_on, fake_ask, release_calls):
    service = stream_app(CountingWorkflow(answer_steps()))
    token = await create_token()
    await chat_request()
    await chat_request(token)
    fake_ask.behaviour = lambda: (_ for _ in ()).throw(RuntimeError("provider failed"))
    await chat_request()  # a failure
    await chat_request(token)  # a session failure
    await stream_request()
    await wait_event(service.started[-1].hooks["worker_exited"])
    await stream_request(token)
    await wait_event(service.started[-1].hooks["worker_exited"])
    wait_until(lambda: len(release_calls) == 6)
    time.sleep(0.05)
    assert len(release_calls) == 6 and len(set(release_calls)) == 6  # one call per ticket
    assert usage(quota.ANONYMOUS)["daily"]["chat"] == 6 and active() == 0


# ---------------------------------------------------------------------------
# Stream release callbacks are guarded
# ---------------------------------------------------------------------------

def test_a_failing_release_callback_skips_nothing(caplog):
    capacity = streaming.StreamCapacity(1)
    ran = []

    def registry():
        ran.append("registry")
        raise RuntimeError("registry failed")

    lease = capacity.try_acquire(registry)
    lease.add_release_callback(lambda: (_ for _ in ()).throw(ValueError("ticket failed")))
    lease.add_release_callback(lambda: ran.append("ticket"))
    lease.mark_worker_done()
    assert ran == []  # nothing runs until both the worker and the consumer are done
    assert capacity.in_use == 1
    lease.mark_consumer_done()
    assert ran == ["registry", "ticket"]
    assert capacity.in_use == 0
    assert "Stream release callback failed (RuntimeError)." in caplog.text
    assert "Stream release callback failed (ValueError)." in caplog.text
    assert "registry failed" not in caplog.text and "ticket failed" not in caplog.text


def test_a_callback_added_after_the_release_runs_once_at_once():
    capacity = streaming.StreamCapacity(1)
    lease = capacity.try_acquire(None)
    lease.mark_worker_done()
    lease.mark_consumer_done()
    ran = []
    lease.add_release_callback(lambda: ran.append(1))
    assert ran == [1]
    lease.mark_consumer_done()  # a later mark releases nothing again
    assert ran == [1]


def test_a_failing_capacity_release_still_runs_the_callbacks(monkeypatch):
    capacity = streaming.StreamCapacity(1)
    ran = []
    lease = capacity.try_acquire(lambda: ran.append("registry"))
    lease.add_release_callback(lambda: ran.append("ticket"))
    monkeypatch.setattr(capacity, "_release", lambda: (_ for _ in ()).throw(RuntimeError("semaphore")))
    lease.mark_worker_done()
    lease.mark_consumer_done()
    assert ran == ["registry", "ticket"]


# ===========================================================================
# S1-4b remediation
# ===========================================================================

# ---------------------------------------------------------------------------
# FIX 1: Pinecone's longest honoured Retry-After must be bounded
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, 6, 30, 21600])
def test_quotas_require_a_small_pinecone_retry_after_bound(value):
    with pytest.raises(ValidationError, match="PINECONE_RETRY_AFTER_MAX_SECONDS"):
        IsolatedSettings(**{**VALID, "pinecone_retry_after_max_seconds": value})


def test_a_negative_pinecone_retry_after_bound_is_refused():
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        IsolatedSettings(**{**VALID, "pinecone_retry_after_max_seconds": -1})


@pytest.mark.parametrize("value", [0, 5])
def test_pinecone_retry_after_bounds_of_0_and_5_are_accepted(value):
    assert IsolatedSettings(**{**VALID, "pinecone_retry_after_max_seconds": value}).quota_enabled


@pytest.mark.parametrize("value", [None, 30, 21600])
def test_with_quotas_off_the_pinecone_setting_is_unchanged(value):
    assert IsolatedSettings(pinecone_retry_after_max_seconds=value).pinecone_retry_after_max_seconds == value


@pytest.mark.parametrize("value", [0, 5])
def test_the_pinecone_client_actually_applies_the_bound(quotas_on, monkeypatch, value):
    from app.rag import vectorstore

    class FakePinecone:  # never connects; only records the configuration
        def __init__(self, api_key):
            self._openapi_config = type("Config", (), {"retries": None})()

    monkeypatch.setattr(quotas_on, "pinecone_retry_after_max_seconds", value)
    monkeypatch.setattr(vectorstore, "Pinecone", FakePinecone)
    retries = vectorstore._pinecone_client()._openapi_config.retries
    assert retries.retry_after_max == value
    assert retries.total == 5 and retries.backoff_factor == 0.25  # the SDK default otherwise


# ---------------------------------------------------------------------------
# FIX 4: the engine is created only from safe settings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, value",
    [
        ("pinecone_retry_after_max_seconds", None),
        ("pinecone_retry_after_max_seconds", 60),
        ("chat_deadline_seconds", None),
        ("stream_timeout_seconds", 120.0),
        ("openai_timeout_seconds", 60.0),
        ("openai_max_retries", 5),
        ("session_turn_max_seconds", 180.0),
        ("quota_chat_per_code_hourly", 30),  # above the daily limit
    ],
)
def test_unsafe_settings_before_the_engine_exists_fail_closed(client, quotas_on, fake_ask, monkeypatch, caplog, name, value):
    monkeypatch.setattr(quotas_on, name, value)  # changed after startup validation
    response = ask_chat(client)
    refused(response, 503, "quota_unavailable")
    assert fake_ask.calls == 0  # no provider call
    assert engine() is None  # nothing was created
    assert "(QuotaConfigError)" in caplog.text
    monkeypatch.undo()


def test_the_engine_is_created_once_settings_are_safe(client, quotas_on, fake_ask, monkeypatch):
    monkeypatch.setattr(quotas_on, "pinecone_retry_after_max_seconds", None)
    refused(ask_chat(client), 503, "quota_unavailable")
    monkeypatch.setattr(quotas_on, "pinecone_retry_after_max_seconds", 0)
    assert ask_chat(client).status_code == 200
    assert engine() is not None and fake_ask.calls == 1


def test_an_existing_engine_is_never_rebuilt_or_reset(client, quotas_on, fake_ask, monkeypatch):
    assert ask_chat(client).status_code == 200
    first = engine()
    monkeypatch.setattr(quotas_on, "pinecone_retry_after_max_seconds", None)  # unsafe afterwards
    assert ask_chat(client).status_code == 200  # checked when the engine was created
    assert engine() is first
    assert usage(quota.ANONYMOUS)["daily"]["chat"] == 2  # counters kept


def test_with_quotas_off_unsafe_quota_settings_change_nothing(client, fake_ask, monkeypatch, settings):
    monkeypatch.setattr(settings, "quota_chat_per_code_hourly", 30)
    monkeypatch.setattr(settings, "pinecone_retry_after_max_seconds", None)
    assert ask_chat(client).status_code == 200
    assert engine() is None


# ---------------------------------------------------------------------------
# FIX 2: session /api/chat — everything after admission is cleaned up
# ---------------------------------------------------------------------------

@pytest.fixture
def lenient_client():
    from fastapi.testclient import TestClient

    with TestClient(app.main.app, raise_server_exceptions=False) as test_client:
        yield test_client


def _failing_once(real, error=RuntimeError):
    state = {"armed": True}

    def wrapper(*args, **kwargs):
        if state["armed"]:
            state["armed"] = False
            raise error("injected")
        return real(*args, **kwargs)

    return wrapper


def test_a_history_failure_after_admission_releases_everything(lenient_client, quotas_on, sessions_on, fake_ask, monkeypatch, release_calls):
    token = new_session(lenient_client)
    monkeypatch.setattr(routes, "history_window", _failing_once(routes.history_window))
    response = ask_chat(lenient_client, {"X-Session-ID": token})
    assert response.status_code == 500 and "injected" not in response.text
    assert fake_ask.calls == 0
    assert active() == 0 and len(release_calls) == 1  # the ticket, exactly once
    row = session_row(quotas_on, token)
    assert row["lease_owner"] is None and row["turn_count"] == 0  # released, nothing saved
    assert ask_chat(lenient_client).status_code == 200  # no leaked slot: anonymous isn't busy
    later = ask_chat(lenient_client, {"X-Session-ID": token})
    assert later.status_code == 200 and later.json()["session"]["turn_saved"] is True  # still usable


def test_a_lease_release_failure_still_releases_the_ticket(lenient_client, quotas_on, sessions_on, fake_ask, monkeypatch, release_calls):
    token = new_session(lenient_client)
    runtime = session_service.get_session_runtime(quotas_on)
    monkeypatch.setattr(routes, "history_window", _failing_once(routes.history_window))
    monkeypatch.setattr(runtime.manager, "finish", _failing_once(runtime.manager.finish))
    assert ask_chat(lenient_client, {"X-Session-ID": token}).status_code == 500
    assert active() == 0 and len(release_calls) == 1


def test_a_session_quota_refusal_is_still_429_not_500(lenient_client, quotas_on, sessions_on, fake_ask, monkeypatch):
    limits(monkeypatch, quotas_on, per_code_hourly=1, per_code_daily=1)
    token = new_session(lenient_client)
    assert ask_chat(lenient_client).status_code == 200
    refused(ask_chat(lenient_client, {"X-Session-ID": token}), 429, "quota_daily_limit")
    assert session_row(quotas_on, token)["lease_owner"] is None


# ---------------------------------------------------------------------------
# FIX 3: session and stateless streams — every boundary up to the worker start
# ---------------------------------------------------------------------------

async def _assert_stream_cleaned(service, settings, token=None):
    wait_until(lambda: active() == 0)
    assert service.capacity.in_use == 0 and service.started == []
    assert service.capacity.acquired_total == service.capacity.released_total
    if token is not None:
        row = session_row(settings, token)
        assert row["lease_owner"] is None and row["turn_count"] == 0


@pytest.mark.anyio
async def test_a_lease_acquisition_failure_returns_the_capacity(stream_app, quotas_on, sessions_on, monkeypatch):
    workflow = CountingWorkflow(answer_steps())
    service = stream_app(workflow)
    token = await create_token()
    monkeypatch.setattr(routes, "_acquire_session_lease", _failing_once(routes._acquire_session_lease))
    result = await stream_request(token)
    assert result.status == 500 or result.exception is not None
    await _assert_stream_cleaned(service, quotas_on, token)
    assert workflow.calls == 0 and usage()["daily"]["chat"] == 0  # no admission either


@pytest.mark.anyio
@pytest.mark.parametrize("target", ["history_window", "StreamSession"])
async def test_a_failure_after_admission_releases_ticket_lease_and_capacity(stream_app, quotas_on, sessions_on, monkeypatch, release_calls, target):
    workflow = CountingWorkflow(answer_steps())
    service = stream_app(workflow)
    token = await create_token()
    if target == "history_window":
        monkeypatch.setattr(routes, "history_window", _failing_once(routes.history_window))
    else:
        monkeypatch.setattr(streaming, "StreamSession", _failing_once(streaming.StreamSession))
    result = await stream_request(token)
    assert result.status == 500 or result.exception is not None
    await _assert_stream_cleaned(service, quotas_on, token)
    assert workflow.calls == 0  # no worker runs without its ticket
    assert len(release_calls) == 1  # the ticket, exactly once (through the slot)
    assert usage(quota.ANONYMOUS)["daily"]["chat"] == 1


@pytest.mark.anyio
async def test_a_lease_release_failure_still_returns_capacity_and_ticket(stream_app, quotas_on, sessions_on, monkeypatch, release_calls):
    service = stream_app(CountingWorkflow(answer_steps()))
    token = await create_token()
    runtime = session_service.get_session_runtime(quotas_on)
    monkeypatch.setattr(routes, "history_window", _failing_once(routes.history_window))
    monkeypatch.setattr(runtime.manager, "finish", _failing_once(runtime.manager.finish))
    await stream_request(token)
    wait_until(lambda: active() == 0)
    assert service.capacity.in_use == 0 and len(release_calls) == 1  # cancel ran despite the failure


@pytest.mark.anyio
@pytest.mark.parametrize("session", [False, True])
async def test_a_ticket_admitted_but_never_attached_is_released(stream_app, quotas_on, sessions_on, monkeypatch, release_calls, session):
    service = stream_app(CountingWorkflow(answer_steps()))
    token = await create_token() if session else None
    monkeypatch.setattr(streaming.StreamLease, "add_release_callback",
                        _failing_once(streaming.StreamLease.add_release_callback))
    await stream_request(token)
    await _assert_stream_cleaned(service, quotas_on, token)
    assert len(release_calls) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("session", [False, True])
async def test_a_failure_creating_the_run_releases_everything(stream_app, quotas_on, sessions_on, monkeypatch, release_calls, session):
    workflow = CountingWorkflow(answer_steps())
    service = stream_app(workflow)
    token = await create_token() if session else None
    monkeypatch.setattr(streaming, "StreamRun", _failing_once(streaming.StreamRun, error=ValueError))
    result = await stream_request(token)
    assert result.status == 503  # start() couldn't start: the usual stream_busy answer
    await _assert_stream_cleaned(service, quotas_on, token)
    assert workflow.calls == 0 and len(release_calls) == 1


@pytest.mark.anyio
async def test_a_started_session_stream_releases_its_ticket_once_after_the_worker(stream_app, quotas_on, sessions_on, release_calls):
    gate = Gate()
    service = stream_app(CountingWorkflow(answer_steps(gate=gate)))
    token = await create_token()
    harness = ASGIHarness(app.main.app, spec_version="2.3")
    harness.disconnect_on(body_contains(b"event: trace"))
    await stream_request(token, harness=harness)
    [run] = service.started
    assert active() == 1 and release_calls == []  # still running: not released
    gate.release.set()
    await wait_event(run.hooks["worker_exited"])
    wait_until(lambda: active() == 0)
    assert len(release_calls) == 1


# ---------------------------------------------------------------------------
# FIX 5: an interrupt in a release step can't skip the other steps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_an_interrupting_callback_still_lets_the_cleanup_finish_and_is_re_raised(interrupt):
    capacity = streaming.StreamCapacity(1)
    ran = []
    lease = capacity.try_acquire(lambda: ran.append("registry"))

    def interrupting():
        ran.append("interrupting")
        raise interrupt()

    lease.add_release_callback(interrupting)
    lease.add_release_callback(lambda: ran.append("ticket"))
    lease.mark_worker_done()
    with pytest.raises(interrupt):
        lease.mark_consumer_done()  # not swallowed
    assert ran == ["registry", "interrupting", "ticket"]
    assert capacity.in_use == 0 and lease.released
    lease.mark_consumer_done()  # nothing runs twice
    assert ran == ["registry", "interrupting", "ticket"]


def test_an_interrupt_releasing_capacity_still_runs_the_callbacks(monkeypatch):
    capacity = streaming.StreamCapacity(1)
    ran = []
    lease = capacity.try_acquire(lambda: ran.append("registry"))
    lease.add_release_callback(lambda: ran.append("ticket"))
    monkeypatch.setattr(capacity, "_release", lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
    lease.mark_worker_done()
    with pytest.raises(KeyboardInterrupt):
        lease.mark_consumer_done()
    assert ran == ["registry", "ticket"]
