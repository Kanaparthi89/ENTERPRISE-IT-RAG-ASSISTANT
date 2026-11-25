"""A2-5: session endpoints, session-aware /api/chat and the router lifespan (revision 8).

The workflow is stubbed (``routes.ask``) except where a test proves real
prompts: the stateless golden through the endpoint, and an end-to-end
follow-up whose history comes from the store.
"""

import asyncio
import logging
import re
import sqlite3
import threading
import time
from pathlib import Path

import anyio
import pytest
from fastapi.testclient import TestClient

import app.main
from app.api import routes, streaming
from app.core.config import BASE_DIR
from app.rag.state import EvidenceGrade, RouteDecision
from app.services import audit
from app.services import sessions as session_service
from app.services.audit import get_recent_audits
from tests.conftest import FakeLLM, FakeRetriever, kb_doc, legacy_owner_ref
from tests.session_support import count_session_rows, find_token
from tests.stateless_scenarios import SCENARIO_NAMES, load_golden, run_scenario

JSON = {"Content-Type": "application/json"}
STATELESS_KEYS = {"request_id", "answer", "source_used", "trace", "citations", "rewritten_query"}


def api_main_app():
    import app.api.main as api_main  # the second, protected entry point (unchanged)

    return api_main.app


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def enabled(monkeypatch, settings):
    monkeypatch.setattr(settings, "sessions_enabled", True)
    monkeypatch.setattr(settings, "session_shutdown_grace_seconds", 0.2)
    monkeypatch.setattr(settings, "session_keeper_join_seconds", 0.3)
    monkeypatch.setattr(settings, "session_release_budget_seconds", 0.5)
    monkeypatch.setattr(settings, "session_db_busy_timeout_seconds", 0.5)
    return settings


class FakeAsk:
    """Stands in for workflow.ask; records calls; can block or act mid-run."""

    def __init__(self, answer="Use GlobalProtect.", source="private_kb"):
        self.calls = []
        self.answer = answer
        self.source = source
        self.gate: threading.Event | None = None
        self.started = threading.Event()
        self.during = None
        self.error: Exception | None = None

    def __call__(self, question, history=None):
        self.calls.append({"question": question, "history": history})
        self.started.set()
        if self.gate is not None:
            assert self.gate.wait(10), "gate never opened"
        if self.during is not None:
            self.during()
        if self.error is not None:
            raise self.error
        return {
            "answer": self.answer,
            "source_used": self.source,
            "trace": ["Router → KB", "Answer generation → PRIVATE KB"],
            "citations": [],
            "current_query": question,
        }


@pytest.fixture
def fake_ask(monkeypatch):
    stub = FakeAsk()
    monkeypatch.setattr(routes, "ask", stub)
    return stub


def new_session(client) -> str:
    response = client.post("/api/sessions", headers=JSON, json={})
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


def ask_in_session(client, token, question="How do I connect to the VPN?"):
    return client.post("/api/chat", headers={"X-Session-ID": token}, json={"question": question})


def runtime():
    found = session_service.peek_session_runtime()
    assert found is not None
    return found


def session_row(settings, token):
    connection = sqlite3.connect(settings.session_db_path)
    connection.row_factory = sqlite3.Row
    try:
        found = connection.execute(
            "SELECT * FROM sessions WHERE token_hash = ?", (session_service.hash_token(token),)
        ).fetchone()
        return dict(found) if found else None
    finally:
        connection.close()


def detail(response) -> dict:
    return response.json()["detail"]


# ===================================================================
# Disabled mode (the default)
# ===================================================================

def test_sessions_are_disabled_by_default(settings):
    assert settings.sessions_enabled is False


@pytest.mark.parametrize("content_type", ["application/json", "application/x-www-form-urlencoded"])
def test_create_returns_404_while_disabled_before_any_check(client, settings, content_type):
    response = client.post("/api/sessions", headers={"Content-Type": content_type,
                                                     "Origin": "https://evil.example"}, content=b"{}")

    assert response.status_code == 404
    assert not settings.session_db_path.exists()


def test_chat_with_a_session_header_gets_410_while_disabled(client, fake_ask):
    response = ask_in_session(client, "anything")

    assert response.status_code == 410
    assert detail(response)["code"] == "session_expired"
    assert fake_ask.calls == []  # never silently answered statelessly


def test_delete_still_works_while_disabled(client, enabled, fake_ask, monkeypatch):
    token = new_session(client)
    monkeypatch.setattr(enabled, "sessions_enabled", False)

    response = client.delete("/api/sessions", headers={"X-Session-ID": token})

    assert response.status_code == 204
    assert session_row(enabled, token) is None


# ===================================================================
# POST /api/sessions
# ===================================================================

def test_create_returns_the_token_once_and_stores_only_its_hash(client, enabled):
    response = client.post("/api/sessions", headers=JSON, json={})

    assert response.status_code == 201
    body = response.json()
    assert set(body) == {"session_id", "idle_timeout_seconds", "expires_at"}
    assert re.fullmatch(r"[A-Za-z0-9_-]{43}", body["session_id"])
    assert body["idle_timeout_seconds"] == 1800
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", body["expires_at"])
    assert response.headers["cache-control"] == "no-store"
    assert not find_token(body["session_id"], db_paths=[enabled.session_db_path])
    assert session_row(enabled, body["session_id"]) is not None


@pytest.mark.parametrize("entry", ["app.main", "app.api.main"])
@pytest.mark.parametrize(
    "content_type, body",
    [
        ("application/x-www-form-urlencoded", b"a=1"),
        ("multipart/form-data; boundary=x", b"--x\r\n\r\n--x--"),
        ("text/plain", b'{"looks": "like json"}'),
        (None, b""),
    ],
)
def test_cross_origin_form_style_posts_never_create_a_session(enabled, entry, content_type, body):
    application = app.main.app if entry == "app.main" else api_main_app()
    headers = {"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"}
    if content_type:
        headers["Content-Type"] = content_type

    response = TestClient(application).post("/api/sessions", headers=headers, content=body)

    # S1-5: the origin is checked (by the route class) before the content type and body.
    assert response.status_code == 403
    assert detail(response) == {"code": "origin_not_allowed", "message": "Request origin not allowed."}
    assert count_session_rows(enabled.session_db_path) == 0


@pytest.mark.parametrize("entry", ["app.main", "app.api.main"])
@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://evil.example"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Sec-Fetch-Site": "same-site"},
        {"Origin": "null"},
        {"Origin": "http://testserver:8081"},  # same host, other port
        {"Origin": "javascript:alert(1)"},
        {"Sec-Fetch-Site": "none"},  # S1-5: only same-origin
        {"Origin": "https://testserver", "Host": "testserver:443"},  # S1-5: Host is never trusted
    ],
)
def test_cross_origin_json_requests_are_refused(enabled, entry, headers):
    application = app.main.app if entry == "app.main" else api_main_app()

    response = TestClient(application).post("/api/sessions", headers={**JSON, **headers}, content=b"{}")

    assert response.status_code == 403
    assert detail(response)["code"] == "origin_not_allowed"
    assert count_session_rows(enabled.session_db_path) == 0


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "http://testserver", "Sec-Fetch-Site": "same-origin"},
        {"Sec-Fetch-Site": "same-origin"},
        # S1-5: compared with PUBLIC_ORIGINS (http://testserver here), never with Host.
        {"Origin": "http://testserver:80"},  # default ports normalised
        {"Origin": "http://TestServer"},  # host comparison ignores case
    ],
)
def test_same_origin_requests_are_allowed(client, enabled, headers):
    response = client.post("/api/sessions", headers={**JSON, **headers}, content=b"{}")

    assert response.status_code == 201


def test_legacy_shim_only_a_request_without_origin_headers_creates_a_session(client, enabled):
    """SHIM-ONLY behaviour, not production: see tests/conftest.py (legacy_access_compat).

    The shim lets this module's header-less test clients through. In
    production, a request with neither Origin nor Sec-Fetch-Site is refused
    with 403 before the body is read:
    tests/test_access_enforcement.py::test_session_creation_without_origin_headers_is_refused_in_production.
    """
    from app.services import access_policy

    assert access_policy.origin_allowed.__module__ == "tests.conftest"  # the shim is what allows it
    response = client.post("/api/sessions", headers=JSON, content=b"{}")

    assert response.status_code == 201


# Body contract (revision 7 §14.3, carried into revision 8): an empty body or
# ``{}``; anything else is 415 unsupported_media_type, before any database access.

@pytest.mark.parametrize("body", [b"", b"{}", b"  {}  ", b"{ }", b"\n{}\n", b"   "])
def test_empty_body_or_empty_object_is_accepted(client, enabled, body):
    response = client.post("/api/sessions", headers=JSON, content=body)

    assert response.status_code == 201
    assert count_session_rows(enabled.session_db_path) == 1


@pytest.fixture
def no_session_storage(monkeypatch):
    """Fail the test if the request reaches the session runtime or database."""

    def forbidden(*args, **kwargs):
        raise AssertionError("the session runtime must not be used for a rejected request")

    monkeypatch.setattr(session_service, "get_session_runtime", forbidden)


@pytest.mark.parametrize("entry", ["app.main", "app.api.main"])
@pytest.mark.parametrize(
    "body",
    [
        b"{",  # malformed
        b"{}}",  # trailing garbage
        b"not json",
        b'{"a": 1}',  # non-empty object
        b'{"session_id": "chosen-by-client"}',  # a client may not choose its token
        b"[]",
        b"null",
        b'""',
        b"0",
        b"true",
        b"\xff\xfe",  # not UTF-8
        b"{" + b" " * 2000 + b"}",  # empty object, but over the 1 KiB limit
    ],
)
def test_other_bodies_get_415_and_never_touch_storage(enabled, no_session_storage, entry, body):
    application = app.main.app if entry == "app.main" else api_main_app()

    response = TestClient(application).post("/api/sessions", headers=JSON, content=body)

    assert response.status_code == 415
    assert detail(response) == {"code": "unsupported_media_type", "message": "Use application/json."}
    assert not enabled.session_db_path.exists()  # no session, and the database was never opened


def test_the_origin_is_checked_before_the_body(enabled, no_session_storage):
    """S1-5 order: access and the origin (403, by the route class), then content type and body (415)."""
    client = TestClient(app.main.app)
    bad_body_cross_origin = client.post(
        "/api/sessions", headers={**JSON, "Origin": "https://evil.example"}, content=b'{"a": 1}')
    good_body_cross_origin = client.post(
        "/api/sessions", headers={**JSON, "Origin": "https://evil.example"}, content=b"{}")
    bad_body_same_origin = client.post(
        "/api/sessions", headers={**JSON, "Origin": "http://testserver"}, content=b'{"a": 1}')

    assert bad_body_cross_origin.status_code == 403
    assert good_body_cross_origin.status_code == 403
    assert bad_body_same_origin.status_code == 415
    assert not enabled.session_db_path.exists()


def test_disabled_is_still_404_whatever_the_body(client, settings):
    response = client.post("/api/sessions", headers=JSON, content=b'{"a": 1}')

    assert response.status_code == 404


def test_create_reports_storage_failures_and_the_cap(client, enabled, monkeypatch):
    monkeypatch.setattr(enabled, "max_active_sessions", 1)
    new_session(client)

    at_cap = client.post("/api/sessions", headers=JSON, json={})
    assert at_cap.status_code == 503 and detail(at_cap)["code"] == "session_unavailable"

    def broken(*, owner_ref):  # the store's real signature: an owner is required (Phase B)
        raise session_service.SessionStoreUnavailable("down")

    monkeypatch.setattr(runtime().store, "create", broken)
    failed = client.post("/api/sessions", headers=JSON, json={})
    assert failed.status_code == 503
    assert detail(failed) == {"code": "session_unavailable",
                              "message": "Conversation memory is temporarily unavailable."}


# ===================================================================
# DELETE /api/sessions
# ===================================================================

def test_delete_clears_memory_and_is_idempotent(client, enabled, fake_ask):
    token = new_session(client)

    for _ in range(2):  # deleted, then already deleted
        response = client.delete("/api/sessions", headers={"X-Session-ID": token})
        assert response.status_code == 204 and response.content == b""
    assert client.delete("/api/sessions", headers={"X-Session-ID": "never-issued"}).status_code == 204
    assert ask_in_session(client, token).status_code == 410


def test_delete_of_an_expired_session_returns_204(client, enabled):
    token = new_session(client)
    connection = sqlite3.connect(enabled.session_db_path)
    connection.execute("UPDATE sessions SET idle_expires_at = 0, absolute_expires_at = 0")
    connection.commit()
    connection.close()

    assert client.delete("/api/sessions", headers={"X-Session-ID": token}).status_code == 204


@pytest.mark.parametrize("headers", [{}, {"X-Session-ID": ""}, {"X-Session-ID": "x" * 129}])
def test_delete_rejects_a_missing_or_malformed_header(client, enabled, headers):
    response = client.delete("/api/sessions", headers=headers)

    assert response.status_code == 400
    assert detail(response)["code"] == "invalid_session_id"


def test_delete_reports_503_when_deletion_cannot_be_confirmed(client, enabled):
    token = new_session(client)
    blocker = sqlite3.connect(enabled.session_db_path, isolation_level=None)
    blocker.execute("BEGIN EXCLUSIVE")
    try:
        response = client.delete("/api/sessions", headers={"X-Session-ID": token})
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()

    assert response.status_code == 503 and detail(response)["code"] == "session_unavailable"
    assert session_row(enabled, token) is not None  # not deleted


def test_delete_never_removes_audit_history(client, enabled, fake_ask):
    token = new_session(client)
    ask_in_session(client, token)
    ask_in_session(client, token, "And on a Mac?")
    before = get_recent_audits(limit=10)

    client.delete("/api/sessions", headers={"X-Session-ID": token})

    assert get_recent_audits(limit=10) == before
    assert len({record["session_id"] for record in before}) == 1


# ===================================================================
# Session-aware /api/chat
# ===================================================================

def test_first_and_follow_up_turns(client, enabled, fake_ask):
    token = new_session(client)

    first = ask_in_session(client, token, "How do I connect to the VPN?")
    second = ask_in_session(client, token, "What about on a Mac?")

    assert first.status_code == second.status_code == 200
    assert set(first.json()) == STATELESS_KEYS | {"session"}
    assert first.json()["session"] == {"history_turns_used": 0, "turn_saved": True}
    assert second.json()["session"] == {"history_turns_used": 1, "turn_saved": True}
    assert fake_ask.calls[0]["history"] == []
    assert fake_ask.calls[1]["history"] == [
        {"question": "How do I connect to the VPN?", "answer": "Use GlobalProtect."}]


def test_history_used_is_capped_at_four_turns(client, enabled, fake_ask):
    token = new_session(client)
    for number in range(6):
        response = ask_in_session(client, token, f"question {number}")

    assert response.json()["session"]["history_turns_used"] == 4
    assert len(fake_ask.calls[-1]["history"]) == 4


def test_follow_up_uses_stored_history_end_to_end(client, enabled, fake_rag):
    llm = FakeLLM(
        structured={RouteDecision: [{"route": "kb"}] * 2, EvidenceGrade: [{"grade": "good"}] * 2},
        text=["Open GlobalProtect.", "How do I connect to the VPN on a Mac?", "Install the macOS client."],
    )
    fake_rag(llm=llm, retriever=FakeRetriever([kb_doc("GlobalProtect has a macOS client.")]))
    token = new_session(client)

    ask_in_session(client, token, "How do I connect to the VPN?")
    response = ask_in_session(client, token, "What about on a Mac?")

    contextualize = llm.calls[3]  # route, grade, answer for turn 1; then the follow-up rewrite
    assert "User: How do I connect to the VPN?" in contextualize["prompt"]
    assert "Assistant: Open GlobalProtect." in contextualize["prompt"]
    assert response.json()["rewritten_query"] == "How do I connect to the VPN on a Mac?"
    assert response.json()["session"] == {"history_turns_used": 1, "turn_saved": True}


def test_busy_session_gets_409_without_running(client, enabled, fake_ask):
    token = new_session(client)
    lease = runtime().manager.acquire(token, "someone-else", owner_ref=legacy_owner_ref())

    response = ask_in_session(client, token)

    assert response.status_code == 409
    assert detail(response)["code"] == "session_busy"
    assert int(response.headers["retry-after"]) >= 1
    assert fake_ask.calls == []
    runtime().manager.finish(lease)
    assert ask_in_session(client, token).status_code == 200


def test_concurrent_requests_on_one_session_run_once(client, enabled, fake_ask):
    token = new_session(client)
    fake_ask.gate = threading.Event()
    results = {}

    first = threading.Thread(target=lambda: results.setdefault("first", ask_in_session(client, token)))
    first.start()
    assert fake_ask.started.wait(5)
    results["second"] = ask_in_session(client, token, "duplicate")
    fake_ask.gate.set()
    first.join(10)

    assert results["second"].status_code == 409
    assert results["first"].status_code == 200
    assert results["first"].json()["session"]["turn_saved"] is True
    assert len(fake_ask.calls) == 1  # the duplicate never ran


@pytest.mark.parametrize("case", ["unknown", "expired", "too_long", "empty"])
def test_unknown_expired_or_malformed_sessions_get_410(client, enabled, fake_ask, case):
    token = new_session(client)
    if case == "unknown":
        token = "never-issued"
    elif case == "expired":
        connection = sqlite3.connect(enabled.session_db_path)
        connection.execute("UPDATE sessions SET idle_expires_at = 0")
        connection.commit()
        connection.close()
    elif case == "too_long":
        token = "x" * 129
    else:
        token = ""

    response = ask_in_session(client, token)

    assert response.status_code == 410
    assert detail(response) == {"code": "session_expired",
                                "message": "This conversation has expired. Please start a new chat."}
    assert fake_ask.calls == []


def test_storage_failure_when_taking_the_lease_gets_503(client, enabled, fake_ask, monkeypatch):
    token = new_session(client)

    def broken(*args, **kwargs):
        raise session_service.SessionStoreUnavailable("down")

    monkeypatch.setattr(runtime().manager, "acquire", broken)
    response = ask_in_session(client, token)

    assert response.status_code == 503 and detail(response)["code"] == "session_unavailable"
    assert fake_ask.calls == []


def test_turn_saved_is_false_when_the_session_is_deleted_mid_answer(client, enabled, fake_ask):
    token = new_session(client)
    fake_ask.during = lambda: runtime().store.delete(token, owner_ref=legacy_owner_ref())

    response = ask_in_session(client, token)

    assert response.status_code == 200
    assert response.json()["answer"] == "Use GlobalProtect."  # the answer is still delivered
    assert response.json()["session"]["turn_saved"] is False


def test_turn_saved_is_false_when_the_save_fails(client, enabled, fake_ask, monkeypatch):
    token = new_session(client)
    new_runtime = session_service.get_session_runtime(enabled)

    def broken(*args, **kwargs):
        raise session_service.SessionStoreUnavailable("down")

    monkeypatch.setattr(new_runtime.store, "save_turn", broken)
    response = ask_in_session(client, token)

    assert response.status_code == 200 and response.json()["session"]["turn_saved"] is False
    assert session_row(enabled, token)["lease_owner"] is None  # lease released anyway
    assert session_row(enabled, token)["turn_count"] == 0


def test_turn_saved_is_false_when_ownership_was_lost(client, enabled, fake_ask):
    token = new_session(client)

    def lose_lease():
        with runtime().manager._lock:
            entry = next(iter(runtime().manager._entries.values()))
            entry.state = session_service.LEASE_LOST

    fake_ask.during = lose_lease
    response = ask_in_session(client, token)

    assert response.json()["session"]["turn_saved"] is False
    assert session_row(enabled, token)["turn_count"] == 0


@pytest.mark.parametrize("failure", ["workflow_error", "empty_answer"])
def test_failures_return_500_audit_the_session_and_release_the_lease(client, enabled, fake_ask, failure):
    token = new_session(client)
    if failure == "workflow_error":
        fake_ask.error = RuntimeError("secret internal detail")
    else:
        fake_ask.answer = ""

    response = ask_in_session(client, token)

    assert response.status_code == 500
    assert detail(response)["message"] == "Unable to process the question."
    assert "secret internal detail" not in response.text
    [record] = get_recent_audits(limit=1)
    assert record["status"] == "failed" and record["session_id"] == session_row(enabled, token)["id"]
    assert session_row(enabled, token)["lease_owner"] is None
    assert session_row(enabled, token)["turn_count"] == 0
    fake_ask.error, fake_ask.answer = None, "ok"
    assert ask_in_session(client, token).status_code == 200  # not left busy


def test_audit_links_session_turns_and_leaves_stateless_rows_unlinked(client, enabled, fake_ask):
    token = new_session(client)
    ask_in_session(client, token)
    client.post("/api/chat", json={"question": "Stateless question"})

    records = {record["question"]: record for record in get_recent_audits(limit=5)}
    assert records["How do I connect to the VPN?"]["session_id"] == session_row(enabled, token)["id"]
    assert records["Stateless question"]["session_id"] is None


def test_audit_failure_does_not_break_the_answer(client, enabled, fake_ask, monkeypatch):
    token = new_session(client)

    def broken(**kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(routes, "write_audit", broken)
    response = ask_in_session(client, token)

    assert response.status_code == 200 and response.json()["session"]["turn_saved"] is True


# ===================================================================
# Stateless compatibility
# ===================================================================

GOLDEN = load_golden()


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_stateless_chat_prompts_match_the_pre_a2_golden(client, fake_rag, name):
    def through_the_endpoint(question):
        body = client.post("/api/chat", json={"question": question}).json()
        assert "session" not in body
        return {"answer": body["answer"], "source_used": body["source_used"], "trace": body["trace"],
                "citations": body["citations"], "current_query": body["rewritten_query"]}

    assert run_scenario(fake_rag, name, through_the_endpoint) == GOLDEN[name]


def test_stateless_chat_touches_no_session_state(client, enabled, fake_ask):
    response = client.post("/api/chat", json={"question": "How do I connect to the VPN?"})

    assert set(response.json()) == STATELESS_KEYS
    assert fake_ask.calls[0]["history"] is None  # ask(question) exactly as before
    assert session_service.peek_session_runtime() is None
    assert not enabled.session_db_path.exists()
    assert get_recent_audits(limit=1)[0]["session_id"] is None


def test_validation_errors_are_unchanged_with_a_session_header(client, enabled, fake_ask):
    token = new_session(client)
    stateless = client.post("/api/chat", json={"question": "a"})
    with_session = client.post("/api/chat", headers={"X-Session-ID": token}, json={"question": "a"})

    assert stateless.status_code == with_session.status_code == 422
    assert stateless.json() == with_session.json()


# ===================================================================
# Token never leaks; no SQLite on the event loop; database isolation
# ===================================================================

def test_token_never_appears_in_logs_audit_errors_or_other_responses(client, enabled, fake_ask, caplog):
    bodies = []
    with caplog.at_level(logging.DEBUG):
        token = new_session(client)  # the only response allowed to contain it
        bodies.append(ask_in_session(client, token).content)
        lease = runtime().manager.acquire(token, "holder", owner_ref=legacy_owner_ref())
        bodies.append(ask_in_session(client, token).content)  # 409
        runtime().manager.finish(lease)
        fake_ask.error = RuntimeError("boom")
        bodies.append(ask_in_session(client, token).content)  # 500
        fake_ask.error = None
        bodies.append(client.delete("/api/sessions", headers={"X-Session-ID": token}).content)
        bodies.append(ask_in_session(client, token).content)  # 410

    hits = find_token(token, records=caplog.records, bodies=bodies,
                      db_paths=[enabled.audit_db_path, enabled.session_db_path])
    assert not hits, hits.locations


def test_no_sqlite_on_the_event_loop_thread(enabled, fake_ask, monkeypatch):
    observed = []
    real_guard = session_service._forbid_event_loop_thread

    def recording_guard():
        try:
            import asyncio

            asyncio.get_running_loop()
            observed.append("event-loop")
        except RuntimeError:
            observed.append("worker")
        real_guard()

    monkeypatch.setattr(session_service, "_forbid_event_loop_thread", recording_guard)
    with TestClient(app.main.app) as client:
        token = new_session(client)
        ask_in_session(client, token)
        client.delete("/api/sessions", headers={"X-Session-ID": token})

    assert observed and set(observed) == {"worker"}


def test_tests_never_touch_the_real_sessions_database(enabled):
    real = BASE_DIR / "data" / "sessions.db"
    assert enabled.session_db_path != real
    assert "pytest" in str(enabled.session_db_path) or "tmp" in str(enabled.session_db_path)


# ===================================================================
# Router lifespan
# ===================================================================

@pytest.mark.parametrize("entry", ["app.main", "app.api.main"])
def test_lifespan_runs_startup_cleanup_on_both_entry_points(enabled, entry, monkeypatch):
    store = session_service.SessionStore.from_settings(enabled)
    old = store.create(owner_ref=legacy_owner_ref())
    connection = sqlite3.connect(enabled.session_db_path)
    connection.execute("UPDATE sessions SET idle_expires_at = 0, absolute_expires_at = 0")
    connection.commit()
    connection.close()
    monkeypatch.setattr(enabled, "sessions_enabled", False)  # cleanup runs even when disabled
    application = app.main.app if entry == "app.main" else api_main_app()

    with TestClient(application):
        pass

    assert count_session_rows(enabled.session_db_path) == 0
    assert store.lookup(old.token, owner_ref=legacy_owner_ref()) is None


def test_lifespan_without_sessions_creates_nothing(settings):
    with TestClient(app.main.app) as client:
        client.get("/api/health")

    assert session_service.peek_session_runtime() is None
    assert not settings.session_db_path.exists()


def test_lifespan_shutdown_is_bounded_and_releases_leases(enabled, fake_ask):
    budget = 0.2 + 0.3 + 0.5
    with TestClient(app.main.app) as client:
        token = new_session(client)
        held = runtime().manager.acquire(token, "still-running", owner_ref=legacy_owner_ref())  # never finished
        started = time.monotonic()
    elapsed = time.monotonic() - started

    assert elapsed <= budget + 0.5
    assert session_row(enabled, token)["lease_owner"] is None  # released by the bulk release
    assert session_service.peek_session_runtime() is None  # cleared for the next app start
    assert not any(t.name.startswith("session-keeper") and t.is_alive() for t in threading.enumerate())
    assert held


class RecordingStreamService:
    """Stands in for the stream service at application shutdown (H2)."""

    def __init__(self, calls, error=None):
        self.calls = calls
        self.error = error

    def close(self, drain_seconds):
        try:
            asyncio.get_running_loop()
            on_loop = True
        except RuntimeError:
            on_loop = False
        self.calls.append(("stream.close", drain_seconds, on_loop))
        if self.error is not None:
            raise self.error
        return {"still_running": 0, "cancelled": 0}


def record_manager_shutdown(calls):
    manager = runtime().manager
    real = manager.shutdown

    def shutdown():
        calls.append(("manager.shutdown",))
        return real()

    manager.shutdown = shutdown


def test_lifespan_closes_the_stream_service_before_the_session_manager(enabled, fake_ask, monkeypatch):
    calls = []
    monkeypatch.setattr(streaming, "_service", RecordingStreamService(calls))
    with TestClient(app.main.app) as client:
        new_session(client)
        record_manager_shutdown(calls)

    assert calls == [
        ("stream.close", routes.STREAM_SHUTDOWN_DRAIN_SECONDS, False),  # off the event loop
        ("manager.shutdown",),
    ]
    assert routes.STREAM_SHUTDOWN_DRAIN_SECONDS == 3.0
    assert streaming.peek_stream_service() is None  # cleared for the next app start


def test_a_failing_stream_close_still_shuts_sessions_down_and_clears_it(enabled, fake_ask, monkeypatch, caplog):
    calls = []
    monkeypatch.setattr(streaming, "_service", RecordingStreamService(calls, RuntimeError("boom")))
    with TestClient(app.main.app) as client:
        token = new_session(client)
        record_manager_shutdown(calls)
        runtime().manager.acquire(token, "still-running", owner_ref=legacy_owner_ref())  # never finished

    assert [call[0] for call in calls] == ["stream.close", "manager.shutdown"]
    assert streaming.peek_stream_service() is None
    assert session_row(enabled, token)["lease_owner"] is None  # the session shutdown still ran
    assert "Stream service shutdown failed." in caplog.text


def test_app_restart_gets_a_fresh_working_stream_service(settings):
    with TestClient(app.main.app):
        first = streaming.get_stream_service()
    assert streaming.peek_stream_service() is None
    with pytest.raises(RuntimeError):
        first._executor.submit(lambda: None)  # shut down with the app

    with TestClient(app.main.app):
        second = streaming.get_stream_service()
        assert second is not first
        assert second._executor.submit(lambda: "ok").result(5) == "ok"
    assert streaming.peek_stream_service() is None


def test_lifespan_without_a_stream_service_creates_none(settings):
    with TestClient(app.main.app):
        pass
    assert streaming.peek_stream_service() is None


class BlockingStreamService:
    """A stream service whose close() blocks until the test releases it (U7a)."""

    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()
        self.closed_with = []

    def close(self, drain_seconds):
        self.entered.set()
        self.release.wait(5)
        self.closed_with.append(drain_seconds)
        return {"still_running": 0, "cancelled": 0}


@pytest.mark.anyio
async def test_a_cancellation_during_stream_close_propagates_and_clears_the_service(monkeypatch):
    """U7a: not swallowed. AnyIO holds a cancellation until the worker thread
    returns, so close() completes, the singleton is cleared, and the
    cancellation reaches the enclosing scope at the next checkpoint."""
    service = BlockingStreamService()
    monkeypatch.setattr(streaming, "_service", service)
    reached = []

    with anyio.CancelScope() as scope:
        async with anyio.create_task_group() as tasks:
            async def cancel_while_closing():
                await anyio.to_thread.run_sync(service.entered.wait, 5)
                scope.cancel()
                service.release.set()

            tasks.start_soon(cancel_while_closing)
            await routes._close_stream_service()
            reached.append("returned")
            await anyio.sleep(0)  # the next checkpoint
            reached.append("past the checkpoint")

    assert scope.cancelled_caught  # propagated to the enclosing scope
    assert reached == ["returned"]  # nothing ran past the checkpoint
    assert service.closed_with == [routes.STREAM_SHUTDOWN_DRAIN_SECONDS]  # close() completed
    assert streaming.peek_stream_service() is None


@pytest.mark.anyio
async def test_a_cancelled_shutdown_raises_out_of_stream_close_and_clears_the_service(monkeypatch):
    """U7a: already cancelled, the cancellation escapes _close_stream_service()
    (``except Exception`` does not catch it) and its ``finally`` still runs."""
    service = BlockingStreamService()
    monkeypatch.setattr(streaming, "_service", service)
    reached = []

    with anyio.CancelScope() as scope:
        scope.cancel()
        await routes._close_stream_service()
        reached.append("returned")

    assert scope.cancelled_caught
    assert reached == []  # raised out of _close_stream_service()
    assert service.closed_with == []  # close() never started
    assert streaming.peek_stream_service() is None


def test_app_restart_in_the_same_process_gets_a_fresh_runtime(enabled, fake_ask):
    with TestClient(app.main.app) as client:
        token = new_session(client)
        assert ask_in_session(client, token).status_code == 200
    with TestClient(app.main.app) as client:
        assert ask_in_session(client, token).status_code == 200  # not refused by the old manager


# ===================================================================
# CORS: same origin only, on both entry points
# ===================================================================

@pytest.mark.parametrize("entry", ["app.main", "app.api.main"])
@pytest.mark.parametrize("path", ["/api/chat", "/api/chat/stream", "/api/sessions"])
def test_foreign_preflight_gets_no_cors_headers(entry, path):
    application = app.main.app if entry == "app.main" else api_main_app()

    response = TestClient(application).options(path, headers={
        "Origin": "https://evil.example",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "x-session-id, content-type",
    })

    assert "access-control-allow-origin" not in response.headers
    assert "access-control-allow-headers" not in response.headers


@pytest.mark.parametrize("entry", ["app.main", "app.api.main"])
def test_foreign_origin_requests_get_no_cors_headers_and_no_middleware(entry, enabled):
    application = app.main.app if entry == "app.main" else api_main_app()
    client = TestClient(application)

    responses = [
        client.post("/api/sessions", headers={**JSON, "Origin": "https://evil.example"}, content=b"{}"),
        client.get("/api/health", headers={"Origin": "https://evil.example"}),
        client.delete("/api/sessions", headers={"Origin": "https://evil.example", "X-Session-ID": "x"}),
    ]

    for response in responses:
        assert "access-control-allow-origin" not in response.headers
    assert not any("CORS" in type(m.cls).__name__ or "CORS" in getattr(m.cls, "__name__", "")
                   for m in application.user_middleware)


def test_router_lifespan_is_registered_on_the_shared_router():
    assert routes.router.lifespan_context is not None
    assert Path(routes.__file__).name == "routes.py"
    assert audit.write_audit  # audit module unchanged in this step
