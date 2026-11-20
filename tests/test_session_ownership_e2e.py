"""Security S1-5 Phase B4: end-to-end session ownership acceptance.

Real signed access cookies, real SQLite persistence, both FastAPI entry
points, through HTTP. Providers are fakes (the suite's network guard and
blocked clients make any real call fail). Every SQL statement the session
store runs is traced, so "no history read / no lease / no mutation" is
checked against what SQLite was actually asked to do.

Not in tests/conftest.py's LEGACY_ACCESS_MODULES: real enforcement.
"""

import sqlite3
import threading
import time

import pytest
from pydantic import SecretStr
from starlette.testclient import TestClient

from app.api import routes
from app.services import access, access_policy, quota, quota_runtime
from app.services import sessions as session_service
from tests.test_access_enforcement import (  # noqa: F401 - enforced and spies are fixtures
    CODE_ALICE,
    CODE_ALICE_NEW,
    CODE_BOB,
    ENTRY_POINTS,
    OTHER_SECRET,
    QUESTION,
    SAME_ORIGIN,
    SECRET,
    cookie_for,
    enforced,
    entries,
    send,
    spies,
)
from tests.test_session_owner import expected_owner
from tests.test_session_ownership_store import LEGACY_SESSIONS_DDL, LEGACY_TURNS_DDL
from tests.test_stream_run import Gate, ScriptedWorkflow, final_state

GUARD = 10.0
HISTORY_SECRET = "Alice-private-history-7f3a"
LEGACY_SECRET = "legacy-private-answer-91c2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def headers(settings, code, **extra):
    return {**SAME_ORIGIN, "cookie": "%s=%s" % cookie_for(settings, code), **extra}


@pytest.fixture
def sql_trace(monkeypatch):
    """Every statement run on a session-store connection, in order."""
    statements = []
    real_open = session_service.SessionStore._open

    def traced_open(self, timeout=None):
        connection = real_open(self, timeout)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(session_service.SessionStore, "_open", traced_open)
    return statements


def reads_history(statements):
    return any("FROM turns" in sql for sql in statements)


def takes_a_lease(statements):
    return any("SET lease_epoch" in sql for sql in statements)


def db_state(settings):
    with sqlite3.connect(settings.session_db_path) as connection:
        connection.row_factory = sqlite3.Row
        sessions = [dict(row) for row in connection.execute(
            "SELECT id, token_hash, lease_owner, lease_epoch, turn_count, access_owner_ref FROM sessions ORDER BY id")]
        turns = [tuple(row) for row in connection.execute("SELECT session_id, seq, question FROM turns ORDER BY 1, 2")]
    return sessions, turns


class HistoryWorkflow(ScriptedWorkflow):
    """ScriptedWorkflow accepting the session history (session streams pass it)."""

    def __call__(self, question, on_chunk, history=None):
        return super().__call__(question, on_chunk)


def session_event(sse_text):
    """The JSON data of the stream's final ``session`` event."""
    import json

    for block in sse_text.split("\n\n"):
        lines = block.splitlines()
        if "event: session" in lines:
            return json.loads("".join(line[len("data: "):] for line in lines if line.startswith("data: ")))
    raise AssertionError("no session event in the stream")


def chat_with_history(spies, client, settings, token, n=2):
    """Give A's session ``n`` saved turns whose questions contain HISTORY_SECRET."""
    for index in range(n):
        body = ('{"question": "%s %d"}' % (HISTORY_SECRET, index)).encode()
        response = send(client, "chat", headers(settings, CODE_ALICE, **{"x-session-id": token}), body=body)
        assert response.status_code == 200 and response.json()["session"]["turn_saved"] is True


# ===========================================================================
# A. Cross-owner isolation (both entry points)
# ===========================================================================

@pytest.mark.parametrize("entry", list(ENTRY_POINTS))
def test_cross_owner_isolation_end_to_end(enforced, spies, stream_app, sql_trace, entry):
    service = stream_app(HistoryWorkflow([("chunk", "x"), ("final", final_state())]))
    with TestClient(ENTRY_POINTS[entry]()) as client:
        token = send(client, "session_create", headers(enforced, CODE_ALICE)).json()["session_id"]
        chat_with_history(spies, client, enforced, token)
        before = db_state(enforced)
        assert len(before[1]) == 2 and before[0][0]["access_owner_ref"] == expected_owner(enforced, "alice")
        asks_before, engine = len(spies.asks), quota_runtime.peek_quota_engine()

        sql_trace.clear()
        bob = headers(enforced, CODE_BOB, **{"x-session-id": token})
        b_chat = send(client, "chat", bob)
        b_stream = send(client, "stream", bob)
        b_delete = client.delete("/api/sessions", headers=bob)
        unknown = send(client, "chat", headers(enforced, CODE_BOB, **{"x-session-id": "Z" * 43}))
        b_statements = list(sql_trace)

        # Same response as an unknown session: status, body and caching headers.
        assert b_chat.status_code == b_stream.status_code == unknown.status_code == 410
        assert b_chat.json() == b_stream.json() == unknown.json()
        assert b_chat.headers.get("cache-control") == unknown.headers.get("cache-control")
        assert b_delete.status_code == 204 and b_delete.content == b""
        for response in (b_chat, b_stream, b_delete):
            assert HISTORY_SECRET not in response.text
        # What SQLite was asked to do for B: owner-qualified look-ups only.
        assert not reads_history(b_statements) and not takes_a_lease(b_statements)
        assert not any("UPDATE sessions" in sql for sql in b_statements)
        assert all("access_owner_ref" in sql for sql in b_statements if "token_hash" in sql)
        assert db_state(enforced) == before  # no lease, epoch, turn or row change
        assert len(spies.asks) == asks_before and service.started == []  # no provider, no worker
        assert engine.usage("bob")["daily"]["chat"] == 0  # never admitted

        # A carries on, then deletes.
        again = send(client, "chat", headers(enforced, CODE_ALICE, **{"x-session-id": token}))
        assert again.status_code == 200 and again.json()["session"]["history_turns_used"] == 2
        assert client.delete("/api/sessions", headers=headers(enforced, CODE_ALICE, **{"x-session-id": token})).status_code == 204
        assert db_state(enforced) == ([], [])


# ===========================================================================
# B. Access-code lifecycle against existing sessions
# ===========================================================================

def _https(enforced, monkeypatch):
    monkeypatch.setattr(enforced, "public_origins", "https://testserver")
    return {"origin": "https://testserver", "sec-fetch-site": "same-origin"}


@pytest.mark.parametrize("entry", list(ENTRY_POINTS))
def test_re_redeeming_the_same_code_keeps_its_sessions(enforced, spies, monkeypatch, entry):
    origin = _https(enforced, monkeypatch)
    app = ENTRY_POINTS[entry]()
    with TestClient(app, base_url="https://testserver") as first, TestClient(app, base_url="https://testserver") as second:
        for client in (first, second):
            assert client.post("/api/access", json={"code": CODE_ALICE}, headers=origin).status_code == 200
        token = first.post("/api/sessions", json={}, headers=origin).json()["session_id"]
        response = second.post("/api/chat", json={"question": QUESTION}, headers={**origin, "x-session-id": token})
        assert response.status_code == 200 and response.json()["session"]["turn_saved"] is True


@pytest.mark.parametrize("entry", list(ENTRY_POINTS))
@pytest.mark.parametrize("change", ["revoked", "expired"])
def test_a_revoked_or_expired_code_is_refused_before_any_session_access(enforced, spies, monkeypatch, sql_trace, entry, change):
    client = TestClient(ENTRY_POINTS[entry]())
    alice = headers(enforced, CODE_ALICE)
    token = send(client, "session_create", alice).json()["session_id"]
    before = db_state(enforced)
    if change == "revoked":
        monkeypatch.setattr(enforced, "access_code_hashes", entries(("bob", CODE_BOB)))
    else:
        monkeypatch.setattr(enforced, "access_code_hashes",
                            f"alice:{access.hash_code(CODE_ALICE)}:2000-01-01,bob:{access.hash_code(CODE_BOB)}")
    sql_trace.clear()
    for route in ("chat", "stream"):
        response = send(client, route, {**alice, "x-session-id": token})
        assert response.status_code == 401 and response.json()["detail"]["code"] == "access_required"
    assert client.delete("/api/sessions", headers={**alice, "x-session-id": token}).status_code == 401
    assert send(client, "session_create", alice).status_code == 401
    assert sql_trace == [] and db_state(enforced) == before  # not one session statement


@pytest.mark.parametrize("entry", list(ENTRY_POINTS))
def test_a_replaced_code_gets_a_new_owner_and_never_the_old_sessions(enforced, spies, monkeypatch, sql_trace, entry):
    client = TestClient(ENTRY_POINTS[entry]())
    old_cookie = headers(enforced, CODE_ALICE)
    token = send(client, "session_create", old_cookie).json()["session_id"]
    old_owner = expected_owner(enforced, "alice")
    monkeypatch.setattr(enforced, "access_code_hashes", entries(("alice", CODE_ALICE_NEW), ("bob", CODE_BOB)))
    new_cookie = headers(enforced, CODE_ALICE_NEW)
    assert expected_owner(enforced, "alice") != old_owner

    assert send(client, "chat", {**old_cookie, "x-session-id": token}).status_code == 401
    sql_trace.clear()
    refused = send(client, "chat", {**new_cookie, "x-session-id": token})
    assert refused.status_code == 410 and not takes_a_lease(sql_trace) and not reads_history(sql_trace)
    assert client.delete("/api/sessions", headers={**new_cookie, "x-session-id": token}).status_code == 204
    assert [row["access_owner_ref"] for row in db_state(enforced)[0]] == [old_owner]  # untouched
    fresh = send(client, "session_create", new_cookie).json()["session_id"]
    assert send(client, "chat", {**new_cookie, "x-session-id": fresh}).status_code == 200


@pytest.mark.parametrize("entry", list(ENTRY_POINTS))
def test_rotating_the_cookie_secret_keeps_session_ownership(enforced, spies, monkeypatch, entry):
    client = TestClient(ENTRY_POINTS[entry]())
    signed_before = headers(enforced, CODE_ALICE)
    token = send(client, "session_create", signed_before).json()["session_id"]
    monkeypatch.setattr(enforced, "access_cookie_secret", SecretStr(OTHER_SECRET))
    monkeypatch.setattr(enforced, "access_cookie_previous_secrets", SecretStr(SECRET))
    signed_after = headers(enforced, CODE_ALICE)
    assert signed_after["cookie"] != signed_before["cookie"]
    for cookie in (signed_before, signed_after):
        assert send(client, "chat", {**cookie, "x-session-id": token}).status_code == 200


FORGED_OWNER_HEADERS = ["x-owner-ref", "x-access-owner-ref", "x-session-owner", "x-access-owner", "owner-ref"]


@pytest.mark.parametrize("entry", list(ENTRY_POINTS))
def test_forged_owner_fields_never_influence_ownership(enforced, spies, monkeypatch, entry):
    client = TestClient(ENTRY_POINTS[entry]())
    token = send(client, "session_create", headers(enforced, CODE_ALICE)).json()["session_id"]
    alice_owner = expected_owner(enforced, "alice")
    forged = headers(enforced, CODE_BOB, **{"x-session-id": token, **{name: alice_owner for name in FORGED_OWNER_HEADERS}})
    forged["cookie"] += f"; owner_ref={alice_owner}; access_owner_ref={alice_owner}"
    body = ('{"question": "%s", "owner_ref": "%s", "access_owner_ref": "%s"}' % (QUESTION, alice_owner, alice_owner)).encode()
    for route in ("chat", "stream"):
        assert send(client, route, forged, body=body).status_code == 410
    response = client.post(f"/api/chat?owner_ref={alice_owner}", content=body,
                           headers={**forged, "content-type": "application/json"})
    assert response.status_code == 410
    client.delete("/api/sessions", headers=forged)
    # A session created with forged fields belongs to the cookie's code, not the field's.
    created = client.post("/api/sessions", content=b"{}", headers={**forged, "content-type": "application/json"})
    assert created.status_code == 201
    owners = sorted(row["access_owner_ref"] for row in db_state(enforced)[0])
    assert owners == sorted([alice_owner, expected_owner(enforced, "bob")])
    assert send(client, "chat", headers(enforced, CODE_ALICE, **{"x-session-id": token})).status_code == 200


# ===========================================================================
# C. A genuine version-0 database served through HTTP
# ===========================================================================

def make_legacy_db(settings, *, expired=False):
    now = time.time()
    idle = now - 60 if expired else now + 1800
    settings.session_db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(settings.session_db_path) as connection:
        connection.execute(LEGACY_SESSIONS_DDL)
        connection.execute(LEGACY_TURNS_DDL)
        for session_id, token in (("L1", "legacy-token-one"), ("L2", "legacy-token-two")):
            connection.execute(
                "INSERT INTO sessions (id, token_hash, created_at, last_activity_at, idle_expires_at, absolute_expires_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, session_service.hash_token(token), now - 100, now - 100, idle, now + 28000))
            connection.execute(
                "INSERT INTO turns VALUES (?, 1, 1, 'r', ?, ?, 'private_kb', ?)",
                (session_id, f"legacy question {LEGACY_SECRET}", f"legacy answer {LEGACY_SECRET}", now - 100))
    assert sqlite3.connect(settings.session_db_path).execute("PRAGMA user_version").fetchone()[0] == 0


@pytest.mark.parametrize("entry", list(ENTRY_POINTS))
def test_a_legacy_database_is_migrated_and_its_sessions_are_refused_to_everyone(enforced, spies, stream_app, sql_trace, entry):
    service = stream_app(HistoryWorkflow([("final", final_state())]))
    make_legacy_db(enforced)
    with TestClient(ENTRY_POINTS[entry]()) as client:
        for code in (CODE_ALICE, CODE_BOB):
            for token in ("legacy-token-one", "legacy-token-two"):
                sql_trace.clear()
                legacy = headers(enforced, code, **{"x-session-id": token})
                for route in ("chat", "stream"):
                    response = send(client, route, legacy)
                    assert response.status_code == 410 and LEGACY_SECRET not in response.text
                assert client.delete("/api/sessions", headers=legacy).status_code == 204
                assert not reads_history(sql_trace) and not takes_a_lease(sql_trace)
    sessions, turns = db_state(enforced)
    assert [(row["id"], row["access_owner_ref"], row["lease_owner"]) for row in sessions] == [("L1", None, None), ("L2", None, None)]
    assert len(turns) == 2 and spies.asks == [] and service.started == []
    with sqlite3.connect(enforced.session_db_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            connection.execute("UPDATE sessions SET access_owner_ref = ? WHERE id = 'L1'", (expected_owner(enforced, "alice"),))


def test_expired_legacy_sessions_are_cleaned_up_normally(enforced, spies):
    make_legacy_db(enforced, expired=True)
    client = TestClient(ENTRY_POINTS["app.main"]())
    assert send(client, "session_create", headers(enforced, CODE_ALICE)).status_code == 201  # runs a cleanup batch
    sessions, turns = db_state(enforced)
    assert [row["access_owner_ref"] for row in sessions] == [expected_owner(enforced, "alice")]
    assert turns == []  # the legacy turns went with their sessions (cascade)


def test_the_migration_is_idempotent_across_restarts(enforced, spies):
    make_legacy_db(enforced)
    alice = headers(enforced, CODE_ALICE)
    for _ in range(3):  # each round is a "new process": a new runtime and store
        session_service.clear_session_runtime()
        assert send(TestClient(ENTRY_POINTS["app.main"]()), "session_create", alice).status_code == 201
    with sqlite3.connect(enforced.session_db_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        columns = [row[1] for row in connection.execute("PRAGMA table_info(sessions)")]
        triggers = connection.execute("SELECT count(*) FROM sqlite_master WHERE type = 'trigger'").fetchone()[0]
    assert columns.count("access_owner_ref") == 1 and triggers == 1


@pytest.mark.parametrize("entry", list(ENTRY_POINTS))
def test_a_future_schema_version_is_refused_through_http(enforced, spies, entry):
    make_legacy_db(enforced)
    with sqlite3.connect(enforced.session_db_path) as connection:
        connection.execute("PRAGMA user_version = 7")
    client = TestClient(ENTRY_POINTS[entry]())
    alice = headers(enforced, CODE_ALICE, **{"x-session-id": "legacy-token-one"})
    for response in (send(client, "session_create", alice), send(client, "chat", alice),
                     client.delete("/api/sessions", headers=alice)):
        assert response.status_code == 503 and response.json()["detail"]["code"] == "session_unavailable"
        assert LEGACY_SECRET not in response.text
    with sqlite3.connect(enforced.session_db_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 7
        assert "access_owner_ref" not in [row[1] for row in connection.execute("PRAGMA table_info(sessions)")]


# ===========================================================================
# D. Concurrency through HTTP
# ===========================================================================

def _stream_in_background(client, request_headers, results):
    def run():
        results["stream"] = send(client, "stream", request_headers)

    thread = threading.Thread(target=run)
    thread.start()
    return thread


@pytest.mark.parametrize("entry", list(ENTRY_POINTS))
def test_b_never_sees_or_takes_a_lease_held_by_a(enforced, spies, stream_app, sql_trace, entry):
    gate = Gate()
    service = stream_app(HistoryWorkflow([("chunk", "a"), ("gate", gate), ("final", final_state())]))
    with TestClient(ENTRY_POINTS[entry]()) as client:
        token = send(client, "session_create", headers(enforced, CODE_ALICE)).json()["session_id"]
        alice = headers(enforced, CODE_ALICE, **{"x-session-id": token})
        results = {}
        thread = _stream_in_background(client, alice, results)
        assert gate.reached.wait(GUARD)  # A's stream holds the lease
        sql_trace.clear()
        bob = headers(enforced, CODE_BOB, **{"x-session-id": token})
        b_responses = [send(client, "chat", bob), send(client, "stream", bob)]
        b_statements = list(sql_trace)
        a_second = send(client, "chat", alice)
        gate.release.set()
        thread.join(GUARD)
    # B gets "unknown session" (410), never "busy" (409), which would reveal the session exists.
    assert [r.status_code for r in b_responses] == [410, 410]
    assert a_second.status_code == 409  # the owner sees its own lease
    assert not takes_a_lease(b_statements) and not reads_history(b_statements)
    assert results["stream"].status_code == 200 and "event: done" in results["stream"].text
    assert len(service.started) == 1
    assert db_state(enforced)[0][0]["lease_owner"] is None and db_state(enforced)[0][0]["turn_count"] == 1


@pytest.mark.parametrize("entry", list(ENTRY_POINTS))
def test_deleting_during_an_active_turn_never_resurrects_the_session(enforced, spies, stream_app, entry):
    gate = Gate()
    stream_app(HistoryWorkflow([("chunk", "a"), ("gate", gate), ("final", final_state())]))
    with TestClient(ENTRY_POINTS[entry]()) as client:
        token = send(client, "session_create", headers(enforced, CODE_ALICE)).json()["session_id"]
        alice = headers(enforced, CODE_ALICE, **{"x-session-id": token})
        results = {}
        thread = _stream_in_background(client, alice, results)
        assert gate.reached.wait(GUARD)
        # B's delete during A's turn changes nothing; A's own delete wins.
        assert client.delete("/api/sessions", headers=headers(enforced, CODE_BOB, **{"x-session-id": token})).status_code == 204
        assert len(db_state(enforced)[0]) == 1
        assert client.delete("/api/sessions", headers=alice).status_code == 204
        gate.release.set()
        thread.join(GUARD)
    assert results["stream"].status_code == 200
    assert session_event(results["stream"].text)["turn_saved"] is False  # the late save was fenced
    assert db_state(enforced) == ([], [])  # nothing came back
    assert send(TestClient(ENTRY_POINTS[entry]()), "chat", alice).status_code == 410


@pytest.mark.parametrize("entry", list(ENTRY_POINTS))
def test_a_cancelled_owner_stream_releases_its_lease_and_b_still_cannot_use_it(enforced, spies, stream_app, entry):
    gate = Gate()
    stream_app(HistoryWorkflow([("chunk", "a"), ("gate", gate), ("final", final_state())]))
    with TestClient(ENTRY_POINTS[entry]()) as client:
        token = send(client, "session_create", headers(enforced, CODE_ALICE)).json()["session_id"]
        alice = headers(enforced, CODE_ALICE, **{"x-session-id": token})
        with client.stream("POST", "/api/chat/stream", content=b'{"question": "VPN?"}',
                           headers={**alice, "content-type": "application/json"}) as response:
            assert response.status_code == 200
            assert gate.reached.wait(GUARD)
        # The response was closed early (a cancelled client); let the worker finish and release.
        gate.release.set()
        deadline = time.monotonic() + GUARD
        while db_state(enforced)[0][0]["lease_owner"] is not None:
            assert time.monotonic() < deadline, "the cancelled stream's lease was never released"
            time.sleep(0.02)
        assert send(client, "chat", headers(enforced, CODE_BOB, **{"x-session-id": token})).status_code == 410
        assert send(client, "chat", alice).status_code == 200


def _parallel_http_creates(app, owner_headers):
    barrier, outcomes, lock = threading.Barrier(len(owner_headers)), [], threading.Lock()

    def run(request_headers):
        client = TestClient(app)
        barrier.wait(GUARD)
        response = send(client, "session_create", request_headers)
        with lock:
            outcomes.append((request_headers["cookie"], response.status_code,
                             response.json().get("detail", {}).get("code") if response.status_code != 201 else None))

    threads = [threading.Thread(target=run, args=(h,)) for h in owner_headers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(GUARD * 3)
    assert len(outcomes) == len(owner_headers)
    return outcomes


@pytest.mark.parametrize("entry", list(ENTRY_POINTS))
def test_50_parallel_http_creates_by_one_owner_admit_exactly_the_cap(enforced, spies, monkeypatch, entry):
    monkeypatch.setattr(enforced, "session_db_busy_timeout_seconds", 15.0)
    alice = headers(enforced, CODE_ALICE)
    outcomes = _parallel_http_creates(ENTRY_POINTS[entry](), [alice] * 50)
    statuses = sorted(status for _, status, _ in outcomes)
    assert statuses.count(201) == enforced.session_max_active_per_owner == 20
    assert statuses.count(429) == 30
    assert {code for _, status, code in outcomes if status == 429} == {"session_limit"}
    assert send(TestClient(ENTRY_POINTS[entry]()), "session_create", headers(enforced, CODE_BOB)).status_code == 201


def test_the_global_cap_still_applies_across_owners(enforced, spies, monkeypatch):
    monkeypatch.setattr(enforced, "max_active_sessions", 25)
    monkeypatch.setattr(enforced, "session_db_busy_timeout_seconds", 15.0)
    app = ENTRY_POINTS["app.main"]()
    outcomes = _parallel_http_creates(app, [headers(enforced, CODE_ALICE)] * 22 + [headers(enforced, CODE_BOB)] * 10)
    created = [cookie for cookie, status, _ in outcomes if status == 201]
    assert len(created) == 25  # the global cap
    assert sum(1 for _, status, code in outcomes if status == 503 and code == "session_unavailable") + \
        sum(1 for _, status, code in outcomes if status == 429) == 7
    with sqlite3.connect(enforced.session_db_path) as connection:
        per_owner = dict(connection.execute(
            "SELECT access_owner_ref, count(*) FROM sessions GROUP BY access_owner_ref").fetchall())
    assert per_owner.get(expected_owner(enforced, "alice"), 0) <= 20 and sum(per_owner.values()) == 25


# ===========================================================================
# E. Fail-closed conditions through HTTP
# ===========================================================================

def _with_decision(monkeypatch, transform):
    real = access_policy.evaluate
    monkeypatch.setattr(access_policy, "evaluate", lambda request, settings: transform(real(request, settings)))


def _decision(decision, **changes):
    fields = dict(outcome=decision.outcome, reason=decision.reason, grant=decision.grant,
                  code_ref=decision.code_ref, owner_ref=decision.owner_ref)
    fields.update(changes)
    return access_policy.AccessDecision(**fields)


FAIL_CLOSED_DECISIONS = {
    "owner-missing": lambda d: _decision(d, owner_ref=None),
    "owner-malformed": lambda d: _decision(d, owner_ref="Z" * 64),
    "owner-short": lambda d: _decision(d, owner_ref=d.owner_ref[:-2]),
    "anonymous-identity": lambda d: _decision(d, grant=access.AccessGrant(code_id=quota.ANONYMOUS, expires_at=2**31 - 1)),
}


@pytest.mark.parametrize("entry", list(ENTRY_POINTS))
@pytest.mark.parametrize("case", list(FAIL_CLOSED_DECISIONS))
def test_a_bad_decision_fails_closed_before_any_session_sql(enforced, spies, monkeypatch, sql_trace, entry, case):
    client = TestClient(ENTRY_POINTS[entry]())
    alice = headers(enforced, CODE_ALICE)
    token = send(client, "session_create", alice).json()["session_id"]
    before = db_state(enforced)
    _with_decision(monkeypatch, FAIL_CLOSED_DECISIONS[case])
    sql_trace.clear()
    with_token = {**alice, "x-session-id": token}
    for response in (send(client, "session_create", alice), send(client, "chat", with_token),
                     send(client, "stream", with_token), client.delete("/api/sessions", headers=with_token)):
        assert response.status_code == 503 and response.json()["detail"]["code"] == "access_unavailable"
    assert sql_trace == [] and db_state(enforced) == before


@pytest.mark.parametrize("cookie", ["forged", "expired", "none"])
def test_bad_cookies_are_refused_before_any_session_sql(enforced, spies, sql_trace, cookie):
    client = TestClient(ENTRY_POINTS["app.main"]())
    token = send(client, "session_create", headers(enforced, CODE_ALICE)).json()["session_id"]
    name, value = cookie_for(enforced, CODE_ALICE)
    if cookie == "forged":
        value = value[:-2] + ("AA" if not value.endswith("AA") else "BB")
    elif cookie == "expired":
        config = access.AccessConfig.from_settings(enforced)
        _, value = cookie_for(enforced, CODE_ALICE, now=access.wall_clock() - config.max_age_seconds - 120)
    request_headers = {**SAME_ORIGIN, "x-session-id": token}
    if cookie != "none":
        request_headers["cookie"] = f"{name}={value}"
    sql_trace.clear()
    for response in (send(client, "chat", request_headers), send(client, "stream", request_headers),
                     client.delete("/api/sessions", headers=request_headers), send(client, "session_create", request_headers)):
        assert response.status_code == 401
    assert sql_trace == []


def test_a_missing_immutability_trigger_fails_closed(enforced, spies):
    client = TestClient(ENTRY_POINTS["app.main"]())
    alice = headers(enforced, CODE_ALICE)
    token = send(client, "session_create", alice).json()["session_id"]
    with sqlite3.connect(enforced.session_db_path) as connection:
        connection.execute("DROP TRIGGER sessions_owner_immutable")
    session_service.clear_session_runtime()  # a restarted process
    for response in (send(client, "session_create", alice), send(client, "chat", {**alice, "x-session-id": token})):
        assert response.status_code == 503 and response.json()["detail"]["code"] == "session_unavailable"
    with sqlite3.connect(enforced.session_db_path) as connection:  # never silently repaired
        assert connection.execute("SELECT count(*) FROM sqlite_master WHERE type = 'trigger'").fetchone()[0] == 0
    assert spies.asks == []


def test_a_sqlite_lock_timeout_is_a_bounded_controlled_503(enforced, spies, monkeypatch):
    monkeypatch.setattr(enforced, "session_db_busy_timeout_seconds", 0.3)
    client = TestClient(ENTRY_POINTS["app.main"]())
    alice = headers(enforced, CODE_ALICE)
    token = send(client, "session_create", alice).json()["session_id"]
    session_service.clear_session_runtime()  # pick up the short busy timeout
    holder = sqlite3.connect(enforced.session_db_path, isolation_level=None, timeout=5)
    holder.execute("BEGIN EXCLUSIVE")
    try:
        started = time.monotonic()
        responses = [send(client, "chat", {**alice, "x-session-id": token}), send(client, "session_create", alice)]
        elapsed = time.monotonic() - started
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    for response in responses:
        assert response.status_code == 503 and response.json()["detail"]["code"] == "session_unavailable"
        assert "locked" not in response.text.lower() and "sqlite" not in response.text.lower()
    assert elapsed < 0.3 * 2 * 4 + 2.0  # each operation bounded by its busy timeout (plus test overhead)
    assert spies.asks == [] and len(db_state(enforced)[0]) == 1


@pytest.mark.parametrize("token, status", [("", 400), (" ", 400), ("x" * 129, 400), ("never-issued", 204)])
def test_invalid_session_tokens_on_delete(enforced, spies, token, status):
    client = TestClient(ENTRY_POINTS["app.main"]())
    response = client.delete("/api/sessions", headers=headers(enforced, CODE_ALICE, **{"x-session-id": token}))
    assert response.status_code == status


@pytest.mark.parametrize("token", ["", " ", "x" * 129, "never-issued"])
def test_invalid_session_tokens_on_chat_are_410(enforced, spies, sql_trace, token):
    client = TestClient(ENTRY_POINTS["app.main"]())
    response = send(client, "chat", headers(enforced, CODE_ALICE, **{"x-session-id": token}))
    assert response.status_code == 410 and spies.asks == []
    assert not takes_a_lease(sql_trace)


# ===========================================================================
# F. SQLite defence in depth
# ===========================================================================

def test_sql_acquire_rejects_a_mismatched_owner_on_its_own(enforced, spies):
    client = TestClient(ENTRY_POINTS["app.main"]())
    send(client, "session_create", headers(enforced, CODE_ALICE))
    (session_id,) = [row["id"] for row in db_state(enforced)[0]]
    store = session_service.SessionStore.from_settings(enforced)
    connection = store._open()
    try:
        params = {"rid": "r-direct", "inst": "i", "gen": 1, "ttl": 30.0, "turn_max": 60.0, "sid": session_id}
        connection.execute("BEGIN IMMEDIATE")
        wrong = connection.execute(session_service.SQL_ACQUIRE, {**params, "owner": expected_owner(enforced, "bob")}).fetchall()
        connection.execute("ROLLBACK")
        assert wrong == []  # the internal id alone is not enough
        assert db_state(enforced)[0][0]["lease_owner"] is None
        connection.execute("BEGIN IMMEDIATE")
        right = connection.execute(session_service.SQL_ACQUIRE, {**params, "owner": expected_owner(enforced, "alice")}).fetchall()
        connection.execute("ROLLBACK")
        assert len(right) == 1
    finally:
        connection.close()


def test_sql_delete_and_lookup_reject_a_mismatched_owner_on_their_own(enforced, spies):
    client = TestClient(ENTRY_POINTS["app.main"]())
    token = send(client, "session_create", headers(enforced, CODE_ALICE)).json()["session_id"]
    store = session_service.SessionStore.from_settings(enforced)
    connection = store._open()
    try:
        bob = expected_owner(enforced, "bob")
        token_hash = session_service.hash_token(token)
        assert connection.execute(session_service.SQL_LOOKUP, {"token_hash": token_hash, "owner": bob}).fetchall() == []
        assert connection.execute(session_service.SQL_DELETE_SESSION, {"token_hash": token_hash, "owner": bob}).rowcount == 0
    finally:
        connection.close()
    assert len(db_state(enforced)[0]) == 1


def test_a_failed_migration_step_rolls_back_and_a_retry_succeeds(enforced, spies, monkeypatch):
    make_legacy_db(enforced)

    class FailingAlter(sqlite3.Connection):
        def execute(self, sql, *args):
            if sql.startswith("ALTER TABLE sessions ADD COLUMN"):
                super().execute(sql, *args)  # the change happens, then the transaction fails
                raise sqlite3.OperationalError("injected failure after the ALTER")
            return super().execute(sql, *args)

    failing = session_service.SessionStore.from_settings(
        enforced, connect=lambda *a, **k: sqlite3.connect(*a, factory=FailingAlter, **k))
    with pytest.raises(session_service.SessionStoreUnavailable):
        failing.create(owner_ref=expected_owner(enforced, "alice"))
    with sqlite3.connect(enforced.session_db_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert "access_owner_ref" not in [row[1] for row in connection.execute("PRAGMA table_info(sessions)")]
        assert connection.execute("SELECT count(*) FROM sqlite_master WHERE type = 'trigger'").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM turns").fetchone()[0] == 2  # data intact
    assert send(TestClient(ENTRY_POINTS["app.main"]()), "session_create", headers(enforced, CODE_ALICE)).status_code == 201
    with sqlite3.connect(enforced.session_db_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1


def test_wal_initialisation_elapsed_time_is_bounded_under_a_real_blocking_lock(enforced, tmp_path):
    path = tmp_path / "blocked.db"
    with sqlite3.connect(path) as connection:
        connection.execute(LEGACY_SESSIONS_DDL)
        connection.execute(LEGACY_TURNS_DDL)
    holder = sqlite3.connect(path, isolation_level=None)
    holder.execute("BEGIN EXCLUSIVE")  # SQLite's own busy handler blocks inside each attempt
    store = session_service.SessionStore(path, idle_timeout_seconds=1800, absolute_timeout_seconds=28800,
                                         max_active_sessions=10, busy_timeout_seconds=0.5)
    try:
        started = time.monotonic()
        with pytest.raises(session_service.SessionStoreUnavailable):
            store.create(owner_ref=expected_owner(enforced, "alice"))
        elapsed = time.monotonic() - started
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    # cleanup (one schema attempt) + create (another): each bounded by the 0.5 s deadline.
    assert elapsed < 0.5 * 2 + 0.75, elapsed
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] != "wal"
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
