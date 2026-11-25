"""Security S1-5 Phase B1: the session owner identity (owner_ref).

owner_ref is derived only from a verified access grant (the code id and the
current configured code's fingerprint, K2: no secret), carried on the
server-side AccessDecision, and read by the fail-closed routes._session_owner
helper. B1 does not use it for sessions yet (that is B3).

Not in tests/conftest.py's LEGACY_ACCESS_MODULES: real enforcement.
"""

import ast
import hashlib
import inspect
import json
import logging
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import SecretStr
from starlette.testclient import TestClient

from app.api import routes
from app.services import access, access_policy, quota, quota_runtime
from app.services import sessions as session_service
from tests.test_stream_run import ScriptedWorkflow, final_state
from tests.test_access_enforcement import (  # noqa: F401 - enforced and spies are fixtures
    CODE_ALICE,
    CODE_ALICE_NEW,
    CODE_BOB,
    ENTRY_POINTS,
    OTHER_SECRET,
    SAME_ORIGIN,
    SECRET,
    cookie_for,
    enforced,
    entries,
    fake_request,
    send,
    spies,
)


def code_for(settings, code_id):
    config = access.AccessConfig.from_settings(settings)
    return next(item for item in config.codes if item.code_id == code_id)


def expected_owner(settings, code_id):
    return access_policy.derive_owner_ref(code_id, access.code_fingerprint(code_for(settings, code_id)))


def decide(settings, cookie_value, name=None):
    config = access.AccessConfig.from_settings(settings)
    request = fake_request([("cookie", f"{name or config.cookie_name}={cookie_value}")])
    return access_policy.evaluate(request, settings)


def cookie_value(settings, code, **kwargs):
    return cookie_for(settings, code, **kwargs)[1]


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------

FP_A = "A" * 22
FP_B = "B" * 22


def test_derivation_is_deterministic_lowercase_hex_and_domain_separated():
    owner = access_policy.derive_owner_ref("alice", FP_A)
    assert owner == access_policy.derive_owner_ref("alice", FP_A)
    assert access_policy.OWNER_REF_PATTERN.fullmatch(owner) and access_policy.is_valid_owner_ref(owner)
    expected = hashlib.sha256(
        b"bluepeak-session-owner-v1\x00" + (5).to_bytes(4, "big") + b"alice" + (22).to_bytes(4, "big") + FP_A.encode()
    ).hexdigest()
    assert owner == expected
    # Not the plain hash of the fields, with or without a separator.
    for undomained in (b"alice" + FP_A.encode(), b"alice:" + FP_A.encode(), b"alice\x00" + FP_A.encode()):
        assert owner != hashlib.sha256(undomained).hexdigest()


def test_every_input_changes_the_owner():
    owners = {
        access_policy.derive_owner_ref(code_id, fingerprint)
        for code_id in ("alice", "bob", "alice2", "a")
        for fingerprint in (FP_A, FP_B, "A" * 21 + "B")
    }
    assert len(owners) == 12


def test_the_encoding_is_unambiguous():
    # Length prefixes: moving characters between the fields changes the bytes.
    assert access_policy._length_prefixed("ab") + access_policy._length_prefixed("c") != (
        access_policy._length_prefixed("a") + access_policy._length_prefixed("bc")
    )
    assert access_policy.derive_owner_ref("ab", FP_A) != access_policy.derive_owner_ref("a", FP_A)


@pytest.mark.parametrize("code_id, fingerprint", [
    ("", FP_A), ("Alice", FP_A), ("al ice", FP_A), ("a" * 33, FP_A), (None, FP_A), (quota.ANONYMOUS, FP_A),
    ("alice", ""), ("alice", "A" * 21), ("alice", "A" * 23), ("alice", "A" * 21 + "="), ("alice", None),
])
def test_derivation_rejects_invalid_inputs(code_id, fingerprint):
    with pytest.raises(ValueError):
        access_policy.derive_owner_ref(code_id, fingerprint)


@pytest.mark.parametrize("value", [None, "", "a" * 63, "a" * 65, "A" * 64, "g" * 64, 123, b"a" * 64, " " + "a" * 63])
def test_malformed_owner_refs_are_invalid(value):
    assert not access_policy.is_valid_owner_ref(value)


# ---------------------------------------------------------------------------
# From a verified grant only, and the code lifecycle
# ---------------------------------------------------------------------------

def test_a_verified_grant_carries_the_owner_of_its_configured_code(enforced):
    decision = decide(enforced, cookie_value(enforced, CODE_ALICE))
    assert decision.granted and decision.owner_ref == expected_owner(enforced, "alice")


def test_different_codes_have_different_owners(enforced):
    alice = decide(enforced, cookie_value(enforced, CODE_ALICE)).owner_ref
    bob = decide(enforced, cookie_value(enforced, CODE_BOB)).owner_ref
    assert alice != bob and access_policy.is_valid_owner_ref(alice) and access_policy.is_valid_owner_ref(bob)


def test_redeeming_the_same_code_again_keeps_the_owner(enforced):
    now = access.wall_clock()
    first = decide(enforced, cookie_value(enforced, CODE_ALICE, now=now - 3600))
    second = decide(enforced, cookie_value(enforced, CODE_ALICE, now=now))
    assert first.granted and second.granted and first.owner_ref == second.owner_ref


def test_replacing_a_code_under_the_same_id_changes_the_owner(enforced, monkeypatch):
    old_owner = decide(enforced, cookie_value(enforced, CODE_ALICE)).owner_ref
    old_cookie = cookie_value(enforced, CODE_ALICE)
    monkeypatch.setattr(enforced, "access_code_hashes", entries(("alice", CODE_ALICE_NEW), ("bob", CODE_BOB)))
    stale = decide(enforced, old_cookie)
    assert stale.outcome == access_policy.UNAUTHENTICATED and stale.owner_ref is None
    new_owner = decide(enforced, cookie_value(enforced, CODE_ALICE_NEW)).owner_ref
    assert access_policy.is_valid_owner_ref(new_owner) and new_owner != old_owner


def test_rotating_the_cookie_secret_does_not_change_the_owner(enforced, monkeypatch):
    issued_before = cookie_value(enforced, CODE_ALICE)
    before = decide(enforced, issued_before).owner_ref
    monkeypatch.setattr(enforced, "access_cookie_secret", SecretStr(OTHER_SECRET))
    monkeypatch.setattr(enforced, "access_cookie_previous_secrets", SecretStr(SECRET))
    old_cookie = decide(enforced, issued_before)  # verified by the previous secret
    new_cookie = decide(enforced, cookie_value(enforced, CODE_ALICE))  # signed by the new one
    assert old_cookie.granted and new_cookie.granted
    assert old_cookie.owner_ref == new_cookie.owner_ref == before


@pytest.mark.parametrize("change", ["revoked", "expired-code", "secret-dropped"])
def test_no_owner_without_a_verified_grant(enforced, monkeypatch, change):
    cookie = cookie_value(enforced, CODE_ALICE)
    if change == "revoked":
        monkeypatch.setattr(enforced, "access_code_hashes", entries(("bob", CODE_BOB)))
    elif change == "expired-code":
        expired = f"alice:{access.hash_code(CODE_ALICE)}:2000-01-01"
        monkeypatch.setattr(enforced, "access_code_hashes", f"{expired},bob:{access.hash_code(CODE_BOB)}")
    else:
        monkeypatch.setattr(enforced, "access_cookie_secret", SecretStr(OTHER_SECRET))
    decision = decide(enforced, cookie)
    assert decision.outcome == access_policy.UNAUTHENTICATED and decision.owner_ref is None


@pytest.mark.parametrize("cookie", ["", "garbage", "a.b"])
def test_unauthenticated_and_unavailable_decisions_have_no_owner(enforced, monkeypatch, cookie):
    assert decide(enforced, cookie).owner_ref is None
    assert access_policy.evaluate(fake_request([]), enforced).owner_ref is None
    monkeypatch.setattr(enforced, "quota_enabled", False)
    unavailable = decide(enforced, cookie_value(enforced, CODE_ALICE))
    assert unavailable.outcome == access_policy.UNAVAILABLE and unavailable.owner_ref is None


def test_a_cookie_cannot_choose_its_owner(enforced):
    # Payload fields other than the signed ones are refused; the owner is
    # derived from configuration, so there is nothing a client could set.
    name, value = cookie_for(enforced, CODE_ALICE)
    payload, signature = value.split(".")
    data = json.loads(access._b64decode(payload))
    data["owner_ref"] = "0" * 64
    forged = f"{access._b64encode(json.dumps(data).encode())}.{signature}"
    decision = decide(enforced, forged, name)
    assert decision.outcome == access_policy.UNAUTHENTICATED and decision.owner_ref is None


# ---------------------------------------------------------------------------
# Never exposed
# ---------------------------------------------------------------------------

def test_the_owner_is_not_in_the_decision_repr(enforced):
    decision = decide(enforced, cookie_value(enforced, CODE_ALICE))
    assert decision.owner_ref not in repr(decision) and "owner_ref" not in repr(decision)


@pytest.mark.parametrize("entry", list(ENTRY_POINTS))
def test_the_owner_is_never_in_responses_or_logs(enforced, spies, caplog, entry):
    caplog.set_level(logging.DEBUG)
    owner = expected_owner(enforced, "alice")
    with TestClient(ENTRY_POINTS[entry]()) as client:
        headers = {**SAME_ORIGIN, "cookie": "%s=%s" % cookie_for(enforced, CODE_ALICE)}
        responses = [
            send(client, "chat", headers),
            send(client, "session_create", headers),
            client.get("/api/access/status", headers=headers),
            send(client, "chat", SAME_ORIGIN),  # refused
        ]
    assert [r.status_code for r in responses] == [200, 201, 200, 401]
    for response in responses:
        assert owner not in response.text and owner not in str(response.headers)
    assert owner not in caplog.text


# ---------------------------------------------------------------------------
# routes._session_owner: fail closed
# ---------------------------------------------------------------------------

VALID_OWNER = access_policy.derive_owner_ref("alice", FP_A)


def granted_decision(code_id="alice", owner_ref=VALID_OWNER, outcome=access_policy.GRANTED):
    grant = access.AccessGrant(code_id=code_id, expires_at=2**31 - 1)
    return access_policy.AccessDecision(outcome, "test", grant=grant, code_ref="ref", owner_ref=owner_ref)


def with_decision(decision):
    return SimpleNamespace(state=SimpleNamespace(access=decision))


def test_session_owner_returns_the_verified_owner():
    assert routes._session_owner(with_decision(granted_decision())) == VALID_OWNER


def test_session_owner_from_a_real_request(enforced):
    request = fake_request([("cookie", "%s=%s" % cookie_for(enforced, CODE_BOB))])
    request.state.access = access_policy.evaluate(request, enforced)
    assert routes._session_owner(request) == expected_owner(enforced, "bob")


@pytest.mark.parametrize("request_like", [
    None,
    SimpleNamespace(),
    SimpleNamespace(state=SimpleNamespace()),
    with_decision(None),
    with_decision({"owner_ref": VALID_OWNER}),
    with_decision(granted_decision(outcome=access_policy.UNAUTHENTICATED)),
    with_decision(granted_decision(outcome=access_policy.UNAVAILABLE)),
    with_decision(access_policy.AccessDecision(access_policy.GRANTED, "no-grant", owner_ref=VALID_OWNER)),
    with_decision(granted_decision(owner_ref=None)),
    with_decision(granted_decision(owner_ref="")),
    with_decision(granted_decision(owner_ref=VALID_OWNER.upper())),
    with_decision(granted_decision(owner_ref=VALID_OWNER[:-1])),
    with_decision(granted_decision(owner_ref=VALID_OWNER + "0")),
    with_decision(granted_decision(owner_ref=b"a" * 64)),
    with_decision(granted_decision(code_id=quota.ANONYMOUS)),
    with_decision(granted_decision(code_id="Not A Code")),
], ids=["none", "no-state", "no-decision", "decision-none", "not-a-decision", "unauthenticated", "unavailable",
        "no-grant", "owner-none", "owner-empty", "owner-upper", "owner-short", "owner-long", "owner-bytes",
        "anonymous-identity", "bad-code-id"])
def test_session_owner_fails_closed(request_like, caplog):
    with pytest.raises(HTTPException) as caught:
        routes._session_owner(request_like)
    assert caught.value.status_code == 503 and caught.value.detail["code"] == "access_unavailable"
    assert VALID_OWNER not in caplog.text


# ---------------------------------------------------------------------------
# B3: every session route resolves the verified owner and passes it on
# ---------------------------------------------------------------------------

SESSION_HANDLERS = {"create_session", "delete_session", "_chat_with_session", "_chat_stream_with_session"}


def _functions(tree):
    return {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def test_every_session_handler_resolves_the_owner_before_any_session_work():
    tree = ast.parse(inspect.getsource(routes))
    functions = _functions(tree)
    for name in SESSION_HANDLERS:
        body = ast.unparse(functions[name])
        owner_at = body.index("owner_ref = _session_owner(request)")
        for marker in ("get_session_runtime", "_acquire_session_lease", "service.reserve()",
                       "_check_session_creation_request"):
            if marker in body:
                assert owner_at < body.index(marker), (name, marker)


def test_every_store_and_lease_call_in_the_routes_passes_the_owner():
    tree = ast.parse(inspect.getsource(routes))
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            text = ast.unparse(node)
            if any(f"{receiver}.{method}" in text for receiver in ("runtime.store", "runtime.manager")
                   for method in ("create", "lookup", "delete", "acquire")):
                calls.append(text)
    assert calls, "no session store calls found"
    for text in calls:
        assert "owner_ref=owner_ref" in text, text
    assert "runtime.store.lookup" not in inspect.getsource(routes)  # no token-only look-up anywhere


def _owner_less(decision):
    return access_policy.AccessDecision(
        decision.outcome, decision.reason, grant=decision.grant, code_ref=decision.code_ref, owner_ref=None)


@pytest.mark.parametrize("entry", list(ENTRY_POINTS))
@pytest.mark.parametrize("route", ["session_create", "session_delete", "chat-session", "stream-session"])
def test_session_routes_require_a_verified_owner(enforced, spies, monkeypatch, entry, route):
    client = TestClient(ENTRY_POINTS[entry]())
    headers = {**SAME_ORIGIN, "cookie": "%s=%s" % cookie_for(enforced, CODE_ALICE)}
    token = send(client, "session_create", headers).json()["session_id"]
    spies.session_runtime = 0
    real = access_policy.evaluate
    monkeypatch.setattr(access_policy, "evaluate", lambda request, settings: _owner_less(real(request, settings)))
    if route in ("session_create", "session_delete"):
        response = send(client, route, {**headers, "x-session-id": token})
    else:
        response = send(client, "chat" if route == "chat-session" else "stream", {**headers, "x-session-id": token})
    assert response.status_code == 503 and response.json()["detail"]["code"] == "access_unavailable"
    assert spies.session_runtime == 0 and spies.asks == [] and spies.stream_service == 0


def _row(settings, token):
    import sqlite3

    with sqlite3.connect(settings.session_db_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT lease_owner, lease_epoch, turn_count, access_owner_ref FROM sessions WHERE token_hash = ?",
            (session_service.hash_token(token),)).fetchone()
        turns = connection.execute("SELECT count(*) FROM turns").fetchone()[0]
    return (None if row is None else dict(row)), turns


@pytest.mark.parametrize("entry", list(ENTRY_POINTS))
def test_owner_b_cannot_use_owner_as_session_with_a_valid_token(enforced, spies, stream_app, entry):
    service = stream_app(ScriptedWorkflow([("chunk", "x"), ("final", final_state())]))
    alice = {**SAME_ORIGIN, "cookie": "%s=%s" % cookie_for(enforced, CODE_ALICE)}
    bob = {**SAME_ORIGIN, "cookie": "%s=%s" % cookie_for(enforced, CODE_BOB)}
    with TestClient(ENTRY_POINTS[entry]()) as client:
        token = send(client, "session_create", alice).json()["session_id"]
        assert send(client, "chat", {**alice, "x-session-id": token}).status_code == 200
        before, turns_before = _row(enforced, token)
        assert before["access_owner_ref"] == expected_owner(enforced, "alice")
        spies.asks.clear()

        chat = send(client, "chat", {**bob, "x-session-id": token})
        stream = send(client, "stream", {**bob, "x-session-id": token})
        deleted = client.delete("/api/sessions", headers={**bob, "x-session-id": token})
        unknown = send(client, "chat", {**bob, "x-session-id": "never-issued-token"})

        assert chat.status_code == stream.status_code == unknown.status_code == 410
        assert chat.json() == unknown.json() == stream.json()  # indistinguishable from an unknown session
        assert deleted.status_code == 204
        assert spies.asks == [] and service.started == []  # no provider, no stream worker
        after, turns_after = _row(enforced, token)
        assert after == before and turns_after == turns_before  # no lease, no epoch change, nothing deleted
        engine = quota_runtime.peek_quota_engine()
        assert engine.usage("bob")["daily"]["chat"] == 0  # refused before quota admission

        assert send(client, "chat", {**alice, "x-session-id": token}).status_code == 200  # still Alice's
        assert client.delete("/api/sessions", headers={**alice, "x-session-id": token}).status_code == 204
        assert _row(enforced, token)[0] is None


def test_the_per_owner_cap_is_a_controlled_429(enforced, spies, monkeypatch):
    monkeypatch.setattr(enforced, "session_max_active_per_owner", 2)
    client = TestClient(ENTRY_POINTS["app.main"]())
    alice = {**SAME_ORIGIN, "cookie": "%s=%s" % cookie_for(enforced, CODE_ALICE)}
    bob = {**SAME_ORIGIN, "cookie": "%s=%s" % cookie_for(enforced, CODE_BOB)}
    assert [send(client, "session_create", alice).status_code for _ in range(2)] == [201, 201]
    refused = send(client, "session_create", alice)
    assert refused.status_code == 429 and refused.json()["detail"]["code"] == "session_limit"
    assert refused.headers["retry-after"] == str(routes.SESSION_LIMIT_RETRY_AFTER_SECONDS)
    assert "sqlite" not in refused.text.lower() and expected_owner(enforced, "alice") not in refused.text
    assert send(client, "session_create", bob).status_code == 201


def test_an_unsupported_schema_is_a_controlled_503(enforced, spies):
    import sqlite3

    client = TestClient(ENTRY_POINTS["app.main"]())
    alice = {**SAME_ORIGIN, "cookie": "%s=%s" % cookie_for(enforced, CODE_ALICE)}
    token = send(client, "session_create", alice).json()["session_id"]
    with sqlite3.connect(enforced.session_db_path) as connection:
        connection.execute("PRAGMA user_version = 99")
    session_service.clear_session_runtime()  # a restarted process sees the newer schema
    responses = [send(client, "session_create", alice), send(client, "chat", {**alice, "x-session-id": token}),
                 client.delete("/api/sessions", headers={**alice, "x-session-id": token})]
    for response in responses:
        assert response.status_code == 503 and response.json()["detail"]["code"] == "session_unavailable"
        assert "schema" not in response.text.lower() and "sqlite" not in response.text.lower()
    assert spies.asks == []
