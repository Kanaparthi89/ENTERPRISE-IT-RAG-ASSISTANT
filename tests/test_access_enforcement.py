"""Security S1-5 Phase A: verified access on every paid AI route.

This module is deliberately NOT in tests/conftest.py's LEGACY_ACCESS_MODULES:
everything here runs with real enforcement (real access settings, real
quotas, real PUBLIC_ORIGINS and real signed cookies). Providers are fakes,
and the suite's network guard and blocked clients make any real provider
call fail, so nothing here can reach a paid API.

Every refusal asserts, where it applies: no provider call, no quota engine
use, no session storage use, no stream capacity use, no upload slot and no
read of the request body.
"""

import json
import logging
import secrets
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import SecretStr, ValidationError
from starlette.requests import Request
from starlette.testclient import TestClient

import app.main
from app.api import routes, streaming
from app.services import access, access_policy, quota, quota_runtime, transcription
from app.services import sessions as session_service
from tests.conftest import IsolatedSettings
from tests.test_chat_api import workflow_result
from tests.test_stream_run import ScriptedWorkflow, final_state
from tests.test_voice_transcribe import MULTIPART, FakeProvider, upload

QUESTION = "How do I connect to the VPN?"
CODE_ALICE = "Alice-enforce-0123456789abcdefGHIJKL"
CODE_BOB = "Bob-enforce-0123456789abcdefghijKLMN"
CODE_ALICE_NEW = "Alice-replaced-0123456789abcdefGHIJ"
SECRET = "S15-test-secret-9fK2pX4mZ8rT1yW6nB3cJ5hL0dSxE4uA"
OTHER_SECRET = "S15-other-secret-7Lq3wE9rT5yU1iO8pA2sD6fG4hJ0kZ"
ORIGIN = "http://testserver"
SAME_ORIGIN = {"origin": ORIGIN, "sec-fetch-site": "same-origin"}
JSON_TYPE = {"content-type": "application/json"}
WELL_FORMED_TOKEN = secrets.token_urlsafe(32)

PAID_ROUTES = ["chat", "stream", "session_create", "session_delete", "transcribe"]
JSON_ROUTES = ["chat", "stream"]


def api_main_app():
    import app.api.main as api_main  # the second entry point (unchanged)

    return api_main.app


ENTRY_POINTS = {"app.main": lambda: app.main.app, "app.api.main": api_main_app}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def entries(*pairs):
    return ",".join(f"{name}:{access.hash_code(code)}" for name, code in pairs)


@pytest.fixture
def enforced(monkeypatch, settings):
    """Everything paid routes need, all valid: access codes, quotas and PUBLIC_ORIGINS."""
    for name, value in (
        ("access_enabled", True),
        ("access_code_hashes", entries(("alice", CODE_ALICE), ("bob", CODE_BOB))),
        ("access_cookie_secret", SecretStr(SECRET)),
        ("access_cookie_previous_secrets", SecretStr("")),
        ("access_cookie_secure", True),
        ("access_client_ip_header", ""),
        ("access_trust_socket_peer", False),
        ("quota_enabled", True),
        ("chat_deadline_seconds", 90.0),
        ("stream_timeout_seconds", 90.0),
        ("session_turn_max_seconds", 105.0),
        ("openai_timeout_seconds", 30.0),
        ("openai_max_retries", 1),
        ("pinecone_retry_after_max_seconds", 5),
        ("public_origins", ORIGIN),
        ("sessions_enabled", True),
        ("session_shutdown_grace_seconds", 0.2),
        ("session_keeper_join_seconds", 0.3),
        ("session_release_budget_seconds", 0.5),
        ("stream_enabled", True),
        ("voice_transcription_enabled", True),
        ("openai_api_key", SecretStr("sk-test-not-a-real-key")),
        ("voice_transcription_timeout_seconds", 30.0),
        ("voice_transcription_max_retries", 0),
    ):
        monkeypatch.setattr(settings, name, value)
    assert access_policy.configuration_problem(settings) is None
    return settings


def cookie_for(settings, code, *, now=None):
    config = access.AccessConfig.from_settings(settings)
    now = access.wall_clock() if now is None else now
    matched = next(item for item in config.codes if item.digest == access.hash_code(code))
    value, _ = access.issue_cookie(config, matched, now)
    return config.cookie_name, value


def cookie_header(settings, code, **kwargs):
    name, value = cookie_for(settings, code, **kwargs)
    return {"cookie": f"{name}={value}"}


class Spies:
    """Counts everything a refused request must never reach."""

    def __init__(self):
        self.asks = []
        self.quota_engine = 0
        self.session_runtime = 0
        self.stream_service = 0
        self.body_reads = 0
        self.upload_slots = 0
        self.provider = FakeProvider()

    def untouched(self):
        return (
            self.asks == [] and self.quota_engine == 0 and self.session_runtime == 0
            and self.stream_service == 0 and self.body_reads == 0 and self.upload_slots == 0
            and self.provider.requests == [] and self.provider.clients == []
            and quota_runtime.peek_quota_engine() is None
        )


@pytest.fixture
def spies(monkeypatch):
    spy = Spies()

    def fake_ask(question, history=None):
        spy.asks.append(question)
        return workflow_result()

    real_engine = quota_runtime.get_quota_engine
    real_runtime = session_service.get_session_runtime
    real_service = streaming.get_stream_service
    real_stream = Request.stream
    real_slot = routes._voice_upload_slot

    def get_quota_engine(settings):
        spy.quota_engine += 1
        return real_engine(settings)

    def get_session_runtime(settings):
        spy.session_runtime += 1
        return real_runtime(settings)

    def get_stream_service():
        spy.stream_service += 1
        return real_service()

    def stream(self):
        spy.body_reads += 1
        return real_stream(self)

    def voice_upload_slot(identity):
        spy.upload_slots += 1
        return real_slot(identity)

    monkeypatch.setattr(routes, "ask", fake_ask)
    monkeypatch.setattr(quota_runtime, "get_quota_engine", get_quota_engine)
    monkeypatch.setattr(session_service, "get_session_runtime", get_session_runtime)
    monkeypatch.setattr(streaming, "get_stream_service", get_stream_service)
    monkeypatch.setattr(Request, "stream", stream)
    monkeypatch.setattr(routes, "_voice_upload_slot", voice_upload_slot)
    monkeypatch.setattr(transcription, "AsyncOpenAI", spy.provider)
    return spy


def send(client, route, headers=None, *, body=None):
    """One request to a paid route with an otherwise valid body for that route."""
    headers = dict(headers or {})
    if route in ("chat", "stream"):
        path = "/api/chat" if route == "chat" else "/api/chat/stream"
        content = json.dumps({"question": QUESTION}).encode() if body is None else body
        return client.post(path, content=content, headers={**JSON_TYPE, **headers})
    if route == "session_create":
        return client.post("/api/sessions", content=b"{}" if body is None else body, headers={**JSON_TYPE, **headers})
    if route == "session_delete":
        return client.delete("/api/sessions", headers={"x-session-id": WELL_FORMED_TOKEN, **headers})
    if route == "transcribe":
        return client.post(
            "/api/voice/transcribe", content=upload() if body is None else body,
            headers={"content-type": MULTIPART, **headers},
        )
    raise AssertionError(route)


def assert_error(response, status, code):
    assert response.status_code == status, response.text
    assert response.json()["detail"]["code"] == code
    assert response.headers["cache-control"] == "no-store"
    if status == 401:
        assert response.headers["www-authenticate"].startswith("AccessCode")
    if status == 503:
        assert response.headers["retry-after"] == str(routes.ACCESS_UNAVAILABLE_RETRY_AFTER_SECONDS)


# ---------------------------------------------------------------------------
# Unauthenticated: every cookie problem, every paid route, both entry points
# ---------------------------------------------------------------------------

def forged(settings):
    name, value = cookie_for(settings, CODE_ALICE)
    payload, signature = value.split(".")
    flipped = ("A" if signature[-1] != "A" else "B")
    return {"cookie": f"{name}={payload}.{signature[:-1]}{flipped}"}


def expired(settings):
    config = access.AccessConfig.from_settings(settings)
    return cookie_header(settings, CODE_ALICE, now=access.wall_clock() - config.max_age_seconds - 120)


def plain_name(settings):
    _, value = cookie_for(settings, CODE_ALICE)
    return {"cookie": f"{access.COOKIE_NAME}={value}"}  # the plain-HTTP name is not the configured one


def duplicated(settings):
    name, alice = cookie_for(settings, CODE_ALICE)
    _, bob = cookie_for(settings, CODE_BOB)
    return {"cookie": f"{name}={alice}; {name}={bob}"}


def payload_tampered(settings):
    name, value = cookie_for(settings, CODE_ALICE)
    payload, signature = value.split(".")
    data = json.loads(access._b64decode(payload))
    data["cid"] = "bob"
    return {"cookie": f"{name}={access._b64encode(json.dumps(data).encode())}.{signature}"}


COOKIE_CASES = {
    "none": lambda settings: {},
    "empty": lambda settings: {"cookie": f"{access.SECURE_COOKIE_NAME}="},
    "garbage": lambda settings: {"cookie": f"{access.SECURE_COOKIE_NAME}=not-a-cookie"},
    "forged-signature": forged,
    "tampered-payload": payload_tampered,
    "expired": expired,
    "plain-http-name": plain_name,
    "two-cookies": duplicated,
}


@pytest.mark.parametrize("entry", list(ENTRY_POINTS))
@pytest.mark.parametrize("route", PAID_ROUTES)
@pytest.mark.parametrize("case", list(COOKIE_CASES))
def test_requests_without_a_verified_cookie_get_401_and_reach_nothing(enforced, spies, entry, route, case):
    headers = {**SAME_ORIGIN, **COOKIE_CASES[case](enforced)}
    response = send(TestClient(ENTRY_POINTS[entry]()), route, headers)
    assert_error(response, 401, "access_required")
    assert spies.untouched()


@pytest.mark.parametrize("route", PAID_ROUTES)
@pytest.mark.parametrize("change", ["revoked", "replaced", "secret-rotated"])
def test_cookies_of_revoked_or_replaced_codes_and_old_secrets_get_401(enforced, spies, monkeypatch, route, change):
    headers = {**SAME_ORIGIN, **cookie_header(enforced, CODE_ALICE)}
    if change == "revoked":
        monkeypatch.setattr(enforced, "access_code_hashes", entries(("bob", CODE_BOB)))
    elif change == "replaced":  # the same id, a new code
        monkeypatch.setattr(enforced, "access_code_hashes", entries(("alice", CODE_ALICE_NEW), ("bob", CODE_BOB)))
    else:
        monkeypatch.setattr(enforced, "access_cookie_secret", SecretStr(OTHER_SECRET))
    assert_error(send(TestClient(app.main.app), route, headers), 401, "access_required")
    assert spies.untouched()


# ---------------------------------------------------------------------------
# No bypass: addresses, Host, forwarded headers and APP_ENV grant nothing
# ---------------------------------------------------------------------------

BYPASS_HEADERS = [
    {"x-forwarded-for": "127.0.0.1"},
    {"forwarded": "for=127.0.0.1;host=localhost;proto=https"},
    {"x-real-ip": "127.0.0.1"},
    {"x-forwarded-host": "localhost"},
    {"host": "localhost"},
    {"host": "127.0.0.1:8000"},
    {"x-quota-identity": "alice"},
    {"x-access-code": CODE_ALICE},
    {"authorization": f"Bearer {CODE_ALICE}"},
]


@pytest.mark.parametrize("app_env", ["development", "testing"])
@pytest.mark.parametrize("peer", ["127.0.0.1", "::1", "testclient"])
@pytest.mark.parametrize("headers", BYPASS_HEADERS)
@pytest.mark.parametrize("route", PAID_ROUTES)
def test_loopback_host_proxy_headers_and_app_env_never_grant_access(
    enforced, spies, monkeypatch, app_env, peer, headers, route
):
    monkeypatch.setattr(enforced, "app_env", app_env)
    client = TestClient(app.main.app, client=(peer, 50000))
    assert_error(send(client, route, {**SAME_ORIGIN, **headers}), 401, "access_required")
    assert spies.untouched()


@pytest.mark.parametrize("route", PAID_ROUTES)
def test_access_disabled_in_development_on_loopback_is_still_read_only(enforced, spies, monkeypatch, route):
    monkeypatch.setattr(enforced, "app_env", "development")
    monkeypatch.setattr(enforced, "access_enabled", False)
    client = TestClient(app.main.app, client=("127.0.0.1", 50000))
    headers = {**SAME_ORIGIN, "host": "localhost:8000", "origin": "http://localhost:8000"}
    assert_error(send(client, route, headers), 503, "access_unavailable")
    assert spies.untouched()


# ---------------------------------------------------------------------------
# Before the body is parsed or read
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("route", JSON_ROUTES)
@pytest.mark.parametrize("body", [b'{"question": "How do I', b"", b"[]", b'{"question": "x"}', b"\xff\xfe"])
def test_an_unauthenticated_invalid_body_gets_401_not_422(enforced, spies, route, body):
    response = send(TestClient(app.main.app), route, SAME_ORIGIN, body=body)
    assert_error(response, 401, "access_required")
    assert spies.untouched()  # body_reads == 0: FastAPI never read or parsed it


@pytest.mark.parametrize("route", JSON_ROUTES)
def test_with_a_grant_the_same_invalid_body_gets_422(enforced, spies, route):
    headers = {**SAME_ORIGIN, **cookie_header(enforced, CODE_ALICE)}
    response = send(TestClient(app.main.app), route, headers, body=b'{"question": "How do I')
    assert response.status_code == 422
    assert spies.asks == [] and spies.quota_engine == 0 and spies.stream_service == 0


def test_an_oversized_unauthenticated_upload_is_refused_unread(enforced, spies):
    big = upload(data=b"\x00" * (3 * 1024 * 1024))
    response = send(TestClient(app.main.app), "transcribe", SAME_ORIGIN, body=big)
    assert_error(response, 401, "access_required")
    assert spies.untouched()


# ---------------------------------------------------------------------------
# Read-only mode: incomplete configuration refuses paid routes, public stays up
# ---------------------------------------------------------------------------

READ_ONLY_CASES = {
    "access-disabled": [("access_enabled", False)],
    "access-invalid": [("access_cookie_secret", SecretStr(""))],
    "access-no-codes": [("access_code_hashes", "")],
    "quota-disabled": [("quota_enabled", False)],
    "quota-unbounded": [("chat_deadline_seconds", None)],
    "quota-unbounded-pinecone": [("pinecone_retry_after_max_seconds", None)],
    "origins-missing": [("public_origins", "")],
    "origins-malformed": [("public_origins", "https://*.example")],
    "origins-with-path": [("public_origins", "http://testserver/app")],
    "production-http-origin": [("app_env", "production"), ("access_trust_socket_peer", True)],
}


@pytest.mark.parametrize("route", PAID_ROUTES)
@pytest.mark.parametrize("case", list(READ_ONLY_CASES))
def test_incomplete_configuration_is_read_only_even_with_a_valid_cookie(enforced, spies, monkeypatch, route, case):
    headers = {**SAME_ORIGIN, **cookie_header(enforced, CODE_ALICE)}
    for name, value in READ_ONLY_CASES[case]:  # changed after startup: rechecked per request
        monkeypatch.setattr(enforced, name, value)
    assert_error(send(TestClient(app.main.app), route, headers), 503, "access_unavailable")
    assert spies.untouched()


@pytest.mark.parametrize("configured", [False, True])
def test_public_routes_stay_available_in_read_only_mode(settings, monkeypatch, configured, request):
    if configured:
        request.getfixturevalue("enforced")
    with TestClient(app.main.app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/health").status_code == 200
        assert client.get("/api/health").status_code == 200
        assert client.get("/static/js/app.js").status_code == 200
        response = client.post("/api/chat", json={"question": QUESTION}, headers=SAME_ORIGIN)
        assert response.status_code == (401 if configured else 503)


def test_the_default_settings_are_read_only(settings, spies):
    assert (settings.access_enabled, settings.quota_enabled, settings.public_origins) == (False, False, "")
    assert access_policy.configuration_problem(settings) == access_policy.ACCESS_DISABLED
    assert_error(send(TestClient(app.main.app), "chat", SAME_ORIGIN), 503, "access_unavailable")
    assert spies.untouched()


# ---------------------------------------------------------------------------
# Origin (CSRF) and content type, with a valid grant
# ---------------------------------------------------------------------------

ORIGIN_REFUSALS = {
    "cross-site": {"sec-fetch-site": "cross-site"},
    "same-site": {"sec-fetch-site": "same-site"},
    "none": {"sec-fetch-site": "none"},
    "evil-origin": {"origin": "https://evil.example"},
    "null-origin": {"origin": "null"},
    "other-scheme": {"origin": "https://testserver"},
    "other-port": {"origin": "http://testserver:8080"},
    "subdomain": {"origin": "http://a.testserver"},
    "origin-with-path": {"origin": "http://testserver/"},
    "neither-header": {},
    "conflict-same-origin-evil": {"sec-fetch-site": "same-origin", "origin": "https://evil.example"},
    "conflict-cross-site-ours": {"sec-fetch-site": "cross-site", "origin": ORIGIN},
    "spoofed-host": {"host": "evil.example", "origin": "http://evil.example", "sec-fetch-site": "same-origin"},
    "forwarded-host": {"x-forwarded-host": "evil.example", "origin": "http://evil.example"},
}


@pytest.mark.parametrize("route", PAID_ROUTES)
@pytest.mark.parametrize("case", list(ORIGIN_REFUSALS))
def test_cross_site_and_missing_origin_evidence_get_403(enforced, spies, route, case):
    headers = {**cookie_header(enforced, CODE_ALICE), **ORIGIN_REFUSALS[case]}
    assert_error(send(TestClient(app.main.app), route, headers), 403, "origin_not_allowed")
    assert spies.untouched()


def test_session_creation_without_origin_headers_is_refused_in_production(enforced, spies):
    # The production counterpart of the shim-only case in tests/test_session_api.py.
    response = send(TestClient(app.main.app), "session_create", cookie_header(enforced, CODE_ALICE))
    assert_error(response, 403, "origin_not_allowed")
    assert spies.untouched()  # no session created, body never read


def test_transcription_without_origin_headers_is_refused_in_production(enforced, spies):
    # The production counterpart of the shim-only case in tests/test_voice_transcribe.py.
    response = send(TestClient(app.main.app), "transcribe", cookie_header(enforced, CODE_ALICE))
    assert_error(response, 403, "origin_not_allowed")
    assert spies.untouched()  # no upload slot, upload never read, no provider


@pytest.mark.parametrize("route", ["chat", "session_delete"])
@pytest.mark.parametrize("duplicate", ["origin", "sec-fetch-site"])
def test_duplicated_origin_evidence_gets_403(enforced, spies, route, duplicate):
    value = ORIGIN if duplicate == "origin" else "same-origin"
    other = ("sec-fetch-site", "same-origin") if duplicate == "origin" else ("origin", ORIGIN)
    client = TestClient(app.main.app)
    # The other header is sent once and valid: only the duplicate makes it ambiguous.
    headers = [(duplicate, value), (duplicate, value), other, *cookie_header(enforced, CODE_ALICE).items()]
    if route == "chat":
        response = client.post("/api/chat", content=b'{"question": "VPN?"}', headers=[*headers, *JSON_TYPE.items()])
    else:
        response = client.delete("/api/sessions", headers=[*headers, ("x-session-id", WELL_FORMED_TOKEN)])
    assert_error(response, 403, "origin_not_allowed")
    assert spies.untouched()


@pytest.mark.parametrize("route", JSON_ROUTES)
@pytest.mark.parametrize("content_type", [
    None, "text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x",
    "application/json-seq", "application/jsonx", "application/json; charset=latin-1", "application/json;",
    "application/json; charset=utf-8; extra=1", "text/json",
])
def test_json_routes_refuse_other_content_types_with_415(enforced, spies, route, content_type):
    client = TestClient(app.main.app)
    path = "/api/chat" if route == "chat" else "/api/chat/stream"
    headers = {**SAME_ORIGIN, **cookie_header(enforced, CODE_ALICE)}
    if content_type is not None:
        headers["content-type"] = content_type
    response = client.post(path, content=json.dumps({"question": QUESTION}).encode(), headers=headers)
    assert_error(response, 415, "unsupported_media_type")
    assert spies.untouched()


def test_a_duplicated_content_type_gets_415(enforced, spies):
    headers = [*SAME_ORIGIN.items(), *cookie_header(enforced, CODE_ALICE).items(),
               ("content-type", "application/json"), ("content-type", "application/json")]
    response = TestClient(app.main.app).post("/api/chat", content=b'{"question": "VPN?"}', headers=headers)
    assert_error(response, 415, "unsupported_media_type")
    assert spies.untouched()


@pytest.mark.parametrize("content_type", ["application/json", "Application/JSON", "application/json; charset=utf-8",
                                          'application/json;charset="UTF-8"'])
def test_json_with_or_without_utf8_charset_is_accepted(enforced, spies, content_type):
    headers = {**SAME_ORIGIN, **cookie_header(enforced, CODE_ALICE), "content-type": content_type}
    response = TestClient(app.main.app).post("/api/chat", content=json.dumps({"question": QUESTION}).encode(),
                                             headers=headers)
    assert response.status_code == 200, response.text
    assert spies.asks == [QUESTION]


@pytest.mark.parametrize("headers", [{"origin": ORIGIN}, {"sec-fetch-site": "same-origin"}, SAME_ORIGIN,
                                     {"origin": "http://TESTSERVER:80"}])
def test_either_piece_of_same_origin_evidence_is_enough(enforced, spies, headers):
    response = send(TestClient(app.main.app), "chat", {**headers, **cookie_header(enforced, CODE_ALICE)})
    assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# Order: flag, access, origin, content type, body, then capacity/quota/provider
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("route, flag", [("stream", "stream_enabled"), ("session_create", "sessions_enabled"),
                                         ("transcribe", "voice_transcription_enabled")])
def test_a_disabled_feature_is_404_before_access(enforced, spies, monkeypatch, route, flag):
    monkeypatch.setattr(enforced, flag, False)
    response = send(TestClient(app.main.app), route, {})  # no cookie, no origin
    assert response.status_code == 404 and response.json() == {"detail": "Not Found"}
    assert spies.untouched()


def test_access_is_checked_before_the_origin(enforced, spies):
    response = send(TestClient(app.main.app), "chat", {"origin": "https://evil.example"})
    assert_error(response, 401, "access_required")


def test_the_origin_is_checked_before_the_content_type(enforced, spies):
    headers = {**cookie_header(enforced, CODE_ALICE), "origin": "https://evil.example", "content-type": "text/plain"}
    response = TestClient(app.main.app).post("/api/chat", content=b"x", headers=headers)
    assert_error(response, 403, "origin_not_allowed")


def test_an_unauthenticated_session_question_never_touches_the_session(enforced, spies):
    headers = {**SAME_ORIGIN, "x-session-id": WELL_FORMED_TOKEN}
    for route in JSON_ROUTES:
        assert_error(send(TestClient(app.main.app), route, headers), 401, "access_required")
    assert spies.untouched()


def test_with_a_grant_an_unknown_session_is_410_after_access(enforced, spies):
    headers = {**SAME_ORIGIN, **cookie_header(enforced, CODE_ALICE), "x-session-id": WELL_FORMED_TOKEN}
    response = send(TestClient(app.main.app), "chat", headers)
    assert response.status_code == 410 and response.json()["detail"]["code"] == "session_expired"
    assert spies.asks == [] and spies.quota_engine == 0


# ---------------------------------------------------------------------------
# Valid grants: every route works, and quota is charged to the verified code
# ---------------------------------------------------------------------------

def granted(settings, code=CODE_ALICE):
    return {**SAME_ORIGIN, **cookie_header(settings, code)}


@pytest.mark.parametrize("entry", list(ENTRY_POINTS))
def test_a_verified_code_can_chat_and_is_charged_for_it(enforced, spies, entry):
    with TestClient(ENTRY_POINTS[entry]()) as client:
        assert send(client, "chat", granted(enforced)).status_code == 200
        assert send(client, "chat", granted(enforced, CODE_BOB)).status_code == 200
    current = quota_runtime.peek_quota_engine()
    assert current.usage("alice")["daily"]["chat"] == 1
    assert current.usage("bob")["daily"]["chat"] == 1
    assert current.usage(quota.ANONYMOUS)["daily"]["chat"] == 0
    assert spies.asks == [QUESTION, QUESTION]


def test_a_verified_code_can_stream_and_is_charged_for_it(enforced, stream_app):
    service = stream_app(ScriptedWorkflow([("chunk", "Install GlobalProtect."), ("final", final_state())]))
    with TestClient(app.main.app) as client:
        response = send(client, "stream", granted(enforced))
    assert response.status_code == 200 and "event: done" in response.text
    assert len(service.started) == 1
    current = quota_runtime.peek_quota_engine()
    assert current.usage("alice")["daily"]["chat"] == 1 and current.usage()["active"]["chat"] == 0


def test_a_verified_code_can_use_sessions(enforced, spies):
    with TestClient(app.main.app) as client:
        created = send(client, "session_create", granted(enforced))
        assert created.status_code == 201, created.text
        token = created.json()["session_id"]
        answered = send(client, "chat", {**granted(enforced), "x-session-id": token})
        assert answered.status_code == 200 and answered.json()["session"]["turn_saved"] is True
        deleted = client.delete("/api/sessions", headers={**granted(enforced), "x-session-id": token})
        assert deleted.status_code == 204
    assert quota_runtime.peek_quota_engine().usage("alice")["daily"]["chat"] == 1


def test_a_verified_code_can_transcribe_and_is_charged_for_it(enforced, spies):
    with TestClient(app.main.app) as client:
        response = send(client, "transcribe", granted(enforced))
    assert response.status_code == 200, response.text
    usage = quota_runtime.peek_quota_engine().usage("alice")
    assert usage["daily"]["transcription"] == 1 and usage["seconds"] == 1
    assert quota_runtime.peek_quota_engine().usage(quota.ANONYMOUS)["seconds"] == 0
    assert routes._voice_upload_identities == set()


def test_redeem_chat_logout_then_401(enforced, spies, monkeypatch):
    monkeypatch.setattr(enforced, "public_origins", "https://testserver")
    https = {"origin": "https://testserver", "sec-fetch-site": "same-origin"}
    with TestClient(app.main.app, base_url="https://testserver") as client:
        redeemed = client.post("/api/access", json={"code": CODE_ALICE}, headers=https)
        assert redeemed.status_code == 200, redeemed.text
        assert client.post("/api/chat", json={"question": QUESTION}, headers=https).status_code == 200
        assert client.post("/api/access/logout", headers=https).status_code == 204
        after = client.post("/api/chat", json={"question": QUESTION}, headers=https)
    assert_error(after, 401, "access_required")
    assert spies.asks == [QUESTION]


def test_production_with_everything_valid_serves_chat_but_still_refuses_transcription(enforced, spies, monkeypatch):
    for name, value in (("app_env", "production"), ("access_trust_socket_peer", True),
                        ("public_origins", "https://testserver")):
        monkeypatch.setattr(enforced, name, value)
    assert access_policy.configuration_problem(enforced) is None
    https = {"origin": "https://testserver", "sec-fetch-site": "same-origin", **cookie_header(enforced, CODE_ALICE)}
    client = TestClient(app.main.app, base_url="https://testserver")
    assert send(client, "chat", https).status_code == 200
    # The S1-4c production transcription guard is kept in Phase A: refused before reading.
    spies.body_reads = 0
    refused = send(client, "transcribe", https)
    assert refused.status_code == 503 and refused.json()["detail"]["code"] == "transcription_unavailable"
    assert spies.body_reads == 0 and spies.upload_slots == 0 and spies.provider.requests == []


# ---------------------------------------------------------------------------
# F4: configuration changed after the access gate, before streaming starts
# ---------------------------------------------------------------------------

AFTER_GATE_CHANGES = {
    "quota-disabled": [("quota_enabled", False)],
    "quota-invalid": [("chat_deadline_seconds", None)],
    "access-disabled": [("access_enabled", False)],
    "access-invalid": [("access_cookie_secret", SecretStr(""))],
    "origins-removed": [("public_origins", "")],
}


@pytest.fixture
def change_after_gate(monkeypatch, enforced):
    """Apply settings changes after the route class has checked access.

    Hooked on the origin check, which every paid route runs after access
    (the JSON content-type check that may follow doesn't read settings):
    the change lands after the gate decided, before the handler runs.
    ``changes`` is a list of (setting, value), or a callable given the request.
    """
    real = access_policy.origin_allowed

    def install(changes):
        def origin_allowed(request, origins):
            result = real(request, origins)
            if callable(changes):
                changes(request)
            else:
                for name, value in changes:
                    monkeypatch.setattr(enforced, name, value)
            return result

        monkeypatch.setattr(access_policy, "origin_allowed", origin_allowed)

    return install


def new_session_token(client, settings):
    created = send(client, "session_create", granted(settings))
    assert created.status_code == 201, created.text
    return created.json()["session_id"]


def assert_nothing_started(spies):
    """After the gate the body was parsed (that is all): nothing else may have run."""
    assert spies.asks == [] and spies.stream_service == 0 and spies.quota_engine == 0
    assert spies.session_runtime == 0 and spies.provider.requests == []
    assert quota_runtime.peek_quota_engine() is None
    assert streaming._service is None  # no stream service, so no capacity reserved and no worker


@pytest.mark.parametrize("case", list(AFTER_GATE_CHANGES))
def test_stateless_streaming_rechecks_the_configuration_before_reserving(enforced, spies, change_after_gate, case):
    change_after_gate(AFTER_GATE_CHANGES[case])
    response = send(TestClient(app.main.app), "stream", granted(enforced))
    assert_error(response, 503, "access_unavailable")
    assert_nothing_started(spies)


@pytest.mark.parametrize("case", list(AFTER_GATE_CHANGES))
def test_session_streaming_rechecks_the_configuration_before_reserving(enforced, spies, change_after_gate, case):
    client = TestClient(app.main.app)
    token = new_session_token(client, enforced)
    spies.session_runtime = 0  # creating the session used it; the stream must not
    change_after_gate(AFTER_GATE_CHANGES[case])
    response = send(client, "stream", {**granted(enforced), "x-session-id": token})
    assert_error(response, 503, "access_unavailable")
    assert_nothing_started(spies)


@pytest.mark.parametrize("route", ["stream", "session-stream"])
@pytest.mark.parametrize("code_id", [quota.ANONYMOUS, "Not A Code"])
def test_streaming_refuses_a_granted_decision_without_a_valid_identity(enforced, spies, monkeypatch, route, code_id):
    client = TestClient(app.main.app)
    headers = granted(enforced)
    if route == "session-stream":
        headers["x-session-id"] = new_session_token(client, enforced)
        spies.session_runtime = 0
    forged_decision = access_policy.AccessDecision(
        access_policy.GRANTED, "forged", grant=access.AccessGrant(code_id=code_id, expires_at=2**31 - 1),
    )
    monkeypatch.setattr(access_policy, "evaluate", lambda request, settings: forged_decision)
    assert_error(send(client, "stream", headers), 503, "access_unavailable")
    assert_nothing_started(spies)


# Phase A hardening: the same recheck before session leases, session
# creation and deletion, and transcription upload slots and body reads.

def _drop_decision(request):
    del request.state.access


def _forge(code_id):
    def change(request):
        request.state.access = access_policy.AccessDecision(
            access_policy.GRANTED, "forged", grant=access.AccessGrant(code_id=code_id, expires_at=2**31 - 1),
        )
    return change


def _malformed(request):
    request.state.access = {"code_id": "alice"}


HANDLER_CHANGES = {
    **AFTER_GATE_CHANGES,
    "quota-unbounded-pinecone": [("pinecone_retry_after_max_seconds", None)],
    "identity-missing": _drop_decision,
    "identity-malformed": _malformed,
    "identity-anonymous": _forge(quota.ANONYMOUS),
    "identity-bad-code-id": _forge("Not A Code"),
}


def session_rows(settings):
    from tests.session_support import count_session_rows

    return count_session_rows(settings.session_db_path) if settings.session_db_path.exists() else 0


@pytest.mark.parametrize("case", list(HANDLER_CHANGES))
def test_session_chat_rechecks_before_taking_the_lease(enforced, spies, change_after_gate, case):
    client = TestClient(app.main.app)
    token = new_session_token(client, enforced)
    spies.session_runtime = 0
    change_after_gate(HANDLER_CHANGES[case])
    response = send(client, "chat", {**granted(enforced), "x-session-id": token})
    assert_error(response, 503, "access_unavailable")
    assert spies.session_runtime == 0  # no lease taken (nor any session store use)
    assert spies.asks == [] and spies.quota_engine == 0 and quota_runtime.peek_quota_engine() is None


@pytest.mark.parametrize("case", list(HANDLER_CHANGES))
def test_session_creation_rechecks_before_reading_or_creating(enforced, spies, change_after_gate, case):
    change_after_gate(HANDLER_CHANGES[case])
    response = send(TestClient(app.main.app), "session_create", granted(enforced))
    assert_error(response, 503, "access_unavailable")
    assert spies.session_runtime == 0 and spies.body_reads == 0
    assert session_rows(enforced) == 0
    assert spies.untouched()


@pytest.mark.parametrize("case", list(HANDLER_CHANGES))
def test_session_deletion_rechecks_before_deleting(enforced, spies, change_after_gate, case):
    client = TestClient(app.main.app)
    token = new_session_token(client, enforced)
    assert session_rows(enforced) == 1
    spies.session_runtime = 0
    change_after_gate(HANDLER_CHANGES[case])
    response = client.delete("/api/sessions", headers={**granted(enforced), "x-session-id": token})
    assert_error(response, 503, "access_unavailable")
    assert spies.session_runtime == 0
    assert session_rows(enforced) == 1  # not deleted


@pytest.mark.parametrize("case", list(HANDLER_CHANGES))
def test_transcription_rechecks_before_the_upload_slot_and_body(enforced, spies, change_after_gate, case):
    change_after_gate(HANDLER_CHANGES[case])
    response = send(TestClient(app.main.app), "transcribe", granted(enforced))
    assert_error(response, 503, "access_unavailable")
    assert spies.upload_slots == 0 and spies.body_reads == 0
    assert routes._voice_upload_identities == set()
    assert spies.untouched()  # no quota engine, no provider client or request


def test_sessions_and_transcription_still_work_when_nothing_changes(enforced, spies):
    with TestClient(app.main.app) as client:
        token = new_session_token(client, enforced)
        assert send(client, "chat", {**granted(enforced), "x-session-id": token}).status_code == 200
        assert send(client, "transcribe", granted(enforced)).status_code == 200
        deleted = client.delete("/api/sessions", headers={**granted(enforced), "x-session-id": token})
        assert deleted.status_code == 204
    assert session_rows(enforced) == 0
    usage = quota_runtime.peek_quota_engine().usage("alice")
    assert usage["daily"]["chat"] == 1 and usage["daily"]["transcription"] == 1


def _unauthenticated_decision(request):
    request.state.access = access_policy.AccessDecision(access_policy.UNAUTHENTICATED, "changed")


# R1 (final audit): every post-gate change, including those quota admission
# alone would not catch while quotas stay on.
STATELESS_CHAT_CHANGES = {
    "access-disabled": [("access_enabled", False)],
    "access-invalid-no-codes": [("access_code_hashes", "")],
    "access-invalid-weak-secret": [("access_cookie_secret", SecretStr("short"))],
    "quota-disabled": [("quota_enabled", False)],
    "quota-invalid-deadline": [("chat_deadline_seconds", None)],
    "quota-invalid-openai-retries": [("openai_max_retries", 5)],
    "origins-missing": [("public_origins", "")],
    "origins-invalid": [("public_origins", "https://*.example")],
    "identity-missing": _drop_decision,
    "identity-malformed": _malformed,
    "identity-anonymous": _forge(quota.ANONYMOUS),
    "identity-bad-code-id": _forge("Not A Code"),
    "identity-empty": _forge(""),
    "decision-unauthenticated": _unauthenticated_decision,
}


@pytest.mark.parametrize("engine_exists", [False, True], ids=["no-engine", "existing-engine"])
@pytest.mark.parametrize("entry", list(ENTRY_POINTS))
@pytest.mark.parametrize("case", list(STATELESS_CHAT_CHANGES))
def test_stateless_chat_rechecks_before_quota_admission(enforced, spies, change_after_gate, entry, case, engine_exists):
    engine = None
    if engine_exists:
        # An engine created while everything was valid is never rebuilt, so
        # invalid quota settings would not be caught by engine creation.
        engine = quota_runtime.get_quota_engine(enforced)
        spies.quota_engine = 0
    change_after_gate(STATELESS_CHAT_CHANGES[case])
    response = send(TestClient(ENTRY_POINTS[entry]()), "chat", granted(enforced))
    assert_error(response, 503, "access_unavailable")
    assert spies.asks == []  # no provider
    assert spies.quota_engine == 0  # no admission attempted
    assert quota_runtime.peek_quota_engine() is engine
    if engine is not None:
        usage = engine.usage("alice")
        assert usage["daily"]["chat"] == 0 and usage["active"]["chat"] == 0
        assert engine.usage()["active"]["chat"] == 0
    assert spies.session_runtime == 0 and session_rows(enforced) == 0
    assert spies.stream_service == 0 and spies.upload_slots == 0 and spies.provider.requests == []


class HistoryWorkflow(ScriptedWorkflow):
    """ScriptedWorkflow that also accepts the session history (session streams pass it)."""

    def __call__(self, question, on_chunk, history=None):
        return super().__call__(question, on_chunk)


def test_streaming_still_works_with_the_configuration_unchanged(enforced, stream_app):
    steps = [("chunk", "Install GlobalProtect."), ("final", final_state())]
    service = stream_app(HistoryWorkflow(steps))  # replays its steps on every call
    with TestClient(app.main.app) as client:
        token = new_session_token(client, enforced)
        stateless = send(client, "stream", granted(enforced))
        stateful = send(client, "stream", {**granted(enforced), "x-session-id": token})
    assert stateless.status_code == stateful.status_code == 200
    assert "event: done" in stateless.text and "event: done" in stateful.text
    assert len(service.started) == 2
    assert quota_runtime.peek_quota_engine().usage("alice")["daily"]["chat"] == 2


# ---------------------------------------------------------------------------
# Quota identity: the verified decision only, never ANONYMOUS
# ---------------------------------------------------------------------------

def decision(outcome=access_policy.GRANTED, code_id="alice"):
    grant = access.AccessGrant(code_id=code_id, expires_at=2**31 - 1) if code_id is not None else None
    return access_policy.AccessDecision(outcome, "test", grant=grant, code_ref="ref")


@pytest.mark.parametrize("request_like", [
    None,
    SimpleNamespace(),
    SimpleNamespace(state=SimpleNamespace()),
    SimpleNamespace(state=SimpleNamespace(access=None)),
    SimpleNamespace(state=SimpleNamespace(access={"code_id": "alice"})),
    SimpleNamespace(state=SimpleNamespace(access=decision(access_policy.UNAUTHENTICATED))),
    SimpleNamespace(state=SimpleNamespace(access=decision(access_policy.UNAVAILABLE))),
    SimpleNamespace(state=SimpleNamespace(access=decision(code_id=None))),
    SimpleNamespace(state=SimpleNamespace(access=decision(code_id=quota.ANONYMOUS))),
    SimpleNamespace(state=SimpleNamespace(access=decision(code_id="Not A Code Id"))),
    SimpleNamespace(state=SimpleNamespace(access=decision(code_id=""))),
], ids=["none", "no-state", "no-decision", "decision-none", "not-a-decision", "unauthenticated",
        "unavailable", "no-grant", "anonymous", "malformed-id", "empty-id"])
def test_quota_identity_fails_closed_without_a_verified_decision(request_like):
    with pytest.raises(HTTPException) as caught:
        routes._quota_identity(request_like)
    assert caught.value.status_code == 503 and caught.value.detail["code"] == "access_unavailable"


def test_quota_identity_is_the_verified_code():
    request_like = SimpleNamespace(state=SimpleNamespace(access=decision()))
    assert routes._quota_identity(request_like) == ("alice", "ref")


@pytest.mark.parametrize("function", ["_admit_chat", "_admit_transcription"])
def test_admission_with_quotas_off_fails_closed(settings, spies, function):
    assert settings.quota_enabled is False
    request_like = SimpleNamespace(state=SimpleNamespace(access=decision()))
    with pytest.raises(HTTPException) as caught:
        if function == "_admit_chat":
            routes._admit_chat(request_like, "request-id")
        else:
            routes._admit_transcription("alice", "ref", "request-id", 1)
    assert caught.value.status_code == 503 and caught.value.detail["code"] == "access_unavailable"
    assert spies.quota_engine == 0


def test_a_handler_called_without_the_route_class_fails_closed(enforced, spies):
    with pytest.raises(HTTPException) as caught:
        routes.chat(routes.ChatRequest(question=QUESTION), None, None)
    assert caught.value.status_code == 503
    assert spies.asks == [] and spies.quota_engine == 0


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------

def test_refusals_never_log_cookies_codes_or_code_ids(enforced, spies, caplog):
    caplog.set_level(logging.DEBUG)
    client = TestClient(app.main.app)
    _, value = cookie_for(enforced, CODE_ALICE)
    send(client, "chat", {**SAME_ORIGIN, **forged(enforced)})
    send(client, "chat", {**cookie_header(enforced, CODE_ALICE), "origin": "https://evil.example"})
    send(client, "chat", {**SAME_ORIGIN, **cookie_header(enforced, CODE_ALICE)})
    text = caplog.text
    for secret in (value, value.split(".")[1], CODE_ALICE, SECRET, access.hash_code(CODE_ALICE)):
        assert secret not in text
    assert "alice" not in text
    assert "access required (cookie_invalid)" in text
    assert "origin not allowed" in text


# ---------------------------------------------------------------------------
# Startup validation of PUBLIC_ORIGINS
# ---------------------------------------------------------------------------

def settings_with(**values):
    return IsolatedSettings(app_env="testing", **values)


@pytest.mark.parametrize("value", ["", "http://localhost:8000", "https://a.example,https://b.example:8443",
                                   " https://a.example , http://127.0.0.1:8000 ", "http://[::1]:8000"])
def test_valid_public_origins_start(value):
    settings_with(public_origins=value)


@pytest.mark.parametrize("value", [
    "*", "https://*.example", "https://a.example/", "https://a.example/path", "a.example", "//a.example",
    "ftp://a.example", "https://user@a.example", "https://a.example?x=1", "https://a.example#x",
    "https://a.example:0", "https://a.example:65536", "https://a.example:0443", "https://a.example.",
    "https://a..example", "https://a_b.example", "null", "https://a.example,", ",".join(["https://a.example"] * 11),
    "HTTPS://a.example", "https://[zz::1]",
])
def test_malformed_public_origins_refuse_to_start(value):
    with pytest.raises(ValidationError):
        settings_with(public_origins=value)


def test_production_requires_https_origins():
    with pytest.raises(ValidationError):
        IsolatedSettings(app_env="production", public_origins="http://a.example")
    IsolatedSettings(app_env="production", public_origins="https://a.example")


# ---------------------------------------------------------------------------
# access_policy units
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [
    ("https://a.example", ("https", "a.example", 443)),
    ("https://A.Example:443", ("https", "a.example", 443)),
    ("http://a.example:80", ("http", "a.example", 80)),
    ("http://localhost:8000", ("http", "localhost", 8000)),
    ("http://127.0.0.1:8000", ("http", "127.0.0.1", 8000)),
    ("http://[::1]:8000", ("http", "[::1]", 8000)),
    (" https://a.example\t", ("https", "a.example", 443)),
    ("null", None), ("", None), (None, None), ("https://a.example/", None), ("https://a.example:", None),
    ("https://256.1.1.1", None), ("https://a.example:99999", None), ("x" * 400, None),
])
def test_parse_origin(value, expected):
    assert access_policy.parse_origin(value) == expected


def fake_request(headers):
    raw = [(name.lower().encode("latin-1"), value.encode("latin-1")) for name, value in headers]
    return Request({"type": "http", "method": "POST", "path": "/", "headers": raw, "query_string": b""})


def test_cookie_values_count_every_cookie_header():
    request = fake_request([("cookie", "a=1; b=2"), ("cookie", "a=3"), ("cookie", "ab=4; =5; a")])
    assert access_policy._cookie_values(request, "a") == ["1", "3"]
    assert access_policy._cookie_values(request, "c") == []


def test_evaluate_is_unavailable_before_it_looks_at_cookies(settings):
    request = fake_request([("cookie", "__Host-bp_access=anything")])
    result = access_policy.evaluate(request, settings)
    assert (result.outcome, result.reason, result.grant) == (access_policy.UNAVAILABLE, "access_disabled", None)
    assert not result.granted


# ---------------------------------------------------------------------------
# This module runs without the legacy compatibility shim
# ---------------------------------------------------------------------------

def test_this_module_runs_with_real_enforcement():
    from tests.conftest import LEGACY_ACCESS_MODULES

    assert __name__ not in LEGACY_ACCESS_MODULES
    assert "tests.test_route_inventory" not in LEGACY_ACCESS_MODULES
    for name in ("evaluate", "configuration_problem", "origin_allowed"):
        assert getattr(access_policy, name).__module__ == "app.services.access_policy", name
        assert getattr(access_policy, name).__qualname__ == name, name
