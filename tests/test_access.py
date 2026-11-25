"""Security S1-3: access codes and signed access cookies (no network).

Codes are redeemed at POST /api/access for an HttpOnly, SameSite=Strict
cookie (``__Host-bp_access`` when Secure); GET /api/access/status and
POST /api/access/logout report and end access. No other route requires
access in this phase.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import threading
from datetime import datetime, timezone

import pytest
from pydantic import SecretStr, ValidationError
from starlette.datastructures import Headers
from starlette.requests import ClientDisconnect, Request
from starlette.responses import Response

from app.api import routes
from app.services import access
from tests.conftest import IsolatedSettings

REDEEM_URL = "/api/access"
STATUS_URL = "/api/access/status"
LOGOUT_URL = "/api/access/logout"
NOT_FOUND = {"detail": "Not Found"}
SECURE_NAME = access.SECURE_COOKIE_NAME  # "__Host-bp_access"

# High-entropy-format test values (never real ones).
CODE_ALICE = "Alice-test-code-0123456789abcdefGHIJ"
CODE_BOB = "Bob-test-code-0123456789abcdefghijKL"
CODE_NEW = "Fresh-test-code-0123456789abcdefGHIJK"
SECRET = "Q7v-test-secret-9fK2pX4mZ8rT1yW6nB3cJ5hL0dSxE4uA"  # 48 characters
OLD_SECRET = "Old-test-secret-4tG8kR2wP6vN1xZ9mQ3bH7jF5yC0eW2"
NOW = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc).timestamp()
JSON = {"content-type": "application/json"}
WRONG = "Wrong-test-code-0123456789abcdefGHIJ"


def entry(code_id, code, expires=None):
    return f"{code_id}:{access.hash_code(code)}" + (f":{expires}" if expires else "")


@pytest.fixture
def clock(monkeypatch):
    """Controls both clocks: wall time (cookies, code expiry) and the throttle's monotonic time."""

    class Clock:
        wall = NOW
        mono = 1_000.0

        def advance(self, seconds):
            self.wall += seconds
            self.mono += seconds

    state = Clock()
    monkeypatch.setattr(access, "wall_clock", lambda: state.wall)
    monkeypatch.setattr(access.time, "monotonic", lambda: state.mono)
    return state


@pytest.fixture
def access_on(monkeypatch, settings, clock):
    monkeypatch.setattr(settings, "access_enabled", True)
    monkeypatch.setattr(
        settings, "access_code_hashes", f"{entry('alice', CODE_ALICE)},{entry('bob', CODE_BOB, '2026-11-20')}"
    )
    monkeypatch.setattr(settings, "access_cookie_secret", SecretStr(SECRET))
    monkeypatch.setattr(settings, "access_cookie_previous_secrets", SecretStr(""))
    monkeypatch.setattr(settings, "access_cookie_secure", True)
    monkeypatch.setattr(settings, "access_client_ip_header", "")
    monkeypatch.setattr(settings, "access_trust_socket_peer", False)
    return settings


def redeem(client, code=CODE_ALICE, headers=None, **kwargs):
    return client.post(REDEEM_URL, content=json.dumps({"code": code}), headers={**JSON, **(headers or {})}, **kwargs)


def cookie_from(response, name=SECURE_NAME) -> str:
    header = response.headers["set-cookie"]
    assert header.startswith(f"{name}=")
    return header.split(";", 1)[0].split("=", 1)[1]


def status(client, cookie=None, name=SECURE_NAME):
    headers = {"cookie": f"{name}={cookie}"} if cookie is not None else {}
    return client.get(STATUS_URL, headers=headers)


def config_for(settings):
    return access.AccessConfig.from_settings(settings)


def valid_settings(**overrides):
    values = dict(
        access_enabled=True,
        access_code_hashes=entry("alice", CODE_ALICE),
        access_cookie_secret=SECRET,
    )
    values.update(overrides)
    return IsolatedSettings(**values)


def signed_cookie(payload: dict, secret: str = SECRET) -> str:
    """A cookie with any payload, correctly signed: tests the payload checks behind the signature."""
    encoded = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()).rstrip(b"=").decode()
    message = f"v{access.COOKIE_VERSION}.".encode() + encoded.encode()
    signature = hmac.new(secret.encode(), message, hashlib.sha256).digest()
    return f"{encoded}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode()}"


def payload_of(cookie: str) -> dict:
    encoded = cookie.split(".")[0]
    return json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))


# ---------------------------------------------------------------------------
# Disabled by default
# ---------------------------------------------------------------------------

def test_access_is_disabled_by_default(settings):
    from app.core import config

    assert config.Settings.model_fields["access_enabled"].default is False
    assert config.Settings.model_fields["access_trust_socket_peer"].default is False
    assert IsolatedSettings().access_enabled is False
    assert settings.access_enabled is False


def test_disabled_routes_are_404_like_unknown_routes(client):
    unknown = client.get("/api/no-such-route")
    for method, url in (("POST", REDEEM_URL), ("GET", STATUS_URL), ("POST", LOGOUT_URL)):
        response = client.request(method, url, content=b"{}", headers=JSON)
        assert (response.status_code, response.json()) == (404, NOT_FOUND), url
        assert response.json() == unknown.json()
        assert "set-cookie" not in response.headers


# ---------------------------------------------------------------------------
# Configuration: never active with missing, weak, insecure or ambiguous settings
# ---------------------------------------------------------------------------

def test_valid_configurations_are_accepted():
    assert valid_settings().access_enabled is True  # development: the socket peer by default
    assert valid_settings(app_env="production", access_trust_socket_peer=True).access_enabled is True
    assert valid_settings(app_env="production", access_client_ip_header="Do-Connecting-IP").access_client_ip_header
    assert valid_settings(app_env="development", access_cookie_secure=False).access_cookie_secure is False


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(access_cookie_secret=""), "ACCESS_COOKIE_SECRET"),
        (dict(access_cookie_secret="short-secret"), "ACCESS_COOKIE_SECRET"),
        (dict(access_cookie_secret="Q7v-test-secret-9fK2pX4mZ8rT1yW6nB3cJ5h"), "ACCESS_COOKIE_SECRET"),  # 39 chars
        (dict(access_cookie_secret="a" * 64), "ACCESS_COOKIE_SECRET"),  # long but not random
        (dict(access_cookie_secret="abababababcdcdcdcdcdefefefefghghghghijijijijk"), "ACCESS_COOKIE_SECRET"),
        (dict(access_code_hashes=""), "at least one code"),
        (dict(access_code_hashes="alice"), "id:sha256"),
        (dict(access_code_hashes="Alice:" + "0" * 64), "code id"),
        (dict(access_code_hashes="alice:" + "A" * 64), "digest"),
        (dict(access_code_hashes="alice:" + "0" * 63), "digest"),
        (dict(access_code_hashes="alice:" + "0" * 64 + ":2026-13-01"), "YYYY-MM-DD"),
        (dict(access_code_hashes=f"alice:{'0' * 64},alice:{'1' * 64}"), "unique"),
        (dict(access_code_hashes=f"alice:{'0' * 64},bob:{'0' * 64}"), "unique"),
        (dict(access_cookie_previous_secrets="weak"), "ACCESS_COOKIE_PREVIOUS_SECRETS"),
        (dict(access_cookie_previous_secrets=SECRET), "must not repeat"),
        # Insecure cookies only for local development.
        (dict(app_env="production", access_cookie_secure=False, access_trust_socket_peer=True), "APP_ENV=development"),
        (dict(app_env="testing", access_cookie_secure=False), "APP_ENV=development"),
        # Production needs an explicit, trusted client identity.
        (dict(app_env="production"), "trusted client identity"),
        (dict(access_client_ip_header="do-connecting-ip", access_trust_socket_peer=True), "not both"),
        (dict(access_client_ip_header="X-Forwarded-For"), "forge"),
        (dict(access_client_ip_header="forwarded"), "forge"),
        (dict(access_client_ip_header="bad header!"), "header name"),
    ],
)
def test_invalid_configurations_refuse_to_start(overrides, message):
    with pytest.raises(ValidationError, match=message):
        valid_settings(**overrides)


@pytest.mark.parametrize("value", ["1", "yes", "on", "", 1])
def test_access_flags_accept_only_true_or_false(value):
    for name in ("access_enabled", "access_cookie_secure", "access_trust_socket_peer"):
        with pytest.raises(ValidationError, match="must be true or false"):
            IsolatedSettings(**{name: value})


def test_access_limits_are_bounded():
    for name, value in (
        ("access_cookie_max_age_seconds", 299),
        ("access_cookie_max_age_seconds", 30 * 24 * 3600 + 1),
        ("access_max_failed_attempts", 0),
        ("access_failure_window_seconds", 59),
        ("access_max_global_failed_attempts", 0),
    ):
        with pytest.raises(ValidationError):
            IsolatedSettings(**{name: value})


def test_disabled_access_ignores_unset_values():
    # The suite's own settings carry placeholder secrets; while disabled they are never used.
    assert IsolatedSettings(access_code_hashes="", access_cookie_secret="x").access_enabled is False


def test_settings_weakened_at_runtime_fail_closed(client, access_on, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    for name, value in (
        ("access_cookie_secret", SecretStr("weak")),
        ("app_env", "production"),  # production without a trusted client identity
    ):
        with monkeypatch.context() as patch:
            patch.setattr(access_on, name, value)
            for response in (redeem(client), status(client)):
                assert response.status_code == 503, name
                assert response.json()["detail"]["code"] == "access_unavailable"
                assert response.headers["cache-control"] == "no-store"
                assert "set-cookie" not in response.headers
    assert "weak" not in caplog.text


# ---------------------------------------------------------------------------
# Codes
# ---------------------------------------------------------------------------

def test_generated_codes_are_random_and_high_entropy():
    codes = {access.generate_code() for _ in range(1000)}
    assert len(codes) == 1000
    for code in list(codes)[:50]:
        assert access.CODE_PATTERN.fullmatch(code)
        assert len(base64.urlsafe_b64decode(code + "=" * (-len(code) % 4))) == 24  # 192 bits


def test_codes_match_only_when_configured_and_unexpired(access_on, clock):
    config = config_for(access_on)

    assert access.match_code(config, CODE_ALICE, clock.wall).code_id == "alice"
    assert access.match_code(config, CODE_BOB, clock.wall).code_id == "bob"
    for candidate in (WRONG, "short", "x" * 129, "space in code 0123456789abcdefgh", None, 42):
        assert access.match_code(config, candidate, clock.wall) is None
    end = datetime(2026, 11, 21, tzinfo=timezone.utc).timestamp()  # Bob: valid through 2026-11-20 UTC
    assert access.match_code(config, CODE_BOB, end - 1) is not None
    assert access.match_code(config, CODE_BOB, end) is None


def test_only_digests_are_configured(access_on):
    assert CODE_ALICE not in access_on.access_code_hashes
    assert hashlib.sha256(CODE_ALICE.encode()).hexdigest() in access_on.access_code_hashes


# ---------------------------------------------------------------------------
# Redemption
# ---------------------------------------------------------------------------

def _attributes(set_cookie: str) -> list[str]:
    return [part.strip().lower() for part in set_cookie.split(";")[1:]]


def test_a_valid_code_sets_a_host_prefixed_strict_http_only_cookie(client, access_on, clock):
    response = redeem(client)

    assert response.status_code == 200
    assert response.json() == {"authenticated": True, "expires_at": "2026-11-25T12:00:00Z"}
    assert response.headers["cache-control"] == "no-store"
    header = response.headers["set-cookie"]
    assert header.startswith("__Host-bp_access=")
    attributes = _attributes(header)
    for required in ("httponly", "secure", "samesite=strict", "path=/", f"max-age={7 * 24 * 3600}"):
        assert required in attributes, required
    assert any(part.startswith("expires=") for part in attributes)
    assert not any(part.startswith("domain") for part in attributes)  # required for __Host-
    assert CODE_ALICE not in response.text and "alice" not in response.text


def test_plain_http_cookies_only_for_local_development(client, access_on, monkeypatch):
    monkeypatch.setattr(access_on, "app_env", "development")
    monkeypatch.setattr(access_on, "access_cookie_secure", False)
    response = redeem(client)
    header = response.headers["set-cookie"]
    assert header.startswith("bp_access=")  # a __Host- cookie must be Secure, so the plain name is used
    attributes = _attributes(header)
    assert "secure" not in attributes and "httponly" in attributes and "samesite=strict" in attributes
    cookie = cookie_from(response, name="bp_access")
    assert status(client, cookie, name="bp_access").json()["authenticated"] is True
    assert status(client, cookie, name=SECURE_NAME).json() == {"authenticated": False}


def test_a_code_expiring_soon_caps_the_cookie(client, access_on):
    assert redeem(client, CODE_BOB).json()["expires_at"] == "2026-11-21T00:00:00Z"


@pytest.mark.parametrize("code", [WRONG, "short", "x" * 200, "has spaces in it 0123456789abcdefgh", ""])
def test_invalid_codes_get_one_generic_answer(client, access_on, code):
    response = redeem(client, code)

    assert response.status_code == 401
    assert response.json() == {"detail": {"code": "access_denied", "message": "Invalid or expired access code."}}
    assert "set-cookie" not in response.headers
    assert response.headers["cache-control"] == "no-store"


def test_expired_and_revoked_codes_are_refused(client, access_on, clock, monkeypatch):
    clock.advance(3 * 24 * 3600)  # past Bob's expiry
    assert redeem(client, CODE_BOB).status_code == 401
    monkeypatch.setattr(access_on, "access_code_hashes", entry("bob", CODE_BOB))  # Alice revoked
    assert redeem(client, CODE_ALICE).status_code == 401


@pytest.mark.parametrize(
    "body, content_type, status_code, code",
    [
        (b'{"code": "x"}', "text/plain", 415, "unsupported_media_type"),
        (b"not json", "application/json", 400, "invalid_request"),
        (b'["code"]', "application/json", 400, "invalid_request"),
        (b'{"code": 1}', "application/json", 400, "invalid_request"),
        (b'{"code": "x", "extra": 1}', "application/json", 400, "invalid_request"),
        (b'{"code": "' + b"a" * 600 + b'"}', "application/json", 400, "invalid_request"),
    ],
)
def test_malformed_redemptions_are_refused(client, access_on, body, content_type, status_code, code):
    response = client.post(REDEEM_URL, content=body, headers={"content-type": content_type})
    assert (response.status_code, response.json()["detail"]["code"]) == (status_code, code)
    assert "set-cookie" not in response.headers


def test_an_oversized_body_is_not_read_beyond_the_limit(client, access_on, monkeypatch):
    seen = []
    real = routes._read_body_up_to

    async def spy(request, limit):
        result = await real(request, limit)
        seen.append((limit, result))
        return result

    monkeypatch.setattr(routes, "_read_body_up_to", spy)
    client.post(REDEEM_URL, content=b"{" + b" " * 100_000 + b"}", headers=JSON)
    assert seen == [(access.MAX_REDEEM_BODY_BYTES, None)]


def test_a_client_that_disconnects_mid_body_still_uses_an_attempt(access_on):
    # The attempt is reserved before the body is read, so hanging up can't be
    # used to guess without being counted.
    async def disconnect():
        return {"type": "http.disconnect"}

    scope = {
        "type": "http", "method": "POST", "path": REDEEM_URL, "query_string": b"",
        "headers": [(b"content-type", b"application/json"), (b"host", b"testserver")],
        "client": ("198.51.100.4", 4321), "server": ("testserver", 80), "scheme": "http",
    }
    with pytest.raises(ClientDisconnect):
        asyncio.run(routes.redeem_access(Request(scope, disconnect), Response()))
    throttle = access.get_throttle(access_on)
    assert len(throttle._clients["198.51.100.4"]) == 1
    assert len(throttle._global) == 1


@pytest.mark.parametrize(
    "headers",
    [{"sec-fetch-site": "cross-site"}, {"sec-fetch-site": "same-site"}, {"origin": "https://evil.example"}, {"origin": "null"}],
)
def test_cross_site_redemptions_and_logouts_are_refused_and_not_counted(client, access_on, headers):
    for _ in range(10):
        response = redeem(client, headers=headers)
        assert (response.status_code, response.json()["detail"]["code"]) == (403, "origin_not_allowed")
    assert client.post(LOGOUT_URL, headers=headers).status_code == 403
    assert redeem(client).status_code == 200  # a cross-site page can't lock its visitor out


# ---------------------------------------------------------------------------
# Brute-force protection
# ---------------------------------------------------------------------------

def test_repeated_failures_are_throttled_even_for_a_valid_code(client, access_on, clock):
    for _ in range(access_on.access_max_failed_attempts):
        assert redeem(client, WRONG).status_code == 401

    blocked = redeem(client)  # the right code, while throttled: no oracle
    assert blocked.status_code == 429
    assert blocked.json()["detail"]["code"] == "access_throttled"
    assert 1 <= int(blocked.headers["retry-after"]) <= 900
    assert "set-cookie" not in blocked.headers

    clock.advance(access_on.access_failure_window_seconds + 1)  # the window expires
    assert redeem(client).status_code == 200


def test_a_success_never_erases_earlier_failures(client, access_on):
    limit = access_on.access_max_failed_attempts
    for _ in range(limit - 1):
        assert redeem(client, WRONG).status_code == 401
    assert redeem(client).status_code == 200  # releases only its own attempt
    assert redeem(client, WRONG).status_code == 401  # the last allowed failure
    assert redeem(client).status_code == 429  # the earlier failures still count


def test_malformed_attempts_count_as_failures(client, access_on):
    for _ in range(access_on.access_max_failed_attempts):
        client.post(REDEEM_URL, content=b"junk", headers=JSON)
    assert redeem(client).status_code == 429


def test_clients_are_throttled_separately_with_a_trusted_platform_header(client, access_on, monkeypatch):
    monkeypatch.setattr(access_on, "access_client_ip_header", "do-connecting-ip")
    for _ in range(access_on.access_max_failed_attempts):
        redeem(client, WRONG, headers={"do-connecting-ip": "203.0.113.7"})
    assert redeem(client, headers={"do-connecting-ip": "203.0.113.7"}).status_code == 429
    assert redeem(client, headers={"do-connecting-ip": "198.51.100.9"}).status_code == 200


@pytest.mark.parametrize("value", [None, "", "not-an-ip", "203.0.113.7, 198.51.100.9", "203.0.113.7 x"])
def test_an_untrustworthy_client_address_fails_closed(client, access_on, monkeypatch, value):
    monkeypatch.setattr(access_on, "access_client_ip_header", "do-connecting-ip")
    headers = {} if value is None else {"do-connecting-ip": value}
    response = redeem(client, headers=headers)
    assert (response.status_code, response.json()["detail"]["code"]) == (503, "access_unavailable")
    assert "set-cookie" not in response.headers


def test_a_repeated_identity_header_fails_closed(client, access_on, monkeypatch):
    # A platform that appends instead of overwriting would let a client add
    # its own value; two copies are never trusted, whichever comes first.
    monkeypatch.setattr(access_on, "access_client_ip_header", "do-connecting-ip")
    response = client.post(
        REDEEM_URL,
        content=json.dumps({"code": CODE_ALICE}),
        headers=[("content-type", "application/json"), ("do-connecting-ip", "198.51.100.1"), ("do-connecting-ip", "203.0.113.9")],
    )
    assert (response.status_code, response.json()["detail"]["code"]) == (503, "access_unavailable")


def test_client_keys_group_ipv6_by_64_and_refuse_duplicates():
    key = lambda value: access.client_key(None, Headers({"x-ip": value}), "x-ip")  # noqa: E731
    assert key("2001:db8:1:2::1") == key("2001:db8:1:2:ffff:ffff:ffff:ffff") == "2001:db8:1:2::/64"
    assert key("2001:db8:1:3::1") != key("2001:db8:1:2::1")  # another /64
    assert key("::ffff:203.0.113.5") == key("203.0.113.5") == "203.0.113.5"  # IPv4-mapped
    assert access.client_key("2001:db8::7", Headers({}), "") == "2001:db8::/64"  # the socket peer too
    duplicated = Headers(raw=[(b"x-ip", b"198.51.100.1"), (b"x-ip", b"203.0.113.9")])
    assert access.client_key("edge", duplicated, "x-ip") is None


def test_ipv6_clients_in_one_64_share_a_throttle(client, access_on, monkeypatch):
    monkeypatch.setattr(access_on, "access_client_ip_header", "do-connecting-ip")
    for index in range(access_on.access_max_failed_attempts):
        redeem(client, WRONG, headers={"do-connecting-ip": f"2001:db8:5:6::{index + 1}"})
    assert redeem(client, headers={"do-connecting-ip": "2001:db8:5:6::abcd"}).status_code == 429
    assert redeem(client, headers={"do-connecting-ip": "2001:db8:5:7::1"}).status_code == 200


def test_a_shared_socket_address_is_one_client(client, access_on):
    # Behind an edge that shows one address for everyone, every visitor is the
    # same client: why production must configure a trusted identity first.
    for _ in range(access_on.access_max_failed_attempts):
        redeem(client, WRONG)
    assert redeem(client).status_code == 429


def test_forwarded_headers_are_never_trusted(client, access_on):
    for index in range(access_on.access_max_failed_attempts):
        redeem(client, WRONG, headers={"x-forwarded-for": f"203.0.113.{index}"})
    assert redeem(client, headers={"x-forwarded-for": "198.51.100.1"}).status_code == 429


def test_a_global_limit_stops_distributed_guessing(client, access_on, monkeypatch):
    monkeypatch.setattr(access_on, "access_client_ip_header", "do-connecting-ip")
    monkeypatch.setattr(access_on, "access_max_global_failed_attempts", 6)
    for index in range(6):
        redeem(client, WRONG, headers={"do-connecting-ip": f"203.0.113.{index}"})
    assert redeem(client, headers={"do-connecting-ip": "198.51.100.200"}).status_code == 429


def test_concurrent_attempts_cannot_exceed_the_limit():
    throttle = access.FailureThrottle(5, 900, 1000, clock=lambda: 1.0)
    results, barrier = [], threading.Barrier(50)

    def attempt():
        barrier.wait(5)
        results.append(throttle.attempt("one-client")[0] is None)

    threads = [threading.Thread(target=attempt) for _ in range(50)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert sum(results) == 5  # exactly the limit, never more


def test_concurrent_successes_and_failures_stay_exact():
    # Successes release only their own reservations, under the same lock.
    throttle = access.FailureThrottle(5, 900, 1000, clock=lambda: 1.0)
    reserved = [throttle.attempt("c")[1] for _ in range(5)]
    assert throttle.attempt("c")[0] is not None
    barrier = threading.Barrier(2)

    def release(stamp):
        barrier.wait(5)
        throttle.succeeded("c", stamp)

    threads = [threading.Thread(target=release, args=(stamp,)) for stamp in reserved[:2]]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert len(throttle._clients["c"]) == 3 and len(throttle._global) == 3
    assert throttle.attempt("c")[0] is None and throttle.attempt("c")[0] is None
    assert throttle.attempt("c")[0] is not None


def test_the_throttle_stays_bounded_and_forgets_after_the_window():
    now = [1.0]
    throttle = access.FailureThrottle(3, 60, 50, max_tracked=4, clock=lambda: now[0])
    for client in ("a", "b", "c", "d"):
        for _ in range(5):
            throttle.attempt(client)
    assert all(len(times) <= 3 for times in throttle._clients.values())  # per client
    assert len(throttle._global) <= 50  # in total
    assert throttle.attempt("e") == (60.0, None)  # can't track a fifth client: fail closed
    now[0] += 61  # the window passes
    assert throttle.attempt("e")[0] is None
    assert set(throttle._clients) == {"e"}  # expired clients were dropped


# ---------------------------------------------------------------------------
# Cookies: status, tampering, payload types, expiry, revocation, rotation, logout
# ---------------------------------------------------------------------------

def test_status_reports_a_valid_cookie(client, access_on):
    cookie = cookie_from(redeem(client))

    response = status(client, cookie)
    assert response.json() == {"authenticated": True, "expires_at": "2026-11-25T12:00:00Z"}
    assert response.headers["cache-control"] == "no-store"
    assert status(client).json() == {"authenticated": False}
    assert status(client, cookie, name="bp_access").json() == {"authenticated": False}  # wrong name


def test_tampered_and_malformed_cookies_are_rejected(client, access_on):
    cookie = cookie_from(redeem(client))
    encoded, signature = cookie.split(".")
    payload = payload_of(cookie)
    forged = signed_cookie({**payload, "cid": "bob"}, "Forged-test-secret-0123456789abcdefghABCDEFG")
    flipped = encoded[:-1] + ("A" if encoded[-1] != "A" else "B") + "." + signature

    for value in (forged, flipped, signature, cookie + "x", "", "a.b.c", "not base64!.x", "x" * 600):
        assert status(client, value).json() == {"authenticated": False}, value[:40]


@pytest.mark.parametrize(
    "change",
    [
        {"v": True},  # a bool is not the version number
        {"v": 1},  # an older format
        {"v": "2"},
        {"iat": float(int(NOW))},
        {"exp": str(int(NOW) + 3600)},
        {"exp": True},
        {"cid": "Alice"},
        {"cid": 7},
        {"cfp": "short"},
        {"cfp": None},
        {"extra": 1},
    ],
)
def test_signed_cookies_with_wrong_payload_types_are_rejected(client, access_on, change):
    # Correctly signed (as if the secret had leaked into a test): the payload
    # checks behind the signature must still hold.
    good = payload_of(cookie_from(redeem(client)))
    assert status(client, signed_cookie(good)).json()["authenticated"] is True  # the baseline is valid
    assert status(client, signed_cookie({**good, **change})).json() == {"authenticated": False}


def test_a_signed_cookie_without_the_code_fingerprint_is_rejected(client, access_on):
    good = payload_of(cookie_from(redeem(client)))
    del good["cfp"]
    assert status(client, signed_cookie(good)).json() == {"authenticated": False}


def test_a_longer_than_allowed_signed_cookie_is_rejected(client, access_on):
    good = payload_of(cookie_from(redeem(client)))
    assert status(client, signed_cookie({**good, "exp": good["exp"] + 30 * 24 * 3600})).json() == {"authenticated": False}


def test_expired_cookies_are_rejected(client, access_on, clock):
    cookie = cookie_from(redeem(client))
    clock.advance(7 * 24 * 3600 - 1)
    assert status(client, cookie).json()["authenticated"] is True
    clock.advance(1)
    assert status(client, cookie).json() == {"authenticated": False}


def test_removing_a_code_invalidates_its_cookies(client, access_on, monkeypatch):
    alice, bob = cookie_from(redeem(client)), cookie_from(redeem(client, CODE_BOB))
    monkeypatch.setattr(access_on, "access_code_hashes", entry("bob", CODE_BOB, "2026-11-20"))
    assert status(client, alice).json() == {"authenticated": False}
    assert status(client, bob).json()["authenticated"] is True


def test_replacing_a_code_under_the_same_id_invalidates_its_cookies(client, access_on, monkeypatch):
    leaked = cookie_from(redeem(client))  # issued from CODE_ALICE
    monkeypatch.setattr(access_on, "access_code_hashes", f"{entry('alice', CODE_NEW)},{entry('bob', CODE_BOB)}")

    assert status(client, leaked).json() == {"authenticated": False}  # the old code's cookie is dead
    assert redeem(client, CODE_ALICE).status_code == 401  # and so is the old code
    fresh = cookie_from(redeem(client, CODE_NEW))
    assert status(client, fresh).json()["authenticated"] is True

    # Restoring the original code (same digest) makes its cookies valid again: bound to the code itself.
    monkeypatch.setattr(access_on, "access_code_hashes", f"{entry('alice', CODE_ALICE)},{entry('bob', CODE_BOB)}")
    assert status(client, leaked).json()["authenticated"] is True
    assert status(client, fresh).json() == {"authenticated": False}


def test_an_expiry_added_later_applies_to_existing_cookies(client, access_on, clock, monkeypatch):
    cookie = cookie_from(redeem(client))
    monkeypatch.setattr(access_on, "access_code_hashes", entry("alice", CODE_ALICE, "2026-11-19"))
    assert status(client, cookie).json()["authenticated"] is True
    clock.advance(36 * 3600)  # 2026-11-20 00:00 UTC
    assert status(client, cookie).json() == {"authenticated": False}


def test_shortening_the_cookie_lifetime_rejects_longer_cookies(client, access_on, monkeypatch):
    cookie = cookie_from(redeem(client))
    monkeypatch.setattr(access_on, "access_cookie_max_age_seconds", 3600)
    assert status(client, cookie).json() == {"authenticated": False}


def test_secret_rotation_and_code_binding_together(client, access_on, monkeypatch):
    monkeypatch.setattr(access_on, "access_cookie_secret", SecretStr(OLD_SECRET))
    old_cookie = cookie_from(redeem(client))

    # Rotate: the new secret signs; the old one only verifies, until removed.
    monkeypatch.setattr(access_on, "access_cookie_secret", SecretStr(SECRET))
    monkeypatch.setattr(access_on, "access_cookie_previous_secrets", SecretStr(OLD_SECRET))
    assert status(client, old_cookie).json()["authenticated"] is True
    new_cookie = cookie_from(redeem(client))
    assert new_cookie.split(".")[1] == access._signature(SECRET.encode(), new_cookie.split(".")[0])

    # A code replaced during rotation invalidates cookies signed with either secret.
    monkeypatch.setattr(access_on, "access_code_hashes", entry("alice", CODE_NEW))
    assert status(client, old_cookie).json() == {"authenticated": False}
    assert status(client, new_cookie).json() == {"authenticated": False}

    monkeypatch.setattr(access_on, "access_code_hashes", entry("alice", CODE_ALICE))
    monkeypatch.setattr(access_on, "access_cookie_previous_secrets", SecretStr(""))
    assert status(client, old_cookie).json() == {"authenticated": False}  # old secret removed
    assert status(client, new_cookie).json()["authenticated"] is True


def test_a_cookie_signed_with_an_unknown_secret_is_rejected(client, access_on):
    good = payload_of(cookie_from(redeem(client)))
    assert status(client, signed_cookie(good, OLD_SECRET)).json() == {"authenticated": False}


def test_logout_deletes_both_cookie_names(client, access_on):
    cookie = cookie_from(redeem(client))
    response = client.post(LOGOUT_URL, headers={"cookie": f"{SECURE_NAME}={cookie}"})

    assert response.status_code == 204
    assert response.headers["cache-control"] == "no-store"
    deletions = {header.split("=", 1)[0]: _attributes(header) for header in response.headers.get_list("set-cookie")}
    assert set(deletions) == {"__Host-bp_access", "bp_access"}
    for attributes in deletions.values():
        for required in ("max-age=0", "httponly", "samesite=strict", "path=/"):
            assert required in attributes
        assert not any(part.startswith("domain") for part in attributes)
    assert "secure" in deletions["__Host-bp_access"]  # browsers only accept a Secure __Host- cookie
    assert client.post(LOGOUT_URL).status_code == 204  # idempotent


# ---------------------------------------------------------------------------
# Nothing secret is logged
# ---------------------------------------------------------------------------

def test_no_code_cookie_secret_id_or_address_is_logged(client, access_on, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(access_on, "access_client_ip_header", "do-connecting-ip")
    ip = {"do-connecting-ip": "203.0.113.77"}

    cookie = cookie_from(redeem(client, headers=ip))
    status(client, cookie)
    for _ in range(access_on.access_max_failed_attempts + 1):
        redeem(client, WRONG, headers=ip)
    client.post(LOGOUT_URL)
    with monkeypatch.context() as patch:  # a misconfiguration at runtime
        patch.setattr(access_on, "access_cookie_secret", SecretStr("weak-secret-value"))
        redeem(client, headers=ip)

    text = caplog.text
    fingerprint = payload_of(cookie)["cfp"]
    for secret in (
        CODE_ALICE, WRONG, access.hash_code(CODE_ALICE), SECRET, "weak-secret-value",
        cookie, cookie.split(".")[1], fingerprint, "alice", "203.0.113.77",
    ):
        assert secret not in text, secret[:12]
    assert "Access granted: code_ref=" in text
    assert "Access code rejected (invalid or expired)." in text
    assert "Access code rejected (throttled)." in text
    assert "Access codes are enabled but their settings are invalid." in text


# ---------------------------------------------------------------------------
# Public routes are unchanged; paid AI routes need the full configuration (S1-5)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("enabled", [False, True])
def test_public_routes_stay_open_and_chat_is_refused_without_quotas(client, monkeypatch, settings, clock, enabled):
    # S1-3 enforced nothing; S1-5 refuses paid AI routes unless access, quotas
    # and PUBLIC_ORIGINS are all configured (tests/test_access_enforcement.py).
    calls = []
    monkeypatch.setattr(settings, "access_enabled", enabled)
    monkeypatch.setattr(settings, "access_code_hashes", entry("alice", CODE_ALICE))
    monkeypatch.setattr(settings, "access_cookie_secret", SecretStr(SECRET))
    monkeypatch.setattr(routes, "ask", lambda question, history=None: calls.append(question))

    assert client.get("/api/health").status_code == 200
    chat = client.post("/api/chat", json={"question": "How do I connect to the VPN?"})
    assert chat.status_code == 503 and chat.json()["detail"]["code"] == "access_unavailable"
    assert calls == []
    assert client.post("/api/chat/stream", json={"question": "VPN?"}).json() == NOT_FOUND
    assert client.post("/api/sessions", json={}).status_code == 404
    assert client.get("/api/voice/status").status_code == 404
    assert client.get("/api/voice/transcription/status").status_code == 404
    assert client.get("/").status_code == 200


# ---------------------------------------------------------------------------
# Operator commands
# ---------------------------------------------------------------------------

def test_new_code_prints_a_code_and_its_matching_entry(capsys):
    assert access._main(["new-code", "acme-recruiter", "--expires", "2026-12-31"]) == 0
    lines = capsys.readouterr().out.splitlines()
    code = lines[0].rsplit(" ", 1)[1]
    configured = lines[1].rsplit(" ", 1)[1]
    assert access.CODE_PATTERN.fullmatch(code)
    assert configured == f"acme-recruiter:{access.hash_code(code)}:2026-12-31"
    assert access.parse_code_entries(configured)[0].code_id == "acme-recruiter"


def test_new_secret_is_strong_enough(capsys):
    assert access._main(["new-secret"]) == 0
    secret = capsys.readouterr().out.strip()
    assert len(secret) == 64
    access._check_secret(secret, "ACCESS_COOKIE_SECRET")  # does not raise


@pytest.mark.parametrize("argv", [["new-code", "Bad Id"], ["new-code", "ok", "--expires", "31-12-2026"]])
def test_new_code_rejects_bad_input(argv):
    with pytest.raises(SystemExit):
        access._main(argv)
