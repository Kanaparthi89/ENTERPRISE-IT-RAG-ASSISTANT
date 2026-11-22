"""Security S1-5 Phase D1: origin checks on /api/access* and API docs off by default.

POST /api/access and /api/access/logout compare browser evidence (Origin,
Sec-Fetch-Site) with PUBLIC_ORIGINS only; the request's Host, forwarded
headers and Referer are never the authority. Headerless non-browser clients
are allowed. Both FastAPI entry points are checked.

Not in tests/conftest.py's LEGACY_ACCESS_MODULES: real enforcement.
"""

import ast
import importlib
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError
from starlette.testclient import TestClient

import app.api.main
import app.main
from app.core import config
from app.services import access

ROOT = Path(__file__).resolve().parents[1]
CODE = "D1-origin-code-0123456789abcdefGHIJKLMN"
SECRET = "D1-origin-secret-Zq8Wm3Kx7Pv2Ty9Lr4Nb6Hc1Jd5Fg0Se"
PUBLIC = "https://bluepeak.example"
APPS = {"app.main": lambda: app.main.app, "app.api.main": lambda: app.api.main.app}
DOCS_PATHS = ("/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect")


@pytest.fixture
def access_on(monkeypatch, settings):
    for name, value in (
        ("access_enabled", True),
        ("access_code_hashes", f"probe:{access.hash_code(CODE)}"),
        ("access_cookie_secret", SecretStr(SECRET)),
        ("access_cookie_previous_secrets", SecretStr("")),
        ("access_cookie_secure", True),
        ("access_client_ip_header", ""),
        ("access_trust_socket_peer", False),
        ("public_origins", PUBLIC),
    ):
        monkeypatch.setattr(settings, name, value)
    return settings


def redeem(entry, headers):
    return TestClient(APPS[entry]()).post("/api/access", json={"code": CODE}, headers=headers)


def logout(entry, headers):
    return TestClient(APPS[entry]()).post("/api/access/logout", headers=headers)


def refused(response):
    return response.status_code == 403 and response.json()["detail"]["code"] == "origin_not_allowed"


# ===========================================================================
# Accepted
# ===========================================================================

ACCEPTED = {
    "headerless non-browser": [],
    "public Origin, Host differs": [("origin", PUBLIC)],
    "public Origin with its default port": [("origin", PUBLIC + ":443")],
    "public Origin upper-case host": [("origin", "https://BluePeak.Example")],
    "Sec-Fetch-Site same-origin only": [("sec-fetch-site", "same-origin")],
    "both, consistent": [("origin", PUBLIC), ("sec-fetch-site", "same-origin")],
    "spoofed Host is irrelevant": [("host", "evil.example"), ("origin", PUBLIC)],
    "Referer alone is ignored (headerless)": [("referer", "https://evil.example/page")],
}


@pytest.mark.parametrize("entry", list(APPS))
@pytest.mark.parametrize("case", list(ACCEPTED))
def test_accepted_requests(access_on, entry, case):
    access.reset_throttle()
    assert redeem(entry, ACCEPTED[case]).status_code == 200
    assert logout(entry, ACCEPTED[case]).status_code == 204


# ===========================================================================
# Refused: cross-origin, malformed, duplicated or contradictory evidence
# ===========================================================================

REFUSED = {
    "evil Origin": [("origin", "https://evil.example")],
    "spoofed Host matching an evil Origin": [("host", "evil.example"), ("origin", "https://evil.example")],
    "X-Forwarded-Host matching an evil Origin": [("x-forwarded-host", "evil.example"), ("origin", "https://evil.example")],
    "Host equal to the public host, evil Origin": [("host", "bluepeak.example"), ("origin", "https://evil.example")],
    "the old Host-based match (Origin == Host)": [("origin", "http://testserver")],
    "public host over http": [("origin", "http://bluepeak.example")],
    "public host, other port": [("origin", PUBLIC + ":8443")],
    "Origin with a path": [("origin", PUBLIC + "/")],
    "Origin null": [("origin", "null")],
    "Origin empty": [("origin", "")],
    "Sec-Fetch-Site empty": [("sec-fetch-site", "")],
    "Sec-Fetch-Site cross-site": [("sec-fetch-site", "cross-site")],
    "Sec-Fetch-Site same-site": [("sec-fetch-site", "same-site")],
    "Sec-Fetch-Site none": [("sec-fetch-site", "none")],
    "two identical Origins": [("origin", PUBLIC), ("origin", PUBLIC)],
    "two different Origins": [("origin", PUBLIC), ("origin", "https://evil.example")],
    "two Sec-Fetch-Site": [("sec-fetch-site", "same-origin"), ("sec-fetch-site", "same-origin")],
    "contradiction: cross-site + public Origin": [("sec-fetch-site", "cross-site"), ("origin", PUBLIC)],
    "contradiction: same-origin + evil Origin": [("sec-fetch-site", "same-origin"), ("origin", "https://evil.example")],
    "Referer never grants": [("origin", "https://evil.example"), ("referer", PUBLIC + "/")],
}


@pytest.mark.parametrize("entry", list(APPS))
@pytest.mark.parametrize("case", list(REFUSED))
def test_refused_requests(access_on, entry, case):
    access.reset_throttle()
    assert refused(redeem(entry, REFUSED[case]))
    assert refused(logout(entry, REFUSED[case]))


@pytest.mark.parametrize("entry", list(APPS))
def test_an_origin_refusal_sets_no_cookie_and_uses_no_throttle_attempt(access_on, entry):
    access.reset_throttle()
    for _ in range(3 * access_on.access_max_failed_attempts):
        response = redeem(entry, [("origin", "https://evil.example")])
        assert refused(response) and "set-cookie" not in response.headers
    assert redeem(entry, []).status_code == 200  # the throttle was never consulted


# ===========================================================================
# PUBLIC_ORIGINS empty or invalid: browser evidence refused, non-browsers allowed
# ===========================================================================

@pytest.mark.parametrize("entry", list(APPS))
@pytest.mark.parametrize("configured", ["", "https://*.example"])
def test_without_valid_public_origins_browser_evidence_is_refused(access_on, monkeypatch, entry, configured):
    monkeypatch.setattr(access_on, "public_origins", configured)  # "*" only possible if changed at runtime
    for headers in ([("origin", PUBLIC)], [("sec-fetch-site", "same-origin")],
                    [("origin", PUBLIC), ("sec-fetch-site", "same-origin")]):
        access.reset_throttle()
        assert refused(redeem(entry, headers)), headers
        assert refused(logout(entry, headers)), headers
    access.reset_throttle()
    assert redeem(entry, []).status_code == 200
    assert logout(entry, []).status_code == 204


# ===========================================================================
# GET /api/access/status is unchanged (read-only, no origin check)
# ===========================================================================

@pytest.mark.parametrize("entry", list(APPS))
def test_access_status_is_unchanged(access_on, entry):
    client = TestClient(APPS[entry]())
    for headers in ({}, {"origin": "https://evil.example"}, {"sec-fetch-site": "cross-site"}):
        response = client.get("/api/access/status", headers=headers)
        assert response.status_code == 200 and response.json() == {"authenticated": False}
        assert response.headers["cache-control"] == "no-store"
        assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize("entry", list(APPS))
def test_access_routes_are_still_404_while_disabled(settings, entry):
    assert settings.access_enabled is False
    assert redeem(entry, [("origin", "https://evil.example")]).status_code == 404
    assert logout(entry, []).status_code == 404


# ===========================================================================
# API docs: off by default, on only by explicit opt-in
# ===========================================================================

@pytest.mark.parametrize("entry", list(APPS))
def test_docs_and_openapi_are_absent_by_default(settings, entry):
    assert settings.api_docs_enabled is False
    application = APPS[entry]()
    assert (application.docs_url, application.redoc_url, application.openapi_url) == (None, None, None)
    paths = {getattr(route, "path", None) for route in application.routes}
    assert not paths & set(DOCS_PATHS)
    client = TestClient(application)
    for path in DOCS_PATHS:
        assert client.get(path).status_code == 404, path


def test_the_docs_option_helper():
    on, off = config.Settings.model_construct(api_docs_enabled=True), config.Settings.model_construct(api_docs_enabled=False)
    assert config.api_docs_options(on) == {"docs_url": "/docs", "redoc_url": "/redoc", "openapi_url": "/openapi.json"}
    assert config.api_docs_options(off) == {"docs_url": None, "redoc_url": None, "openapi_url": None}
    assert config.Settings.model_fields["api_docs_enabled"].default is False


@pytest.mark.parametrize("value", ["yes", "1", "on", 1])
def test_the_docs_flag_is_strict(value):
    from tests.conftest import IsolatedSettings

    with pytest.raises(ValidationError):
        IsolatedSettings(api_docs_enabled=value)


def test_explicit_opt_in_enables_docs_on_both_entry_points(settings, monkeypatch):
    monkeypatch.setattr(settings, "api_docs_enabled", True)
    try:
        for module in (app.main, app.api.main):
            importlib.reload(module)  # the docs settings are read when the app is constructed
            client = TestClient(module.app)
            for path in DOCS_PATHS:
                assert client.get(path).status_code == 200, (module.__name__, path)
            assert "/api/ingest" in client.get("/openapi.json").json()["paths"]
    finally:
        monkeypatch.setattr(settings, "api_docs_enabled", False)
        for module in (app.main, app.api.main):
            importlib.reload(module)
    for module in (app.main, app.api.main):
        assert module.app.openapi_url is None


@pytest.mark.parametrize("module", ["app/main.py", "app/api/main.py"])
def test_both_entry_points_construct_with_the_docs_options(module):
    tree = ast.parse((ROOT / module).read_text())
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "FastAPI"]
    assert len(calls) == 1
    (call,) = calls
    spread = [kw.value for kw in call.keywords if kw.arg is None]
    assert len(spread) == 1 and ast.unparse(spread[0]) == "api_docs_options(settings)"
    assert not {kw.arg for kw in call.keywords} & {"docs_url", "redoc_url", "openapi_url"}
