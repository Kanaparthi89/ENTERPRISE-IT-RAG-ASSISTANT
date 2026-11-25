"""Security S2b: browser security headers on every response, in both entry points (D3-2).

app/core/security_headers.py wraps each app's whole middleware stack, outside
ServerErrorMiddleware, so the headers are on normal responses, early
rejections, validation errors, 404/405, redirects, static files, streaming
responses and unhandled-exception 500s. These tests pin the exact policies,
the forbidden sources, the stack order, pass-through streaming, Cache-Control
preservation, the development docs pages' explicit policies, and the
template/JS properties the strict CSP depends on.

Real-browser behaviour (no CSP violations in DevTools, fonts, dictation,
rendered Markdown) is not verified here.
"""

import asyncio
import base64
import hashlib
import importlib
import re
import threading
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr
from starlette.middleware.errors import ServerErrorMiddleware
from starlette.testclient import TestClient

import app.api.main
import app.main
from app.core import security_headers as sh
from app.services import access

ROOT = Path(__file__).resolve().parents[1]
APPS = {"app.main": lambda: app.main.app, "app.api.main": lambda: app.api.main.app}
MODULES = {"app.main": app.main, "app.api.main": app.api.main}
COMMON = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
    "permissions-policy": "microphone=(self), camera=(), geolocation=(), payment=(), usb=()",
    "cross-origin-opener-policy": "same-origin",
}
DOCUMENT_DIRECTIVES = {
    "default-src": ["'none'"],
    "script-src": ["'self'"],
    "style-src": ["'self'", "https://fonts.googleapis.com"],
    "font-src": ["https://fonts.gstatic.com"],
    "img-src": ["'self'"],
    "connect-src": ["'self'"],
    "worker-src": ["'self'"],
    "manifest-src": ["'none'"],
    "media-src": ["'none'"],
    "object-src": ["'none'"],
    "frame-src": ["'none'"],
    "base-uri": ["'none'"],
    "form-action": ["'self'"],
    "frame-ancestors": ["'none'"],
}
NON_DOCUMENT_DIRECTIVES = {"default-src": ["'none'"], "base-uri": ["'none'"],
                           "form-action": ["'none'"], "frame-ancestors": ["'none'"]}
FORBIDDEN_TOKENS = ("'unsafe-inline'", "'unsafe-eval'", "'unsafe-hashes'", "'strict-dynamic'", "*", "http:",
                    "https:", "data:", "blob:", "cdn.jsdelivr.net", "unpkg.com", "cdnjs")


def directives(policy: str) -> dict:
    parsed = {}
    for part in filter(None, (piece.strip() for piece in policy.split(";"))):
        name, *sources = part.split()
        assert name not in parsed, f"duplicate directive {name}"
        parsed[name] = sources
    return parsed


def assert_security_headers(response, *, document: bool):
    for name, value in COMMON.items():
        assert response.headers.get(name) == value, (name, response.headers.get(name))
    assert len(response.headers.get_list("content-security-policy")) == 1
    expected = DOCUMENT_DIRECTIVES if document else NON_DOCUMENT_DIRECTIVES
    assert directives(response.headers["content-security-policy"]) == expected
    assert "strict-transport-security" not in response.headers  # not before the edge's HTTPS is verified


# ===========================================================================
# The policies themselves
# ===========================================================================

def test_the_document_policy_is_exactly_the_approved_one():
    assert directives(sh.DOCUMENT_CSP) == DOCUMENT_DIRECTIVES


def test_the_non_document_policy_allows_nothing():
    assert directives(sh.NON_DOCUMENT_CSP) == NON_DOCUMENT_DIRECTIVES


@pytest.mark.parametrize("policy", [sh.DOCUMENT_CSP, sh.NON_DOCUMENT_CSP])
def test_no_policy_contains_a_weakening_source(policy):
    sources = [source for values in directives(policy).values() for source in values]
    for token in FORBIDDEN_TOKENS:
        assert token not in sources, token
    assert all(source.startswith(("'", "https://fonts.googleapis.com", "https://fonts.gstatic.com"))
               for source in sources), sources


def test_framing_is_refused_twice_and_hsts_is_absent():
    assert directives(sh.DOCUMENT_CSP)["frame-ancestors"] == ["'none'"]
    assert dict(sh.COMMON_HEADERS)["x-frame-options"] == "DENY"
    assert "strict-transport-security" not in dict(sh.COMMON_HEADERS)


def test_the_microphone_stays_available_to_this_origin_only():
    policy = dict(sh.COMMON_HEADERS)["permissions-policy"]
    features = dict(item.strip().split("=", 1) for item in policy.split(","))
    assert features == {"microphone": "(self)", "camera": "()", "geolocation": "()", "payment": "()", "usb": "()"}


# ===========================================================================
# Both entry points, every response type
# ===========================================================================

@pytest.mark.parametrize("entry", list(APPS))
@pytest.mark.parametrize("method, path, status, document", [
    ("GET", "/", 200, True),
    ("HEAD", "/", 405, False),                  # pre-existing: "/" is GET-only (FastAPI adds no HEAD)
    ("GET", "/api/health", 200, False),
    ("GET", "/api/health/", 307, False),        # trailing-slash redirect
    ("GET", "/static/js/app.js", 200, False),
    ("GET", "/static/vendor/dompurify-3.4.18/purify.min.js", 200, False),
    ("GET", "/static/css/style.css", 200, False),
    ("GET", "/no-such-page", 404, False),
    ("GET", "/api/chat", 405, False),
    ("POST", "/api/chat", 503, False),          # AccessEnforcedRoute: read-only mode, before the body
    ("POST", "/api/ingest", 503, False),        # AdminIngestRoute: no admin key
    ("GET", "/api/voice/status", None, False),
])
def test_every_response_type_carries_the_headers(entry, method, path, status, document):
    response = TestClient(APPS[entry]()).request(method, path, follow_redirects=False)
    if status is not None:
        assert response.status_code == status, (path, response.status_code)
    assert_security_headers(response, document=document)


def test_health_on_the_production_entry_point():
    response = TestClient(app.main.app).get("/health")
    assert response.status_code == 200
    assert_security_headers(response, document=False)


@pytest.fixture
def access_on(monkeypatch, settings):
    for name, value in (("access_enabled", True), ("access_code_hashes", f"probe:{access.hash_code('S2b-code-0123456789abcdefghijk')}"),
                        ("access_cookie_secret", SecretStr("S2b-secret-Zq8Wm3Kx7Pv2Ty9Lr4Nb6Hc1Jd5Fg0Se")),
                        ("access_cookie_previous_secrets", SecretStr("")), ("access_cookie_secure", True),
                        ("access_client_ip_header", ""), ("access_trust_socket_peer", False),
                        ("public_origins", "https://bluepeak.example")):
        monkeypatch.setattr(settings, name, value)
    access.reset_throttle()
    return settings


@pytest.mark.parametrize("entry", list(APPS))
def test_bad_requests_and_cache_control_are_preserved(access_on, entry):
    client = TestClient(APPS[entry]())
    invalid = client.post("/api/access", json={"not_code": 1})  # the route parses its own bounded body: 400
    assert invalid.status_code == 400
    assert_security_headers(invalid, document=False)
    status = client.get("/api/access/status")
    assert status.status_code == 200 and status.headers["cache-control"] == "no-store"  # untouched
    assert_security_headers(status, document=False)


@pytest.mark.parametrize("entry", list(APPS))
def test_request_validation_errors_carry_the_headers(entry):
    # A temporary probe route with a validated body exercises RequestValidationError
    # (app.api.main has its own handler for it); it is removed afterwards.
    from pydantic import BaseModel

    class Probe(BaseModel):
        number: int

    async def probe(body: Probe):
        return {"ok": True}

    application = APPS[entry]()
    before = list(application.router.routes)
    application.add_api_route("/__s2b_validation_probe", probe, methods=["POST"])
    try:
        response = TestClient(application).post("/__s2b_validation_probe", json={"number": "not a number"})
    finally:
        application.router.routes[:] = before
    assert response.status_code == 422
    assert_security_headers(response, document=False)
    assert "/__s2b_validation_probe" not in {getattr(route, "path", "") for route in application.router.routes}


# ===========================================================================
# Unhandled exceptions: the headers are outside ServerErrorMiddleware
# ===========================================================================

@pytest.mark.parametrize("entry", list(APPS))
def test_the_middleware_wraps_server_error_middleware(entry):
    application = APPS[entry]()
    assert getattr(application.build_middleware_stack, "security_headers_installed", False)
    stack = application.build_middleware_stack()
    assert isinstance(stack, sh.SecurityHeadersMiddleware)
    assert isinstance(stack.app, ServerErrorMiddleware)  # outermost: covers unhandled 500s


@pytest.mark.parametrize("entry, body_kind", [("app.main", "text/plain"), ("app.api.main", "application/json")])
def test_an_unhandled_exception_500_carries_the_headers(monkeypatch, entry, body_kind):
    def boom(*args, **kwargs):
        raise RuntimeError("S2b probe: unhandled")

    monkeypatch.setattr(MODULES[entry].templates, "TemplateResponse", boom)
    response = TestClient(APPS[entry](), raise_server_exceptions=False).get("/")
    assert response.status_code == 500
    assert response.headers["content-type"].startswith(body_kind)  # plain Starlette 500 / app.api.main's JSON handler
    assert "S2b probe" not in response.text  # no internal detail leaks
    assert_security_headers(response, document=False)


# ===========================================================================
# Pure ASGI behaviour (no buffering, pass-through, set-if-absent)
# ===========================================================================

def run_asgi(application, scope, receive_messages=()):
    sent = []
    queue = list(receive_messages) or [{"type": "http.request", "body": b"", "more_body": False}]

    async def receive():
        return queue.pop(0) if queue else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(application(scope, receive, send))
    return sent


HTTP_SCOPE = {"type": "http", "method": "GET", "path": "/x", "headers": [], "query_string": b""}


def test_body_messages_are_forwarded_immediately_not_buffered():
    first_chunk_seen = asyncio.Event

    async def streaming_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]})
        await send({"type": "http.response.body", "body": b"one", "more_body": True})
        await asyncio.wait_for(state["first"].wait(), timeout=2)  # deadlocks if the middleware buffered
        await send({"type": "http.response.body", "body": b"two", "more_body": False})

    state = {}
    sent = []

    async def main():
        state["first"] = first_chunk_seen()

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)
            if message.get("body") == b"one":
                state["first"].set()

        await sh.SecurityHeadersMiddleware(streaming_app)(HTTP_SCOPE, receive, send)

    asyncio.run(main())
    assert [message.get("body") for message in sent[1:]] == [b"one", b"two"]
    assert [message["more_body"] for message in sent[1:]] == [True, False]
    assert dict(sent[0]["headers"])[b"content-security-policy"] == sh.NON_DOCUMENT_CSP.encode()


def test_headers_a_response_sets_itself_are_kept_and_never_duplicated():
    async def custom(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": [
            (b"content-type", b"text/html; charset=utf-8"), (b"cache-control", b"no-store"),
            (b"X-Frame-Options", b"SAMEORIGIN"), (b"content-security-policy", b"default-src 'self'")]})
        await send({"type": "http.response.body", "body": b"<p>x</p>"})

    [start, _] = run_asgi(sh.SecurityHeadersMiddleware(custom), HTTP_SCOPE)
    names = [name.lower() for name, _ in start["headers"]]
    assert len(names) == len(set(names))  # no duplicates
    headers = {name.lower(): value for name, value in start["headers"]}
    assert headers[b"x-frame-options"] == b"SAMEORIGIN" and headers[b"content-security-policy"] == b"default-src 'self'"
    assert headers[b"cache-control"] == b"no-store"
    assert headers[b"x-content-type-options"] == b"nosniff"


def test_html_responses_get_the_document_policy_whatever_the_charset():
    async def html(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": [(b"Content-Type", b"TEXT/HTML; charset=utf-8")]})
        await send({"type": "http.response.body", "body": b""})

    [start, _] = run_asgi(sh.SecurityHeadersMiddleware(html), HTTP_SCOPE)
    assert dict(start["headers"])[b"content-security-policy"] == sh.DOCUMENT_CSP.encode()


@pytest.mark.parametrize("scope_type", ["lifespan", "websocket"])
def test_other_scopes_pass_through_untouched(scope_type):
    seen = {}

    async def inner(scope, receive, send):
        seen["scope"] = scope
        await send({"type": f"{scope_type}.probe"})

    sent = run_asgi(sh.SecurityHeadersMiddleware(inner), {"type": scope_type})
    assert seen["scope"] == {"type": scope_type} and sent == [{"type": f"{scope_type}.probe"}]


# ===========================================================================
# A real streaming response over TCP: headers present, deltas still incremental
# ===========================================================================

def test_a_real_stream_carries_the_headers_and_stays_incremental(stream_app, loopback_runtime, monkeypatch):
    from app.services import access_policy
    from tests.conftest import legacy_access_decision
    from tests.streaming_support import LoopbackServer, read_sse_until
    from tests.test_stream_run import Gate, ScriptedWorkflow, final_state

    # A test-only grant (as the legacy shim in conftest does): this test is about headers, not access.
    monkeypatch.setattr(access_policy, "configuration_problem", lambda settings: None)
    monkeypatch.setattr(access_policy, "evaluate", lambda request, settings: legacy_access_decision())

    gate = Gate()
    service = stream_app(ScriptedWorkflow([("chunk", "first "), ("gate", gate), ("chunk", "second"),
                                           ("final", final_state(answer="first second"))]))
    with LoopbackServer(app.main.app, runtime=loopback_runtime) as server, \
            httpx.Client(base_url=server.base_url, trust_env=False, timeout=5.0) as http:
        # A browser's same-origin evidence: the paid-route origin check stays real (not patched).
        with http.stream("POST", "/api/chat/stream", json={"question": "How do I connect to the VPN?"},
                         headers={"Sec-Fetch-Site": "same-origin"}) as response:
            assert response.status_code == 200
            assert_security_headers(response, document=False)
            chunks = response.iter_raw()
            buffer = bytearray()
            read_sse_until(chunks, buffer, b"event: delta")
            assert not gate.release.is_set()  # a delta arrived while the graph is still held: not buffered
            gate.release.set()
            buffer += b"".join(chunks)
    [run] = service.started
    assert run.hooks["worker_exited"].wait(5.0)
    assert b"event: done" in bytes(buffer)


# ===========================================================================
# Development API docs: explicit per-page policies, only when enabled
# ===========================================================================

DOCS_ALLOWED_URLS = {
    "/docs": {sh.SWAGGER_JS, sh.SWAGGER_CSS, sh.FASTAPI_FAVICON},
    "/redoc": {sh.REDOC_JS, sh.FASTAPI_FAVICON, "https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700"},
    "/docs/oauth2-redirect": set(),
}


@pytest.fixture
def docs_enabled(settings, monkeypatch):
    monkeypatch.setattr(settings, "api_docs_enabled", True)
    try:
        for module in MODULES.values():
            importlib.reload(module)
        yield
    finally:
        monkeypatch.setattr(settings, "api_docs_enabled", False)
        for module in MODULES.values():
            importlib.reload(module)


def test_docs_pages_are_not_special_cased_while_docs_are_disabled(settings):
    assert settings.api_docs_enabled is False
    assert sh.docs_pages_for(settings) == {}
    for entry in APPS:
        response = TestClient(APPS[entry]()).get("/docs")
        assert response.status_code == 404
        assert_security_headers(response, document=False)


@pytest.mark.parametrize("path, kind", [("/docs", "swagger"), ("/redoc", "redoc"), ("/docs/oauth2-redirect", "oauth2-redirect")])
def test_each_docs_page_gets_its_explicit_policy(docs_enabled, path, kind):
    for module in MODULES.values():
        response = TestClient(module.app).get(path)
        assert response.status_code == 200, (module.__name__, path)
        policy = response.headers["content-security-policy"]
        assert policy == sh.docs_csp(kind, response.content)
        parsed = directives(policy)
        assert parsed["frame-ancestors"] == ["'none'"] and parsed["default-src"] == ["'none'"]
        assert "'unsafe-eval'" not in policy and "'unsafe-inline'" not in parsed["script-src"]
        if kind == "redoc":
            assert parsed["style-src"] == ["'unsafe-inline'", "https://fonts.googleapis.com/css"]  # the one documented exception
        else:
            assert "'unsafe-inline'" not in policy
        # every inline script is allowed by its own hash, and only that
        inline = re.findall(rb"<script>(.*?)</script>", response.content, re.S)
        hashes = [f"'sha256-{base64.b64encode(hashlib.sha256(script).digest()).decode()}'" for script in inline]
        assert [source for source in parsed["script-src"] if source.startswith("'sha256-")] == hashes
        # the generated page loads nothing outside its allowlist (catches FastAPI changing its CDN URLs)
        urls = set(re.findall(r'(?:src|href)="(https?://[^"]+)"', response.text))
        assert urls == DOCS_ALLOWED_URLS[path], urls
        for name, value in COMMON.items():
            assert response.headers.get(name) == value


@pytest.mark.parametrize("kind, expected", [("swagger", 1), ("redoc", 0), ("oauth2-redirect", 1)])
def test_an_unexpected_inline_script_is_never_authorized(kind, expected):
    legit = b"<script>legit()</script>"
    extra = b"<script>alert(document.cookie)</script>"
    exact = legit * expected
    assert sh.docs_csp(kind, exact).count("'sha256-") == expected  # the generator's own script(s) only
    for body in (exact + extra, extra * (expected + 2), b""):
        if body.count(b"<script>") == expected:
            continue
        policy = sh.docs_csp(kind, body)
        assert "'sha256-" not in policy, (kind, body)  # wrong count: nothing is hashed (fail closed)
        assert "'unsafe-inline'" not in directives(policy)["script-src"]
    if kind == "oauth2-redirect":
        assert directives(sh.docs_csp(kind, extra * 2))["script-src"] == ["'none'"]
    with pytest.raises(ValueError):
        sh.docs_csp("unknown", exact)


def test_installing_after_the_first_request_is_refused():
    from fastapi import FastAPI
    from types import SimpleNamespace

    late = FastAPI()
    late.get("/q")(lambda: {"ok": True})
    TestClient(late).get("/q")  # the stack is now built and cached
    with pytest.raises(RuntimeError, match="before the application serves its first request"):
        sh.install_security_headers(late, SimpleNamespace(api_docs_enabled=False))


def test_installing_twice_adds_one_layer_only():
    from fastapi import FastAPI
    from types import SimpleNamespace

    twice = FastAPI()
    twice.get("/q")(lambda: {"ok": True})
    for _ in range(2):
        sh.install_security_headers(twice, SimpleNamespace(api_docs_enabled=False))
    stack = twice.build_middleware_stack()
    assert isinstance(stack, sh.SecurityHeadersMiddleware) and isinstance(stack.app, ServerErrorMiddleware)
    assert_security_headers(TestClient(twice).get("/q"), document=False)


def test_both_entry_points_install_before_serving(monkeypatch):
    # Importing (or reloading) an entry point must not raise: install runs at construction.
    for module in MODULES.values():
        assert getattr(module.app.build_middleware_stack, "security_headers_installed", False)


def test_the_docs_policy_is_scoped_to_the_three_pages(docs_enabled):
    client = TestClient(app.main.app)
    assert_security_headers(client.get("/"), document=True)
    assert_security_headers(client.get("/openapi.json"), document=False)
    redirect = client.get("/docs/", follow_redirects=False)  # not a docs page path: a plain redirect
    assert redirect.status_code == 307
    assert_security_headers(redirect, document=False)


# ===========================================================================
# The page and its scripts stay compatible with the strict policy
# ===========================================================================

TEMPLATE = (ROOT / "templates" / "index.html").read_text()
SCRIPTS = {path.name: path.read_text() for path in (ROOT / "static" / "js").glob("*.js")}


def test_the_template_has_no_inline_code_the_policy_would_block():
    # Case-insensitive: HTML tag and attribute names are (ONCLICK= is an event handler too).
    assert not re.search(r"<script\b(?![^>]*\bsrc=)[^>]*>", TEMPLATE, re.I)  # no inline <script>
    assert not re.search(r"\son[a-z]+\s*=", TEMPLATE, re.I)                # no inline event handlers
    assert not re.search(r"\sstyle\s*=", TEMPLATE, re.I)                    # no style attributes
    assert not re.search(r"<style\b|javascript:", TEMPLATE, re.I)
    assert all(src.startswith("/static/") for src in re.findall(r'<script[^>]+src="([^"]+)"', TEMPLATE))
    stylesheets = re.findall(r'<link[^>]+rel="stylesheet"[^>]*>|<link\s+href="[^"]+"\s+rel="stylesheet"', TEMPLATE, re.S)
    hrefs = re.findall(r'href="([^"]+)"', " ".join(stylesheets))
    assert all(href.startswith(("/static/", "https://fonts.googleapis.com/")) for href in hrefs), hrefs


def test_the_frontend_scripts_use_no_code_evaluation():
    for name, text in SCRIPTS.items():
        assert not re.search(r"\beval\s*\(|new\s+Function\s*\(|set(Timeout|Interval)\(\s*['\"`]", text), name


def test_the_only_html_sink_is_the_sanitized_markdown_answer():
    sinks = [(name, match.start()) for name, text in SCRIPTS.items()
             for match in re.finditer(r"innerHTML|outerHTML|insertAdjacentHTML|document\.write", text)]
    assert [name for name, _ in sinks] == ["app.js"]
    [(_, offset)] = sinks
    app_js = SCRIPTS["app.js"]
    assert app_js[offset:].startswith("innerHTML = html;")
    before = app_js[max(0, offset - 700):offset]
    # html is only ever DOMPurify's output with the explicit allowlist, and only a string.
    assert "purify.sanitize(parsed, MARKDOWN_SANITIZE_CONFIG)" in before
    assert 'typeof html === "string"' in before
    assert app_js.count("renderAnswerMarkdown(answerText, normalizedAnswer)") == 1
    assert "window.DOMPurify.sanitize(" not in app_js  # no unconfigured (default-profile) call remains


def test_network_access_is_same_origin_only():
    for name, text in SCRIPTS.items():
        assert not re.search(r"new\s+(WebSocket|EventSource|Worker)\s*\(", text), name
        assert not re.search(r"fetch\(\s*['\"`]https?:", text), name
