"""Security S1-5 Phase A: every route of both entry points is classified.

``app.main:app`` (what the Dockerfile runs) and ``app.api.main:app`` (a
second entry point that includes the same router) must expose exactly the
routes below. Every paid AI route must use its AccessEnforcedRoute
subclass; a new route, or a paid route registered without enforcement on
either app, fails here. A static scan of app/api/routes.py also checks that
only the classified handlers can reach a provider.

Not in tests/conftest.py's LEGACY_ACCESS_MODULES: real enforcement.
"""

import ast
from pathlib import Path

import pytest
from fastapi.routing import APIRoute, APIWebSocketRoute
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.testclient import TestClient

import app.main
from app.api import routes

ROOT = Path(__file__).resolve().parents[1]

# (method, path): (classification, route class, endpoint name)
PAID = {
    ("POST", "/api/chat"): (routes.ChatRoute, "chat"),
    ("POST", "/api/chat/stream"): (routes.StreamingFlagRoute, "chat_stream"),
    ("POST", "/api/sessions"): (routes.SessionCreateRoute, "create_session"),
    ("DELETE", "/api/sessions"): (routes.SessionDeleteRoute, "delete_session"),
    ("POST", "/api/voice/transcribe"): (routes.TranscriptionRoute, "voice_transcribe"),
}
ADMIN = {("POST", "/api/ingest"): "ingest"}  # its own admin key (S1-5 Phase C adds a throttle)
PUBLIC = {
    ("GET", "/api/health"): "health",
    ("GET", "/api/voice/status"): "voice_status",
    ("GET", "/api/voice/transcription/status"): "voice_transcription_status",
    ("POST", "/api/access"): "redeem_access",
    ("GET", "/api/access/status"): "access_status",
    ("POST", "/api/access/logout"): "access_logout",
    ("GET", "/"): "home",
}
APP_MAIN_ONLY = {("GET", "/health"): "health_check"}
# FastAPI's documentation routes: public on both apps until Phase D disables them in production.
DOCS = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}

EXPECTED_FLAGS = {
    routes.ChatRoute: (None, True),
    routes.StreamingFlagRoute: ("stream_enabled", True),
    routes.SessionCreateRoute: ("sessions_enabled", False),
    routes.SessionDeleteRoute: (None, False),
    routes.TranscriptionRoute: ("voice_transcription_enabled", False),
}


def api_main_app():
    import app.api.main as api_main

    return api_main.app


APPS = {"app.main": lambda: app.main.app, "app.api.main": api_main_app}


def inventory(application):
    """{(method, path): route} for API routes; docs routes and mounts separately."""
    api, docs, mounts, other = {}, set(), set(), []
    for route in application.routes:
        if isinstance(route, (APIWebSocketRoute, WebSocketRoute)):
            other.append(("websocket", route.path))
        elif isinstance(route, APIRoute):
            for method in route.methods:
                key = (method, route.path)
                assert key not in api, f"{key} is registered twice"
                api[key] = route
        elif isinstance(route, Route) and route.path in DOCS:
            docs.add(route.path)
        elif isinstance(route, Mount):
            mounts.add(route.path)
        else:
            other.append((type(route).__name__, getattr(route, "path", None)))
    return api, docs, mounts, other


@pytest.mark.parametrize("entry", list(APPS))
def test_every_route_is_classified(entry):
    api, docs, mounts, other = inventory(APPS[entry]())
    expected = {**{key: name for key, (_, name) in PAID.items()}, **ADMIN, **PUBLIC}
    if entry == "app.main":
        expected.update(APP_MAIN_ONLY)
    assert {key: route.endpoint.__name__ for key, route in api.items()} == expected
    assert docs == DOCS
    assert mounts == {"/static"}
    assert other == []  # no websockets, no unclassified routes


@pytest.mark.parametrize("entry", list(APPS))
@pytest.mark.parametrize("key", list(PAID))
def test_every_paid_route_enforces_access_before_the_body(entry, key):
    api, *_ = inventory(APPS[entry]())
    route = api[key]
    route_class, _ = PAID[key]
    assert type(route) is route_class
    assert isinstance(route, routes.AccessEnforcedRoute)
    assert (route_class.feature_flag, route_class.json_body) == EXPECTED_FLAGS[route_class]


@pytest.mark.parametrize("entry", list(APPS))
def test_no_other_route_uses_the_paid_route_classes_or_skips_them(entry):
    api, *_ = inventory(APPS[entry]())
    for key, route in api.items():
        assert isinstance(route, routes.AccessEnforcedRoute) == (key in PAID), key


@pytest.mark.parametrize("entry", list(APPS))
@pytest.mark.parametrize("key", list(PAID))
def test_every_paid_route_refuses_a_direct_request_on_both_entry_points(settings, monkeypatch, entry, key):
    """With the default (read-only) settings, nothing paid answers on either app."""
    method, path = key
    for flag in ("stream_enabled", "sessions_enabled", "voice_transcription_enabled"):
        monkeypatch.setattr(settings, flag, True)
    response = TestClient(APPS[entry]()).request(
        method, path, content=b'{"question": "How do I connect to the VPN?"}',
        headers={"content-type": "application/json", "origin": "http://testserver", "x-session-id": "x" * 43},
    )
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "access_unavailable"


# ---------------------------------------------------------------------------
# Static scan: which handlers can reach a provider
# ---------------------------------------------------------------------------

# Names in app/api/routes.py that lead to a paid provider or private
# knowledge: ask (chat RAG workflow), the stream service (stream_ask in its
# workers), transcription.transcribe (OpenAI) and add_documents (embeddings
# and Pinecone upserts).
PROVIDER_NAMES = {"ask", "get_stream_service", "transcribe", "add_documents"}


def _references(node) -> set[str]:
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            names.add(child.attr)
    return names


def provider_capable_functions() -> set[str]:
    tree = ast.parse((ROOT / "app" / "api" / "routes.py").read_text())
    functions = {
        node.name: _references(node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    capable = {name for name, refs in functions.items() if refs & PROVIDER_NAMES}
    changed = True
    while changed:  # everything that calls a provider-capable function, transitively
        changed = False
        for name, refs in functions.items():
            if name not in capable and refs & capable:
                capable.add(name)
                changed = True
    return capable


def test_only_classified_handlers_can_reach_a_provider():
    capable = provider_capable_functions()
    handlers = {name for _, name in PAID.values()} | set(ADMIN.values())
    public = set(PUBLIC.values()) | set(APP_MAIN_ONLY.values())
    assert {"chat", "chat_stream", "voice_transcribe", "ingest"} <= capable
    assert capable & public == set()
    # Every provider-capable route handler is a paid (enforced) or admin route.
    endpoints = {route.endpoint.__name__ for route in app.main.app.routes if isinstance(route, APIRoute)}
    assert (capable & endpoints) <= handlers


@pytest.mark.parametrize("module", ["app/main.py", "app/api/main.py"])
def test_the_entry_points_reach_no_provider_directly(module):
    tree = ast.parse((ROOT / module).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not any(name.startswith(("app.rag", "app.services.transcription", "openai", "pinecone",
                                    "langchain", "tavily")) for name in imported), imported
    assert not (_references(tree) & PROVIDER_NAMES)


def test_no_background_tasks_or_websockets_in_the_router():
    source = (ROOT / "app" / "api" / "routes.py").read_text()
    assert "BackgroundTasks" not in source
    assert "websocket" not in source.lower()
