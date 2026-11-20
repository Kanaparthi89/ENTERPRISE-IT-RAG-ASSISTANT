"""Shared fixtures, fakes, and isolation for the test suite.

Isolation guarantees (enforced here, verified in tests/test_isolation.py):

* Settings never come from the repository .env or the shell environment.
  Every app module receives ``TEST_SETTINGS``, built only from explicit
  values, with a dummy value for every ``SecretStr`` field.
* Storage defaults to a pytest-managed session directory, and each test
  gets its own ``tmp_path`` for the audit database and uploads.
* Real OpenAI, Pinecone, and Tavily clients cannot be constructed.
* Any non-loopback network access is blocked, recorded, and fails the
  test, even if application code catches the exception. Wildcard bind
  lookups (``getaddrinfo`` with ``AI_PASSIVE``) are allowed; connecting or
  sending to wildcard addresses is not.

Event-loop limitation: the network guard wraps Python's ``socket`` module.
uvloop (libuv) performs DNS resolution and connections in C and would bypass
it. Async tests therefore run on the standard asyncio loop only
(``anyio_backend`` below), and ``tests/test_isolation.py`` asserts that uvloop
is not active. Loopback servers run the baseline runtime (asyncio + h11) by
default; the production runtime (uvloop + httptools, H3) runs on
``guarded_uvloop_class()`` below, which applies the same rules to uvloop's own
network methods. Its limits are documented there.
"""

import ipaddress
import os
import socket
import threading
import time
from collections import deque
from pathlib import Path

import pytest
from pydantic import SecretStr

# Only the config module may be imported before test settings are installed.
import app.core.config as config


# -------------------------------------------------------------------
# Isolated settings (must run before any other app module is imported)
# -------------------------------------------------------------------

class IsolatedSettings(config.Settings):
    """Settings built only from explicit values: never .env or os.environ."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (init_settings,)


SECRET_FIELDS = tuple(
    name
    for name, field in config.Settings.model_fields.items()
    if field.annotation is SecretStr
)

# Until the session fixture assigns pytest-managed directories, storage
# points beneath os.devnull. That path is a file, so any premature write
# fails loudly instead of touching production data.
UNCONFIGURED_STORAGE = Path(os.devnull) / "bluepeak-tests-unconfigured"

TEST_SETTINGS = IsolatedSettings(
    app_env="testing",
    max_retries=2,
    audit_db_path=UNCONFIGURED_STORAGE / "audit.db",
    session_db_path=UNCONFIGURED_STORAGE / "sessions.db",
    upload_dir=UNCONFIGURED_STORAGE / "uploads",
    **{name: f"test-{name}" for name in SECRET_FIELDS},
)


def get_test_settings() -> config.Settings:
    return TEST_SETTINGS


config.get_settings.cache_clear()
config.get_settings = get_test_settings

# app/core/logging.py reads these directly from the process environment.
for _name in ("LOG_LEVEL", "LOG_TO_FILE", "LOG_DIR"):
    os.environ.pop(_name, None)


# -------------------------------------------------------------------
# Outbound network guard (installed before app modules load)
# -------------------------------------------------------------------

class NetworkAccessBlocked(OSError):
    """Raised for outbound network access attempted during tests."""


# Descriptions of blocked attempts. Each test's guard fixture claims the
# attempts made during that test; anything left over happened outside a
# test (for example at import time) and fails the session.
NETWORK_ATTEMPTS: list[str] = []

_REAL_SOCKET_FUNCTIONS = {
    "connect": socket.socket.connect,
    "connect_ex": socket.socket.connect_ex,
    "sendto": socket.socket.sendto,
    "getaddrinfo": socket.getaddrinfo,
}
_UNIX_FAMILY = getattr(socket, "AF_UNIX", None)
_INET_FAMILIES = (socket.AF_INET, socket.AF_INET6)


def _is_loopback(host) -> bool:
    if host is None:
        return True  # Wildcard lookups resolve locally.
    if isinstance(host, bytes):
        host = host.decode("ascii", "ignore")
    host = str(host).strip("[]").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _block(description: str):
    NETWORK_ATTEMPTS.append(description)
    raise NetworkAccessBlocked(
        f"Outbound network access is blocked in tests: {description}"
    )


def _check_socket_address(sock, address, operation: str) -> None:
    if _UNIX_FAMILY is not None and sock.family == _UNIX_FAMILY:
        return
    host = address[0] if isinstance(address, tuple) else address
    if sock.family in _INET_FAMILIES and _is_loopback(host):
        return
    _block(f"{operation} {address!r}")


def _guarded_connect(self, address):
    _check_socket_address(self, address, "connect")
    return _REAL_SOCKET_FUNCTIONS["connect"](self, address)


def _guarded_connect_ex(self, address):
    _check_socket_address(self, address, "connect_ex")
    return _REAL_SOCKET_FUNCTIONS["connect_ex"](self, address)


def _guarded_sendto(self, data, *args):
    # sendto(data, address) or sendto(data, flags, address)
    _check_socket_address(self, args[-1], "sendto")
    return _REAL_SOCKET_FUNCTIONS["sendto"](self, data, *args)


_WILDCARD_HOSTS = {"", "0.0.0.0", "::"}


def _is_passive_wildcard_lookup(host, args, kwargs) -> bool:
    """A lookup for binding a local server: wildcard host with AI_PASSIVE."""
    if isinstance(host, bytes):
        host = host.decode("ascii", "ignore")
    if host not in _WILDCARD_HOSTS:
        return False
    # getaddrinfo(host, port, family=0, type=0, proto=0, flags=0)
    flags = args[4] if len(args) > 4 else kwargs.get("flags", 0)
    return bool(flags & socket.AI_PASSIVE)


def _guarded_getaddrinfo(host, *args, **kwargs):
    if not (_is_loopback(host) or _is_passive_wildcard_lookup(host, args, kwargs)):
        _block(f"getaddrinfo {host!r}")
    return _REAL_SOCKET_FUNCTIONS["getaddrinfo"](host, *args, **kwargs)


socket.socket.connect = _guarded_connect
socket.socket.connect_ex = _guarded_connect_ex
socket.socket.sendto = _guarded_sendto
socket.getaddrinfo = _guarded_getaddrinfo


_guarded_uvloop_class = None


def guarded_uvloop_class():
    """A ``uvloop.Loop`` with the network guard on its own network methods (H3).

    uvloop resolves names and connects in C (libuv), so the socket-module
    guard above cannot see those calls. This subclass applies the same rules
    to the loop's Python-level entry points: ``getaddrinfo``,
    ``getnameinfo``, ``create_connection``, ``sock_connect``,
    ``sock_sendto`` and ``create_datagram_endpoint``. Blocked attempts are
    recorded like any other, so the test fails even if the error is caught.
    Inbound sockets, pipes and Unix-domain connections are allowed, as in the
    socket guard.

    Limitation: network work that libuv performs without going through
    these methods is not intercepted. The application opens no connections
    on its event loop (its provider clients are synchronous and use Python
    sockets in worker threads); container runs with ``--network none``
    cover the rest. Used only by loopback servers; async tests still run on
    the standard asyncio loop.
    """
    global _guarded_uvloop_class
    if _guarded_uvloop_class is not None:
        return _guarded_uvloop_class

    import uvloop

    class GuardedUVLoop(uvloop.Loop):
        async def getaddrinfo(self, host, port, *, family=0, type=0, proto=0, flags=0):
            if not (
                _is_loopback(host)
                or _is_passive_wildcard_lookup(host, (port, family, type, proto, flags), {})
            ):
                _block(f"uvloop getaddrinfo {host!r}")
            return await super().getaddrinfo(
                host, port, family=family, type=type, proto=proto, flags=flags
            )

        async def getnameinfo(self, sockaddr, flags=0):
            if not _is_loopback(sockaddr[0]):
                _block(f"uvloop getnameinfo {sockaddr!r}")
            return await super().getnameinfo(sockaddr, flags)

        async def create_connection(self, protocol_factory, host=None, port=None, *args, **kwargs):
            sock = kwargs.get("sock")
            if sock is not None:
                _check_socket_address(sock, sock.getpeername(), "uvloop create_connection")
            elif not _is_loopback(host):
                _block(f"uvloop create_connection {(host, port)!r}")
            return await super().create_connection(protocol_factory, host, port, *args, **kwargs)

        async def sock_connect(self, sock, address):
            _check_socket_address(sock, address, "uvloop sock_connect")
            return await super().sock_connect(sock, address)

        async def sock_sendto(self, sock, data, address):
            _check_socket_address(sock, address, "uvloop sock_sendto")
            return await super().sock_sendto(sock, data, address)

        async def create_datagram_endpoint(self, protocol_factory, local_addr=None, remote_addr=None, **kwargs):
            if remote_addr is not None and not _is_loopback(remote_addr[0]):
                _block(f"uvloop create_datagram_endpoint {remote_addr!r}")
            return await super().create_datagram_endpoint(
                protocol_factory, local_addr=local_addr, remote_addr=remote_addr, **kwargs
            )

    _guarded_uvloop_class = GuardedUVLoop
    return GuardedUVLoop


def check_network_attempts(attempts: list[str]) -> None:
    """Fail the current test if it attempted outbound network access."""
    if attempts:
        pytest.fail(
            "Blocked outbound network access: " + "; ".join(attempts),
            pytrace=False,
        )


# -------------------------------------------------------------------
# App imports (after settings and the network guard are in place)
# -------------------------------------------------------------------

import app.core.logging as app_logging  # noqa: E402
import app.main  # noqa: E402
import ingest_sample_kb  # noqa: E402
from app.api import routes  # noqa: E402
from app.rag import cancellation, vectorstore, workflow  # noqa: E402
from app.services import audit  # noqa: E402
from app.api import streaming  # noqa: E402
from app.services import sessions as session_service  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from tests.streaming_support import ChunkScript, ScriptedChatModel  # noqa: E402

SETTINGS_CONSUMERS = (app.main, routes, workflow, vectorstore, audit)
GET_SETTINGS_CONSUMERS = (app_logging, ingest_sample_kb, streaming)

if any(module.settings is not TEST_SETTINGS for module in SETTINGS_CONSUMERS) or any(
    module.get_settings is not get_test_settings for module in GET_SETTINGS_CONSUMERS
):
    raise RuntimeError(
        "An app module captured real settings before tests/conftest.py "
        "installed isolated settings; tests could read real configuration."
    )

ADMIN_KEY = TEST_SETTINGS.admin_api_key.get_secret_value()

# Real client classes that must never be constructed in tests.
EXTERNAL_CLIENTS = {
    vectorstore: ("Pinecone", "OpenAIEmbeddings", "PineconeVectorStore"),
    workflow: ("ChatOpenAI", "TavilySearch"),
}


# -------------------------------------------------------------------
# Test-outcome helpers
# -------------------------------------------------------------------

class KnownBugStillPresent(Exception):
    """Raised by a known-bug xfail test when the documented bug reproduces.

    Known-bug xfails use ``raises=KnownBugStillPresent``, so any other
    exception, including failed preconditions, is reported as a failure.
    """


class ScriptExhausted(Exception):
    """A fake was called more times than its script allows."""


class ExternalClientBlocked(RuntimeError):
    """A real external client was constructed during a test."""


def require(condition, message: str) -> None:
    """Fail the test outright when a precondition does not hold."""
    if not condition:
        pytest.fail(f"Precondition failed: {message}", pytrace=False)


# -------------------------------------------------------------------
# Hooks
# -------------------------------------------------------------------

BUG_PHASES = {"B1", "B5"}


def pytest_collection_modifyitems(config, items):
    for item in items:
        for marker in item.iter_markers("characterizes_bug"):
            if len(marker.args) != 1 or marker.args[0] not in BUG_PHASES:
                raise pytest.UsageError(
                    f"{item.nodeid}: characterizes_bug needs one phase "
                    f"from {sorted(BUG_PHASES)}, got {marker.args!r}"
                )

    if os.getenv("RUN_LIVE_TESTS") == "1":
        return

    skip_live = pytest.mark.skip(reason="set RUN_LIVE_TESTS=1 to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


def pytest_sessionfinish(session, exitstatus):
    if NETWORK_ATTEMPTS:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter:
            reporter.write_line(
                "Blocked outbound network access outside any test: "
                + "; ".join(NETWORK_ATTEMPTS),
                red=True,
            )
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


# -------------------------------------------------------------------
# Isolation fixtures
# -------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def session_storage(tmp_path_factory):
    """Pytest-managed default storage for anything outside a test's tmp_path."""
    root = tmp_path_factory.mktemp("session-storage")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(TEST_SETTINGS, "audit_db_path", root / "audit.db")
        patch.setattr(TEST_SETTINGS, "session_db_path", root / "sessions.db")
        patch.setattr(TEST_SETTINGS, "upload_dir", root / "uploads")
        yield root


@pytest.fixture(autouse=True)
def network_guard(request, monkeypatch):
    """Fail any test that attempted outbound network access."""
    if request.node.get_closest_marker("live") and os.getenv("RUN_LIVE_TESTS") == "1":
        monkeypatch.setattr(socket.socket, "connect", _REAL_SOCKET_FUNCTIONS["connect"])
        monkeypatch.setattr(socket.socket, "connect_ex", _REAL_SOCKET_FUNCTIONS["connect_ex"])
        monkeypatch.setattr(socket.socket, "sendto", _REAL_SOCKET_FUNCTIONS["sendto"])
        monkeypatch.setattr(socket, "getaddrinfo", _REAL_SOCKET_FUNCTIONS["getaddrinfo"])

    start = len(NETWORK_ATTEMPTS)
    yield
    attempts = NETWORK_ATTEMPTS[start:]
    del NETWORK_ATTEMPTS[start:]
    check_network_attempts(attempts)


@pytest.fixture
def blocked_network():
    """For tests that deliberately trigger the guard.

    Returns a function that hands back, and clears, the attempts recorded
    since the fixture was created, so the guard does not fail the test.
    """
    start = len(NETWORK_ATTEMPTS)

    def consume() -> list[str]:
        attempts = NETWORK_ATTEMPTS[start:]
        del NETWORK_ATTEMPTS[start:]
        return attempts

    return consume


def _blocked_client(name: str):
    def factory(*args, **kwargs):
        raise ExternalClientBlocked(
            f"{name} cannot be constructed in tests; install a fake instead."
        )

    return factory


@pytest.fixture(autouse=True)
def stream_service_singleton(monkeypatch):
    """Each test starts without a stream service; one created is shut down after.

    A test that triggers ``get_stream_service()`` must release its runs;
    leftover worker threads are joined with a guard so none leak.
    """
    monkeypatch.setattr(streaming, "_service", None)
    yield
    service = streaming._service
    if service is not None:
        service.shutdown(wait=False)
        deadline = time.monotonic() + 5.0
        while any(t.name.startswith(streaming.WORKER_THREAD_PREFIX) for t in threading.enumerate()):
            if time.monotonic() > deadline:
                pytest.fail("Stream worker threads leaked from the service singleton.")
            time.sleep(0.01)


STREAM_GUARD = 5.0


def _stream_worker_threads():
    return [t for t in threading.enumerate() if t.name.startswith(streaming.WORKER_THREAD_PREFIX)]


@pytest.fixture
def stream_app(monkeypatch, settings):
    """Enable streaming and install a test service; returns a configurator.

    After the test, every service created is checked: no worker threads
    left, capacity released, one lease release and exactly one audit write
    per started worker, and the defensive worker-exit claim never fired.
    """
    from tests.test_stream_run import AuditLog  # test helper; imported lazily

    monkeypatch.setattr(settings, "stream_enabled", True)
    created = []

    def configure(workflow, *, max_concurrent=2, timeout=120.0, clock=time.monotonic, audit=None):
        service = streaming.StreamService(
            max_concurrent=max_concurrent,
            timeout_seconds=timeout,
            clock=clock,
            poll_seconds=0.01,
            workflow_stream=workflow,
            audit_writer=audit if audit is not None else AuditLog(),
        )
        service.audit = service.audit_writer
        service.started = []
        real_start = service.start

        # start_run() goes through start(), and so do session runs (A2-6), so
        # wrapping start() records every run either way.
        def start(*args, **kwargs):
            run = real_start(*args, **kwargs)
            if run is not None:
                service.started.append(run)
            return run

        service.start = start
        monkeypatch.setattr(streaming, "get_stream_service", lambda: service)
        created.append(service)
        return service

    yield configure

    for service in created:
        for run in service.started:
            run.hooks["worker_exited"].wait(STREAM_GUARD)
        service.shutdown(wait=False)
    deadline = time.monotonic() + STREAM_GUARD
    while _stream_worker_threads() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert _stream_worker_threads() == [], "leaked stream worker threads"
    for service in created:
        assert service.capacity.in_use == 0
        assert service.capacity.acquired_total == service.capacity.released_total
        for run in service.started:
            assert run.lease.released
            assert run.audit_writes == (1 if run.worker_started else 0)
            assert not run.defensive_claim_fired


@pytest.fixture(autouse=True)
def access_throttle_cleared():
    """The S1-3 access-code throttle is process-wide; each test starts with a fresh one."""
    from app.services import access

    access.reset_throttle()
    yield
    access.reset_throttle()


# -------------------------------------------------------------------
# Security S1-5: legacy access compatibility (explicit modules only)
# -------------------------------------------------------------------
#
# S1-5 makes every paid AI route require a verified access cookie, quotas,
# PUBLIC_ORIGINS and same-origin evidence. The test modules below predate it:
# they test chat, streaming, sessions and transcription themselves, with
# access and quotas off and clients (TestClient, httpx, raw ASGI) that send
# no cookie and no Origin. For those modules only, ``legacy_access_compat``
# replaces three functions of app/services/access_policy.py, in tests only:
#
# * ``configuration_problem``: None (paid routes run with access or quotas off);
# * ``evaluate``: a verified cookie still grants its own code, exactly as in
#   production; any other request is granted the reserved test identity
#   ``LEGACY_IDENTITY`` (where S1-4 used the anonymous bucket);
# * ``origin_allowed``: a request with neither Origin nor Sec-Fetch-Site is
#   allowed (a non-browser test client); one that sends either is checked
#   for real, against PUBLIC_ORIGINS = LEGACY_PUBLIC_ORIGIN.
#
# The content-type check, the route class, its order and everything after
# it are unchanged. Every other module, and every new one, runs with real
# enforcement: tests/test_access_enforcement.py and
# tests/test_route_inventory.py are deliberately not listed. Production code
# has no switch for any of this.

LEGACY_ACCESS_MODULES = frozenset({
    "tests.test_audit_sessions",
    "tests.test_cancellation",
    "tests.test_chat_api",
    "tests.test_chat_stream",
    "tests.test_graceful_shutdown",
    "tests.test_quota_routes",
    "tests.test_session_api",
    "tests.test_session_stream",
    "tests.test_stream_backpressure_http",
    "tests.test_stream_http",
    "tests.test_stream_lifecycle",
    "tests.test_voice_quota",
    "tests.test_voice_status",
    "tests.test_voice_transcribe",
    "tests.test_workflow_history",
})
LEGACY_IDENTITY = "legacy-compat"
LEGACY_PUBLIC_ORIGIN = "http://testserver"  # TestClient's default Host


def legacy_access_decision():
    """The decision the compatibility shim grants a request without a verified cookie."""
    from app.services import access, access_policy

    return access_policy.AccessDecision(
        access_policy.GRANTED,
        "legacy-compat",
        grant=access.AccessGrant(code_id=LEGACY_IDENTITY, expires_at=2**31 - 1),
        code_ref="legacy",
    )


def legacy_access_request():
    """A stand-in request for tests that call a route function directly."""
    from types import SimpleNamespace

    return SimpleNamespace(state=SimpleNamespace(access=legacy_access_decision()))


@pytest.fixture(autouse=True)
def legacy_access_compat(request, monkeypatch):
    """No-op outside LEGACY_ACCESS_MODULES; see the comment above."""
    if request.module is None or request.module.__name__ not in LEGACY_ACCESS_MODULES:
        yield None
        return
    from app.services import access, access_policy

    real_origin_allowed = access_policy.origin_allowed

    def evaluate(http_request, settings):
        if settings.access_enabled:
            try:
                config = access.AccessConfig.from_settings(settings)
            except access.AccessConfigError:
                config = None
            if config is not None:
                grant = access.verify_cookie(
                    config, http_request.cookies.get(config.cookie_name), access.wall_clock()
                )
                if grant is not None:
                    return access_policy.AccessDecision(
                        access_policy.GRANTED, access_policy.VERIFIED, grant=grant,
                        code_ref=access.code_reference(config, grant.code_id),
                    )
        return legacy_access_decision()

    def origin_allowed(http_request, origins):
        headers = http_request.headers
        if "origin" not in headers and "sec-fetch-site" not in headers:
            return True
        return real_origin_allowed(http_request, origins)

    monkeypatch.setattr(TEST_SETTINGS, "public_origins", LEGACY_PUBLIC_ORIGIN)
    monkeypatch.setattr(access_policy, "configuration_problem", lambda settings: None)
    monkeypatch.setattr(access_policy, "evaluate", evaluate)
    monkeypatch.setattr(access_policy, "origin_allowed", origin_allowed)
    yield LEGACY_IDENTITY


@pytest.fixture(autouse=True)
def quota_engine_cleared():
    """The S1-4b chat quota engine is process-wide; each test starts without one."""
    from app.services import quota_runtime

    quota_runtime.reset_quota_engine()
    yield
    quota_runtime.reset_quota_engine()


@pytest.fixture(autouse=True)
def shutdown_signal_cleared():
    """The H1-3a shutdown signal is process-wide; an app lifespan's shutdown sets it."""
    cancellation.reset_shutdown()
    yield
    cancellation.reset_shutdown()


@pytest.fixture(autouse=True)
def session_runtime_singleton():
    """Each test starts without a session runtime; one created is shut down after.

    Fails the test if a lease keeper thread outlives the shutdown.
    """
    session_service.clear_session_runtime()
    yield
    runtime = session_service.peek_session_runtime()
    if runtime is not None:
        runtime.manager.shutdown()
        session_service.clear_session_runtime(runtime)
    deadline = time.monotonic() + 5.0
    while any(t.name.startswith("session-keeper") and t.is_alive() for t in threading.enumerate()):
        if time.monotonic() > deadline:
            pytest.fail("A session keeper thread leaked from the session runtime.")
        time.sleep(0.01)


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    """Give each test its own storage and forbid real external clients."""
    monkeypatch.setattr(TEST_SETTINGS, "audit_db_path", tmp_path / "audit.db")
    monkeypatch.setattr(TEST_SETTINGS, "session_db_path", tmp_path / "sessions.db")
    monkeypatch.setattr(TEST_SETTINGS, "upload_dir", tmp_path / "uploads")

    for module, names in EXTERNAL_CLIENTS.items():
        for name in names:
            monkeypatch.setattr(
                module, name, _blocked_client(f"{module.__name__}.{name}")
            )

    # Never let a cached client leak between tests.
    monkeypatch.setattr(vectorstore, "_embeddings", None)
    monkeypatch.setattr(vectorstore, "_vectorstore", None)
    monkeypatch.setattr(workflow, "_llm", None)
    monkeypatch.setattr(workflow, "_internal_llm", None)
    monkeypatch.setattr(workflow, "_web_search", {})

    return TEST_SETTINGS


@pytest.fixture
def settings(isolate):
    return isolate


@pytest.fixture
def client():
    """Test client with lifespan enabled so the audit DB is initialized."""
    with TestClient(app.main.app) as test_client:
        yield test_client


@pytest.fixture(params=["A", "P"])
def loopback_runtime(request):
    """H3: run a real-server test on runtime A (asyncio + h11, the baseline)
    and runtime P (uvloop + httptools, what production uvicorn selects)."""
    return request.param


@pytest.fixture
def anyio_backend():
    """Run async tests on the standard asyncio loop, never uvloop.

    See the module docstring: the network guard cannot see uvloop sockets.
    """
    return ("asyncio", {"use_uvloop": False})


# -------------------------------------------------------------------
# Fakes for the LangGraph workflow
# -------------------------------------------------------------------

class FakeLLM:
    """Scripted stand-in for ChatOpenAI.

    ``structured`` maps a Pydantic schema class to a queue of responses
    (dicts or model instances) consumed by ``with_structured_output``.
    ``text`` is a queue of plain ``invoke`` responses: strings, or
    ``ChunkScript`` objects that are emitted as model output chunks through
    LangChain callbacks. With ``stream=True``, strings are split into chunks
    too. Every call is recorded in ``calls`` with the tags from its config.
    """

    def __init__(self, structured=None, text=None, *, stream=False):
        self.structured = {
            schema: deque(responses)
            for schema, responses in (structured or {}).items()
        }
        self.text = deque(text or [])
        self.stream = stream
        self.prompts = []
        self.calls = []

    def _record(self, kind, prompt, config, schema=None):
        self.prompts.append(prompt)
        self.calls.append(
            {
                "kind": kind,
                "schema": schema,
                "prompt": prompt,
                "tags": list((config or {}).get("tags", [])),
            }
        )

    def with_structured_output(self, schema, **kwargs):
        return _FakeStructured(self, schema)

    def invoke(self, prompt, config=None):
        self._record("text", prompt, config)
        if not self.text:
            raise ScriptExhausted("FakeLLM ran out of text responses.")
        response = self.text.popleft()
        if self.stream and isinstance(response, str):
            response = ChunkScript.from_text(response)
        if isinstance(response, ChunkScript):
            return ScriptedChatModel(script=response).invoke(prompt, config=config)
        return AIMessage(content=response)


class _FakeStructured:
    def __init__(self, parent, schema):
        self.parent = parent
        self.schema = schema

    def invoke(self, prompt, config=None):
        self.parent._record("structured", prompt, config, schema=self.schema)
        queue = self.parent.structured.get(self.schema)
        if not queue:
            raise ScriptExhausted(
                f"FakeLLM has no scripted {self.schema.__name__} response."
            )
        response = queue.popleft()
        if isinstance(response, dict):
            return self.schema(**response)
        return response


class FakeRetriever:
    def __init__(self, docs=None):
        self.docs = list(docs or [])
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return list(self.docs)


class FakeWebSearch:
    """Returns the scripted Tavily payloads in order (last one repeats)."""

    def __init__(self, *payloads, error=None):
        self.payloads = list(payloads) or [{"results": []}]
        self.error = error
        self.queries = []
        self.include_domains = []

    def factory(self, include_domains=None):
        self.include_domains.append(include_domains)
        return self

    def invoke(self, payload):
        self.queries.append(payload["query"])
        if self.error:
            raise self.error
        index = min(len(self.queries) - 1, len(self.payloads) - 1)
        return self.payloads[index]


@pytest.fixture
def fake_rag(monkeypatch):
    """Install fakes into the workflow and return a configurator.

    The same ``FakeLLM`` serves answer generation (``llm``) and internal
    routing, grading, and rewriting calls (``internal_llm``); its ``calls``
    record which tags each call carried.
    """

    def install(llm=None, retriever=None, web=None):
        llm = llm or FakeLLM()
        retriever = retriever or FakeRetriever()
        web = web or FakeWebSearch()

        monkeypatch.setattr(workflow, "llm", lambda: llm)
        monkeypatch.setattr(workflow, "internal_llm", lambda: llm)
        monkeypatch.setattr(workflow, "get_retriever", lambda: retriever)
        monkeypatch.setattr(workflow, "web_search_tool", web.factory)
        return llm, retriever, web

    return install


def kb_doc(content, source="/kb/handbook.md", **metadata):
    return Document(
        page_content=content,
        metadata={"source": source, "filename": source.split("/")[-1], **metadata},
    )


def web_result(title, url, content="Relevant evidence."):
    return {"title": title, "url": url, "content": content}
