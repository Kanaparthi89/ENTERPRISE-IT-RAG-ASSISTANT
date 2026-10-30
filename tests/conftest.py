"""Shared fixtures, fakes, and isolation for the test suite.

Isolation guarantees (enforced here, verified in tests/test_isolation.py):

* Settings never come from the repository .env or the shell environment.
  Every app module receives ``TEST_SETTINGS``, built only from explicit
  values, with a dummy value for every ``SecretStr`` field.
* Storage defaults to a pytest-managed session directory, and each test
  gets its own ``tmp_path`` for the audit database and uploads.
* Real OpenAI, Pinecone, and Tavily clients cannot be constructed.
* Any non-loopback network access is blocked, recorded, and fails the
  test, even if application code catches the exception.
"""

import ipaddress
import os
import socket
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


def _guarded_getaddrinfo(host, *args, **kwargs):
    if not _is_loopback(host):
        _block(f"getaddrinfo {host!r}")
    return _REAL_SOCKET_FUNCTIONS["getaddrinfo"](host, *args, **kwargs)


socket.socket.connect = _guarded_connect
socket.socket.connect_ex = _guarded_connect_ex
socket.socket.sendto = _guarded_sendto
socket.getaddrinfo = _guarded_getaddrinfo


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
from app.rag import vectorstore, workflow  # noqa: E402
from app.services import audit  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

SETTINGS_CONSUMERS = (app.main, routes, workflow, vectorstore, audit)
GET_SETTINGS_CONSUMERS = (app_logging, ingest_sample_kb)

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
def isolate(monkeypatch, tmp_path):
    """Give each test its own storage and forbid real external clients."""
    monkeypatch.setattr(TEST_SETTINGS, "audit_db_path", tmp_path / "audit.db")
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


# -------------------------------------------------------------------
# Fakes for the LangGraph workflow
# -------------------------------------------------------------------

class FakeLLM:
    """Scripted stand-in for ChatOpenAI.

    ``structured`` maps a Pydantic schema class to a queue of responses
    (dicts or model instances) consumed by ``with_structured_output``.
    ``text`` is a queue of strings returned by plain ``invoke`` calls.
    """

    def __init__(self, structured=None, text=None):
        self.structured = {
            schema: deque(responses)
            for schema, responses in (structured or {}).items()
        }
        self.text = deque(text or [])
        self.prompts = []

    def with_structured_output(self, schema, **kwargs):
        return _FakeStructured(self, schema)

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if not self.text:
            raise ScriptExhausted("FakeLLM ran out of text responses.")
        return AIMessage(content=self.text.popleft())


class _FakeStructured:
    def __init__(self, parent, schema):
        self.parent = parent
        self.schema = schema

    def invoke(self, prompt):
        self.parent.prompts.append(prompt)
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
    """Install fakes into the workflow and return a configurator."""

    def install(llm=None, retriever=None, web=None):
        llm = llm or FakeLLM()
        retriever = retriever or FakeRetriever()
        web = web or FakeWebSearch()

        monkeypatch.setattr(workflow, "llm", lambda: llm)
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
