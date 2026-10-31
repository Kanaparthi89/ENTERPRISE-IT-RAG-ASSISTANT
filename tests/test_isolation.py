"""Proves the suite is isolated from real configuration, storage and networks."""

import asyncio
import re
import socket

import pytest
from pydantic import SecretStr

import app.core.config as config
from app.core.config import BASE_DIR
from app.rag import vectorstore, workflow
from tests.conftest import (
    GET_SETTINGS_CONSUMERS,
    SETTINGS_CONSUMERS,
    TEST_SETTINGS,
    UNCONFIGURED_STORAGE,
    ExternalClientBlocked,
    IsolatedSettings,
    NetworkAccessBlocked,
    check_network_attempts,
    get_test_settings,
)

# TEST-NET-1 (RFC 5737): reserved for documentation, never routable.
UNROUTABLE_IP = "192.0.2.1"


# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------

def test_every_app_module_uses_test_settings():
    assert config.get_settings() is TEST_SETTINGS
    for module in SETTINGS_CONSUMERS:
        assert module.settings is TEST_SETTINGS, module.__name__
    for module in GET_SETTINGS_CONSUMERS:
        assert module.get_settings is get_test_settings, module.__name__


def test_every_secret_field_has_a_dummy_value():
    secret_fields = [
        name
        for name, field in config.Settings.model_fields.items()
        if field.annotation is SecretStr
    ]

    assert {"openai_api_key", "pinecone_api_key", "tavily_api_key", "admin_api_key"} <= set(
        secret_fields
    )
    for name in secret_fields:
        assert getattr(TEST_SETTINGS, name).get_secret_value() == f"test-{name}"


def test_isolated_settings_ignore_dotenv_and_shell(monkeypatch):
    # Simulate a developer shell exporting real-looking configuration.
    monkeypatch.setenv("OPENAI_API_KEY", "from-shell")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "from-shell")
    monkeypatch.setenv("PINECONE_NAMESPACE", "from-shell")
    monkeypatch.setenv("TOP_K", "49")

    isolated = IsolatedSettings()

    # Every field equals its declared default, so neither the shell nor the
    # repository .env (which the base class would read) contributed a value.
    for name, field in config.Settings.model_fields.items():
        assert getattr(isolated, name) == field.get_default(
            call_default_factory=True
        ), name


def test_unset_test_settings_fall_back_to_declared_defaults():
    fields = config.Settings.model_fields

    assert TEST_SETTINGS.pinecone_index_name == fields["pinecone_index_name"].default
    assert TEST_SETTINGS.pinecone_namespace == fields["pinecone_namespace"].default
    assert TEST_SETTINGS.openai_model == fields["openai_model"].default


# -------------------------------------------------------------------
# Storage
# -------------------------------------------------------------------

def test_each_test_gets_private_storage(settings, tmp_path):
    assert settings.audit_db_path.parent == tmp_path
    assert settings.upload_dir.parent == tmp_path
    assert not settings.audit_db_path.is_relative_to(BASE_DIR)
    assert not settings.upload_dir.is_relative_to(BASE_DIR)


def test_session_storage_is_pytest_managed(session_storage, tmp_path_factory):
    assert session_storage.is_relative_to(tmp_path_factory.getbasetemp())
    assert not session_storage.is_relative_to(BASE_DIR)


def test_storage_before_session_setup_is_unwritable():
    with pytest.raises(OSError):
        UNCONFIGURED_STORAGE.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# Network guard
# -------------------------------------------------------------------

def test_hostname_resolution_is_blocked(blocked_network):
    with pytest.raises(NetworkAccessBlocked):
        socket.create_connection(("api.openai.com", 443), timeout=1)

    assert blocked_network() == ["getaddrinfo 'api.openai.com'"]


@pytest.mark.parametrize("operation", ["connect", "connect_ex"])
def test_direct_ip_connection_is_blocked(blocked_network, operation):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(NetworkAccessBlocked):
            getattr(sock, operation)((UNROUTABLE_IP, 443))

    assert blocked_network() == [f"{operation} ('{UNROUTABLE_IP}', 443)"]


def test_udp_send_is_blocked(blocked_network):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        with pytest.raises(NetworkAccessBlocked):
            sock.sendto(b"x", (UNROUTABLE_IP, 53))

    assert blocked_network() == [f"sendto ('{UNROUTABLE_IP}', 53)"]


def test_swallowed_attempt_is_still_recorded(blocked_network):
    try:
        socket.getaddrinfo("api.pinecone.io", 443)
    except Exception:
        pass  # Simulates application code that catches every exception.

    assert blocked_network() == ["getaddrinfo 'api.pinecone.io'"]


def test_recorded_attempts_fail_the_test():
    check_network_attempts([])

    with pytest.raises(pytest.fail.Exception, match="Blocked outbound network access"):
        check_network_attempts([f"connect ('{UNROUTABLE_IP}', 443)"])


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_loopback_tcp_is_allowed(host):
    with socket.create_server(("127.0.0.1", 0)) as server:
        port = server.getsockname()[1]
        with socket.create_connection((host, port), timeout=2) as client:
            peer, _ = server.accept()
            with peer:
                client.sendall(b"ping")
                assert peer.recv(4) == b"ping"


def test_unix_socketpair_is_allowed():
    left, right = socket.socketpair()
    with left, right:
        left.sendall(b"ping")
        assert right.recv(4) == b"ping"


def test_test_client_needs_no_network(client, blocked_network):
    assert client.get("/health").status_code == 200
    assert blocked_network() == []


# -------------------------------------------------------------------
# Wildcard (bind) lookups
# -------------------------------------------------------------------

@pytest.mark.parametrize("host", ["0.0.0.0", "::", ""])
def test_passive_wildcard_lookup_is_allowed(host, blocked_network):
    try:
        socket.getaddrinfo(host, 0, flags=socket.AI_PASSIVE)
    except NetworkAccessBlocked:
        pytest.fail(f"AI_PASSIVE lookup for {host!r} was blocked")
    except socket.gaierror:
        pass  # The platform may reject the literal; only the guard matters here.

    assert blocked_network() == []


def test_passive_flag_as_positional_argument_is_honoured(blocked_network):
    # getaddrinfo(host, port, family, type, proto, flags)
    socket.getaddrinfo("0.0.0.0", 0, 0, 0, 0, socket.AI_PASSIVE)

    assert blocked_network() == []


@pytest.mark.parametrize("host", ["0.0.0.0", "::", ""])
def test_wildcard_lookup_without_passive_flag_is_blocked(host, blocked_network):
    with pytest.raises(NetworkAccessBlocked):
        socket.getaddrinfo(host, 80)

    assert blocked_network() == [f"getaddrinfo {host!r}"]


def test_passive_flag_does_not_allow_public_hosts(blocked_network):
    with pytest.raises(NetworkAccessBlocked):
        socket.getaddrinfo("api.openai.com", 443, flags=socket.AI_PASSIVE)

    assert blocked_network() == ["getaddrinfo 'api.openai.com'"]


def test_connect_to_wildcard_address_is_blocked(blocked_network):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(NetworkAccessBlocked):
            sock.connect(("0.0.0.0", 9))

    assert blocked_network() == ["connect ('0.0.0.0', 9)"]


def test_send_to_wildcard_address_is_blocked(blocked_network):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        with pytest.raises(NetworkAccessBlocked):
            sock.sendto(b"x", ("0.0.0.0", 53))

    assert blocked_network() == ["sendto ('0.0.0.0', 53)"]


# -------------------------------------------------------------------
# Event loop
# -------------------------------------------------------------------

def test_anyio_backend_is_asyncio_without_uvloop(anyio_backend):
    assert anyio_backend == ("asyncio", {"use_uvloop": False})


@pytest.mark.anyio
async def test_async_tests_run_on_the_standard_asyncio_loop():
    loop = asyncio.get_running_loop()

    assert type(loop).__module__.startswith("asyncio."), type(loop)
    assert "uvloop" not in type(asyncio.get_event_loop_policy()).__module__


# -------------------------------------------------------------------
# External clients
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    "factory",
    [
        workflow.llm,
        workflow.internal_llm,
        workflow.web_search_tool,
        vectorstore.get_embeddings,
        vectorstore.ensure_index,
        vectorstore.get_vectorstore,
        vectorstore.get_retriever,
    ],
    ids=lambda function: f"{function.__module__}.{function.__name__}",
)
def test_real_clients_cannot_be_constructed(factory, blocked_network):
    with pytest.raises(ExternalClientBlocked):
        factory()

    assert blocked_network() == []


def test_client_caches_start_empty():
    assert vectorstore._embeddings is None
    assert vectorstore._vectorstore is None
    assert workflow._llm is None
    assert workflow._internal_llm is None
    assert workflow._web_search == {}


def test_only_logging_reads_the_process_environment():
    readers = {
        path.relative_to(BASE_DIR).as_posix()
        for path in (BASE_DIR / "app").rglob("*.py")
        if re.search(r"\bos\.(getenv|environ)\b", path.read_text(encoding="utf-8"))
    }

    assert readers == {"app/core/logging.py"}
