"""H1: finite timeouts for Pinecone and Tavily calls (no external network).

Pinecone wrapper tests live in tests/test_vectorstore.py. Here: the settings,
and the Tavily wrapper, its injection, and failed-search handling; and (H1-2)
the retry behavior the README documents for OpenAI, Tavily and Pinecone.
"""

import logging
import selectors
import socket
import threading
import time

import pytest
import requests
from pydantic import ValidationError

from app.core import config
from app.rag import vectorstore, workflow
from tests.conftest import FakeWebSearch, IsolatedSettings

Wrapper = workflow._TimeoutTavilySearchAPIWrapper


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def test_timeout_defaults():
    fields = config.Settings.model_fields

    assert fields["pinecone_timeout_seconds"].default == 10.0
    assert fields["tavily_timeout_seconds"].default == 15.0


def test_timeouts_are_separate_from_the_openai_settings():
    fields = config.Settings.model_fields

    assert fields["openai_timeout_seconds"].default == 60.0
    assert fields["openai_max_retries"].default == 2


@pytest.mark.parametrize(
    "name, value",
    [
        ("pinecone_timeout_seconds", 0.1),
        ("pinecone_timeout_seconds", 15),
        ("tavily_timeout_seconds", 0.1),
        ("tavily_timeout_seconds", 30),
    ],
)
def test_timeouts_within_range_are_accepted(name, value):
    assert getattr(IsolatedSettings(**{name: value}), name) == value


@pytest.mark.parametrize(
    "name, value",
    [
        ("pinecone_timeout_seconds", 0),
        ("pinecone_timeout_seconds", -1),
        ("pinecone_timeout_seconds", 15.01),
        ("tavily_timeout_seconds", 0),
        ("tavily_timeout_seconds", -1),
        ("tavily_timeout_seconds", 30.01),
    ],
)
def test_timeouts_out_of_range_are_rejected(name, value):
    with pytest.raises(ValidationError):
        IsolatedSettings(**{name: value})


# H1-5: both new limits are off unless configured.
def test_h1_5_settings_are_unset_by_default():
    fields = config.Settings.model_fields

    assert fields["pinecone_retry_after_max_seconds"].default is None
    assert fields["chat_deadline_seconds"].default is None
    settings = IsolatedSettings()
    assert settings.pinecone_retry_after_max_seconds is None
    assert settings.chat_deadline_seconds is None


@pytest.mark.parametrize(
    "name, value",
    [
        ("pinecone_retry_after_max_seconds", 0),
        ("pinecone_retry_after_max_seconds", 30),
        ("pinecone_retry_after_max_seconds", 21600),
        ("chat_deadline_seconds", 0.5),
        ("chat_deadline_seconds", 3600),
    ],
)
def test_h1_5_settings_within_range_are_accepted(name, value):
    assert getattr(IsolatedSettings(**{name: value}), name) == value


@pytest.mark.parametrize(
    "name, value",
    [
        ("pinecone_retry_after_max_seconds", -1),
        ("pinecone_retry_after_max_seconds", 21601),
        ("pinecone_retry_after_max_seconds", 1.5),  # whole seconds, as urllib3's retry_after_max
        ("chat_deadline_seconds", 0),
        ("chat_deadline_seconds", -1),
        ("chat_deadline_seconds", 3600.01),
    ],
)
def test_h1_5_settings_out_of_range_are_rejected(name, value):
    with pytest.raises(ValidationError):
        IsolatedSettings(**{name: value})


# ---------------------------------------------------------------------------
# Tavily wrapper
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.payload = payload if payload is not None else {"results": []}

    def json(self):
        return self.payload


@pytest.fixture
def recorded_post(monkeypatch):
    calls = []

    def install(response):
        def post(url, **kwargs):
            calls.append((url, kwargs))
            return response

        monkeypatch.setattr(workflow.requests, "post", post)
        return calls

    return install


def test_wrapper_sends_the_search_with_the_configured_timeout(recorded_post):
    calls = recorded_post(FakeResponse(payload={"results": [{"url": "https://cisa.gov"}]}))
    wrapper = Wrapper(tavily_api_key="test-key", timeout_seconds=4.0)

    result = wrapper.raw_results(
        query="phishing", max_results=5, include_domains=["cisa.gov"], time_range=None
    )

    assert result == {"results": [{"url": "https://cisa.gov"}]}
    [(url, kwargs)] = calls
    assert url == f"{workflow.TAVILY_API_URL}/search"
    assert kwargs["timeout"] == (4.0, 4.0)
    assert kwargs["json"] == {"query": "phishing", "max_results": 5, "include_domains": ["cisa.gov"]}
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"


def test_wrapper_uses_a_configured_base_url(recorded_post):
    calls = recorded_post(FakeResponse())
    wrapper = Wrapper(
        tavily_api_key="test-key", timeout_seconds=4.0, api_base_url="https://tavily.example.invalid"
    )

    wrapper.raw_results(query="q")

    assert calls[0][0] == "https://tavily.example.invalid/search"


def test_non_200_raises_without_the_response_body_or_key(recorded_post):
    recorded_post(FakeResponse(401, {"detail": {"error": "Invalid key test-key for private-query"}}))
    wrapper = Wrapper(tavily_api_key="test-key", timeout_seconds=4.0)

    with pytest.raises(ValueError) as raised:
        wrapper.raw_results(query="private-query")

    assert str(raised.value) == "Tavily search failed with HTTP 401."


def test_a_stalled_server_times_out_instead_of_blocking():
    # Loopback only (allowed by the suite's network guard). The server never
    # accepts or answers; the connection completes in the listen backlog.
    # The request runs in a daemon thread so a missing timeout fails this
    # test instead of hanging the suite.
    outcome = {}

    with socket.create_server(("127.0.0.1", 0)) as server:
        port = server.getsockname()[1]
        wrapper = Wrapper(
            tavily_api_key="test-key", timeout_seconds=0.2, api_base_url=f"http://127.0.0.1:{port}"
        )

        def search():
            try:
                wrapper.raw_results(query="q")
            except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
                outcome["error"] = exc

        started = time.monotonic()
        thread = threading.Thread(target=search, daemon=True)
        thread.start()
        thread.join(5.0)

    assert not thread.is_alive(), "the request was still blocked after 5 seconds"
    assert isinstance(outcome.get("error"), requests.exceptions.Timeout)
    assert time.monotonic() - started < 2.0


# ---------------------------------------------------------------------------
# Injection into TavilySearch
# ---------------------------------------------------------------------------

class RecordingTavilySearch:
    """Records constructor arguments, like the suite's other client stand-ins."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        RecordingTavilySearch.instances.append(self)


@pytest.fixture
def recording_tavily(monkeypatch):
    RecordingTavilySearch.instances = []
    monkeypatch.setattr(workflow, "TavilySearch", RecordingTavilySearch)
    return RecordingTavilySearch.instances


def test_web_search_tool_injects_the_timeout_wrapper(recording_tavily, settings):
    tool = workflow.web_search_tool(include_domains=["cisa.gov", "cisa.gov"])

    assert workflow.web_search_tool(include_domains=["cisa.gov"]) is tool  # cached
    [instance] = recording_tavily
    wrapper = instance.kwargs["api_wrapper"]
    assert isinstance(wrapper, Wrapper)
    assert wrapper.timeout_seconds == settings.tavily_timeout_seconds
    assert wrapper.tavily_api_key.get_secret_value() == settings.tavily_api_key.get_secret_value()
    # Passing the key to TavilySearch would replace the wrapper with the
    # library's default, which has no timeout.
    assert "tavily_api_key" not in instance.kwargs
    assert instance.kwargs["include_domains"] == ["cisa.gov"]
    assert instance.kwargs["max_results"] == 5


# ---------------------------------------------------------------------------
# {"error": ...} results from TavilySearch
# ---------------------------------------------------------------------------

def test_an_error_result_is_a_failed_search_logged_by_type_only(fake_rag, settings, monkeypatch, caplog):
    monkeypatch.setattr(settings, "max_retries", 0)
    secret = "GET https://api.tavily.com/search?q=private-query Authorization: Bearer test-key"
    fake_rag(web=FakeWebSearch({"error": requests.exceptions.ReadTimeout(secret)}))
    caplog.set_level(logging.INFO, logger=workflow.__name__)

    result = workflow.ask("Cite official guidance on backups.")

    assert "Web search failed" in result["trace"]
    assert "Web fallback → Tavily search" not in result["trace"]
    assert result["source_used"] == "insufficient_evidence"
    assert result["citations"] == []

    [record] = [r for r in caplog.records if r.getMessage().startswith("External web search failed")]
    assert record.getMessage() == "External web search failed (ReadTimeout)"
    assert record.levelno == logging.WARNING
    assert record.exc_info is None  # no traceback, so no exception message
    assert "private-query" not in caplog.text
    assert "test-key" not in caplog.text
    assert "backups" not in record.getMessage()


def test_a_normal_result_is_not_treated_as_an_error(fake_rag, settings, monkeypatch, caplog):
    monkeypatch.setattr(settings, "max_retries", 0)
    fake_rag(web=FakeWebSearch({"results": []}))
    caplog.set_level(logging.INFO, logger=workflow.__name__)

    result = workflow.ask("Cite official guidance on backups.")

    assert "Web fallback → Tavily search" in result["trace"]
    assert "Web search failed" not in result["trace"]
    assert not [r for r in caplog.records if r.getMessage().startswith("External web search failed")]


# ---------------------------------------------------------------------------
# H1-2: retry behavior (documented in the README's timeout and retry budget)
# ---------------------------------------------------------------------------

class StalledHttpServer:
    """A loopback HTTP server that reads each request's headers and never answers.

    One selector thread accepts connections and counts complete request
    headers, so every attempt a client makes is counted, on whichever
    connection it arrives. Leaving the ``with`` block stops the thread
    (bounded join) and closes every socket, which also ends any client
    still waiting on a response.
    """

    def __init__(self):
        self._listener = socket.create_server(("127.0.0.1", 0))
        self._listener.setblocking(False)
        self.port = self._listener.getsockname()[1]
        self.headers = []  # lowercased header lines of each request, in order
        self._buffers = {}
        self._changed = threading.Condition()
        self._stop = threading.Event()
        self._selector = selectors.DefaultSelector()
        self._selector.register(self._listener, selectors.EVENT_READ)
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._stop.set()
        self._thread.join(5.0)
        for conn in list(self._buffers):
            conn.close()
        self._selector.close()
        self._listener.close()
        assert not self._thread.is_alive(), "the server thread did not stop"

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}"

    def _serve(self):
        while not self._stop.is_set():
            for key, _ in self._selector.select(timeout=0.05):
                if key.fileobj is self._listener:
                    conn, _ = self._listener.accept()
                    conn.setblocking(False)
                    self._buffers[conn] = b""
                    self._selector.register(conn, selectors.EVENT_READ)
                    continue
                conn = key.fileobj
                try:
                    data = conn.recv(65536)
                except OSError:
                    data = b""
                if not data:
                    self._selector.unregister(conn)
                    self._buffers.pop(conn, None)
                    conn.close()
                    continue
                buffer = self._buffers[conn] + data
                if b"\r\n\r\n" in buffer:  # complete headers: one request
                    head = buffer.split(b"\r\n\r\n", 1)[0]
                    with self._changed:
                        self.headers.append(head.decode("latin-1").lower().split("\r\n"))
                        self._changed.notify_all()
                    buffer = b""  # the body is ignored, and no response is sent
                self._buffers[conn] = buffer

    def wait_for_requests(self, count, timeout=5.0):
        with self._changed:
            self._changed.wait_for(lambda: len(self.headers) >= count, timeout)
            return len(self.headers)


def run_bounded(call, timeout=15.0):
    """Run ``call`` in a daemon thread; a missing timeout fails the test instead of hanging it."""
    outcome = {}

    def target():
        try:
            outcome["result"] = call()
        except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
            outcome["error"] = exc

    started = time.monotonic()
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), f"the call was still running after {timeout} seconds"
    outcome["elapsed"] = time.monotonic() - started
    return outcome


@pytest.fixture
def no_proxy(monkeypatch):
    """Loopback requests must not be sent through a proxy from the environment."""
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


class RecordingClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_chat_models_get_the_configured_openai_timeout_and_retries(settings, monkeypatch):
    # Non-default values, so a hard-coded 60 s / 2 retries would fail.
    monkeypatch.setattr(settings, "openai_timeout_seconds", 7.5)
    monkeypatch.setattr(settings, "openai_max_retries", 4)
    monkeypatch.setattr(workflow, "ChatOpenAI", RecordingClient)

    for client in (workflow.llm(), workflow.internal_llm()):
        assert client.kwargs["timeout"] == 7.5
        assert client.kwargs["max_retries"] == 4


def test_embeddings_get_the_configured_openai_timeout_and_retries(settings, monkeypatch):
    monkeypatch.setattr(settings, "openai_timeout_seconds", 7.5)
    monkeypatch.setattr(settings, "openai_max_retries", 4)
    monkeypatch.setattr(vectorstore, "OpenAIEmbeddings", RecordingClient)

    embeddings = vectorstore.get_embeddings()

    assert embeddings.kwargs["timeout"] == 7.5
    assert embeddings.kwargs["max_retries"] == 4


def test_openai_makes_three_attempts_against_a_stalled_server(settings, monkeypatch, no_proxy):
    # The application's chat model, built by _chat_model() from the settings,
    # against a loopback server: 1 attempt + OPENAI_MAX_RETRIES (2) retries.
    import openai
    from langchain_openai import ChatOpenAI as RealChatOpenAI

    monkeypatch.setattr(workflow, "ChatOpenAI", RealChatOpenAI)  # the conftest blocks it
    monkeypatch.setattr(settings, "openai_timeout_seconds", 0.2)
    monkeypatch.setattr(settings, "openai_max_retries", 2)

    with StalledHttpServer() as server:
        model = workflow._chat_model(base_url=f"{server.base_url}/v1")
        outcome = run_bounded(lambda: model.invoke("ping"))
        received = server.wait_for_requests(3)
        headers = list(server.headers)

    assert isinstance(outcome.get("error"), openai.APITimeoutError)
    assert received == 3
    assert len(headers) == 3  # the client had finished, so no request can follow
    assert [_header(h, "x-stainless-retry-count") for h in headers] == ["0", "1", "2"]
    assert all(h[0].startswith("post /v1/chat/completions") for h in headers)
    # 3 × 0.2 s per attempt plus about 1.1–1.5 s of backoff.
    assert outcome["elapsed"] < 10.0


def _header(lines, name):
    for line in lines[1:]:
        key, _, value = line.partition(":")
        if key.strip() == name:
            return value.strip()
    return None


def test_tavily_makes_exactly_one_request_against_a_stalled_server(no_proxy):
    with StalledHttpServer() as server:
        wrapper = Wrapper(tavily_api_key="test-key", timeout_seconds=0.2, api_base_url=server.base_url)
        outcome = run_bounded(lambda: wrapper.raw_results(query="q"))
        received = server.wait_for_requests(1)
        headers = list(server.headers)

    assert isinstance(outcome.get("error"), requests.exceptions.ReadTimeout)
    assert received == 1
    assert len(headers) == 1  # no HTTP retry: the client had finished
    assert headers[0][0].startswith("post /search")
    assert outcome["elapsed"] < 5.0


# The README's Pinecone retry figures come from the SDK's default policy.
# Reading it uses private SDK attributes (the same generated control-plane
# API as ensure_index(), via vectorstore._index_api, and the client the data
# plane builds); if an SDK upgrade changes them, this test fails and the
# README's retry budget must be re-checked. No request is sent.
def test_installed_pinecone_default_retry_policy_matches_the_readme():
    from pinecone import Pinecone as RealPinecone
    from urllib3.util.retry import Retry

    pc = RealPinecone(api_key="test-key")
    control = vectorstore._index_api(pc).api_client.rest_client
    data = pc.Index(host="test-index.example.invalid")._vector_api.api_client.rest_client

    for rest_client in (control, data):
        retry = rest_client.pool_manager.connection_pool_kw["retries"]
        assert isinstance(retry, Retry)
        assert retry.total == 5  # 6 attempts
        assert (retry.connect, retry.read, retry.status) == (None, None, None)  # only total limits
        assert retry.backoff_factor == 0.25
        assert retry.backoff_max == 120
        assert tuple(retry.status_forcelist) == (500, 502, 503, 504)
        assert retry.allowed_methods is None  # every method, POST included
        assert retry.respect_retry_after_header is True
        assert retry.retry_after_max == 21600  # Retry-After honored up to 6 hours per retry
        assert set(Retry.RETRY_AFTER_STATUS_CODES) == {413, 429, 503}
        # 0 s before the first retry, then 0.5, 1, 2, 4 s, each plus up to
        # 0.25 s of jitter (Pinecone's JitterRetry).
        backoffs = []
        for _ in range(5):
            retry = retry.increment(method="POST", url="/", error=None, response=_FakeResponse(503))
            backoffs.append(retry.get_backoff_time())
        assert all(0 <= b - base <= 0.25 for b, base in zip(backoffs, [0, 0.5, 1, 2, 4], strict=True))


# ---------------------------------------------------------------------------
# H1-5 (A): optional cap on Pinecone's Retry-After wait
# ---------------------------------------------------------------------------

# Every setting urllib3's Retry.new() copies from one retry to the next.
RETRY_SETTINGS = (
    "total", "connect", "read", "redirect", "status", "other", "allowed_methods",
    "status_forcelist", "backoff_factor", "backoff_max", "retry_after_max",
    "raise_on_redirect", "raise_on_status", "remove_headers_on_redirect",
    "respect_retry_after_header", "backoff_jitter",
)


def _policies(pc):
    """The retry policy each Pinecone HTTP client uses: (control plane, data plane)."""
    control = vectorstore._index_api(pc).api_client.rest_client
    data = pc.Index(host="test-index.example.invalid")._vector_api.api_client.rest_client
    return [client.pool_manager.connection_pool_kw["retries"] for client in (control, data)]


@pytest.fixture
def real_pinecone_client(monkeypatch, settings):
    """vectorstore._pinecone_client() with the real SDK class (no request is sent)."""
    from pinecone import Pinecone as RealPinecone
    from pydantic import SecretStr

    monkeypatch.setattr(vectorstore, "Pinecone", RealPinecone)
    monkeypatch.setattr(settings, "pinecone_api_key", SecretStr("test-key"))
    return vectorstore._pinecone_client


def test_without_a_cap_the_client_keeps_the_sdk_retry_policy(real_pinecone_client, settings):
    from pinecone import Pinecone as RealPinecone

    assert settings.pinecone_retry_after_max_seconds is None
    pc = real_pinecone_client()

    assert pc._openapi_config.retries is None  # nothing installed: the SDK chooses
    sdk_default = _policies(RealPinecone(api_key="test-key"))[0]
    for policy in _policies(pc):
        assert type(policy) is type(sdk_default)
        assert {name: getattr(policy, name) for name in RETRY_SETTINGS} == {
            name: getattr(sdk_default, name) for name in RETRY_SETTINGS
        }
        assert policy.retry_after_max == 21600


@pytest.mark.parametrize("cap", [0, 30, 21600])
def test_a_cap_changes_only_retry_after_max(real_pinecone_client, settings, monkeypatch, cap):
    # Compared setting by setting with the installed SDK's own default, so
    # an SDK upgrade that changes its default policy fails here.
    from pinecone import Pinecone as RealPinecone

    monkeypatch.setattr(settings, "pinecone_retry_after_max_seconds", cap)
    sdk_default = _policies(RealPinecone(api_key="test-key"))[0]

    for policy in _policies(real_pinecone_client()):  # setup calls and queries alike
        assert type(policy) is type(sdk_default)  # Pinecone's JitterRetry
        assert policy.retry_after_max == cap
        assert {name: getattr(policy, name) for name in RETRY_SETTINGS if name != "retry_after_max"} == {
            name: getattr(sdk_default, name) for name in RETRY_SETTINGS if name != "retry_after_max"
        }


def test_a_cap_survives_each_retry_and_shortens_a_long_retry_after(real_pinecone_client, settings, monkeypatch):
    monkeypatch.setattr(settings, "pinecone_retry_after_max_seconds", 30)
    [policy, _] = _policies(real_pinecone_client())

    retry = policy.increment(method="GET", url="/", error=None, response=_FakeResponse(429))
    retry = retry.increment(method="GET", url="/", error=None, response=_FakeResponse(503))

    assert type(retry) is type(policy) and retry.retry_after_max == 30  # Retry.new() keeps both
    assert retry.parse_retry_after("3600") == 30  # shortened, not refused
    assert retry.parse_retry_after("12") == 12  # a shorter value is kept


class _FakeResponse:
    """Just enough of a urllib3 response for Retry.increment()."""

    def __init__(self, status):
        self.status = status

    def get_redirect_location(self):
        return False
