"""H1: finite timeouts for Pinecone and Tavily calls (no external network).

Pinecone wrapper tests live in tests/test_vectorstore.py. Here: the settings,
and the Tavily wrapper, its injection, and failed-search handling.
"""

import logging
import socket
import threading
import time

import pytest
import requests
from pydantic import ValidationError

from app.core import config
from app.rag import workflow
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
