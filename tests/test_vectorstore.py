"""Tests for Pinecone index validation and document upserts (no network)."""

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore as LangChainPineconeVectorStore

from app.rag import vectorstore


HOST = "test-index-abc123.svc.example.invalid"


class FakeIndexApi:
    """Stand-in for the SDK's generated control-plane API (``pc.db.index._index_api``)."""

    def __init__(self, pinecone):
        self.pinecone = pinecone
        self.calls = []

    def list_indexes(self, **kwargs):
        self.calls.append(("list_indexes", (), kwargs))
        return {"indexes": [SimpleNamespace(name=name) for name in self.pinecone.indexes]}

    def describe_index(self, name, **kwargs):
        self.calls.append(("describe_index", (name,), kwargs))
        self.pinecone.on_describe()
        info = self.pinecone.indexes[name]
        if self.pinecone.polls_until_ready > 0:
            self.pinecone.polls_until_ready -= 1
            return index_info(dimension=info.dimension, ready=False, host=info.host)
        return info

    def create_index(self, create_index_request, **kwargs):
        # Recorded in ``pinecone.created`` (not ``calls``), as the public
        # create_index() used to be: (name, dimension, metric, kwargs).
        request = create_index_request
        self.pinecone.created.append((request.name, request.dimension, request.metric, kwargs))
        self.pinecone.create_requests.append(request)
        if self.pinecone.create_error is not None:
            error, existing = self.pinecone.create_error
            if existing is not None:  # another creator got there first
                self.pinecone.indexes[request.name] = existing
            raise error
        self.pinecone.indexes[request.name] = index_info(dimension=request.dimension)


class FakePinecone:
    """Minimal stand-in for the Pinecone client."""

    def __init__(self, indexes=None, polls_until_ready=0, create_error=None):
        self.indexes = dict(indexes or {})
        self.created = []
        self.create_requests = []
        self.create_error = create_error  # (exception, index_info that exists afterwards or None)
        self.opened = []
        self.polls_until_ready = polls_until_ready
        self.on_describe = lambda: None
        self.db = SimpleNamespace(index=SimpleNamespace(_index_api=FakeIndexApi(self)))
        self._openapi_config = SimpleNamespace(retries=None)  # H1-5 (A) sets a policy here

    def __call__(self, api_key):
        return self

    @property
    def control_calls(self):
        return self.db.index._index_api.calls

    # The public list, describe and create calls take no HTTP timeout;
    # ensure_index() must not use them.
    def list_indexes(self):
        raise AssertionError("untimed list_indexes() was called")

    def describe_index(self, name):
        raise AssertionError("untimed describe_index() was called")

    def create_index(self, *args, **kwargs):
        raise AssertionError("untimed create_index() was called")

    def Index(self, name="", host=""):
        # Without a host, the SDK would describe the index again (untimed).
        assert host, "Index() was called without the known host"
        self.opened.append((name, host))
        return f"index:{name}"


@pytest.fixture
def pinecone(monkeypatch):
    def install(indexes=None, **kwargs):
        fake = FakePinecone(indexes, **kwargs)
        monkeypatch.setattr(vectorstore, "Pinecone", fake)
        return fake

    return install


def index_info(dimension=1536, ready=True, host=HOST):
    return SimpleNamespace(dimension=dimension, status=SimpleNamespace(ready=ready), host=host)


def ensure_index():
    # The real function runs; the conftest blocks the Pinecone class, and
    # the ``pinecone`` fixture replaces it with FakePinecone.
    return vectorstore.ensure_index()


@pytest.mark.parametrize(
    "model, dimension",
    [
        ("text-embedding-3-small", 1536),
        ("TEXT-EMBEDDING-3-LARGE", 3072),
        ("text-embedding-ada-002", 1536),
    ],
)
def test_embedding_dimension_lookup(model, dimension):
    assert vectorstore.get_embedding_dimension(model) == dimension


def test_unknown_embedding_model_is_rejected():
    with pytest.raises(ValueError, match="Unsupported embedding model"):
        vectorstore.get_embedding_dimension("mystery-embed")


def test_existing_index_with_matching_dimension_is_reused(pinecone, settings):
    fake = pinecone({settings.pinecone_index_name: index_info()})

    assert ensure_index() == f"index:{settings.pinecone_index_name}"
    assert fake.created == []


def test_dimension_mismatch_fails_without_touching_index(pinecone, settings):
    fake = pinecone({settings.pinecone_index_name: index_info(dimension=3072)})

    with pytest.raises(RuntimeError, match="Index dimension is 3072"):
        ensure_index()

    assert settings.pinecone_index_name in fake.indexes
    assert fake.created == []


def test_index_that_is_not_ready_fails(pinecone, settings):
    pinecone({settings.pinecone_index_name: index_info(ready=False)})

    with pytest.raises(RuntimeError, match="not ready"):
        ensure_index()


def test_missing_index_is_created_with_cosine_metric(pinecone, settings):
    fake = pinecone({})

    ensure_index()

    assert [created[:3] for created in fake.created] == [(settings.pinecone_index_name, 1536, "cosine")]


# ---------------------------------------------------------------------------
# H1-1: timeouts for the Pinecone setup calls in ensure_index()
# ---------------------------------------------------------------------------

def test_existing_index_setup_calls_get_the_timeout_and_reuse_the_host(pinecone, settings):
    name = settings.pinecone_index_name
    fake = pinecone({name: index_info()})

    ensure_index()

    timeout = {"_request_timeout": settings.pinecone_timeout_seconds}
    assert fake.control_calls == [("list_indexes", (), timeout), ("describe_index", (name,), timeout)]
    assert fake.opened == [(name, HOST)]


def test_missing_index_skips_sdk_polling_and_polls_with_the_timeout(pinecone, settings, clock):
    name = settings.pinecone_index_name
    fake = pinecone({}, polls_until_ready=2)

    ensure_index()

    timeout = {"_request_timeout": settings.pinecone_timeout_seconds}
    # H1-4: the create request goes through the generated API with the
    # per-attempt timeout, so the SDK's own (unlimited) polling never runs.
    assert fake.created == [(name, 1536, "cosine", timeout)]
    assert fake.control_calls == [("list_indexes", (), timeout)] + [("describe_index", (name,), timeout)] * 3
    assert clock.sleeps == [2, 2]
    assert fake.opened == [(name, HOST)]


# ---------------------------------------------------------------------------
# H1-4: the index create request has a per-attempt timeout; HTTP 409 waits
# ---------------------------------------------------------------------------

def conflict():
    from pinecone.exceptions import PineconeApiException

    return PineconeApiException(status=409, reason="Conflict")


def test_the_create_request_body_matches_the_public_sdk_method(pinecone, settings):
    # The request the public create_index() builds, from the same SDK factory
    # and with its default arguments spelled out.
    from pinecone import ServerlessSpec
    from pinecone.db_control.request_factory import PineconeDBControlRequestFactory

    fake = pinecone({})

    ensure_index()

    expected = PineconeDBControlRequestFactory.create_index_request(
        name=settings.pinecone_index_name,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        dimension=1536,
        metric="cosine",
        deletion_protection="disabled",
        vector_type="dense",
        tags=None,
    )
    [request] = fake.create_requests
    assert request.to_dict() == expected.to_dict()
    assert request.to_dict() == {
        "name": settings.pinecone_index_name,
        "dimension": 1536,
        "metric": "cosine",
        "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}},
        "deletion_protection": "disabled",
        "vector_type": "dense",
        "tags": {},  # the factory always sends tags, as the public method does
    }


def test_a_conflict_on_create_waits_for_the_existing_index(pinecone, settings, clock, caplog):
    name = settings.pinecone_index_name
    fake = pinecone({}, polls_until_ready=1, create_error=(conflict(), index_info()))

    with caplog.at_level("INFO", logger="app.rag.vectorstore"):
        assert ensure_index() == f"index:{name}"

    timeout = {"_request_timeout": settings.pinecone_timeout_seconds}
    assert len(fake.created) == 1  # not sent again
    assert fake.control_calls == [("list_indexes", (), timeout)] + [("describe_index", (name,), timeout)] * 2
    assert clock.sleeps == [2]
    assert fake.opened == [(name, HOST)]
    assert "already exists (HTTP 409)" in caplog.text


def test_a_conflicting_index_with_another_dimension_is_refused(pinecone, settings, clock):
    fake = pinecone({}, create_error=(conflict(), index_info(dimension=3072)))

    with pytest.raises(RuntimeError, match="Index dimension is 3072"):
        ensure_index()

    assert settings.pinecone_index_name in fake.indexes  # never deleted
    assert fake.opened == []


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 429, 500])
def test_other_create_errors_still_fail(pinecone, clock, status):
    from pinecone.exceptions import PineconeApiException

    fake = pinecone({}, create_error=(PineconeApiException(status=status, reason="error"), None))

    with pytest.raises(PineconeApiException) as raised:
        ensure_index()

    assert raised.value.status == status
    assert [call[0] for call in fake.control_calls] == ["list_indexes"]  # no readiness wait
    assert fake.opened == []


def test_a_create_timeout_is_not_treated_as_a_conflict(pinecone, clock):
    from urllib3.exceptions import MaxRetryError

    fake = pinecone({}, create_error=(MaxRetryError(None, "/indexes", "read timed out"), None))

    with pytest.raises(MaxRetryError):
        ensure_index()

    assert [call[0] for call in fake.control_calls] == ["list_indexes"]
    assert fake.opened == []


class FakeClock:
    """A monotonic clock that only moves when the code sleeps or a describe call takes time."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(vectorstore, "time", fake)
    return fake


def test_readiness_wait_stops_at_the_deadline(pinecone, clock):
    fake = pinecone({}, polls_until_ready=1000)

    def slow_describe():  # each describe takes 25 s, like a slow call with retries
        clock.now += 25

    fake.on_describe = slow_describe

    with pytest.raises(TimeoutError, match="within 120 seconds"):
        ensure_index()

    describes = [call for call in fake.control_calls if call[0] == "describe_index"]
    assert len(describes) == 5  # well before the 60-poll limit
    assert clock.now - 1000.0 == 5 * 25 + 4 * 2  # the deadline is checked between polls
    assert fake.opened == []


def test_last_sleep_is_cut_to_the_time_left(pinecone, clock):
    fake = pinecone({}, polls_until_ready=1000)
    fake.on_describe = lambda: setattr(clock, "now", clock.now + 58.5)

    with pytest.raises(TimeoutError):
        ensure_index()

    # After two describes and one sleep (119 s) only 1 s is left before the deadline.
    assert clock.sleeps == [2, 1]
    assert clock.now - 1000.0 == 120 + 58.5  # one more describe ran past the deadline


def test_readiness_wait_keeps_the_poll_limit(pinecone, clock):
    fake = pinecone({}, polls_until_ready=1000)
    clock.sleep = clock.sleeps.append  # time does not move

    with pytest.raises(TimeoutError, match="within 120 seconds"):
        ensure_index()

    describes = [call for call in fake.control_calls if call[0] == "describe_index"]
    assert len(describes) == vectorstore.INDEX_READY_POLLS == 60
    assert fake.opened == []


def test_index_that_becomes_ready_on_the_last_poll_is_used(pinecone, clock, settings):
    fake = pinecone({}, polls_until_ready=59)

    ensure_index()

    assert fake.opened == [(settings.pinecone_index_name, HOST)]


# Private SDK dependency (H1-1). ensure_index() reaches the generated
# control-plane API through ``pc.db.index._index_api``, because the public
# list_indexes() and describe_index() take no HTTP timeout. This checks the
# shape it relies on in the installed SDK, without contacting Pinecone.
def test_installed_pinecone_sdk_has_the_control_plane_methods_ensure_index_uses():
    import inspect

    from pinecone import Pinecone as RealPinecone
    from pinecone.core.openapi.db_control.api.manage_indexes_api import ManageIndexesApi

    pc = RealPinecone(api_key="test-key")
    api = vectorstore._index_api(pc)

    assert isinstance(api, ManageIndexesApi)
    assert "_request_timeout" in api.list_indexes.params_map["all"]
    assert "_request_timeout" in api.describe_index.params_map["all"]
    assert api.describe_index.params_map["required"] == ["index_name"]
    assert "host" in inspect.signature(RealPinecone.Index).parameters
    # H1-4: the create request uses the generated method and the SDK's own
    # request factory, and recognizes a conflict by its HTTP status.
    from pinecone.db_control.request_factory import PineconeDBControlRequestFactory
    from pinecone.exceptions import PineconeApiException

    assert "_request_timeout" in api.create_index.params_map["all"]
    assert api.create_index.params_map["required"] == ["create_index_request"]
    assert callable(PineconeDBControlRequestFactory.create_index_request)
    assert PineconeApiException(status=409).status == 409


# ---------------------------------------------------------------------------
# H1-1 against loopback servers, with the real Pinecone SDK
# ---------------------------------------------------------------------------

def index_json(name, ready=True):
    return {
        "name": name,
        "metric": "cosine",
        "dimension": 1536,
        "host": HOST,
        "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}},
        "status": {"ready": ready, "state": "Ready" if ready else "Initializing"},
        "deletion_protection": "disabled",
        "vector_type": "dense",
    }


@pytest.fixture
def loopback_pinecone(monkeypatch):
    """Points the real Pinecone client's control plane at a loopback URL."""
    from pinecone import Pinecone as RealPinecone

    def install(base_url):
        monkeypatch.setattr(
            vectorstore, "Pinecone", lambda api_key: RealPinecone(api_key=api_key, host=base_url)
        )

    return install


def test_real_sdk_reads_the_raw_control_plane_responses(loopback_pinecone, settings):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    name = settings.pinecone_index_name
    paths = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            paths.append(self.path)
            body = {"/indexes": {"indexes": [index_json(name)]}, f"/indexes/{name}": index_json(name)}
            payload = json.dumps(body[self.path]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            loopback_pinecone(f"http://127.0.0.1:{server.server_address[1]}")
            index = ensure_index()
        finally:
            server.shutdown()
            thread.join(5)

    assert paths == ["/indexes", f"/indexes/{name}"]  # no extra describe for the host
    assert index.config.host == f"https://{HOST}"


def test_a_stalled_setup_call_times_out_instead_of_blocking(loopback_pinecone, settings, monkeypatch):
    # The server accepts connections (in the listen backlog) and never
    # answers. Each attempt times out after 0.2 s; the SDK's retries and
    # their backoff (about 8 s) still apply, so this test takes ~10 s.
    import socket
    import threading
    import time

    monkeypatch.setattr(settings, "pinecone_timeout_seconds", 0.2)
    outcome = {}

    with socket.create_server(("127.0.0.1", 0), backlog=32) as server:
        loopback_pinecone(f"http://127.0.0.1:{server.getsockname()[1]}")

        def setup():
            try:
                ensure_index()
            except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
                outcome["error"] = exc

        started = time.monotonic()
        thread = threading.Thread(target=setup, daemon=True)
        thread.start()
        thread.join(30.0)
        elapsed = time.monotonic() - started

    assert not thread.is_alive(), "the setup call was still blocked after 30 seconds"
    assert type(outcome.get("error")).__name__ == "MaxRetryError"
    assert elapsed < 20.0


# ---------------------------------------------------------------------------
# H1-4 against a loopback server, with the real Pinecone SDK
# ---------------------------------------------------------------------------

@pytest.fixture
def create_server(settings):
    """A loopback control plane without the configured index.

    ``mode`` decides how ``POST /indexes`` answers: ``created`` (201),
    ``conflict`` (409, as Pinecone answers for an existing name) or
    ``stall`` (never answers until the test ends). Readiness describes
    answer ready, so the readiness wait does not sleep.
    """
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    name = settings.pinecone_index_name
    state = {"mode": "created", "requests": [], "bodies": [], "release": threading.Event()}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, body):
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            state["requests"].append(("GET", self.path))
            if self.path == "/indexes":
                self._send(200, {"indexes": []})
            else:
                self._send(200, index_json(name))

        def do_POST(self):
            state["requests"].append(("POST", self.path))
            length = int(self.headers.get("Content-Length", 0))
            state["bodies"].append(json.loads(self.rfile.read(length)))
            if state["mode"] == "stall":
                state["release"].wait(30)
                return
            if state["mode"] == "conflict":
                self._send(409, {"error": {"code": "ALREADY_EXISTS", "message": "Resource already exists"}, "status": 409})
                return
            self._send(201, index_json(name, ready=False))

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["base_url"] = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield state
    finally:
        state["release"].set()
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_real_sdk_sends_the_create_request_with_the_expected_body(loopback_pinecone, create_server, settings):
    name = settings.pinecone_index_name
    loopback_pinecone(create_server["base_url"])

    index = ensure_index()

    assert create_server["requests"] == [("GET", "/indexes"), ("POST", "/indexes"), ("GET", f"/indexes/{name}")]
    assert create_server["bodies"] == [
        {
            "name": name,
            "dimension": 1536,
            "metric": "cosine",
            "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}},
            "deletion_protection": "disabled",
            "vector_type": "dense",
            "tags": {},
        }
    ]
    assert index.config.host == f"https://{HOST}"

    # The public method (what ensure_index() used before H1-4) sends the same body.
    from pinecone import Pinecone as RealPinecone, ServerlessSpec

    RealPinecone(api_key="test-key", host=create_server["base_url"]).create_index(
        name=name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        timeout=-1,
    )
    assert len(create_server["bodies"]) == 2
    assert create_server["bodies"][1] == create_server["bodies"][0]


def test_real_sdk_conflict_on_create_uses_the_existing_index(loopback_pinecone, create_server, settings):
    name = settings.pinecone_index_name
    create_server["mode"] = "conflict"
    loopback_pinecone(create_server["base_url"])

    index = ensure_index()

    assert create_server["requests"] == [("GET", "/indexes"), ("POST", "/indexes"), ("GET", f"/indexes/{name}")]
    assert index.config.host == f"https://{HOST}"


def test_a_stalled_create_request_times_out_instead_of_blocking(loopback_pinecone, create_server, settings, monkeypatch):
    # The server reads the create request and never answers. Each attempt
    # times out after 0.2 s. The SDK retries a POST after a read error, so
    # the create request is sent 6 times (1 + 5 retries) with about 8 s of
    # backoff; this test takes ~10 s. Before H1-4 this call never returned.
    import threading
    import time

    monkeypatch.setattr(settings, "pinecone_timeout_seconds", 0.2)
    create_server["mode"] = "stall"
    loopback_pinecone(create_server["base_url"])
    outcome = {}

    def setup():
        try:
            ensure_index()
        except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
            outcome["error"] = exc

    started = time.monotonic()
    thread = threading.Thread(target=setup, daemon=True)
    thread.start()
    thread.join(30.0)
    elapsed = time.monotonic() - started

    assert not thread.is_alive(), "the create request was still blocked after 30 seconds"
    assert type(outcome.get("error")).__name__ == "MaxRetryError"
    assert elapsed < 20.0
    posts = [request for request in create_server["requests"] if request[0] == "POST"]
    assert len(posts) == 6  # the SDK resends the create request after each read timeout


# ---------------------------------------------------------------------------
# H1-5 (A): the optional Retry-After cap
# ---------------------------------------------------------------------------

def test_ensure_index_leaves_the_retry_policy_alone_by_default(pinecone, settings):
    fake = pinecone({settings.pinecone_index_name: index_info()})

    ensure_index()

    assert settings.pinecone_retry_after_max_seconds is None
    assert fake._openapi_config.retries is None


def test_ensure_index_installs_the_capped_policy_before_its_first_call(pinecone, settings, monkeypatch):
    monkeypatch.setattr(settings, "pinecone_retry_after_max_seconds", 30)
    fake = pinecone({settings.pinecone_index_name: index_info()})
    seen = []
    fake.on_describe = lambda: seen.append(fake._openapi_config.retries)

    ensure_index()

    [policy] = seen  # already installed when the describe call ran
    assert policy.retry_after_max == 30
    assert fake._openapi_config.retries is policy


@pytest.fixture
def retry_after_server(settings):
    """A loopback control plane whose first answer is ``429`` with
    ``Retry-After: 3600``; later requests succeed for an existing index."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    name = settings.pinecone_index_name
    state = {"requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(self.path)
            if len(state["requests"]) == 1:
                status, body, extra = 429, {"error": {"code": "RESOURCE_EXHAUSTED"}}, {"Retry-After": "3600"}
            elif self.path == "/indexes":
                status, body, extra = 200, {"indexes": [index_json(name)]}, {}
            else:
                status, body, extra = 200, index_json(name), {}
            payload = json.dumps(body).encode()
            self.send_response(status)
            for header, value in {"Content-Type": "application/json", "Content-Length": str(len(payload)), **extra}.items():
                self.send_header(header, value)
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["base_url"] = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


@pytest.fixture
def retry_sleeps(monkeypatch):
    """Records the waits urllib3's retry logic asks for, without sleeping."""
    import time as real_time

    import urllib3.util.retry as retry_module

    sleeps = []
    monkeypatch.setattr(
        retry_module, "time", SimpleNamespace(sleep=sleeps.append, time=real_time.time)
    )
    return sleeps


@pytest.mark.parametrize(
    "cap, expected",
    [
        (None, [3600]),  # default: the SDK honors the header (up to 6 hours)
        (30, [30]),  # shortened to the cap
        (2, [2]),
    ],
)
def test_real_sdk_retry_after_wait_follows_the_cap(
    loopback_pinecone, retry_after_server, retry_sleeps, settings, monkeypatch, cap, expected
):
    monkeypatch.setattr(settings, "pinecone_retry_after_max_seconds", cap)
    loopback_pinecone(retry_after_server["base_url"])

    index = ensure_index()

    name = settings.pinecone_index_name
    assert retry_after_server["requests"] == ["/indexes", "/indexes", f"/indexes/{name}"]  # retried once
    assert retry_sleeps == expected
    assert index.config.host == f"https://{HOST}"


def test_real_sdk_cap_of_zero_falls_back_to_the_normal_backoff(
    loopback_pinecone, retry_after_server, retry_sleeps, settings, monkeypatch
):
    monkeypatch.setattr(settings, "pinecone_retry_after_max_seconds", 0)
    loopback_pinecone(retry_after_server["base_url"])

    ensure_index()

    # The first retry's backoff is 0 s plus up to 0.25 s of jitter (JitterRetry).
    assert len(retry_after_server["requests"]) == 3
    assert all(0 <= seconds <= 0.25 for seconds in retry_sleeps)


def test_add_documents_with_no_chunks_skips_pinecone():
    assert vectorstore.add_documents([]) == []


def test_add_documents_rejects_mismatched_ids():
    with pytest.raises(ValueError, match="number of IDs"):
        vectorstore.add_documents(["chunk"], ids=["a", "b"])


def test_add_documents_passes_ids_through(monkeypatch):
    calls = []

    class FakeStore:
        def add_documents(self, chunks, **kwargs):
            calls.append((chunks, kwargs))
            return kwargs.get("ids", ["generated"])

    monkeypatch.setattr(vectorstore, "get_vectorstore", FakeStore)

    assert vectorstore.add_documents(["a"], ids=["id-a"]) == ["id-a"]
    assert vectorstore.add_documents(["b"]) == ["generated"]
    assert calls == [(["a"], {"ids": ["id-a"]}), (["b"], {})]


# ---------------------------------------------------------------------------
# H1: finite timeouts for Pinecone queries and upserts
# ---------------------------------------------------------------------------

TIMEOUT = 7.5


class FakeAsyncResult:
    """What the SDK returns for ``async_req=True``: a result fetched with get()."""

    def __init__(self, value):
        self.value = value
        self.fetched = False

    def get(self):
        self.fetched = True
        return self.value


class FakeIndex:
    """Records data-plane calls; never touches the network."""

    def __init__(self):
        self.calls = []
        self.config = SimpleNamespace(host="https://example-index.invalid", api_key="test-key")
        self.async_results = []

    def query(self, *args, **kwargs):
        self.calls.append(("query", args, kwargs))
        return {"matches": [{"id": "d1", "score": 0.9, "metadata": {"text": "Use GlobalProtect."}}]}

    def upsert(self, *args, **kwargs):
        self.calls.append(("upsert", args, kwargs))
        if kwargs.get("async_req"):
            result = FakeAsyncResult({"upserted_count": len(kwargs["vectors"])})
            self.async_results.append(result)
            return result
        return {"upserted_count": len(kwargs["vectors"])}

    def describe_index_stats(self):
        return {"namespaces": {}}


def timed(index=None, timeout=TIMEOUT):
    index = index or FakeIndex()
    return index, vectorstore._TimeoutIndex(index, timeout)


def test_query_gets_the_default_timeout():
    index, wrapped = timed()

    result = wrapped.query(vector=[0.1], top_k=2, namespace="ns")

    assert index.calls == [("query", (), {"vector": [0.1], "top_k": 2, "namespace": "ns", "_request_timeout": TIMEOUT})]
    assert result["matches"][0]["id"] == "d1"


@pytest.mark.parametrize("explicit", [3, (1.0, 2.0), None])
def test_an_explicit_query_timeout_is_kept(explicit):
    index, wrapped = timed()

    wrapped.query(vector=[0.1], top_k=1, _request_timeout=explicit)

    assert index.calls[0][2]["_request_timeout"] == explicit


def test_synchronous_upsert_gets_the_default_timeout():
    index, wrapped = timed()

    result = wrapped.upsert(vectors=[("a", [0.1])], namespace="ns")

    assert index.calls[0][2]["_request_timeout"] == TIMEOUT
    assert "async_req" not in index.calls[0][2]
    assert result == {"upserted_count": 1}


def test_async_upsert_gets_the_default_timeout_and_returns_the_async_result():
    index, wrapped = timed()

    result = wrapped.upsert(vectors=[("a", [0.1])], namespace="ns", async_req=True)

    assert index.calls[0][2]["_request_timeout"] == TIMEOUT
    assert index.calls[0][2]["async_req"] is True
    assert result is index.async_results[0]


def test_an_explicit_upsert_timeout_is_kept():
    index, wrapped = timed()

    wrapped.upsert(vectors=[("a", [0.1])], _request_timeout=2)

    assert index.calls[0][2]["_request_timeout"] == 2


def test_other_attributes_including_config_are_the_wrapped_index():
    index, wrapped = timed()

    assert wrapped.config is index.config
    assert wrapped.config.host == "https://example-index.invalid"
    assert wrapped.describe_index_stats() == {"namespaces": {}}
    with pytest.raises(AttributeError):
        wrapped.no_such_attribute


def test_get_vectorstore_wraps_the_index_and_keeps_index_and_namespace(pinecone, settings, monkeypatch):
    fake = pinecone({settings.pinecone_index_name: index_info()})
    created = []

    def recording_store(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(**kwargs)

    # Recording stand-ins, as elsewhere in the suite; the conftest guard
    # still blocks the real classes in every other test.
    monkeypatch.setattr(vectorstore, "PineconeVectorStore", recording_store)
    monkeypatch.setattr(vectorstore, "get_embeddings", lambda: "embeddings")

    store = vectorstore.get_vectorstore()

    assert vectorstore.get_vectorstore() is store  # still cached
    [kwargs] = created
    assert isinstance(kwargs["index"], vectorstore._TimeoutIndex)
    assert kwargs["index"]._index == f"index:{settings.pinecone_index_name}"
    assert kwargs["index"]._timeout_seconds == settings.pinecone_timeout_seconds
    assert kwargs["namespace"] == settings.pinecone_namespace
    assert kwargs["embedding"] == "embeddings"
    assert fake.created == []  # the existing index was reused, not created


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[0.1, 0.2] for _ in texts]


def library_store(wrapped):
    """The attributes LangChain's PineconeVectorStore methods read from ``self``.

    The library's methods run unchanged against our wrapper; no
    PineconeVectorStore or Pinecone client is constructed.
    """
    return SimpleNamespace(
        index=wrapped, _namespace="company-it-kb", _text_key="text", _embedding=FakeEmbeddings()
    )


@pytest.mark.parametrize("async_req", [True, False])
def test_langchain_upserts_through_the_wrapper_get_the_timeout(async_req):
    index, wrapped = timed()

    ids = LangChainPineconeVectorStore.add_texts(
        library_store(wrapped), ["one", "two"], ids=["a", "b"], async_req=async_req
    )

    assert ids == ["a", "b"]
    upserts = [kwargs for name, _, kwargs in index.calls if name == "upsert"]
    assert upserts and all(kwargs["_request_timeout"] == TIMEOUT for kwargs in upserts)
    assert all(kwargs["async_req"] is async_req for kwargs in upserts)
    assert all(kwargs["namespace"] == "company-it-kb" for kwargs in upserts)
    if async_req:
        assert index.async_results and all(result.fetched for result in index.async_results)


def test_langchain_retrieval_query_through_the_wrapper_gets_the_timeout():
    index, wrapped = timed()

    docs = LangChainPineconeVectorStore.similarity_search_by_vector_with_score(
        library_store(wrapped), [0.1, 0.2], k=2
    )

    [(name, _, kwargs)] = index.calls
    assert name == "query"
    assert kwargs["_request_timeout"] == TIMEOUT
    assert kwargs["namespace"] == "company-it-kb"
    assert docs == [(Document(id="d1", page_content="Use GlobalProtect.", metadata={}), 0.9)]


def test_api_and_cli_ingestion_use_the_vectorstore_add_documents():
    import ingest_sample_kb
    from app.api import routes

    assert routes.add_documents is vectorstore.add_documents
    assert ingest_sample_kb.add_documents is vectorstore.add_documents
