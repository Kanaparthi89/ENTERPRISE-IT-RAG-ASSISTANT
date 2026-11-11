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


class FakePinecone:
    """Minimal stand-in for the Pinecone client."""

    def __init__(self, indexes=None, polls_until_ready=0):
        self.indexes = dict(indexes or {})
        self.created = []
        self.opened = []
        self.polls_until_ready = polls_until_ready
        self.on_describe = lambda: None
        self.db = SimpleNamespace(index=SimpleNamespace(_index_api=FakeIndexApi(self)))

    def __call__(self, api_key):
        return self

    @property
    def control_calls(self):
        return self.db.index._index_api.calls

    # The public list and describe calls take no HTTP timeout; ensure_index()
    # must not use them.
    def list_indexes(self):
        raise AssertionError("untimed list_indexes() was called")

    def describe_index(self, name):
        raise AssertionError("untimed describe_index() was called")

    def create_index(self, name, dimension, metric, spec, timeout=None):
        self.created.append((name, dimension, metric, timeout))
        self.indexes[name] = index_info(dimension=dimension)

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

    [(_, _, _, sdk_timeout)] = fake.created
    assert sdk_timeout == -1  # the SDK returns at once instead of polling without a limit
    timeout = {"_request_timeout": settings.pinecone_timeout_seconds}
    assert fake.control_calls == [("list_indexes", (), timeout)] + [("describe_index", (name,), timeout)] * 3
    assert clock.sleeps == [2, 2]
    assert fake.opened == [(name, HOST)]


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
    assert "timeout" in inspect.signature(RealPinecone.create_index).parameters
    assert "host" in inspect.signature(RealPinecone.Index).parameters


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
