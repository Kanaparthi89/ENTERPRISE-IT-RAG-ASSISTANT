"""Tests for Pinecone index validation and document upserts (no network)."""

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore as LangChainPineconeVectorStore

from app.rag import vectorstore


class FakePinecone:
    """Minimal stand-in for the Pinecone client."""

    def __init__(self, indexes=None):
        self.indexes = dict(indexes or {})
        self.created = []

    def __call__(self, api_key):
        return self

    def list_indexes(self):
        return [{"name": name} for name in self.indexes]

    def describe_index(self, name):
        return self.indexes[name]

    def create_index(self, name, dimension, metric, spec):
        self.created.append((name, dimension, metric))
        self.indexes[name] = SimpleNamespace(
            dimension=dimension, status={"ready": True}
        )

    def Index(self, name):
        return f"index:{name}"


@pytest.fixture
def pinecone(monkeypatch):
    def install(indexes=None):
        fake = FakePinecone(indexes)
        monkeypatch.setattr(vectorstore, "Pinecone", fake)
        return fake

    return install


def index_info(dimension=1536, ready=True):
    return SimpleNamespace(dimension=dimension, status={"ready": ready})


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

    assert fake.created == [(settings.pinecone_index_name, 1536, "cosine")]


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
