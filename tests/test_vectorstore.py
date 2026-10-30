"""Tests for Pinecone index validation and document upserts (no network)."""

from types import SimpleNamespace

import pytest

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
