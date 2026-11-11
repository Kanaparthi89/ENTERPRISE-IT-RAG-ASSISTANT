import logging
import time

from pinecone import Pinecone, ServerlessSpec
from pinecone.db_control.request_factory import PineconeDBControlRequestFactory
from pinecone.exceptions import PineconeApiException
from pinecone.openapi_support.retry_urllib3 import JitterRetry
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.core.config import get_settings
from app.rag.cancellation import raise_if_stopped

logger = logging.getLogger(__name__)

settings = get_settings()

_embeddings = None
_vectorstore = None

EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def get_embedding_dimension(model_name: str | None = None) -> int:
    name = (model_name or settings.embedding_model or "").strip()

    if not name:
        raise RuntimeError("Embedding model is not configured.")

    if name.lower() not in EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Unsupported embedding model '{name}'. "
            "Add its correct dimension to EMBEDDING_DIMENSIONS."
        )

    return EMBEDDING_DIMENSIONS[name.lower()]


def get_embeddings():
    global _embeddings

    if _embeddings is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing.")

        _embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
        )

    return _embeddings


INDEX_READY_POLLS = 60
INDEX_READY_POLL_SECONDS = 2
INDEX_READY_DEADLINE_SECONDS = 120


def _retry_policy(retry_after_max: int) -> JitterRetry:
    """The SDK's default retry policy with a capped ``Retry-After`` (H1-5 A).

    These are the arguments ``pinecone.openapi_support.rest_urllib3`` uses
    when no policy is configured (pinecone 7.3.0); a test compares every
    retry setting against the installed SDK's own default, so an SDK
    upgrade that changes them fails loudly. Only ``retry_after_max``
    differs: urllib3 shortens a longer ``Retry-After`` to it.
    """
    return JitterRetry(
        total=5,
        backoff_factor=0.25,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=None,
        retry_after_max=retry_after_max,
    )


def _pinecone_client():
    """A Pinecone client, with the optional retry cap installed (H1-5 A).

    Without ``PINECONE_RETRY_AFTER_MAX_SECONDS`` the client is unchanged.
    With it, the policy is set on the client's (private) OpenAPI
    configuration before any request: the SDK builds its HTTP clients
    lazily from that configuration, and both the control plane (setup
    calls) and ``pc.Index()`` (queries and upserts) read it.
    """
    pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
    retry_after_max = settings.pinecone_retry_after_max_seconds
    if retry_after_max is not None:
        pc._openapi_config.retries = _retry_policy(retry_after_max)
    return pc


def _index_api(pc):
    """The SDK's generated control-plane API (private in pinecone 7.x; H1-1).

    The public ``list_indexes()``, ``describe_index()`` and
    ``create_index()`` accept no HTTP timeout. The generated methods
    underneath them accept ``_request_timeout``, which applies to each HTTP
    attempt; the SDK's retries still apply. All three calls go through this
    one private access.
    """
    return pc.db.index._index_api


def _list_index_names(pc) -> list[str]:
    raise_if_stopped("Pinecone list indexes")
    response = _index_api(pc).list_indexes(
        _request_timeout=settings.pinecone_timeout_seconds
    )
    return [index.name for index in response.get("indexes", [])]


def _describe_index(pc, name: str):
    raise_if_stopped("Pinecone describe index")
    return _index_api(pc).describe_index(
        name, _request_timeout=settings.pinecone_timeout_seconds
    )


def _create_index(pc, name: str, dimension: int) -> None:
    """Send the index create request with a per-attempt timeout (H1-4).

    The request body is built by the SDK's own request factory, as the
    public ``create_index()`` builds it (same defaults: cosine metric,
    dense vectors, deletion protection disabled, empty tags). The SDK retries
    a POST after a read error, so a create that timed out can be sent
    again after Pinecone already created the index; Pinecone then answers
    HTTP 409. A 409 is therefore not a failure here: the caller waits for
    the existing index to become ready and checks its dimension. This also
    covers two processes creating the index at the same time.
    """
    raise_if_stopped("Pinecone create index")
    request = PineconeDBControlRequestFactory.create_index_request(
        name=name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    try:
        _index_api(pc).create_index(
            create_index_request=request,
            _request_timeout=settings.pinecone_timeout_seconds,
        )
    except PineconeApiException as exc:
        if exc.status != 409:
            raise
        logger.info(
            "Pinecone index %r already exists (HTTP 409); waiting for it to be ready.",
            name,
        )


def _check_dimension(index_info, dimension: int) -> None:
    if index_info.dimension != dimension:
        raise RuntimeError(
            f"Index dimension is {index_info.dimension}, but model "
            f"'{settings.embedding_model}' requires {dimension}. "
            "Use a matching index or model. The index was not deleted."
        )


def ensure_index():
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing.")

    if not settings.pinecone_index_name:
        raise RuntimeError("Pinecone index name is not configured.")

    dimension = get_embedding_dimension()

    pc = _pinecone_client()

    index_names = _list_index_names(pc)

    if settings.pinecone_index_name in index_names:
        index_info = _describe_index(pc, settings.pinecone_index_name)

        _check_dimension(index_info, dimension)

        if not index_info.status.ready:
            raise RuntimeError("The Pinecone index is not ready.")

    else:
        # The create request has a per-attempt timeout (H1-4). It goes
        # through the generated API, so the SDK's own readiness polling
        # (which has no limit when no timeout is given) never runs; the
        # loop below waits instead.
        _create_index(pc, settings.pinecone_index_name, dimension)

        # The deadline is checked between polls, so one describe call in
        # progress (with its retries) can run past it.
        deadline = time.monotonic() + INDEX_READY_DEADLINE_SECONDS

        for _ in range(INDEX_READY_POLLS):
            index_info = _describe_index(pc, settings.pinecone_index_name)

            if index_info.status.ready:
                # After a 409 the index may have been created elsewhere.
                _check_dimension(index_info, dimension)
                break

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "Pinecone index did not become ready within 120 seconds."
                )

            time.sleep(min(INDEX_READY_POLL_SECONDS, remaining))
        else:
            raise TimeoutError(
                "Pinecone index did not become ready within 120 seconds."
            )

    # The known host avoids another, untimed describe call in the SDK.
    return pc.Index(settings.pinecone_index_name, host=index_info.host)


class _TimeoutIndex:
    """A Pinecone index whose queries and upserts have a finite timeout (H1).

    The SDK sets no timeout by default, so a stalled connection blocks its
    thread indefinitely. ``_request_timeout`` applies to each HTTP attempt;
    the SDK's own retries still apply. A timeout passed by the caller is kept.
    Everything else (for example ``config``) is the wrapped index's.
    """

    def __init__(self, index, timeout_seconds: float):
        self._index = index
        self._timeout_seconds = timeout_seconds

    def query(self, *args, **kwargs):
        raise_if_stopped("Pinecone query")  # H1-3a: no-op outside a stoppable run
        kwargs.setdefault("_request_timeout", self._timeout_seconds)
        return self._index.query(*args, **kwargs)

    def upsert(self, *args, **kwargs):
        kwargs.setdefault("_request_timeout", self._timeout_seconds)
        return self._index.upsert(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._index, name)


def get_vectorstore():
    global _vectorstore

    if _vectorstore is None:
        _vectorstore = PineconeVectorStore(
            index=_TimeoutIndex(ensure_index(), settings.pinecone_timeout_seconds),
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )

    return _vectorstore


def get_retriever():
    return get_vectorstore().as_retriever(
        search_kwargs={"k": settings.top_k}
    )


def add_documents(chunks, ids=None):
    """Add document chunks to Pinecone with optional stable IDs."""
    if not chunks:
        return []

    if ids is not None and len(ids) != len(chunks):
        raise ValueError(
            "The number of IDs must match the number of document chunks."
        )

    if ids is not None:
        return get_vectorstore().add_documents(chunks, ids=ids)

    return get_vectorstore().add_documents(chunks)
