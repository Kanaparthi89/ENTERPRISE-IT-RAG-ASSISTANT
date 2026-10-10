import time

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.core.config import get_settings

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
        )

    return _embeddings


def ensure_index():
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing.")

    if not settings.pinecone_index_name:
        raise RuntimeError("Pinecone index name is not configured.")

    dimension = get_embedding_dimension()

    pc = Pinecone(
        api_key=settings.pinecone_api_key.get_secret_value()
    )

    index_names = [item["name"] for item in pc.list_indexes()]

    if settings.pinecone_index_name in index_names:
        index_info = pc.describe_index(settings.pinecone_index_name)

        if index_info.dimension != dimension:
            raise RuntimeError(
                f"Index dimension is {index_info.dimension}, but model "
                f"'{settings.embedding_model}' requires {dimension}. "
                "Use a matching index or model. The index was not deleted."
            )

        if not index_info.status["ready"]:
            raise RuntimeError("The Pinecone index is not ready.")

    else:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

        for _ in range(60):
            index_info = pc.describe_index(settings.pinecone_index_name)

            if index_info.status["ready"]:
                break

            time.sleep(2)
        else:
            raise TimeoutError(
                "Pinecone index did not become ready within 120 seconds."
            )

    return pc.Index(settings.pinecone_index_name)


def get_vectorstore():
    global _vectorstore

    if _vectorstore is None:
        _vectorstore = PineconeVectorStore(
            index=ensure_index(),
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
