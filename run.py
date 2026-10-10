import time

from langchain_core.documents import Document

from app.rag.vectorstore import (
    ensure_index,
    get_embeddings,
    get_vectorstore,
    get_retriever,
)


def main():
    print("=" * 60)
    print("BLUEPEAK TECHNOLOGIES - VECTOR STORE TEST")
    print("=" * 60)

    # Step 1: Test Pinecone connection and index
    try:
        print("\n[1/5] Checking Pinecone connection and index...")

        index = ensure_index()
        stats = index.describe_index_stats()

        print("SUCCESS: Pinecone index is accessible.")
        print(f"Index statistics: {stats}")

    except Exception as exc:
        print(f"FAILED: Pinecone connection/index: {exc}")
        return

    # Step 2: Test OpenAI embeddings
    try:
        print("\n[2/5] Testing OpenAI embeddings...")

        embeddings = get_embeddings()
        test_vector = embeddings.embed_query(
            "How do I reset my company password?"
        )

        print("SUCCESS: Embeddings generated.")
        print(f"Embedding dimensions: {len(test_vector)}")

    except Exception as exc:
        print(f"FAILED: Embedding generation: {exc}")
        return

    # Step 3: Initialize the vector store
    try:
        print("\n[3/5] Initializing Pinecone vector store...")

        vectorstore = get_vectorstore()

        print("SUCCESS: Vector store initialized.")

    except Exception as exc:
        print(f"FAILED: Vector store initialization: {exc}")
        return

    # Step 4: Insert a test document
    test_id = "vectorstore-connectivity-test-001"

    test_document = Document(
        page_content=(
            "BluePeak Technologies employees who need to reset "
            "their company password should contact the IT Service "
            "Desk and follow the approved password reset procedure."
        ),
        metadata={
            "source": "run.py",
            "filename": "vectorstore_test",
            "customer": "BluePeak Technologies",
            "test_document": True,
        },
    )

    try:
        print("\n[4/5] Adding a test document to Pinecone...")

        vectorstore.add_documents(
            documents=[test_document],
            ids=[test_id],
        )

        print("SUCCESS: Test document uploaded.")

    except Exception as exc:
        print(f"FAILED: Document insertion: {exc}")
        return

    # Step 5: Test semantic retrieval
    try:
        print("\n[5/5] Testing semantic retrieval...")

        retriever = get_retriever()

        # Allow a short time for the vector to become searchable.
        matching_document = None

        for attempt in range(5):
            results = retriever.invoke(
                "How can I reset my company password?"
            )

            matching_document = next(
                (
                    doc for doc in results
                    if doc.metadata.get("source") == "run.py"
                    and doc.metadata.get("test_document") is True
                ),
                None,
            )

            if matching_document:
                break

            if attempt < 4:
                time.sleep(2)

        if matching_document:
            print("SUCCESS: Test document retrieved!")

            print("\nRetrieved content:")
            print(matching_document.page_content)

            print("\nRetrieved metadata:")
            print(matching_document.metadata)

        else:
            print(
                "WARNING: The test document was not found in the "
                "retrieval results. Check the namespace, index, and "
                "TOP_K configuration."
            )
            print(f"Results returned: {len(results)}")

    except Exception as exc:
        print(f"FAILED: Semantic retrieval: {exc}")
        return

    print("\n" + "=" * 60)
    print("VECTOR STORE TEST FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()