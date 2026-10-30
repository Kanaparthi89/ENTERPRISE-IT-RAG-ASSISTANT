"""Tests for POST /api/ingest with Pinecone indexing mocked out."""

import pytest
from pydantic import SecretStr

from app.api import routes
from tests.conftest import ADMIN_KEY, KnownBugStillPresent, require

MARKDOWN = b"# VPN\n\nInstall the VPN client and sign in with MFA.\n"


@pytest.fixture
def indexed(monkeypatch):
    """Capture chunks sent to the vector store."""
    captured = []

    def add_documents(chunks, ids=None):
        captured.extend(chunks)
        return [f"id-{index}" for index in range(len(chunks))]

    monkeypatch.setattr(routes, "add_documents", add_documents)
    return captured


def upload(client, name="vpn.md", content=MARKDOWN, key=ADMIN_KEY):
    headers = {"X-Admin-Key": key} if key is not None else {}
    return client.post(
        "/api/ingest",
        files={"file": (name, content, "application/octet-stream")},
        headers=headers,
    )


def test_ingest_indexes_markdown(client, indexed, settings):
    response = upload(client)

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Document indexed successfully."
    assert body["file"] == "vpn.md"
    assert body["chunks"] == len(indexed) > 0
    assert body["ids_created"] == len(indexed)
    assert "MFA" in indexed[0].page_content

    stored = list(settings.upload_dir.iterdir())
    assert len(stored) == 1
    assert stored[0].suffix == ".md"
    assert stored[0].name != "vpn.md"


def test_ingest_disabled_without_admin_key(client, indexed, monkeypatch, settings):
    monkeypatch.setattr(settings, "admin_api_key", SecretStr(""))

    response = upload(client)

    assert response.status_code == 503
    assert indexed == []


@pytest.mark.parametrize("key", [None, "", "wrong-key"])
def test_ingest_rejects_bad_admin_key(client, indexed, key):
    response = upload(client, key=key)

    assert response.status_code == 401
    assert indexed == []


def test_ingest_rejects_unsupported_type(client, indexed):
    response = upload(client, name="malware.exe")

    assert response.status_code == 400
    assert "Supported file types" in response.json()["detail"]


def test_ingest_rejects_empty_file(client, indexed):
    response = upload(client, content=b"")

    assert response.status_code == 400
    assert response.json()["detail"] == "The uploaded file is empty."


def test_ingest_rejects_whitespace_only_file(client, indexed):
    response = upload(client, content=b"   \n\n  ")

    assert response.status_code == 400
    assert "No readable content" in response.json()["detail"]
    assert indexed == []


def test_ingest_enforces_size_limit(client, indexed, monkeypatch):
    monkeypatch.setattr(routes, "MAX_UPLOAD_SIZE", 10)

    response = upload(client, content=b"x" * 11)

    assert response.status_code == 413
    assert indexed == []


def test_ingest_strips_path_from_filename(client, indexed, settings):
    response = upload(client, name="../../etc/vpn.md")

    assert response.status_code == 200
    assert response.json()["file"] == "vpn.md"
    stored = list(settings.upload_dir.iterdir())
    assert len(stored) == 1
    assert stored[0].parent == settings.upload_dir.resolve()


def test_ingest_hides_indexing_errors(client, monkeypatch):
    def add_documents(chunks, ids=None):
        raise RuntimeError("pinecone 403 for key pc-secret")

    monkeypatch.setattr(routes, "add_documents", add_documents)

    response = upload(client)

    assert response.status_code == 500
    assert "pc-secret" not in response.text


@pytest.mark.characterizes_bug("B1")
@pytest.mark.xfail(
    raises=KnownBugStillPresent,
    reason="Known issue: uploads are cited by their UUID storage name. "
    "Fixed in Phase B1.",
)
def test_uploaded_chunks_keep_original_filename(client, indexed):
    response = upload(client, name="vpn-guide.md")
    require(response.status_code == 200, f"upload returned {response.status_code}")
    require(indexed, "no chunks were sent to the vector store")

    filename = indexed[0].metadata["filename"]
    if filename != "vpn-guide.md":
        raise KnownBugStillPresent(
            f"chunk is labelled {filename!r} instead of 'vpn-guide.md'"
        )
