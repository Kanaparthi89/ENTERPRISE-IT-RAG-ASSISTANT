"""Tests for document loading, chunking, and the sample-KB script helpers."""

import pytest
from docx import Document as DocxDocument
from langchain_core.documents import Document

from app.services.ingestion import chunk_documents, load_file
from ingest_sample_kb import discover_files, generate_chunk_id


@pytest.mark.characterizes_bug("B1")
def test_load_markdown_attaches_metadata(tmp_path):
    path = tmp_path / "vpn.md"
    path.write_text("# VPN\n\nConnect with GlobalProtect.", encoding="utf-8")

    [document] = load_file(path)

    assert "GlobalProtect" in document.page_content
    assert document.metadata["filename"] == "vpn.md"
    assert document.metadata["file_type"] == "md"
    assert document.metadata["source"] == str(path.resolve())


def test_load_docx_skips_blank_paragraphs(tmp_path):
    path = tmp_path / "policy.docx"
    docx = DocxDocument()
    docx.add_paragraph("Reset passwords every 90 days.")
    docx.add_paragraph("   ")
    docx.add_paragraph("Use MFA for remote access.")
    docx.save(path)

    [document] = load_file(path)

    assert document.page_content == (
        "Reset passwords every 90 days.\nUse MFA for remote access."
    )
    assert document.metadata["file_type"] == "docx"


def test_load_file_drops_empty_documents(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("  \n ", encoding="utf-8")

    assert load_file(path) == []


def test_load_file_rejects_unsupported_type(tmp_path):
    path = tmp_path / "notes.csv"
    path.write_text("a,b", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_file(path)


def test_load_file_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_file(tmp_path / "missing.md")


def test_load_file_rejects_directory(tmp_path):
    folder = tmp_path / "folder.md"
    folder.mkdir()

    with pytest.raises(ValueError, match="Expected a file"):
        load_file(folder)


def test_load_file_wraps_loader_errors(tmp_path):
    path = tmp_path / "broken.docx"
    path.write_bytes(b"not a real docx")

    with pytest.raises(RuntimeError, match="Failed to load 'broken.docx'"):
        load_file(path)


def test_chunk_documents_indexes_chunks_per_source():
    docs = [
        Document(page_content="alpha " * 300, metadata={"source": "a.md"}),
        Document(page_content="beta " * 300, metadata={"source": "b.md"}),
    ]

    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)

    for source in ("a.md", "b.md"):
        indexes = [
            chunk.metadata["chunk_index"]
            for chunk in chunks
            if chunk.metadata["source"] == source
        ]
        assert indexes == list(range(len(indexes)))
        assert len(indexes) > 1

    assert all(len(chunk.page_content) <= 200 for chunk in chunks)
    assert all("start_index" in chunk.metadata for chunk in chunks)


def test_chunk_documents_handles_empty_input():
    assert chunk_documents([]) == []


@pytest.mark.parametrize(
    "size, overlap",
    [(0, 0), (100, -1), (100, 100), (100, 150)],
)
def test_chunk_documents_validates_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunk_documents(
            [Document(page_content="text", metadata={})],
            chunk_size=size,
            chunk_overlap=overlap,
        )


@pytest.mark.characterizes_bug("B1")
def test_sample_kb_chunk_ids_are_deterministic():
    chunk = Document(
        page_content="VPN steps",
        metadata={"source": "/kb/handbook.md", "chunk_index": 3},
    )
    same = Document(
        page_content="VPN steps",
        metadata={"source": "/kb/handbook.md", "chunk_index": 3},
    )
    different = Document(
        page_content="VPN steps",
        metadata={"source": "/kb/handbook.md", "chunk_index": 4},
    )

    assert generate_chunk_id(chunk) == generate_chunk_id(same)
    assert generate_chunk_id(chunk) != generate_chunk_id(different)


def test_discover_files_skips_hidden_and_unsupported(tmp_path):
    for name in ("b.md", "a.pdf", ".hidden.md", "notes.csv"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    (tmp_path / "subdir.md").mkdir()

    assert [path.name for path in discover_files(tmp_path)] == ["a.pdf", "b.md"]
