"""Ingest sample knowledge-base documents."""

import hashlib
import logging
import sys
from pathlib import Path

from app.core.config import get_settings
from app.services.ingestion import load_file, chunk_documents
from app.rag.vectorstore import add_documents

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bluepeak.ingestion")


def discover_files(folder: Path) -> list[Path]:
    """Return supported, non-hidden files from the knowledge-base folder."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def generate_chunk_id(chunk) -> str:
    """Create a deterministic ID for a source chunk."""
    source = str(chunk.metadata.get("source", "unknown"))
    chunk_index = str(chunk.metadata.get("chunk_index", 0))
    content = chunk.page_content

    unique_value = f"{source}|{chunk_index}|{content}"
    return hashlib.sha256(unique_value.encode("utf-8")).hexdigest()


def main() -> int:
    settings = get_settings()
    folder = Path(settings.sample_kb_dir).expanduser()

    logger.info("=" * 60)
    logger.info("BLUEPEAK TECHNOLOGIES | SAMPLE KNOWLEDGE-BASE INGESTION")
    logger.info("=" * 60)

    if not folder.is_dir():
        logger.error("Knowledge-base directory does not exist: %s", folder)
        return 1

    files = discover_files(folder)

    if not files:
        logger.warning("No supported documents found in %s", folder)
        return 0

    logger.info("Knowledge-base directory: %s", folder.resolve())
    logger.info("Supported files found: %d", len(files))

    all_documents = []
    failed_files = []

    for position, path in enumerate(files, start=1):
        logger.info("[%d/%d] Loading %s", position, len(files), path.name)

        try:
            documents = load_file(path)

            if not documents:
                logger.warning("No readable content found in %s", path.name)
                continue

            all_documents.extend(documents)
            logger.info("Loaded %d document section(s)", len(documents))

        except Exception:
            logger.exception("Failed to load %s", path.name)
            failed_files.append(path.name)

    if not all_documents:
        logger.error("No usable document content was loaded.")
        return 1

    try:
        chunks = chunk_documents(all_documents)
    except Exception:
        logger.exception("Failed to split documents into chunks.")
        return 1

    if not chunks:
        logger.error("Chunking produced no results.")
        return 1

    ids = [generate_chunk_id(chunk) for chunk in chunks]

    if len(ids) != len(set(ids)):
        logger.error("Duplicate chunk IDs detected; indexing was stopped.")
        return 1

    logger.info("-" * 60)
    logger.info("Document sections loaded: %d", len(all_documents))
    logger.info("Chunks generated: %d", len(chunks))
    logger.info("Starting Pinecone indexing...")

    try:
        indexed_ids = add_documents(chunks, ids=ids)
    except Exception:
        logger.exception("Failed to index chunks in Pinecone.")
        return 1

    logger.info("Vectors indexed successfully: %d", len(indexed_ids))

    if failed_files:
        logger.warning("Files that failed: %s", ", ".join(failed_files))
        logger.warning("Ingestion finished with errors.")
        return 1

    logger.info("Ingestion completed successfully.")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
