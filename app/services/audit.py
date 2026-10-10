
"""SQLite audit logging for the BluePeak Agentic RAG Copilot."""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT,
    created_at TEXT NOT NULL,
    question TEXT NOT NULL,
    source_used TEXT NOT NULL,
    trace_json TEXT NOT NULL,
    citation_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'success',
    error_type TEXT
)
"""

_ALLOWED_STATUSES = {"success", "failed", "insufficient_evidence"}


def _database_path() -> Path:
    """Resolve the configured database path and create its parent folder."""
    path = Path(settings.audit_db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    """Open a configured SQLite connection."""
    connection = sqlite3.connect(
        str(_database_path()),
        timeout=10.0,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 10000")
    return connection


def init_db() -> None:
    """Create the audit table and add supported columns to older databases."""
    with _connect() as connection:
        connection.execute(_SCHEMA)

        existing_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(query_audit)"
            ).fetchall()
        }

        migrations = {
            "request_id": "TEXT",
            "citation_json": "TEXT NOT NULL DEFAULT '[]'",
            "status": "TEXT NOT NULL DEFAULT 'success'",
            "error_type": "TEXT",
        }

        for column, definition in migrations.items():
            if column not in existing_columns:
                connection.execute(
                    f"ALTER TABLE query_audit ADD COLUMN {column} {definition}"
                )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_query_audit_created_at
            ON query_audit(created_at)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_query_audit_request_id
            ON query_audit(request_id)
            """
        )


def write_audit(
    question: str,
    source_used: str,
    trace: list[str] | None = None,
    *,
    request_id: str | None = None,
    citations: list[dict[str, Any]] | None = None,
    status: str = "success",
    error_type: str | None = None,
) -> int:
    """Persist an audit event and return its database row ID.

    Avoid passing secrets, authentication tokens, or sensitive personal
    information in questions, traces, citations, or error messages.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Audit question must not be empty.")

    if not isinstance(source_used, str) or not source_used.strip():
        raise ValueError("Audit source_used must not be empty.")

    if status not in _ALLOWED_STATUSES:
        raise ValueError(
            f"Unsupported audit status: {status!r}"
        )

    trace = trace or []
    citations = citations or []

    created_at = datetime.now(timezone.utc).isoformat()

    try:
        trace_json = json.dumps(trace, ensure_ascii=False)
        citation_json = json.dumps(citations, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Trace and citations must be JSON serializable.") from exc

    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO query_audit (
                request_id,
                created_at,
                question,
                source_used,
                trace_json,
                citation_json,
                status,
                error_type
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                created_at,
                question.strip(),
                source_used.strip(),
                trace_json,
                citation_json,
                status,
                error_type,
            ),
        )
        audit_id = cursor.lastrowid

    logger.info(
        "Audit event recorded: id=%s source=%s status=%s",
        audit_id,
        source_used,
        status,
    )

    return audit_id


def get_recent_audits(limit: int = 20) -> list[dict[str, Any]]:
    """Return recent audit records without exposing database internals."""
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500.")

    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, request_id, created_at, question, source_used,
                   trace_json, citation_json, status, error_type
            FROM query_audit
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    records = []

    for row in rows:
        record = dict(row)
        record["trace"] = json.loads(record.pop("trace_json"))
        record["citations"] = json.loads(record.pop("citation_json"))
        records.append(record)

    return records
