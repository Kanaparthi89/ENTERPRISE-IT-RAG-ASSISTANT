"""Tests for the SQLite audit store."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.services import audit
from app.services.audit import get_recent_audits, init_db, write_audit


@pytest.fixture
def opened_connections(monkeypatch):
    """Record every connection the audit service opens."""
    connections = []
    real_connect = audit._connect

    def tracking_connect():
        connection = real_connect()
        connections.append(connection)
        return connection

    monkeypatch.setattr(audit, "_connect", tracking_connect)
    return connections


def is_closed(connection):
    try:
        connection.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        return True
    return False


def count_rows(settings):
    with sqlite3.connect(settings.audit_db_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM query_audit").fetchone()[0]
    connection.close()
    return count


def test_init_db_is_idempotent(settings):
    init_db()
    init_db()

    assert settings.audit_db_path.exists()


def test_init_db_migrates_legacy_schema(settings):
    with sqlite3.connect(settings.audit_db_path) as connection:
        connection.execute(
            """
            CREATE TABLE query_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                question TEXT NOT NULL,
                source_used TEXT NOT NULL,
                trace_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO query_audit (created_at, question, source_used, trace_json)"
            " VALUES ('2026-01-01T00:00:00+00:00', 'old', 'kb', '[]')"
        )

    init_db()

    with sqlite3.connect(settings.audit_db_path) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(query_audit)")
        }

    assert {"request_id", "citation_json", "status", "error_type"} <= columns

    [legacy] = get_recent_audits(limit=1)
    assert legacy["question"] == "old"
    assert legacy["status"] == "success"
    assert legacy["citations"] == []


def test_write_and_read_round_trip():
    init_db()

    audit_id = write_audit(
        question="  How do I reset MFA?  ",
        source_used="private_kb",
        trace=["Router → KB"],
        request_id="req-1",
        citations=[{"title": "handbook.md", "type": "private_kb"}],
    )

    [record] = get_recent_audits(limit=1)
    assert record["id"] == audit_id
    assert record["request_id"] == "req-1"
    assert record["question"] == "How do I reset MFA?"
    assert record["trace"] == ["Router → KB"]
    assert record["citations"] == [{"title": "handbook.md", "type": "private_kb"}]
    assert record["status"] == "success"
    assert record["error_type"] is None


def test_recent_audits_are_newest_first_and_limited():
    init_db()
    for number in range(3):
        write_audit(question=f"question {number}", source_used="direct")

    records = get_recent_audits(limit=2)

    assert [record["question"] for record in records] == [
        "question 2",
        "question 1",
    ]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"question": "   ", "source_used": "kb"}, "question"),
        ({"question": "q", "source_used": ""}, "source_used"),
        ({"question": "q", "source_used": "kb", "status": "weird"}, "status"),
        ({"question": "q", "source_used": "kb", "trace": [object()]}, "JSON"),
    ],
)
def test_write_audit_validates_input(kwargs, message):
    init_db()

    with pytest.raises(ValueError, match=message):
        write_audit(**kwargs)


@pytest.mark.parametrize("limit", [0, 501])
def test_recent_audits_rejects_bad_limit(limit):
    with pytest.raises(ValueError):
        get_recent_audits(limit=limit)


# -------------------------------------------------------------------
# Connection lifecycle and concurrency
# -------------------------------------------------------------------

def test_every_operation_closes_its_connection(opened_connections):
    init_db()
    write_audit(question="q", source_used="kb")
    get_recent_audits(limit=5)

    assert len(opened_connections) == 3
    assert all(is_closed(connection) for connection in opened_connections)


def test_failed_transaction_rolls_back_and_closes(opened_connections, settings):
    init_db()

    with pytest.raises(RuntimeError, match="boom"):
        with audit._transaction() as connection:
            connection.execute(
                "INSERT INTO query_audit (created_at, question, source_used, trace_json)"
                " VALUES ('2026-01-01T00:00:00+00:00', 'q', 'kb', '[]')"
            )
            raise RuntimeError("boom")

    assert count_rows(settings) == 0
    assert all(is_closed(connection) for connection in opened_connections)


def test_write_failure_closes_connection(opened_connections, settings):
    init_db()
    with sqlite3.connect(settings.audit_db_path) as connection:
        connection.execute("DROP TABLE query_audit")
    connection.close()

    with pytest.raises(sqlite3.OperationalError):
        write_audit(question="q", source_used="kb")

    assert all(is_closed(connection) for connection in opened_connections)


def test_concurrent_writes_are_all_persisted(opened_connections, settings):
    init_db()
    workers, writes_per_worker = 8, 25

    def write_batch(worker):
        return [
            write_audit(question=f"worker {worker} write {n}", source_used="kb")
            for n in range(writes_per_worker)
        ]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        ids = [audit_id for batch in pool.map(write_batch, range(workers)) for audit_id in batch]

    total = workers * writes_per_worker
    assert len(ids) == len(set(ids)) == total
    assert count_rows(settings) == total
    assert {record["question"] for record in get_recent_audits(limit=total)} == {
        f"worker {worker} write {n}"
        for worker in range(workers)
        for n in range(writes_per_worker)
    }
    assert all(is_closed(connection) for connection in opened_connections)
