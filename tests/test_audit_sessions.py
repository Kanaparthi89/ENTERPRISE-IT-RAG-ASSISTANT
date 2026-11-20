"""A2-4: session-aware audit records (design revision 8, §15, §21).

Additive only: a nullable ``session_id`` column (the internal session UUID,
never the bearer token) and its index, added by ``init_db``'s existing
migration mechanism. Stateless writes use the original statement unchanged,
and deleting a session never touches audit rows.
"""

import logging
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.services import audit
from app.services.audit import get_recent_audits, init_db, write_audit
from app.services.sessions import SessionStore
from tests.conftest import legacy_owner_ref
from tests.session_support import find_token
from tests.streaming_support import FakeClock

IDLE, ABSOLUTE = 1800.0, 28800.0

# The query_audit schema as committed at 5d1e45a (before A2-4).
PRE_A2_SCHEMA = """
CREATE TABLE query_audit (
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


@pytest.fixture
def session_store(tmp_path):
    return SessionStore(
        tmp_path / "sessions.db",
        idle_timeout_seconds=IDLE,
        absolute_timeout_seconds=ABSOLUTE,
        max_active_sessions=100,
        clock=FakeClock(),
    )


def columns(settings) -> dict[str, tuple]:
    connection = sqlite3.connect(settings.audit_db_path)
    try:
        return {row[1]: row for row in connection.execute("PRAGMA table_info(query_audit)")}
    finally:
        connection.close()


def indexes(settings) -> set[str]:
    connection = sqlite3.connect(settings.audit_db_path)
    try:
        return {row[1] for row in connection.execute("PRAGMA index_list(query_audit)")}
    finally:
        connection.close()


def raw_rows(settings) -> list[tuple]:
    connection = sqlite3.connect(settings.audit_db_path)
    try:
        return connection.execute("SELECT * FROM query_audit ORDER BY id").fetchall()
    finally:
        connection.close()


# ===================================================================
# Schema: additive and backwards compatible
# ===================================================================

def test_init_db_adds_a_nullable_session_column_and_index(settings):
    init_db()
    init_db()  # idempotent

    session_column = columns(settings)["session_id"]
    assert session_column[2] == "TEXT" and session_column[3] == 0  # type, NOT NULL flag
    assert session_column[4] is None  # no default
    assert "idx_query_audit_session_id" in indexes(settings)
    assert {"idx_query_audit_created_at", "idx_query_audit_request_id"} <= indexes(settings)


def test_pre_a2_database_migrates_without_touching_existing_rows(settings):
    settings.audit_db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(settings.audit_db_path)
    connection.execute(PRE_A2_SCHEMA)
    connection.execute(
        "INSERT INTO query_audit (request_id, created_at, question, source_used, trace_json, status)"
        " VALUES ('old', '2026-10-01T00:00:00+00:00', 'old question', 'private_kb', '[\"Router → KB\"]', 'success')"
    )
    connection.commit()
    connection.close()
    before = raw_rows(settings)

    init_db()

    after = raw_rows(settings)
    assert [row[:-1] for row in after] == before  # every original value kept
    assert after[0][-1] is None  # session_id NULL for history
    [record] = get_recent_audits(limit=1)
    assert record["question"] == "old question" and record["session_id"] is None


# ===================================================================
# Stateless behaviour unchanged
# ===================================================================

def test_stateless_write_is_unchanged_and_has_no_session(settings):
    init_db()

    audit_id = write_audit(
        question="  How do I reset MFA?  ", source_used="private_kb", trace=["Router → KB"],
        request_id="req-1", citations=[{"title": "handbook.md", "type": "private_kb"}],
    )

    [record] = get_recent_audits(limit=1)
    assert record == {
        "id": audit_id, "request_id": "req-1", "created_at": record["created_at"],
        "question": "How do I reset MFA?", "source_used": "private_kb", "status": "success",
        "error_type": None, "trace": ["Router → KB"],
        "citations": [{"title": "handbook.md", "type": "private_kb"}], "session_id": None,
    }


def test_stateless_write_uses_the_original_insert_statement(settings, monkeypatch):
    """A stateless write never names the new column, so it is byte-for-byte the old statement."""
    init_db()
    seen = []
    real_connect = audit._connect

    def spying_connect():
        connection = real_connect()
        connection.set_trace_callback(seen.append)
        return connection

    monkeypatch.setattr(audit, "_connect", spying_connect)
    write_audit(question="q", source_used="kb")

    inserts = [sql for sql in seen if "INSERT INTO query_audit" in sql]
    assert len(inserts) == 1 and "session_id" not in inserts[0]


def test_stateless_write_works_on_a_database_before_migration(settings):
    """Even if init_db hasn't added the column yet, stateless writes still succeed."""
    settings.audit_db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(settings.audit_db_path)
    connection.execute(PRE_A2_SCHEMA)
    connection.commit()
    connection.close()

    write_audit(question="q", source_used="kb", request_id="r")

    assert len(raw_rows(settings)) == 1


# ===================================================================
# Session-linked records
# ===================================================================

def test_session_write_records_the_internal_session_id(settings, session_store):
    init_db()
    created = session_store.create(owner_ref=legacy_owner_ref())

    write_audit(question="What about on a Mac?", source_used="private_kb",
                request_id="req-2", session_id=created.session_id)
    write_audit(question="Stateless", source_used="direct", request_id="req-3")

    records = {record["request_id"]: record for record in get_recent_audits(limit=5)}
    assert records["req-2"]["session_id"] == created.session_id
    assert records["req-3"]["session_id"] is None
    assert str(uuid.UUID(records["req-2"]["session_id"])) == created.session_id


def test_all_statuses_can_be_linked_to_a_session(settings, session_store):
    init_db()
    session_id = session_store.create(owner_ref=legacy_owner_ref()).session_id
    for status, error_type in (("success", None), ("insufficient_evidence", None),
                               ("failed", "TimeoutError"), ("cancelled", "ClientDisconnected")):
        write_audit(question="q", source_used="kb", status=status, error_type=error_type,
                    session_id=session_id)

    records = get_recent_audits(limit=10)
    assert {record["session_id"] for record in records} == {session_id}
    assert {record["status"] for record in records} == {"success", "insufficient_evidence", "failed", "cancelled"}


# ===================================================================
# Never the raw token
# ===================================================================

@pytest.mark.parametrize(
    "bad",
    [
        "TOKEN_PLACEHOLDER",  # the real bearer token (filled in below)
        "not-a-uuid",
        "S1",
        "",
        "8E1C2A3B-4D5E-4F60-8A7B-9C0D1E2F3A4B",  # a UUID, but not the canonical lowercase form
        "{8e1c2a3b-4d5e-4f60-8a7b-9c0d1e2f3a4b}",
        "8e1c2a3b4d5e4f608a7b9c0d1e2f3a4b",
        42,
    ],
)
def test_anything_but_an_internal_session_id_is_refused(settings, session_store, bad, caplog):
    init_db()
    created = session_store.create(owner_ref=legacy_owner_ref())
    value = created.token if bad == "TOKEN_PLACEHOLDER" else bad

    with caplog.at_level(logging.DEBUG), pytest.raises(ValueError) as refused:
        write_audit(question="q", source_used="kb", session_id=value)

    assert str(refused.value) == "Audit session_id must be an internal session ID."
    assert raw_rows(settings) == []  # nothing written
    if isinstance(value, str) and value:
        assert value not in str(refused.value)
        assert not find_token(value, records=caplog.records)


def test_token_never_reaches_audit_rows_files_or_logs(settings, session_store, caplog):
    init_db()
    with caplog.at_level(logging.DEBUG):
        created = session_store.create(owner_ref=legacy_owner_ref())
        for status in ("success", "failed", "cancelled"):
            write_audit(question="Follow-up question", source_used="kb", status=status,
                        error_type=None if status == "success" else "X", session_id=created.session_id)
        with pytest.raises(ValueError):
            write_audit(question="q", source_used="kb", session_id=created.token)
        session_store.delete(created.token, owner_ref=legacy_owner_ref())

    assert not find_token(created.token, records=caplog.records,
                          db_paths=[settings.audit_db_path, session_store.path])
    assert not any(created.token in repr(row) for row in raw_rows(settings))
    assert any(created.session_id in r.getMessage() for r in caplog.records)  # internal ID is fine


# ===================================================================
# Retention: deleting a session never deletes audit history
# ===================================================================

def test_deleting_a_session_keeps_its_audit_records(settings, session_store):
    init_db()
    created = session_store.create(owner_ref=legacy_owner_ref())
    for number in range(3):
        write_audit(question=f"turn {number}", source_used="kb", request_id=f"r{number}",
                    session_id=created.session_id)
    before = raw_rows(settings)

    session_store.delete(created.token, owner_ref=legacy_owner_ref())

    assert session_store.lookup(created.token, owner_ref=legacy_owner_ref()) is None  # session memory is gone
    assert raw_rows(settings) == before  # audit history untouched, including session_id


def test_session_expiry_and_cleanup_keep_audit_records(settings, session_store):
    init_db()
    created = session_store.create(owner_ref=legacy_owner_ref())
    write_audit(question="q", source_used="kb", session_id=created.session_id)
    before = raw_rows(settings)

    session_store._clock.advance(ABSOLUTE)
    assert session_store.cleanup().deleted == 1

    assert raw_rows(settings) == before


def test_audit_module_has_no_deletion_path():
    """Retention is unchanged: the audit module never deletes or updates rows."""
    import inspect

    source = inspect.getsource(audit)
    assert "DELETE FROM query_audit" not in source
    assert "UPDATE query_audit" not in source


# ===================================================================
# Failures
# ===================================================================

def test_session_write_failure_raises_and_writes_nothing(settings, session_store, monkeypatch):
    init_db()
    session_id = session_store.create(owner_ref=legacy_owner_ref()).session_id
    opened = []
    real_connect = audit._connect

    def tracking_connect():
        connection = real_connect()
        opened.append(connection)
        return connection

    monkeypatch.setattr(audit, "_connect", tracking_connect)
    connection = sqlite3.connect(settings.audit_db_path)
    connection.execute("DROP TABLE query_audit")
    connection.commit()
    connection.close()

    with pytest.raises(sqlite3.OperationalError):
        write_audit(question="q", source_used="kb", session_id=session_id)

    for item in opened:  # the connection was closed on failure
        with pytest.raises(sqlite3.ProgrammingError):
            item.execute("SELECT 1")


def test_session_write_against_an_unmigrated_database_fails_loudly(settings, session_store):
    """A session write needs init_db's migration; it fails rather than dropping the link."""
    settings.audit_db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(settings.audit_db_path)
    connection.execute(PRE_A2_SCHEMA)
    connection.commit()
    connection.close()

    with pytest.raises(sqlite3.OperationalError, match="session_id"):
        write_audit(question="q", source_used="kb", session_id=session_store.create(owner_ref=legacy_owner_ref()).session_id)
    assert raw_rows(settings) == []


def test_invalid_session_id_is_rejected_before_any_database_access(settings, monkeypatch):
    def no_connect():
        raise AssertionError("the database must not be opened")

    monkeypatch.setattr(audit, "_connect", no_connect)

    with pytest.raises(ValueError, match="internal session ID"):
        write_audit(question="q", source_used="kb", session_id="raw-token-value")


def test_concurrent_session_writes_are_all_persisted(settings, session_store):
    init_db()
    session_ids = [session_store.create(owner_ref=legacy_owner_ref()).session_id for _ in range(4)]

    def write_batch(index):
        return [
            write_audit(question=f"s{index} q{n}", source_used="kb", session_id=session_ids[index])
            for n in range(25)
        ]

    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = [audit_id for batch in pool.map(write_batch, range(4)) for audit_id in batch]

    assert len(set(ids)) == 100
    records = get_recent_audits(limit=100)
    for index, session_id in enumerate(session_ids):
        assert {r["question"] for r in records if r["session_id"] == session_id} == {
            f"s{index} q{n}" for n in range(25)}
