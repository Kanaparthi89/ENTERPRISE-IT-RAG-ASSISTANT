"""A2-1: the SQLite session store (design revision 8, §3, §5, §8, §10.1, §15, §16).

Store operations only: no leases, keeper, endpoints or workflow. Lease
columns are written directly where a test needs a session with a live lease.
"""

import asyncio
import logging
import re
import sqlite3
import threading
import time

import pytest
from pydantic import ValidationError

from app.core import config
from app.services import sessions
from app.services.sessions import (
    SessionCapacityReached,
    SessionStore,
    SessionStoreUnavailable,
    hash_token,
    is_well_formed_token,
)
from tests import session_sql
from tests.conftest import TEST_SETTINGS
from tests.session_support import TEST_OWNER, AdvancingClock, ConnectionSpy, count_session_rows, find_token
from tests.streaming_support import FakeClock

IDLE, ABSOLUTE = 1800.0, 28800.0


@pytest.fixture
def make_store(tmp_path):
    def build(**overrides):
        options = dict(
            idle_timeout_seconds=IDLE,
            absolute_timeout_seconds=ABSOLUTE,
            max_active_sessions=100,
            busy_timeout_seconds=0.5,
            clock=FakeClock(),
        )
        options.update(overrides)
        return SessionStore(tmp_path / "sessions.db", **options)

    return build


def raw(store) -> sqlite3.Connection:
    connection = sqlite3.connect(store.path)
    connection.row_factory = sqlite3.Row
    return connection


def set_lease(store, session_id, *, expires_at, owner="r1"):
    with raw(store) as connection:
        connection.execute(
            "UPDATE sessions SET lease_owner = ?, lease_epoch = 1, lease_expires_at = ? WHERE id = ?",
            (owner, expires_at, session_id),
        )


def normalized(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip()


# -------------------------------------------------------------------
# Design conformance and configuration
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    "store_sql, catalogue_name",
    [
        (sessions.SQL_CREATE_SESSION, "create_session"),
        (sessions.SQL_CAP_CHECK, "cap_check"),
        (sessions.SQL_OWNER_CAP_CHECK, "owner_cap_check"),
        (sessions.SQL_LOOKUP, "lookup"),
        (sessions.SQL_CLEANUP, "cleanup"),
        (sessions.SQL_DELETE_SESSION, "delete_session"),
    ],
)
def test_store_sql_is_the_verified_design_sql(store_sql, catalogue_name):
    assert normalized(store_sql) == normalized(session_sql.STATEMENTS[catalogue_name][0])


def test_store_schema_matches_the_verified_schema():
    as_created = [normalized(ddl).replace("IF NOT EXISTS ", "") for ddl in sessions.SCHEMA]
    assert as_created == [normalized(ddl) for ddl in session_sql.SCHEMA]


def test_session_settings_are_off_by_default_and_validated():
    defaults = config.Settings.model_fields
    assert defaults["sessions_enabled"].default is False
    assert defaults["session_idle_timeout_seconds"].default == 1800.0
    assert defaults["session_absolute_timeout_seconds"].default == 28800.0
    assert defaults["max_active_sessions"].default == 5000

    with pytest.raises(ValidationError, match="at least session_idle_timeout_seconds"):
        config.Settings(session_idle_timeout_seconds=3600, session_absolute_timeout_seconds=600)


def test_from_settings_uses_the_isolated_test_path():
    store = SessionStore.from_settings(TEST_SETTINGS)

    assert store.path == TEST_SETTINGS.session_db_path
    assert "pytest" in str(store.path) or "tmp" in str(store.path)
    assert (store.idle_timeout_seconds, store.max_active_sessions) == (1800.0, 5000)


# -------------------------------------------------------------------
# Version gate, lazy creation, connections and threads
# -------------------------------------------------------------------

def test_old_sqlite_disables_every_operation(make_store, caplog):
    store = make_store(sqlite_version="3.34.1")

    with caplog.at_level(logging.ERROR, logger="app.services.sessions"):
        for operation in (lambda: store.create(owner_ref=TEST_OWNER), lambda: store.lookup("t", owner_ref=TEST_OWNER), lambda: store.delete("t", owner_ref=TEST_OWNER), store.cleanup):
            with pytest.raises(SessionStoreUnavailable):
                operation()

    assert not store.path.exists()
    assert [r.getMessage() for r in caplog.records if "older than" in r.getMessage()] and len(
        [r for r in caplog.records if "older than" in r.getMessage()]
    ) == 1  # logged once
    assert store.startup_cleanup() is True  # nothing to do, nothing created


def test_version_parsing():
    assert sessions.sqlite_version_tuple("3.35.0") == (3, 35, 0)
    assert sessions.sqlite_version_tuple("3.46") == (3, 46, 0)
    assert SessionStore.__init__  # the default reads the runtime version
    assert sessions.sqlite_version_tuple() >= sessions.MIN_SQLITE_VERSION


def test_nothing_is_created_until_a_session_is(make_store):
    store = make_store()

    assert store.lookup("unknown", owner_ref=TEST_OWNER) is None
    store.delete("unknown", owner_ref=TEST_OWNER)
    assert store.cleanup().deleted == 0
    assert store.startup_cleanup() is True
    assert not store.path.exists()

    store.create(owner_ref=TEST_OWNER)
    assert store.path.exists()


def test_every_operation_opens_its_own_connection_with_the_lock_timeout(make_store):
    spy = ConnectionSpy()
    store = make_store(connect=spy, busy_timeout_seconds=0.75)

    created = store.create(owner_ref=TEST_OWNER)
    store.lookup(created.token, owner_ref=TEST_OWNER)
    store.delete(created.token, owner_ref=TEST_OWNER)

    assert len(spy.connects) >= 4  # schema, create, cleanup-on-create, lookup, delete
    assert set(spy.timeouts()) == {0.75}


def test_connection_pragmas(make_store):
    spy = ConnectionSpy()
    store = make_store(connect=spy)
    store.create(owner_ref=TEST_OWNER)

    pragmas = [r.sql for r in spy.executes if r.sql.startswith("PRAGMA")]
    for expected in ("journal_mode = WAL", "synchronous = NORMAL", "foreign_keys = ON", "secure_delete = ON"):
        assert any(expected in sql for sql in pragmas), expected
    with raw(store) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


@pytest.mark.anyio
async def test_store_refuses_the_event_loop_thread_but_works_in_a_worker(make_store):
    store = make_store()

    with pytest.raises(RuntimeError, match="event-loop thread"):
        store.create(owner_ref=TEST_OWNER)

    created = await asyncio.to_thread(store.create, owner_ref=TEST_OWNER)
    info = await asyncio.to_thread(store.lookup, created.token, owner_ref=TEST_OWNER)
    assert info.session_id == created.session_id


def test_operations_run_from_many_threads(make_store):
    store = make_store(clock=time.time)
    tokens, errors = [], []

    def worker():
        try:
            tokens.append(store.create(owner_ref=TEST_OWNER).token)
        except Exception as exc:  # noqa: BLE001 - recorded for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert errors == []
    assert count_session_rows(store.path) == 20
    assert all(store.lookup(token, owner_ref=TEST_OWNER) for token in tokens)


# -------------------------------------------------------------------
# Create
# -------------------------------------------------------------------

def test_create_stores_only_the_token_hash(make_store):
    store = make_store()

    created = store.create(owner_ref=TEST_OWNER)

    assert re.fullmatch(r"[A-Za-z0-9_-]{43}", created.token)  # 256 bits, URL-safe
    with raw(store) as connection:
        stored = dict(connection.execute("SELECT * FROM sessions").fetchone())
    assert stored["token_hash"] == hash_token(created.token)
    assert stored["id"] == created.session_id and stored["id"] != created.token
    assert not find_token(created.token, db_paths=[store.path])


def test_create_uses_one_time_value(make_store):
    clock = AdvancingClock(start=5_000.0)
    store = make_store(clock=clock)

    created = store.create(owner_ref=TEST_OWNER)

    with raw(store) as connection:
        row = dict(connection.execute("SELECT * FROM sessions").fetchone())
    assert row["created_at"] == row["last_activity_at"] == created.created_at
    assert row["idle_expires_at"] == created.created_at + IDLE
    assert row["absolute_expires_at"] == created.absolute_expires_at == created.created_at + ABSOLUTE


def test_each_statement_reads_the_clock_at_most_once(make_store):
    """One read per statement that touches a row; none when nothing matches.

    SQLite only computes the materialised ``t`` when a row needs it, so a
    look-up of an unknown token reads the clock zero times. A statement never
    reads it more than once.
    """
    clock = AdvancingClock()
    store = make_store(clock=clock)
    known = store.create(owner_ref=TEST_OWNER)  # schema in place
    spy = ConnectionSpy()
    store._connect_fn = spy
    counts = {}

    for name, call in (
        ("lookup (known)", lambda: store.lookup(known.token, owner_ref=TEST_OWNER)),
        ("lookup (unknown)", lambda: store.lookup("never-issued", owner_ref=TEST_OWNER)),
        ("cleanup", store.cleanup),
        ("create", lambda: store.create(owner_ref=TEST_OWNER)),
    ):
        clock.reset_count()
        before = len(spy.executes)
        call()
        timed = [r.sql for r in spy.executes[before:] if "app_now()" in r.sql]
        counts[name] = (clock.calls, len(timed))

    assert counts["lookup (known)"] == (1, 1)
    assert counts["lookup (unknown)"] == (0, 1)
    # cleanup, global cap check, per-owner cap check (Phase B), insert: one read each
    assert counts["create"] == (4, 4)
    assert all(calls <= statements for calls, statements in counts.values()), counts


def test_cap_counts_only_active_sessions(make_store):
    clock = FakeClock()
    store = make_store(clock=clock, max_active_sessions=2)
    first = store.create(owner_ref=TEST_OWNER)
    store.create(owner_ref=TEST_OWNER)

    with pytest.raises(SessionCapacityReached):
        store.create(owner_ref=TEST_OWNER)
    assert count_session_rows(store.path) == 2

    clock.advance(IDLE)  # both idle-expired: no longer active, and cleaned up on create
    store.create(owner_ref=TEST_OWNER)
    assert store.lookup(first.token, owner_ref=TEST_OWNER) is None


def test_session_with_a_live_lease_counts_towards_the_cap(make_store):
    clock = FakeClock()
    store = make_store(clock=clock, max_active_sessions=1)
    created = store.create(owner_ref=TEST_OWNER)
    set_lease(store, created.session_id, expires_at=clock() + IDLE + 60)
    clock.advance(IDLE)  # idle-expired, but leased

    with pytest.raises(SessionCapacityReached):
        store.create(owner_ref=TEST_OWNER)


# -------------------------------------------------------------------
# Look-up and the expiry rules (§8)
# -------------------------------------------------------------------

def test_lookup_reports_unknown_valid_and_expired(make_store):
    clock = FakeClock()
    store = make_store(clock=clock)
    created = store.create(owner_ref=TEST_OWNER)

    assert store.lookup("never-issued", owner_ref=TEST_OWNER) is None
    info = store.lookup(created.token, owner_ref=TEST_OWNER)
    assert (info.session_id, info.expired, info.live_lease, info.lease_remaining) == (
        created.session_id, False, False, None)

    clock.advance(IDLE - 0.001)
    assert store.lookup(created.token, owner_ref=TEST_OWNER).expired is False
    clock.advance(0.001)  # exactly at idle_expires_at: expired (strict)
    assert store.lookup(created.token, owner_ref=TEST_OWNER).expired is True


def test_live_lease_keeps_an_idle_session_valid_but_not_past_the_absolute_limit(make_store):
    clock = FakeClock()
    store = make_store(clock=clock)
    created = store.create(owner_ref=TEST_OWNER)
    set_lease(store, created.session_id, expires_at=clock() + IDLE + 30)

    clock.advance(IDLE + 10)
    info = store.lookup(created.token, owner_ref=TEST_OWNER)
    assert (info.expired, info.live_lease) == (False, True)
    assert info.lease_remaining == pytest.approx(20)

    clock.advance(20)  # lease expired at exactly this instant
    assert store.lookup(created.token, owner_ref=TEST_OWNER).live_lease is False
    assert store.lookup(created.token, owner_ref=TEST_OWNER).expired is True

    other = store.create(owner_ref=TEST_OWNER)
    set_lease(store, other.session_id, expires_at=clock() + ABSOLUTE + 100)
    clock.advance(ABSOLUTE)
    assert store.lookup(other.token, owner_ref=TEST_OWNER).expired is True  # absolute limit ignores leases


# -------------------------------------------------------------------
# Delete (§10.1)
# -------------------------------------------------------------------

def test_delete_removes_the_session_and_its_turns(make_store):
    store = make_store()
    created = store.create(owner_ref=TEST_OWNER)
    keep = store.create(owner_ref=TEST_OWNER)
    with raw(store) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executemany(
            "INSERT INTO turns (session_id, seq, question, answer) VALUES (?, ?, 'q', 'a')",
            [(created.session_id, 1), (created.session_id, 2), (keep.session_id, 1)],
        )

    store.delete(created.token, owner_ref=TEST_OWNER)

    assert store.lookup(created.token, owner_ref=TEST_OWNER) is None
    with raw(store) as connection:
        assert [tuple(r) for r in connection.execute("SELECT session_id FROM turns")] == [(keep.session_id,)]


def test_delete_is_idempotent_and_reveals_nothing(make_store):
    clock = FakeClock()
    store = make_store(clock=clock)
    created = store.create(owner_ref=TEST_OWNER)

    store.delete(created.token, owner_ref=TEST_OWNER)
    store.delete(created.token, owner_ref=TEST_OWNER)  # already deleted
    store.delete("never-issued", owner_ref=TEST_OWNER)  # unknown
    expired = store.create(owner_ref=TEST_OWNER)
    clock.advance(ABSOLUTE)
    store.delete(expired.token, owner_ref=TEST_OWNER)  # expired

    assert count_session_rows(store.path) == 0


def test_delete_during_an_active_lease_still_deletes(make_store):
    store = make_store()
    created = store.create(owner_ref=TEST_OWNER)
    set_lease(store, created.session_id, expires_at=1e12)

    store.delete(created.token, owner_ref=TEST_OWNER)

    assert store.lookup(created.token, owner_ref=TEST_OWNER) is None


def test_delete_reports_unavailable_when_the_lock_cannot_be_taken(make_store, caplog):
    store = make_store(busy_timeout_seconds=0.2)
    created = store.create(owner_ref=TEST_OWNER)
    blocker = sqlite3.connect(store.path, isolation_level=None)
    blocker.execute("BEGIN EXCLUSIVE")
    try:
        started = time.monotonic()
        with caplog.at_level(logging.ERROR, logger="app.services.sessions"):
            with pytest.raises(SessionStoreUnavailable):
                store.delete(created.token, owner_ref=TEST_OWNER)
        assert time.monotonic() - started < 2.0  # bounded by the lock timeout
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()

    assert store.lookup(created.token, owner_ref=TEST_OWNER) is not None  # still there: deletion not confirmed
    assert not find_token(created.token, records=caplog.records)


def test_delete_reports_unavailable_when_the_database_cannot_be_opened(tmp_path):
    bad_path = tmp_path / "not-a-db"
    bad_path.write_bytes(b"this is not a sqlite database" * 100)
    store = SessionStore(bad_path, idle_timeout_seconds=IDLE, absolute_timeout_seconds=ABSOLUTE,
                         max_active_sessions=10, busy_timeout_seconds=0.2)

    with pytest.raises(SessionStoreUnavailable):
        store.delete("anything", owner_ref=TEST_OWNER)


def test_token_shape_check():
    assert is_well_formed_token("a" * 43)
    assert is_well_formed_token("a" * 128)
    for bad in (None, "", "   ", "a" * 129, 42):
        assert not is_well_formed_token(bad)


# -------------------------------------------------------------------
# Cleanup, secure_delete and checkpointing (§8, §16)
# -------------------------------------------------------------------

def test_cleanup_removes_expired_sessions_but_never_a_live_lease(make_store):
    clock = FakeClock()
    store = make_store(clock=clock)
    idle_expired = store.create(owner_ref=TEST_OWNER)
    leased = store.create(owner_ref=TEST_OWNER)
    set_lease(store, leased.session_id, expires_at=clock() + IDLE + ABSOLUTE + 100)
    clock.advance(IDLE)
    fresh = store.create(owner_ref=TEST_OWNER)  # create runs a cleanup batch first

    assert store.lookup(idle_expired.token, owner_ref=TEST_OWNER) is None
    assert store.lookup(leased.token, owner_ref=TEST_OWNER) is not None
    assert store.lookup(fresh.token, owner_ref=TEST_OWNER) is not None

    clock.advance(ABSOLUTE)  # leased is absolute-expired, but its lease is still live
    assert store.cleanup().deleted == 1  # fresh (idle-expired) only
    assert store.lookup(fresh.token, owner_ref=TEST_OWNER) is None
    assert store.lookup(leased.token, owner_ref=TEST_OWNER) is not None

    set_lease(store, leased.session_id, expires_at=clock())  # lease ends now (strict)
    assert store.cleanup().deleted == 1  # now leased goes too
    assert count_session_rows(store.path) == 0


def test_cleanup_is_batched(make_store):
    clock = FakeClock()
    store = make_store(clock=clock, cleanup_batch=2)
    for _ in range(5):
        store._token_factory  # noqa: B018 - distinct tokens come from the default factory
        store.create(owner_ref=TEST_OWNER)
    clock.advance(IDLE)

    assert [store.cleanup().deleted for _ in range(4)] == [2, 2, 1, 0]


def test_cleanup_cascades_to_turns_and_checkpoints_the_wal(make_store):
    clock = FakeClock()
    spy = ConnectionSpy()
    store = make_store(clock=clock, connect=spy)
    created = store.create(owner_ref=TEST_OWNER)
    with raw(store) as connection:
        connection.execute(
            "INSERT INTO turns (session_id, seq, question, answer) VALUES (?, 1, 'secret question', 'a')",
            (created.session_id,),
        )
    clock.advance(IDLE)

    result = store.cleanup()

    assert result.deleted == 1 and result.checkpoint_busy is False
    assert any("wal_checkpoint(TRUNCATE)" in r.sql for r in spy.executes)
    with raw(store) as connection:
        assert connection.execute("SELECT count(*) FROM turns").fetchone()[0] == 0


def test_deleted_content_is_not_left_in_the_database_files(make_store):
    """With secure_delete and a successful checkpoint, deleted text is overwritten.

    Best effort only (design §16): this checks the database and WAL files, not
    the filesystem, SSD or backups.
    """
    clock = FakeClock()
    store = make_store(clock=clock)
    created = store.create(owner_ref=TEST_OWNER)
    marker = "UNIQUE-SECRET-QUESTION-TEXT-7f3a"
    with raw(store) as connection:
        connection.execute(
            "INSERT INTO turns (session_id, seq, question, answer) VALUES (?, 1, ?, 'a')",
            (created.session_id, marker),
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    assert find_token(marker, db_paths=[store.path])  # present before deletion

    clock.advance(IDLE)
    assert store.cleanup().checkpoint_busy is False

    assert not find_token(marker, db_paths=[store.path])
    assert not find_token(hash_token(created.token), db_paths=[store.path])


def test_cleanup_reports_a_busy_checkpoint_without_failing(make_store):
    clock = FakeClock()
    store = make_store(clock=clock)
    store.create(owner_ref=TEST_OWNER)
    clock.advance(IDLE)
    reader = sqlite3.connect(store.path, isolation_level=None)
    reader.execute("BEGIN")
    reader.execute("SELECT count(*) FROM sessions").fetchone()  # holds a read snapshot
    try:
        result = store.cleanup()
    finally:
        reader.execute("ROLLBACK")
        reader.close()

    assert result.deleted == 1
    assert result.checkpoint_busy is True  # retried at the next cleanup


# -------------------------------------------------------------------
# Startup cleanup (§7.5)
# -------------------------------------------------------------------

def test_startup_cleanup_expires_leftover_memory_even_when_disabled(make_store):
    clock = FakeClock()
    store = make_store(clock=clock)
    old = store.create(owner_ref=TEST_OWNER)
    clock.advance(ABSOLUTE)

    restarted = make_store(clock=clock)  # a new process; sessions may be disabled
    assert restarted.startup_cleanup(wait_seconds=2.0) is True

    assert restarted.lookup(old.token, owner_ref=TEST_OWNER) is None


def test_startup_cleanup_does_not_create_the_database(make_store):
    store = make_store()

    assert store.startup_cleanup() is True
    assert not store.path.exists()


def test_startup_cleanup_never_blocks_startup_beyond_its_wait(make_store):
    clock = FakeClock()
    store = make_store(clock=clock)
    store.create(owner_ref=TEST_OWNER)
    clock.advance(IDLE)
    spy = ConnectionSpy()
    spy.stall(matching=lambda sql: "DELETE FROM sessions" in sql)
    stalled = make_store(clock=clock, connect=spy)

    started = time.monotonic()
    finished = stalled.startup_cleanup(wait_seconds=0.3)
    elapsed = time.monotonic() - started

    assert finished is False and elapsed < 1.0  # startup was not held up
    assert spy.stall_reached.wait(5)  # the pass is stuck in its DELETE (under load it may get there late)
    spy.release_stall()

    def helpers_alive():
        return [t for t in threading.enumerate() if t.name == "session-startup-cleanup" and t.is_alive()]

    deadline = time.monotonic() + 5
    while (count_session_rows(store.path) or helpers_alive()) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert count_session_rows(store.path) == 0  # the pass completed in the background
    assert helpers_alive() == []


# -------------------------------------------------------------------
# No token in logs
# -------------------------------------------------------------------

def test_store_never_logs_the_token(make_store, caplog):
    clock = FakeClock()
    store = make_store(clock=clock, max_active_sessions=1)
    with caplog.at_level(logging.DEBUG):
        created = store.create(owner_ref=TEST_OWNER)
        store.lookup(created.token, owner_ref=TEST_OWNER)
        with pytest.raises(SessionCapacityReached):
            store.create(owner_ref=TEST_OWNER)
        store.delete(created.token, owner_ref=TEST_OWNER)
        clock.advance(IDLE)
        store.cleanup()

    assert caplog.records  # something was logged
    assert not find_token(created.token, records=caplog.records)
