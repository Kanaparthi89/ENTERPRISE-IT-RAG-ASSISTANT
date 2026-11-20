"""Security S1-5 Phase B2: session ownership in the SQLite store.

Persistence layer only (routes are wired in B3). Every token operation
(create, lookup, acquire, delete, and SessionLeaseManager.acquire) requires
an explicit owner_ref, validated before any SQL, and matches it in SQL in
the same transaction. Covers owner isolation, legacy ownerless rows, the
versioned migration (new, migrated, twice, concurrent processes, future and
tampered schemas), the immutability trigger and the per-owner cap.
"""

import re
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from app.core import config
from app.services import sessions
from app.services.sessions import (
    SessionCapacityReached,
    SessionExpired,
    SessionLeaseManager,
    SessionOwnerCapacityReached,
    SessionOwnerError,
    SessionSchemaError,
    SessionStore,
    SessionStoreUnavailable,
    hash_token,
)
from tests.conftest import TEST_SETTINGS
from tests.session_support import GUARD, OTHER_OWNER, TEST_OWNER
from tests.streaming_support import FakeClock

A, B = TEST_OWNER, OTHER_OWNER
IDLE, ABSOLUTE = 1800.0, 28800.0
ROOT = Path(__file__).resolve().parents[1]

LEGACY_SESSIONS_DDL = """
    CREATE TABLE sessions (
        id TEXT PRIMARY KEY,
        token_hash TEXT UNIQUE NOT NULL,
        created_at REAL, last_activity_at REAL, idle_expires_at REAL, absolute_expires_at REAL,
        turn_count INTEGER NOT NULL DEFAULT 0,
        lease_epoch INTEGER NOT NULL DEFAULT 0,
        lease_owner TEXT, lease_instance TEXT, lease_keeper_gen INTEGER,
        lease_started_at REAL, lease_expires_at REAL
    )
"""
LEGACY_TURNS_DDL = """
    CREATE TABLE turns (
        session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
        seq INTEGER, epoch INTEGER, request_id TEXT,
        question TEXT, answer TEXT, source_used TEXT, created_at REAL,
        PRIMARY KEY (session_id, seq)
    )
"""


class CountingConnect:
    """sqlite3.connect that counts connections (to prove nothing was opened)."""

    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return sqlite3.connect(*args, **kwargs)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def make_store(tmp_path, clock):
    def build(path=None, **overrides):
        options = dict(
            idle_timeout_seconds=IDLE, absolute_timeout_seconds=ABSOLUTE,
            max_active_sessions=1000, busy_timeout_seconds=2.0, clock=clock,
        )
        options.update(overrides)
        return SessionStore(path or tmp_path / "sessions.db", **options)

    return build


def raw(path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def acquire(store, token, owner, request_id="r1"):
    return store.acquire(token, request_id, owner_ref=owner, instance="i1", keeper_gen=1,
                         ttl=300.0, turn_max=600.0, history_turns=4)


def save(store, lease, question="q"):
    return store.save_turn(lease, question=question, answer="a", source_used="private_kb",
                           turn_max=600.0, max_turns=20, answer_chars=8000)


def owner_of(path, session_id):
    with raw(path) as connection:
        return connection.execute("SELECT access_owner_ref FROM sessions WHERE id = ?", (session_id,)).fetchone()[0]


def schema_facts(path):
    with raw(path) as connection:
        columns = [tuple(row) for row in connection.execute("PRAGMA table_info(sessions)")]
        turn_columns = [tuple(row) for row in connection.execute("PRAGMA table_info(turns)")]
        triggers = [(r[0], re.sub(r"\s+", " ", r[1]).strip()) for r in connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' ORDER BY name")]
        (version,) = connection.execute("PRAGMA user_version").fetchone()
    return columns, turn_columns, triggers, version


def make_legacy_db(path, clock, rows=(("L1", "legacy-token-1"), ("L2", "legacy-token-2")), version=0):
    """A database as created before Phase B: no owner column, user_version 0."""
    now = clock()
    with sqlite3.connect(path) as connection:
        connection.execute(LEGACY_SESSIONS_DDL)
        connection.execute(LEGACY_TURNS_DDL)
        for session_id, token in rows:
            connection.execute(
                "INSERT INTO sessions (id, token_hash, created_at, last_activity_at, idle_expires_at,"
                " absolute_expires_at) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, hash_token(token), now, now, now + IDLE, now + ABSOLUTE),
            )
            connection.execute(
                "INSERT INTO turns (session_id, seq, epoch, request_id, question, answer, source_used, created_at)"
                " VALUES (?, 1, 1, 'r', 'legacy question', 'legacy answer', 'private_kb', ?)",
                (session_id, now),
            )
        connection.execute(f"PRAGMA user_version = {version}")


# ===================================================================
# Owner validation: explicit, exact, before any SQL
# ===================================================================

INVALID_OWNERS = [None, "", "a" * 63, "a" * 65, "A" * 64, "g" * 64, "a" * 63 + "\n", " " + "a" * 63,
                  b"a" * 64, 123, "a1" * 31 + "-1"]


@pytest.mark.parametrize("owner", INVALID_OWNERS, ids=repr)
def test_every_token_operation_rejects_an_invalid_owner_before_any_sql(make_store, owner):
    connect = CountingConnect()
    store = make_store(connect=connect)
    for operation in (
        lambda: store.create(owner_ref=owner),
        lambda: store.lookup("t", owner_ref=owner),
        lambda: store.delete("t", owner_ref=owner),
        lambda: acquire(store, "t", owner),
    ):
        with pytest.raises(SessionOwnerError) as caught:
            operation()
        assert str(owner) not in str(caught.value) or owner in ("", None)
    assert connect.calls == 0 and not store.path.exists()


def test_there_is_no_default_owner(make_store):
    store = make_store()
    with pytest.raises(TypeError):
        store.create()
    with pytest.raises(TypeError):
        store.lookup("t")
    with pytest.raises(TypeError):
        store.delete("t")
    with pytest.raises(TypeError):
        store.acquire("t", "r1", instance="i1", keeper_gen=1, ttl=300.0, turn_max=600.0, history_turns=0)
    with pytest.raises(TypeError):
        store.create(A)  # keyword-only: no positional owner either
    assert not store.path.exists()


def test_the_lease_manager_requires_a_valid_owner(make_store):
    store = make_store()
    token = store.create(owner_ref=A).token
    manager = SessionLeaseManager.from_settings(store, TEST_SETTINGS)
    try:
        with pytest.raises(TypeError):
            manager.acquire(token, "r1")
        for owner in INVALID_OWNERS:
            with pytest.raises(SessionOwnerError):
                manager.acquire(token, "r1", owner_ref=owner)
        with pytest.raises(SessionExpired):
            manager.acquire(token, "r1", owner_ref=B)  # not B's
        lease = manager.acquire(token, "r1", owner_ref=A)
        assert lease.session_id
        manager.finish(lease)
    finally:
        manager.shutdown()


# ===================================================================
# Owner isolation: a valid token is not enough
# ===================================================================

def test_the_owner_is_recorded_at_creation(make_store):
    store = make_store()
    created = store.create(owner_ref=A)
    assert owner_of(store.path, created.session_id) == A


def test_a_non_owner_cannot_look_up_acquire_or_delete_with_a_valid_token(make_store):
    store = make_store()
    created = store.create(owner_ref=A)

    assert store.lookup(created.token, owner_ref=B) is None
    with pytest.raises(SessionExpired):
        acquire(store, created.token, B)
    store.delete(created.token, owner_ref=B)

    info = store.lookup(created.token, owner_ref=A)
    assert info is not None and info.session_id == created.session_id and not info.live_lease
    with raw(store.path) as connection:
        row = connection.execute("SELECT lease_owner, lease_epoch FROM sessions").fetchone()
    assert tuple(row) == (None, 0)  # B's attempt left no lease and no epoch change


def test_a_non_owner_never_reads_the_history(make_store):
    store = make_store()
    created = store.create(owner_ref=A)
    lease = acquire(store, created.token, A)
    assert save(store, lease, question="A's private question")

    statements = []
    spy = make_store(connect=lambda *a, **k: _tracing(statements, *a, **k))
    with pytest.raises(SessionExpired):
        acquire(spy, created.token, B)
    assert not any("FROM turns" in sql for sql in statements)
    assert acquire(store, created.token, A, "r2").history[0].question == "A's private question"


def _tracing(statements, *args, **kwargs):
    connection = sqlite3.connect(*args, **kwargs)
    connection.set_trace_callback(statements.append)
    return connection


def test_a_non_owner_is_indistinguishable_from_an_unknown_token(make_store):
    store = make_store()
    created = store.create(owner_ref=A)
    outcomes = []
    for token, owner in ((created.token, B), ("never-issued", B), ("never-issued", A)):
        outcome = [store.lookup(token, owner_ref=owner)]
        try:
            acquire(store, token, owner)
        except SessionExpired as exc:
            outcome.append((type(exc).__name__, str(exc)))
        outcomes.append(outcome)
    assert outcomes[0] == outcomes[1] == outcomes[2] == [None, ("SessionExpired", "")]


def test_the_owner_may_delete_during_their_own_turn_and_the_save_is_fenced(make_store):
    store = make_store()
    created = store.create(owner_ref=A)
    lease = acquire(store, created.token, A)
    store.delete(created.token, owner_ref=A)
    assert save(store, lease) is False  # fenced: the session is gone
    with raw(store.path) as connection:
        assert connection.execute("SELECT count(*) FROM turns").fetchone()[0] == 0


def test_a_non_owner_delete_during_the_owners_turn_changes_nothing(make_store):
    store = make_store()
    created = store.create(owner_ref=A)
    lease = acquire(store, created.token, A)
    store.delete(created.token, owner_ref=B)
    assert save(store, lease) is True
    assert store.lookup(created.token, owner_ref=A).session_id == created.session_id


def test_concurrent_owner_and_non_owner_acquires(make_store):
    store = make_store()
    created = store.create(owner_ref=A)
    barrier = threading.Barrier(20)
    results, lock = [], threading.Lock()

    def attempt(owner, index):
        barrier.wait(GUARD)
        try:
            acquire(store, created.token, owner, f"{owner[:2]}-{index}")
            outcome = ("won", owner)
        except sessions.SessionBusy:
            outcome = ("busy", owner)
        except SessionExpired:
            outcome = ("expired", owner)
        with lock:
            results.append(outcome)

    threads = [threading.Thread(target=attempt, args=(A if i % 2 else B, i)) for i in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(GUARD)
    assert len(results) == 20
    assert [o for o in results if o[1] == B] == [("expired", B)] * 10  # B never wins or sees "busy"
    assert sum(1 for o in results if o == ("won", A)) == 1


# ===================================================================
# Immutability trigger
# ===================================================================

@pytest.mark.parametrize("new_owner", [B, None, A])
def test_the_owner_can_never_be_changed(make_store, new_owner):
    store = make_store()
    created = store.create(owner_ref=A)
    with raw(store.path) as connection:
        with pytest.raises(sqlite3.DatabaseError, match="session owner is immutable"):
            connection.execute("UPDATE sessions SET access_owner_ref = ? WHERE id = ?", (new_owner, created.session_id))
    assert owner_of(store.path, created.session_id) == A


def test_a_legacy_row_can_never_be_adopted(make_store, clock, tmp_path):
    path = tmp_path / "sessions.db"
    make_legacy_db(path, clock)
    store = make_store(path)
    store.cleanup()  # migrates
    with raw(path) as connection:
        with pytest.raises(sqlite3.DatabaseError, match="session owner is immutable"):
            connection.execute("UPDATE sessions SET access_owner_ref = ? WHERE id = 'L1'", (A,))
    assert owner_of(path, "L1") is None


# ===================================================================
# Migration
# ===================================================================

def test_a_new_database_is_created_at_version_1(make_store):
    store = make_store()
    store.create(owner_ref=A)
    columns, _, triggers, version = schema_facts(store.path)
    assert version == sessions.SCHEMA_VERSION == 1
    assert columns[-1][1:] == ("access_owner_ref", "TEXT", 0, None, 0)
    assert [name for name, _ in triggers] == ["sessions_owner_immutable"]


def test_a_legacy_database_is_migrated_keeping_its_rows_and_turns(make_store, clock, tmp_path):
    path = tmp_path / "sessions.db"
    make_legacy_db(path, clock)
    make_store(path).create(owner_ref=A)
    columns, _, triggers, version = schema_facts(path)
    assert version == 1 and columns[-1][1] == "access_owner_ref"
    assert [name for name, _ in triggers] == ["sessions_owner_immutable"]
    with raw(path) as connection:
        legacy = connection.execute(
            "SELECT id, access_owner_ref FROM sessions WHERE id LIKE 'L%' ORDER BY id").fetchall()
        assert [tuple(r) for r in legacy] == [("L1", None), ("L2", None)]  # kept, never assigned
        assert connection.execute("SELECT count(*) FROM turns").fetchone()[0] == 2


def test_new_and_migrated_schemas_are_equivalent(make_store, clock, tmp_path):
    fresh, legacy = tmp_path / "fresh.db", tmp_path / "legacy.db"
    make_store(fresh).create(owner_ref=A)
    make_legacy_db(legacy, clock)
    make_store(legacy).create(owner_ref=A)
    assert schema_facts(fresh) == schema_facts(legacy)


def test_migration_twice_is_harmless(make_store, clock, tmp_path):
    path = tmp_path / "sessions.db"
    make_legacy_db(path, clock)
    make_store(path).cleanup()
    first = schema_facts(path)
    make_store(path).cleanup()  # a new store instance: runs the version-1 verification
    assert schema_facts(path) == first


def test_concurrent_processes_migrate_once_and_agree(clock, tmp_path):
    path = tmp_path / "sessions.db"
    make_legacy_db(path, clock)
    script = (
        "import sys; sys.path.insert(0, sys.argv[1])\n"
        "from app.services.sessions import SessionStore\n"
        "store = SessionStore(sys.argv[2], idle_timeout_seconds=1800, absolute_timeout_seconds=28800,"
        " max_active_sessions=10, busy_timeout_seconds=15)\n"
        "store._ensure_schema()\n"
        "print('ok')\n"
    )
    processes = [
        subprocess.Popen([sys.executable, "-I", "-c", script, str(ROOT), str(path)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for _ in range(6)
    ]
    outputs = [process.communicate(timeout=60) for process in processes]
    assert [p.returncode for p in processes] == [0] * 6, outputs
    assert all(out.strip() == "ok" for out, _ in outputs)
    columns, _, triggers, version = schema_facts(path)
    assert version == 1
    assert [c[1] for c in columns].count("access_owner_ref") == 1
    assert [name for name, _ in triggers] == ["sessions_owner_immutable"]
    with raw(path) as connection:
        assert connection.execute("SELECT count(*) FROM sessions").fetchone()[0] == 2


def test_concurrent_threads_with_separate_stores_migrate_once(make_store, clock, tmp_path):
    path = tmp_path / "sessions.db"
    make_legacy_db(path, clock)
    stores = [make_store(path) for _ in range(8)]
    barrier, errors = threading.Barrier(8), []

    def run(store):
        barrier.wait(GUARD)
        try:
            store.create(owner_ref=A)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(store,)) for store in stores]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(GUARD * 2)
    assert errors == []
    assert schema_facts(path)[3] == 1
    with raw(path) as connection:
        assert connection.execute("SELECT count(*) FROM sessions WHERE access_owner_ref = ?", (A,)).fetchone()[0] == 8


# ===================================================================
# Refused schemas: future versions and failed integrity checks
# ===================================================================

def _assert_refused(store, path, before):
    for operation in (lambda: store.create(owner_ref=A), lambda: store.lookup("t", owner_ref=A),
                      lambda: store.delete("t", owner_ref=A), lambda: acquire(store, "t", A), store.cleanup):
        with pytest.raises(SessionSchemaError):
            operation()
    assert issubclass(SessionSchemaError, SessionStoreUnavailable)
    assert schema_facts(path) == before  # nothing changed, nothing repaired


@pytest.mark.parametrize("version", [2, 99, -1])
def test_an_unsupported_schema_version_is_refused(make_store, clock, tmp_path, version, caplog):
    path = tmp_path / "sessions.db"
    make_legacy_db(path, clock, version=version)
    _assert_refused(make_store(path), path, schema_facts(path))
    assert "unsupported or invalid" in caplog.text


def _version_1_database(make_store, path, tamper):
    make_store(path).create(owner_ref=A)
    with sqlite3.connect(path) as connection:
        tamper(connection)


TAMPERING = {
    "trigger dropped": lambda c: c.execute("DROP TRIGGER sessions_owner_immutable"),
    "trigger replaced": lambda c: (c.execute("DROP TRIGGER sessions_owner_immutable"), c.execute(
        "CREATE TRIGGER sessions_owner_immutable BEFORE UPDATE OF access_owner_ref ON sessions BEGIN SELECT 1; END")),
    "extra trigger": lambda c: c.execute(
        "CREATE TRIGGER sneaky AFTER INSERT ON sessions BEGIN UPDATE sessions SET turn_count = 0; END"),
    "turns table dropped": lambda c: c.execute("DROP TABLE turns"),
}


@pytest.mark.parametrize("case", list(TAMPERING))
def test_a_version_1_database_that_fails_verification_is_refused(make_store, tmp_path, case):
    path = tmp_path / "sessions.db"
    _version_1_database(make_store, path, TAMPERING[case])
    _assert_refused(make_store(path), path, schema_facts(path))


@pytest.mark.parametrize("column", ["missing", "INTEGER", "TEXT NOT NULL DEFAULT ''"])
def test_a_version_1_database_without_a_valid_owner_column_is_refused(make_store, clock, tmp_path, column):
    path = tmp_path / "sessions.db"
    make_legacy_db(path, clock, version=1)  # claims version 1
    if column != "missing":
        with sqlite3.connect(path) as connection:
            connection.execute(f"ALTER TABLE sessions ADD COLUMN access_owner_ref {column}")
            connection.execute(sessions.SCHEMA_OWNER_TRIGGER)
    _assert_refused(make_store(path), path, schema_facts(path))


def test_a_failed_migration_rolls_back_completely(make_store, clock, tmp_path):
    path = tmp_path / "sessions.db"
    make_legacy_db(path, clock)
    with sqlite3.connect(path) as connection:  # an unexpected trigger fails verification after the ALTER
        connection.execute("CREATE TRIGGER sneaky AFTER INSERT ON sessions BEGIN SELECT 1; END")
    before = schema_facts(path)
    _assert_refused(make_store(path), path, before)
    columns, _, _, version = before
    assert version == 0 and "access_owner_ref" not in [c[1] for c in columns]


# ===================================================================
# Legacy ownerless rows: never usable, never adopted, expire normally
# ===================================================================

def test_legacy_rows_are_unusable_by_every_owner(make_store, clock, tmp_path):
    path = tmp_path / "sessions.db"
    make_legacy_db(path, clock)
    store = make_store(path)
    for owner in (A, B):
        assert store.lookup("legacy-token-1", owner_ref=owner) is None
        with pytest.raises(SessionExpired):
            acquire(store, "legacy-token-1", owner)
        store.delete("legacy-token-1", owner_ref=owner)
    with raw(path) as connection:
        rows = connection.execute("SELECT id, access_owner_ref, lease_owner FROM sessions ORDER BY id").fetchall()
    assert [tuple(r) for r in rows] == [("L1", None, None), ("L2", None, None)]


def test_legacy_rows_expire_and_are_cleaned_up_normally(make_store, clock, tmp_path):
    path = tmp_path / "sessions.db"
    make_legacy_db(path, clock)
    store = make_store(path)
    assert store.cleanup().deleted == 0  # still within their lifetime: kept
    clock.advance(IDLE + 1)
    assert store.cleanup().deleted == 2
    with raw(path) as connection:
        assert connection.execute("SELECT count(*) FROM turns").fetchone()[0] == 0


def test_legacy_rows_count_toward_the_global_cap_but_no_owners_cap(make_store, clock, tmp_path):
    path = tmp_path / "sessions.db"
    make_legacy_db(path, clock)
    store = make_store(path, max_active_sessions=3, max_active_sessions_per_owner=1)
    store.create(owner_ref=A)  # 2 legacy + 1: A's own cap was free
    with pytest.raises(SessionCapacityReached) as caught:
        store.create(owner_ref=B)
    assert type(caught.value) is SessionCapacityReached  # the global cap, not B's


# ===================================================================
# Per-owner cap
# ===================================================================

def test_the_per_owner_cap_defaults_to_20_and_is_configurable():
    assert config.Settings.model_fields["session_max_active_per_owner"].default == 20
    assert sessions.DEFAULT_MAX_ACTIVE_SESSIONS_PER_OWNER == 20
    assert SessionStore.from_settings(TEST_SETTINGS).max_active_sessions_per_owner == 20
    store = SessionStore.from_settings(TEST_SETTINGS.model_copy(update={"session_max_active_per_owner": 3}))
    assert store.max_active_sessions_per_owner == 3
    assert store.max_active_sessions == 5000  # the global cap is unchanged


@pytest.mark.parametrize("value", [0, -1, 1.5, None])
def test_an_invalid_per_owner_cap_is_rejected(make_store, value):
    with pytest.raises(ValueError):
        make_store(max_active_sessions_per_owner=value)


def test_an_owner_with_20_active_sessions_is_refused_others_are_not(make_store, caplog):
    store = make_store()
    for _ in range(20):
        store.create(owner_ref=A)
    with pytest.raises(SessionOwnerCapacityReached) as caught:
        store.create(owner_ref=A)
    assert str(caught.value) == "Session storage is unavailable."  # generic, no SQLite detail
    assert isinstance(caught.value, SessionCapacityReached) and isinstance(caught.value, SessionStoreUnavailable)
    store.create(owner_ref=B)
    with raw(store.path) as connection:
        assert connection.execute("SELECT count(*) FROM sessions WHERE access_owner_ref = ?", (A,)).fetchone()[0] == 20
    assert A not in caplog.text and B not in caplog.text
    assert "Per-owner session cap reached." in caplog.text


def test_expired_sessions_no_longer_count_toward_the_owners_cap(make_store, clock):
    store = make_store(max_active_sessions_per_owner=2)
    store.create(owner_ref=A)
    store.create(owner_ref=A)
    with pytest.raises(SessionOwnerCapacityReached):
        store.create(owner_ref=A)
    clock.advance(IDLE + 1)  # idle-expired (no live lease): not active
    store.create(owner_ref=A)


def test_a_live_lease_keeps_an_idle_session_counted(make_store, clock):
    store = make_store(max_active_sessions_per_owner=1)
    created = store.create(owner_ref=A)
    with raw(store.path) as connection:  # past idle, but still leased: active by the existing rule
        connection.execute("UPDATE sessions SET lease_owner = 'r1', lease_expires_at = ? WHERE id = ?",
                           (clock() + IDLE * 2, created.session_id))
    clock.advance(IDLE + 1)
    with pytest.raises(SessionOwnerCapacityReached):
        store.create(owner_ref=A)


def test_the_global_cap_still_applies(make_store):
    store = make_store(max_active_sessions=3, max_active_sessions_per_owner=20)
    store.create(owner_ref=A)
    store.create(owner_ref=B)
    store.create(owner_ref=A)
    with pytest.raises(SessionCapacityReached) as caught:
        store.create(owner_ref=B)
    assert type(caught.value) is SessionCapacityReached


def _parallel_creates(store, owners):
    barrier, outcomes, lock = threading.Barrier(len(owners)), [], threading.Lock()

    def run(owner):
        barrier.wait(GUARD)
        try:
            store.create(owner_ref=owner)
            outcome = ("created", owner)
        except SessionOwnerCapacityReached:
            outcome = ("owner-cap", owner)
        except SessionCapacityReached:
            outcome = ("global-cap", owner)
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=run, args=(owner,)) for owner in owners]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(GUARD * 4)
    assert len(outcomes) == len(owners)
    return outcomes


def test_50_parallel_creates_by_one_owner_admit_exactly_20(make_store):
    store = make_store(busy_timeout_seconds=15.0)
    outcomes = _parallel_creates(store, [A] * 50)
    assert outcomes.count(("created", A)) == 20 and outcomes.count(("owner-cap", A)) == 30


def test_a_full_owner_stays_full_under_parallel_attempts_while_others_proceed(make_store):
    store = make_store(busy_timeout_seconds=15.0)
    for _ in range(20):
        store.create(owner_ref=A)
    outcomes = _parallel_creates(store, [A, B] * 10)
    assert outcomes.count(("owner-cap", A)) == 10 and outcomes.count(("created", B)) == 10
    with raw(store.path) as connection:
        counts = dict(connection.execute(
            "SELECT access_owner_ref, count(*) FROM sessions GROUP BY access_owner_ref").fetchall())
    assert counts == {A: 20, B: 10}


# ===================================================================
# B3-R1: WAL initialisation under lock contention
# ===================================================================

def _sqlite_error(code, message="database is locked", kind=sqlite3.OperationalError):
    exc = kind(message)
    if code is not None:
        exc.sqlite_errorcode = code
    return exc


@pytest.fixture
def scripted():
    """A store connection whose ``PRAGMA journal_mode = WAL`` follows a script, on a driven clock."""
    from types import SimpleNamespace

    state = SimpleNamespace(script=[], attempts=0, busy_timeouts=[], sleeps=[], now=0.0, advance_on_sleep=True)

    class ScriptedConnection(sqlite3.Connection):
        def execute(self, sql, *args):
            if sql.startswith("PRAGMA busy_timeout ="):
                state.busy_timeouts.append(int(sql.split("=", 1)[1]))
            if sql == "PRAGMA journal_mode = WAL":
                state.attempts += 1
                if state.script:
                    step = state.script.pop(0)
                    if isinstance(step, BaseException):
                        raise step
                    return super().execute("SELECT ?", (step,))
            return super().execute(sql, *args)

    def sleep(seconds):
        assert seconds > 0  # never a zero-length busy spin
        state.sleeps.append(seconds)
        if state.advance_on_sleep:
            state.now += seconds

    state.connect = lambda *a, **k: sqlite3.connect(*a, factory=ScriptedConnection, **k)
    original = (sessions._monotonic, sessions._sleep)
    sessions._monotonic, sessions._sleep = (lambda: state.now), sleep
    try:
        yield state
    finally:
        sessions._monotonic, sessions._sleep = original


def _journal_mode(path):
    with sqlite3.connect(path) as connection:
        return connection.execute("PRAGMA journal_mode").fetchone()[0]


def test_wal_initialisation_retries_transient_busy_and_locked(make_store, scripted):
    busy_snapshot = 517  # SQLITE_BUSY_SNAPSHOT: an extended BUSY code
    scripted.script = [_sqlite_error(5), _sqlite_error(6, "database table is locked"), _sqlite_error(busy_snapshot)]
    store = make_store(connect=scripted.connect, busy_timeout_seconds=2.0)
    store.create(owner_ref=A)
    assert scripted.attempts == 4 and len(scripted.sleeps) == 3
    caps = [sessions.WAL_INIT_FIRST_DELAY_SECONDS * 2 ** i for i in range(3)]
    assert all(0 < slept <= cap for slept, cap in zip(scripted.sleeps, caps))  # jittered exponential backoff
    assert _journal_mode(store.path) == "wal" and schema_facts(store.path)[3] == 1
    assert scripted.busy_timeouts[-1] == 2000  # the connection's busy timeout is restored


def test_each_attempts_busy_wait_never_exceeds_the_remaining_deadline(make_store, scripted):
    scripted.script = [_sqlite_error(5)] * 3
    make_store(connect=scripted.connect, busy_timeout_seconds=1.0).create(owner_ref=A)
    *per_attempt, restored = scripted.busy_timeouts  # one per attempt (3 busy + 1 success), then the restore
    assert len(per_attempt) == 4 and per_attempt == sorted(per_attempt, reverse=True) and per_attempt[0] <= 1000
    assert per_attempt[-1] < per_attempt[0] and restored == 1000


def test_wal_initialisation_stops_at_the_deadline(make_store, scripted):
    scripted.script = [_sqlite_error(5)] * 10_000
    store = make_store(connect=scripted.connect, busy_timeout_seconds=0.25)
    with pytest.raises(SessionStoreUnavailable):
        store.create(owner_ref=A)
    assert sum(scripted.sleeps) <= 0.25 + 1e-9
    assert 1 < scripted.attempts < sessions.WAL_INIT_MAX_ATTEMPTS
    assert all(slept <= sessions.WAL_INIT_MAX_DELAY_SECONDS for slept in scripted.sleeps)
    with raw(store.path) as connection:  # nothing migrated: fail closed before the schema transaction
        assert connection.execute("SELECT count(*) FROM sqlite_master").fetchone()[0] == 0
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0


def test_wal_initialisation_stops_at_the_attempt_limit(make_store, scripted):
    scripted.advance_on_sleep = False  # a clock that never reaches the deadline
    scripted.script = [_sqlite_error(5)] * 10_000
    store = make_store(connect=scripted.connect, busy_timeout_seconds=1000.0)
    with pytest.raises(SessionStoreUnavailable):
        store.create(owner_ref=A)
    assert scripted.attempts == sessions.WAL_INIT_MAX_ATTEMPTS
    assert len(scripted.sleeps) == sessions.WAL_INIT_MAX_ATTEMPTS - 1


@pytest.mark.parametrize("error", [
    _sqlite_error(10, "disk I/O error"),               # SQLITE_IOERR
    _sqlite_error(11, "database disk image is malformed", sqlite3.DatabaseError),  # SQLITE_CORRUPT
    _sqlite_error(8, "attempt to write a readonly database"),  # SQLITE_READONLY
    _sqlite_error(14, "unable to open database file"),  # SQLITE_CANTOPEN
    _sqlite_error(26, "file is not a database", sqlite3.DatabaseError),  # SQLITE_NOTADB
    _sqlite_error(13, "database or disk is full"),      # SQLITE_FULL
    _sqlite_error(3, "access permission denied"),       # SQLITE_PERM
    _sqlite_error(None, "database is locked"),          # a lock message without a code: not trusted
    _sqlite_error(5, "database is locked", sqlite3.DatabaseError),  # BUSY code, but not an OperationalError
], ids=["ioerr", "corrupt", "readonly", "cantopen", "notadb", "full", "perm", "message-only", "not-operational"])
def test_other_errors_fail_at_once_without_retrying(make_store, scripted, error):
    scripted.script = [error]
    store = make_store(connect=scripted.connect)
    with pytest.raises(SessionStoreUnavailable):
        store.create(owner_ref=A)
    assert scripted.attempts == 1 and scripted.sleeps == []


@pytest.mark.parametrize("mode", ["delete", "memory", "truncate", None])
def test_a_journal_mode_other_than_wal_fails_closed(make_store, scripted, mode, caplog):
    scripted.script = [mode]
    store = make_store(connect=scripted.connect)
    with pytest.raises(SessionStoreUnavailable):
        store.create(owner_ref=A)
    assert scripted.attempts == 1 and "WAL journal mode could not be enabled" in caplog.text
    with raw(store.path) as connection:
        assert connection.execute("SELECT count(*) FROM sqlite_master").fetchone()[0] == 0


def test_real_contention_is_waited_out_then_migrated(make_store, clock, tmp_path):
    path = tmp_path / "sessions.db"
    make_legacy_db(path, clock)
    holder = sqlite3.connect(path, isolation_level=None, check_same_thread=False)  # released by the timer
    holder.execute("BEGIN EXCLUSIVE")  # another process holding the database
    releaser = threading.Timer(0.3, lambda: holder.execute("COMMIT"))
    releaser.start()
    try:
        make_store(path, busy_timeout_seconds=5.0).create(owner_ref=A)
    finally:
        releaser.join(GUARD)
        holder.close()
    assert _journal_mode(path) == "wal" and schema_facts(path)[3] == 1


def test_repeated_concurrent_wal_initialisation_by_six_processes(clock, tmp_path):
    script = (
        "import sys; sys.path.insert(0, sys.argv[1])\n"
        "from app.services.sessions import SessionStore\n"
        "SessionStore(sys.argv[2], idle_timeout_seconds=1800, absolute_timeout_seconds=28800,"
        " max_active_sessions=10, busy_timeout_seconds=15)._ensure_schema()\n"
    )
    for round_number in range(5):
        path = tmp_path / f"round-{round_number}.db"
        make_legacy_db(path, clock)
        processes = [subprocess.Popen([sys.executable, "-I", "-c", script, str(ROOT), str(path)],
                                      stderr=subprocess.PIPE, text=True) for _ in range(6)]
        errors = [process.communicate(timeout=60)[1] for process in processes]
        assert [p.returncode for p in processes] == [0] * 6, (round_number, errors)
        assert _journal_mode(path) == "wal" and schema_facts(path)[3] == 1
