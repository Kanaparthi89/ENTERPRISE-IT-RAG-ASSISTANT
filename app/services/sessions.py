"""SQLite storage for conversation sessions (A2, design revision 8).

This module is the store only: schema, create, look-up, delete, expiry and
cleanup. Leases, the lease keeper, API endpoints and the workflow are added
in later steps and are not used here.

Rules every operation follows:

* Only ``SHA-256(token)`` is stored; the bearer token itself is never stored
  or logged. Logs use the internal session ID.
* Each operation opens a **new** connection whose ``busy_timeout`` (a
  connection-wide setting) is the store's lock-wait limit, runs one short
  transaction and closes it. Connections are never shared or reused.
* Every statement that needs the time reads ``app_now()`` once, through
  ``WITH t(now) AS MATERIALIZED (SELECT app_now())``.
* SQLite is never used on an event-loop thread; callers run store
  operations in a worker thread.
* SQLite older than 3.35.0 (needed for ``MATERIALIZED`` and ``RETURNING``)
  disables the store: every operation raises ``SessionStoreUnavailable``.
"""

import asyncio
import hashlib
import logging
import random
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)

MIN_SQLITE_VERSION = (3, 35, 0)
MAX_TOKEN_LENGTH = 128

SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        token_hash TEXT UNIQUE NOT NULL,
        created_at REAL, last_activity_at REAL, idle_expires_at REAL, absolute_expires_at REAL,
        turn_count INTEGER NOT NULL DEFAULT 0,
        lease_epoch INTEGER NOT NULL DEFAULT 0,
        lease_owner TEXT, lease_instance TEXT, lease_keeper_gen INTEGER,
        lease_started_at REAL, lease_expires_at REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS turns (
        session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
        seq INTEGER, epoch INTEGER, request_id TEXT,
        question TEXT, answer TEXT, source_used TEXT, created_at REAL,
        PRIMARY KEY (session_id, seq)
    )
    """,
)

# Statements from design revision 8, §6.1 (verified in A2-0 on SQLite
# 3.45.3, 3.46.1 and 3.51.0). tests/test_session_store.py checks they stay
# identical to tests/session_sql.py.
SQL_CREATE_SESSION = """
        WITH t(now) AS MATERIALIZED (SELECT app_now())
        INSERT INTO sessions (id, token_hash, created_at, last_activity_at,
                              idle_expires_at, absolute_expires_at)
        SELECT :id, :token_hash, t.now, t.now, t.now + :idle, t.now + :absolute
          FROM t
        """

SQL_CAP_CHECK = """
        WITH t(now) AS MATERIALIZED (SELECT app_now())
        SELECT count(*)
          FROM sessions AS s, t
         WHERE s.absolute_expires_at > t.now
           AND (s.idle_expires_at > t.now
                OR (s.lease_owner IS NOT NULL AND s.lease_expires_at > t.now))
        """

SQL_LOOKUP = """
        WITH t(now) AS MATERIALIZED (SELECT app_now())
        SELECT s.id,
               (s.lease_owner IS NOT NULL AND s.lease_expires_at > t.now) AS live_lease,
               (s.absolute_expires_at <= t.now
                OR (s.idle_expires_at <= t.now
                    AND NOT (s.lease_owner IS NOT NULL AND s.lease_expires_at > t.now))) AS expired,
               s.lease_expires_at - t.now AS lease_remaining
          FROM sessions AS s, t
         WHERE s.token_hash = :token_hash
        """

SQL_CLEANUP = """
        WITH t(now) AS MATERIALIZED (SELECT app_now())
        DELETE FROM sessions
         WHERE id IN (
               SELECT s.id
                 FROM sessions AS s, t
                WHERE (s.lease_owner IS NULL OR s.lease_expires_at <= t.now)
                  AND (s.idle_expires_at <= t.now OR s.absolute_expires_at <= t.now)
                LIMIT :batch)
        """

SQL_DELETE_SESSION = "DELETE FROM sessions WHERE token_hash = :token_hash"

# Lease statements, design revision 8 §6 / §6.1 (A2-2).
SQL_ACQUIRE = """
        WITH t(now) AS MATERIALIZED (SELECT app_now())
        UPDATE sessions
           SET lease_epoch      = lease_epoch + 1,
               lease_owner      = :rid,
               lease_instance   = :inst,
               lease_keeper_gen = :gen,
               lease_started_at = t.now,
               lease_expires_at = min(t.now + :ttl, t.now + :turn_max),
               last_activity_at = t.now
          FROM t
         WHERE sessions.id = :sid
           AND sessions.idle_expires_at > t.now
           AND sessions.absolute_expires_at > t.now
           AND (sessions.lease_owner IS NULL OR sessions.lease_expires_at <= t.now)
        RETURNING lease_epoch, lease_expires_at, lease_started_at
        """

SQL_RENEW = """
        WITH t(now) AS MATERIALIZED (SELECT app_now())
        UPDATE sessions
           SET lease_expires_at = min(t.now + :ttl, lease_started_at + :turn_max),
               last_activity_at = t.now,
               idle_expires_at  = max(idle_expires_at, t.now + :idle)
          FROM t
         WHERE sessions.id = :sid
           AND sessions.lease_epoch = :epoch
           AND sessions.lease_owner = :rid
           AND sessions.lease_instance = :inst
           AND sessions.lease_keeper_gen = :gen
           AND sessions.lease_expires_at > t.now
           AND sessions.lease_started_at + :turn_max > t.now
        RETURNING lease_expires_at
        """

SQL_SAVE_UPDATE = """
        WITH t(now) AS MATERIALIZED (SELECT app_now())
        UPDATE sessions
           SET turn_count       = turn_count + 1,
               last_activity_at = t.now,
               idle_expires_at  = max(idle_expires_at, t.now + :idle),
               lease_owner      = NULL,
               lease_expires_at = NULL,
               lease_keeper_gen = NULL
          FROM t
         WHERE sessions.id = :sid
           AND sessions.lease_epoch = :epoch
           AND sessions.lease_owner = :rid
           AND sessions.lease_expires_at > t.now
           AND sessions.lease_started_at + :turn_max > t.now
        RETURNING turn_count, last_activity_at
        """

SQL_SAVE_INSERT_TURN = """
        INSERT INTO turns (session_id, seq, epoch, request_id, question, answer, source_used, created_at)
        VALUES (:sid, :turn_count, :epoch, :rid, :question, :answer, :source_used, :last_activity_at)
        """

SQL_SAVE_TRIM = "DELETE FROM turns WHERE session_id = :sid AND seq <= :turn_count - :max_turns"

SQL_RELEASE = """
        UPDATE sessions
           SET lease_owner = NULL, lease_expires_at = NULL, lease_keeper_gen = NULL
         WHERE id = :sid AND lease_epoch = :epoch AND lease_owner = :rid
        """

SQL_KEEPER_MOVE = """
        WITH t(now) AS MATERIALIZED (SELECT app_now())
        UPDATE sessions
           SET lease_keeper_gen = :new_gen
          FROM t
         WHERE sessions.id = :sid
           AND sessions.lease_epoch = :epoch
           AND sessions.lease_owner = :rid
           AND sessions.lease_instance = :inst
           AND sessions.lease_expires_at > t.now
        RETURNING lease_expires_at
        """

# Bulk release: the VALUES list is generated for each chunk of up to 300 leases.
SQL_BULK_RELEASE_PREFIX = """
        UPDATE sessions
           SET lease_owner = NULL, lease_expires_at = NULL, lease_keeper_gen = NULL
         WHERE lease_instance = ?
           AND (id, lease_epoch, lease_owner) IN (VALUES """
BULK_RELEASE_CHUNK = 300

SQL_HISTORY = """
        SELECT seq, question, answer, source_used FROM turns
         WHERE session_id = ? ORDER BY seq DESC LIMIT ?
        """

SQL_LEASE_STATE = """
        WITH t(now) AS MATERIALIZED (SELECT app_now())
        SELECT s.lease_epoch, s.lease_owner, s.lease_instance, s.lease_keeper_gen,
               s.lease_started_at, s.lease_expires_at, t.now
          FROM sessions AS s, t
         WHERE s.id = ?
        """


class SessionStoreUnavailable(Exception):
    """The session database can't be used: version, open, lock or I/O failure.

    The message is generic and never contains a token.
    """


class SessionCapacityReached(SessionStoreUnavailable):
    """The active-session cap has been reached; no session was created."""


@dataclass(frozen=True)
class CreatedSession:
    token: str  # returned to the client once; never stored
    session_id: str  # internal UUID used in logs and audit rows
    created_at: float
    absolute_expires_at: float
    idle_timeout_seconds: float


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    live_lease: bool
    expired: bool
    lease_remaining: float | None


@dataclass(frozen=True)
class CleanupResult:
    deleted: int
    checkpoint_busy: bool | None  # None when no checkpoint was attempted


class SessionExpired(Exception):
    """Unknown, deleted or expired session (the API answers 410 for all of these)."""


class SessionBusy(Exception):
    """Another request holds the session's live lease (the API answers 409)."""

    def __init__(self, retry_after: float):
        super().__init__("A previous question in this conversation is still being answered.")
        self.retry_after = max(0.0, retry_after)


@dataclass(frozen=True)
class Turn:
    seq: int
    question: str
    answer: str
    source_used: str | None


@dataclass(frozen=True)
class Lease:
    session_id: str
    request_id: str
    epoch: int
    instance: str
    keeper_gen: int
    started_at: float
    expires_at: float
    history: tuple[Turn, ...] = ()


@dataclass(frozen=True)
class BulkReleaseResult:
    status: str  # "committed" | "rolled_back"
    released: int


def sqlite_version_tuple(version: str | None = None) -> tuple[int, int, int]:
    parts = (version or sqlite3.sqlite_version).split(".")
    return tuple(int(part) for part in (parts + ["0", "0"])[:3])


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def is_well_formed_token(value: str | None) -> bool:
    """Header shape only (non-empty, at most 128 characters); says nothing about existence."""
    return isinstance(value, str) and 0 < len(value.strip()) <= MAX_TOKEN_LENGTH


def _run_change(connection: sqlite3.Connection, sql: str, params) -> int:
    """Run a data-changing statement to completion and return the rows it changed.

    Python's ``Cursor.rowcount`` is -1 for statements that start with ``WITH``
    (it only recognises a leading INSERT/UPDATE/DELETE), and such a statement
    is left open until it is fully read, which blocks a later checkpoint. So
    the statement is always read to the end and SQLite's ``changes()`` used.
    """
    connection.execute(sql, params).fetchall()
    return connection.execute("SELECT changes()").fetchone()[0]


def _forbid_event_loop_thread() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError("The session store must not be used on an event-loop thread.")


class SessionStore:
    """SQLite-backed session memory. All methods block; call them off the event loop."""

    def __init__(
        self,
        path: Path,
        *,
        idle_timeout_seconds: float,
        absolute_timeout_seconds: float,
        max_active_sessions: int,
        busy_timeout_seconds: float = 2.0,
        cleanup_batch: int = 500,
        clock: Callable[[], float] = time.time,
        connect: Callable[..., sqlite3.Connection] = sqlite3.connect,
        token_factory: Callable[[], str] = lambda: secrets.token_urlsafe(32),
        id_factory: Callable[[], str] = lambda: str(uuid4()),
        sqlite_version: str | None = None,
        hooks=None,
    ):
        self.path = Path(path)
        self.idle_timeout_seconds = idle_timeout_seconds
        self.absolute_timeout_seconds = absolute_timeout_seconds
        self.max_active_sessions = max_active_sessions
        self.busy_timeout_seconds = busy_timeout_seconds
        self.cleanup_batch = cleanup_batch
        self._clock = clock
        self._connect_fn = connect
        self._token_factory = token_factory
        self._id_factory = id_factory
        self.supported = sqlite_version_tuple(sqlite_version) >= MIN_SQLITE_VERSION
        self._schema_ready = False
        self._schema_lock = threading.Lock()
        self._unsupported_logged = False
        # Test-only pause points (``hooks.hit(name)``); a no-op in production.
        self._hooks = hooks

    @classmethod
    def from_settings(cls, settings, **overrides) -> "SessionStore":
        options = dict(
            idle_timeout_seconds=settings.session_idle_timeout_seconds,
            absolute_timeout_seconds=settings.session_absolute_timeout_seconds,
            max_active_sessions=settings.max_active_sessions,
            busy_timeout_seconds=settings.session_db_busy_timeout_seconds,
            cleanup_batch=settings.session_cleanup_batch,
        )
        options.update(overrides)
        return cls(settings.session_db_path, **options)

    # -- connections -----------------------------------------------------

    def _check_supported(self) -> None:
        if self.supported:
            return
        if not self._unsupported_logged:
            self._unsupported_logged = True
            logger.error(
                "Session store disabled: SQLite %s is older than the required %s.",
                sqlite3.sqlite_version,
                ".".join(map(str, MIN_SQLITE_VERSION)),
            )
        raise SessionStoreUnavailable("Session storage is unavailable.")

    def _open(self, timeout: float | None = None) -> sqlite3.Connection:
        """A new connection; its busy_timeout applies to every statement on it."""
        _forbid_event_loop_thread()
        connection = self._connect_fn(
            str(self.path),
            timeout=self.busy_timeout_seconds if timeout is None else timeout,
            isolation_level=None,  # explicit BEGIN/COMMIT
        )
        try:
            # journal_mode is stored in the database file and set once in
            # _ensure_schema; setting it here would wait for the write lock on
            # every connection. These per-connection settings never wait.
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA secure_delete = ON")
            connection.create_function("app_now", 0, self._clock, deterministic=False)
        except BaseException:
            connection.close()
            raise
        return connection

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = self._open()
            try:
                connection.execute("PRAGMA journal_mode = WAL")  # persistent: once per file
                connection.execute("BEGIN IMMEDIATE")
                for ddl in SCHEMA:
                    connection.execute(ddl)
                connection.execute("COMMIT")
            finally:
                connection.close()
            self._schema_ready = True

    def _run(self, operation: str, work: Callable[[sqlite3.Connection], object]):
        """Open, ensure the schema, run ``work`` and close; map errors to a generic one."""
        self._check_supported()
        try:
            self._ensure_schema()
            connection = self._open()
            try:
                return work(connection)
            finally:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                connection.close()
        except SessionCapacityReached:
            raise
        except sqlite3.Error as exc:
            logger.error("Session store %s failed (%s).", operation, type(exc).__name__)
            raise SessionStoreUnavailable("Session storage is unavailable.") from exc

    # -- operations --------------------------------------------------------

    def create(self) -> CreatedSession:
        """Create a session. Runs one cleanup batch first, then enforces the cap."""
        try:
            self.cleanup()
        except SessionStoreUnavailable:
            pass  # best effort (already logged); the cap check below still applies
        token = self._token_factory()
        session_id = self._id_factory()

        def work(connection):
            connection.execute("BEGIN IMMEDIATE")
            (active,) = connection.execute(SQL_CAP_CHECK).fetchone()
            if active >= self.max_active_sessions:
                connection.execute("ROLLBACK")
                logger.warning("Session cap reached: %d active sessions.", active)
                raise SessionCapacityReached("Session storage is unavailable.")
            connection.execute(
                SQL_CREATE_SESSION,
                {
                    "id": session_id,
                    "token_hash": hash_token(token),
                    "idle": self.idle_timeout_seconds,
                    "absolute": self.absolute_timeout_seconds,
                },
            )
            created_at, absolute = connection.execute(
                "SELECT created_at, absolute_expires_at FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            connection.execute("COMMIT")
            return created_at, absolute

        created_at, absolute = self._run("create", work)
        logger.info("Session created: id=%s", session_id)
        return CreatedSession(
            token=token,
            session_id=session_id,
            created_at=created_at,
            absolute_expires_at=absolute,
            idle_timeout_seconds=self.idle_timeout_seconds,
        )

    def lookup(self, token: str) -> SessionInfo | None:
        """The session for ``token``, or None if unknown. Expiry is reported, not enforced."""
        self._check_supported()
        if not self.path.exists():
            return None  # nothing stored yet; never create the file to answer a look-up

        def work(connection):
            return connection.execute(SQL_LOOKUP, {"token_hash": hash_token(token)}).fetchone()

        row = self._run("lookup", work)
        if row is None:
            return None
        session_id, live_lease, expired, lease_remaining = row
        return SessionInfo(
            session_id=session_id,
            live_lease=bool(live_lease),
            expired=bool(expired),
            lease_remaining=lease_remaining if live_lease else None,
        )

    def delete(self, token: str) -> None:
        """Delete the session and its turns. Idempotent: unknown tokens are not an error.

        Raises ``SessionStoreUnavailable`` if deletion can't be confirmed.
        """
        self._check_supported()
        if not self.path.exists():
            return  # nothing to delete; never create the file

        def work(connection):
            connection.execute("BEGIN IMMEDIATE")
            deleted = _run_change(connection, SQL_DELETE_SESSION, {"token_hash": hash_token(token)})
            connection.execute("COMMIT")
            return deleted

        if self._run("delete", work):
            logger.info("Session deleted.")

    def cleanup(self, batch: int | None = None, *, timeout: float | None = None) -> CleanupResult:
        """Delete one batch of expired sessions without a live lease, then checkpoint the WAL.

        Does nothing (and creates nothing) if the database file doesn't exist yet.
        """
        self._check_supported()
        if not self.path.exists():
            return CleanupResult(deleted=0, checkpoint_busy=None)
        try:
            self._ensure_schema()
            connection = self._open(timeout)
            try:
                connection.execute("BEGIN IMMEDIATE")
                deleted = _run_change(
                    connection, SQL_CLEANUP, {"batch": batch or self.cleanup_batch}
                )
                connection.execute("COMMIT")
                busy = None
                if deleted:
                    busy, _log, _checkpointed = connection.execute(
                        "PRAGMA wal_checkpoint(TRUNCATE)"
                    ).fetchone()
                    busy = bool(busy)
            finally:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                connection.close()
        except sqlite3.Error as exc:
            logger.error("Session store cleanup failed (%s).", type(exc).__name__)
            raise SessionStoreUnavailable("Session storage is unavailable.") from exc
        if deleted:
            logger.info("Session cleanup removed %d expired sessions.", deleted)
        return CleanupResult(deleted=deleted, checkpoint_busy=busy)

    def startup_cleanup(self, wait_seconds: float = 2.0) -> bool:
        """Run one cleanup pass on a helper thread and wait at most ``wait_seconds``.

        Only runs if the database file already exists (so memory from before
        the feature was disabled still expires); never creates the file.
        Returns True if the pass finished within the wait. A pass that is still
        running continues on its daemon thread and never blocks startup.
        """
        if not self.path.exists() or not self.supported:
            return True
        finished = threading.Event()

        def run():
            try:
                self.cleanup(timeout=min(self.busy_timeout_seconds, wait_seconds))
            except Exception:  # noqa: BLE001 - logged; the next cleanup retries
                logger.exception("Startup session cleanup failed.")
            finally:
                finished.set()

        threading.Thread(target=run, name="session-startup-cleanup", daemon=True).start()
        if not finished.wait(wait_seconds):
            logger.warning("Startup session cleanup still running after %.1fs.", wait_seconds)
            return False
        return True

    # -- leases (A2-2) -----------------------------------------------------

    def _hit(self, name: str) -> None:
        if self._hooks is not None:
            self._hooks.hit(name)

    def acquire(
        self,
        token: str,
        request_id: str,
        *,
        instance: str,
        keeper_gen: int,
        ttl: float,
        turn_max: float,
        history_turns: int,
    ) -> Lease:
        """Take the session's turn lease and read its history in one transaction.

        Raises ``SessionExpired`` (unknown, deleted or expired), ``SessionBusy``
        (live lease held by another request) or ``SessionStoreUnavailable``.
        """
        self._check_supported()
        if not self.path.exists():
            raise SessionExpired()
        token_hash = hash_token(token)

        def classify(connection):
            row = connection.execute(SQL_LOOKUP, {"token_hash": token_hash}).fetchone()
            if row is None or row[2]:
                raise SessionExpired()
            if row[1]:
                raise SessionBusy(row[3])
            return row[0]

        def work(connection):
            connection.execute("BEGIN IMMEDIATE")
            session_id = classify(connection)
            rows = connection.execute(
                SQL_ACQUIRE,
                {
                    "rid": request_id, "inst": instance, "gen": keeper_gen,
                    "ttl": ttl, "turn_max": turn_max, "sid": session_id,
                },
            ).fetchall()
            if not rows:  # time moved on between the two statements
                classify(connection)
                raise SessionExpired()
            epoch, expires_at, started_at = rows[0]
            history = [
                Turn(*row)
                for row in connection.execute(SQL_HISTORY, (session_id, history_turns)).fetchall()
            ] if history_turns > 0 else []
            connection.execute("COMMIT")
            return Lease(
                session_id=session_id,
                request_id=request_id,
                epoch=epoch,
                instance=instance,
                keeper_gen=keeper_gen,
                started_at=started_at,
                expires_at=expires_at,
                history=tuple(reversed(history)),
            )

        lease = self._run("acquire", work)
        logger.info("Session lease taken: id=%s epoch=%s", lease.session_id, lease.epoch)
        return lease

    def renew_once(self, lease: Lease, *, keeper_gen: int, ttl: float, turn_max: float,
                   timeout: float) -> float | None:
        """One renewal attempt on a new connection whose busy_timeout is ``timeout``.

        Returns the new ``lease_expires_at``, or None if no row matched (the
        lease is lost). SQLite errors propagate so the keeper can tell a
        temporary lock from other failures.
        """
        self._check_supported()
        connection = self._open(timeout)
        try:
            self._hit("renew:before_update")
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                SQL_RENEW,
                {
                    "ttl": ttl, "turn_max": turn_max, "idle": self.idle_timeout_seconds,
                    "sid": lease.session_id, "epoch": lease.epoch, "rid": lease.request_id,
                    "inst": lease.instance, "gen": keeper_gen,
                },
            ).fetchall()
            connection.execute("COMMIT")
            return rows[0][0] if rows else None
        finally:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            connection.close()

    def save_turn(self, lease: Lease, *, question: str, answer: str, source_used: str | None,
                  turn_max: float, max_turns: int, answer_chars: int) -> bool:
        """Save the turn and release the lease, only while the lease is still owned and valid.

        Returns False (nothing written) if the lease was lost, taken over,
        expired, past ``turn_max`` or the session was deleted.
        """

        def work(connection):
            self._hit("save:before_begin")  # test pause point; a no-op in production
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                SQL_SAVE_UPDATE,
                {
                    "idle": self.idle_timeout_seconds, "sid": lease.session_id,
                    "epoch": lease.epoch, "rid": lease.request_id, "turn_max": turn_max,
                },
            ).fetchall()
            if not rows:
                connection.execute("ROLLBACK")
                return False
            turn_count, last_activity_at = rows[0]
            connection.execute(
                SQL_SAVE_INSERT_TURN,
                {
                    "sid": lease.session_id, "turn_count": turn_count, "epoch": lease.epoch,
                    "rid": lease.request_id, "question": question,
                    "answer": answer[:answer_chars], "source_used": source_used,
                    "last_activity_at": last_activity_at,
                },
            )
            _run_change(
                connection, SQL_SAVE_TRIM,
                {"sid": lease.session_id, "turn_count": turn_count, "max_turns": max_turns},
            )
            self._hit("save:before_commit")  # test pause point; a no-op in production
            connection.execute("COMMIT")
            return True

        saved = self._run("save", work)
        logger.info(
            "Session turn %s: id=%s epoch=%s",
            "saved" if saved else "not saved (lease no longer owned)",
            lease.session_id, lease.epoch,
        )
        return saved

    def release(self, lease: Lease, *, timeout: float | None = None) -> bool:
        """Release without saving; checks epoch and owner only. True if a row was cleared."""

        def work(connection):
            connection.execute("BEGIN IMMEDIATE")
            changed = _run_change(
                connection, SQL_RELEASE,
                {"sid": lease.session_id, "epoch": lease.epoch, "rid": lease.request_id},
            )
            connection.execute("COMMIT")
            return changed == 1

        self._check_supported()
        try:
            connection = self._open(timeout)
        except sqlite3.Error as exc:
            logger.error("Session store release failed (%s).", type(exc).__name__)
            raise SessionStoreUnavailable("Session storage is unavailable.") from exc
        try:
            return work(connection)
        except sqlite3.Error as exc:
            logger.error("Session store release failed (%s).", type(exc).__name__)
            raise SessionStoreUnavailable("Session storage is unavailable.") from exc
        finally:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            connection.close()

    def move_generation(self, lease: Lease, *, new_gen: int, timeout: float | None = None) -> float | None:
        """Move a still-valid lease to a new keeper generation; None if it no longer qualifies."""

        def work(connection):
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                SQL_KEEPER_MOVE,
                {
                    "new_gen": new_gen, "sid": lease.session_id, "epoch": lease.epoch,
                    "rid": lease.request_id, "inst": lease.instance,
                },
            ).fetchall()
            connection.execute("COMMIT")
            return rows[0][0] if rows else None

        return self._run("keeper move", work)

    def lost_reason(self, lease: Lease, *, keeper_gen: int, turn_max: float) -> str:
        """Why a renewal matched no row (one read; for logs only)."""

        def work(connection):
            return connection.execute(SQL_LEASE_STATE, (lease.session_id,)).fetchone()

        try:
            row = self._run("lease state", work)
        except SessionStoreUnavailable:
            return "unknown"
        if row is None:
            return "deleted"
        epoch, owner, instance, gen, started_at, expires_at, now = row
        if epoch != lease.epoch:
            return "taken_over"
        if owner is None:
            return "released"
        if started_at is not None and started_at + turn_max <= now:
            return "turn_max"
        if expires_at is None or expires_at <= now:
            return "expired"
        if instance != lease.instance or gen != keeper_gen:
            return "generation"
        return "unknown"

    def bulk_release(
        self,
        instance: str,
        keys: list[tuple[str, int, str]],
        *,
        timeout: float,
        within_deadline: Callable[[], bool],
        on_connection: Callable[[sqlite3.Connection], None] = lambda connection: None,
    ) -> BulkReleaseResult:
        """Release many leases in one transaction, in chunks of 300, checking the deadline.

        Each row must still match its own (id, epoch, owner). If the deadline
        passes, the lock can't be taken or anything fails, the whole
        transaction is rolled back: the effect is all-or-nothing.
        """
        self._check_supported()
        connection = self._open(timeout)
        on_connection(connection)
        try:
            connection.execute("BEGIN IMMEDIATE")
            released = 0
            for start in range(0, len(keys), BULK_RELEASE_CHUNK):
                if not within_deadline():
                    connection.execute("ROLLBACK")
                    return BulkReleaseResult("rolled_back", 0)
                chunk = keys[start:start + BULK_RELEASE_CHUNK]
                sql = SQL_BULK_RELEASE_PREFIX + ", ".join(["(?, ?, ?)"] * len(chunk)) + ")"
                params = [instance] + [value for key in chunk for value in key]
                released += _run_change(connection, sql, params)
            if not within_deadline():
                connection.execute("ROLLBACK")
                return BulkReleaseResult("rolled_back", 0)
            connection.execute("COMMIT")
            return BulkReleaseResult("committed", released)
        except sqlite3.Error as exc:
            logger.error("Session bulk release failed (%s).", type(exc).__name__)
            try:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass  # rolled back by SQLite when the connection closes
            return BulkReleaseResult("rolled_back", 0)
        finally:
            connection.close()


# ---------------------------------------------------------------------------
# Lease manager and keeper (design revision 8, §6–§7)
# ---------------------------------------------------------------------------

LEASE_OK = "OK"
LEASE_UNCONFIRMED = "UNCONFIRMED"
LEASE_LOST = "LOST"

# The bulk release's lock wait ends this long before the shutdown deadline,
# so a rollback after a lock timeout is observed in time.
_RELEASE_SAFETY_MARGIN_SECONDS = 0.05

_TEMPORARY_ERROR_MARKERS = ("database is locked", "database table is locked", "busy")
_BACKOFF_SECONDS = (0.2, 0.5, 1.0, 2.0)


def _is_temporary(exc: sqlite3.Error) -> bool:
    return isinstance(exc, sqlite3.OperationalError) and any(
        marker in str(exc).lower() for marker in _TEMPORARY_ERROR_MARKERS
    )


@dataclass
class _Entry:
    lease: Lease
    confirmed_until: float
    state: str = LEASE_OK
    reported_state: str = LEASE_OK


@dataclass(frozen=True)
class ShutdownReport:
    released: int
    left_to_expire: int
    undetermined: int
    keeper_join_timed_out: bool
    release_status: str  # "none" | "committed" | "rolled_back" | "unfinished"
    elapsed: float


class _RateLimitedLog:
    def __init__(self, monotonic: Callable[[], float], interval: float = 60.0):
        self._monotonic = monotonic
        self._interval = interval
        self._last: dict[str, float] = {}

    def error(self, key: str, message: str, *args, exc_info=False) -> None:
        now = self._monotonic()
        if now - self._last.get(key, float("-inf")) >= self._interval:
            self._last[key] = now
            logger.error(message, *args, exc_info=exc_info)


class SessionLeaseManager:
    """Per-process lease registry, lease keeper and bounded shutdown.

    Requests call ``acquire()`` and always ``finish()``. One keeper thread
    renews every registered lease each TTL/3. Whenever a keeper starts or is
    stopped, the keeper *generation* is increased; renewals are fenced on it
    in the database and their results are applied only by the current
    generation, so a stale keeper can neither write nor change the registry.

    Clocks: ``store`` time (``app_now``) for lease windows, so they compare
    with values from SQL; ``monotonic`` for keeper liveness and deadlines.
    """

    def __init__(
        self,
        store: SessionStore,
        *,
        ttl: float,
        margin: float,
        turn_max: float,
        max_turns: int,
        history_turns: int,
        answer_chars: int,
        shutdown_grace: float,
        keeper_join: float,
        release_budget: float,
        cleanup_interval: float = 300.0,
        health_interval: float = 300.0,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        wait: Callable[[float], bool] | None = None,
        hooks=None,
        instance: str | None = None,
        jitter: Callable[[], float] = random.random,
    ):
        self.store = store
        self.ttl = ttl
        self.interval = ttl / 3
        self.margin = margin
        self.turn_max = turn_max
        self.max_turns = max_turns
        self.history_turns = history_turns
        self.answer_chars = answer_chars
        self.shutdown_grace = shutdown_grace
        self.keeper_join = keeper_join
        self.release_budget = release_budget
        self.cleanup_interval = cleanup_interval
        self.health_interval = health_interval
        self.instance = instance or str(uuid4())
        self._clock = store._clock
        self._monotonic = monotonic
        self._sleep = sleep
        self._wait = wait  # tests: returns True when a cycle should run
        self._hooks = hooks
        self._jitter = jitter
        self._errors = _RateLimitedLog(monotonic)

        self._lock = threading.RLock()  # registry, generation, flags
        self._drained = threading.Condition(self._lock)
        self._admission = threading.Lock()  # acquire+register vs keeper restart and shutdown
        # Admission to turn saves (H2). ``finish`` checks ``_closed`` and runs
        # the whole save transaction while holding this gate, and waits for it
        # at most the store's busy timeout (a timeout means "not saved"). The
        # shutdown fence sets ``_closed`` while holding it if it gets it within
        # the grace period, and without it otherwise; see ``shutdown`` step 3.
        # Taken before ``_lock``, never inside it.
        self._save_gate = threading.Lock()
        self._entries: dict[str, _Entry] = {}
        self._generation = 0
        self._keeper: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_cycle: float | None = None
        self._accepting = True
        self._closed = False
        self.cycles = 0
        self.renewal_failures = 0
        self.restarts = 0
        self._next_cleanup = monotonic() + cleanup_interval
        self._next_health = monotonic() + health_interval
        self._slowest_cycle = 0.0

    @classmethod
    def from_settings(cls, store: SessionStore, settings, **overrides) -> "SessionLeaseManager":
        options = dict(
            ttl=settings.session_lease_ttl_seconds,
            margin=settings.session_lease_margin_seconds,
            turn_max=settings.session_turn_max_seconds,
            max_turns=settings.session_max_turns,
            history_turns=settings.session_history_turns,
            answer_chars=settings.session_stored_answer_chars,
            shutdown_grace=settings.session_shutdown_grace_seconds,
            keeper_join=settings.session_keeper_join_seconds,
            release_budget=settings.session_release_budget_seconds,
            cleanup_interval=settings.session_cleanup_interval_seconds,
        )
        options.update(overrides)
        return cls(store, **options)

    # -- inspection (tests and health) -------------------------------------

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def state_of(self, request_id: str) -> str | None:
        with self._lock:
            entry = self._entries.get(request_id)
            return entry.state if entry else None

    def confirmed_until(self, request_id: str) -> float | None:
        with self._lock:
            entry = self._entries.get(request_id)
            return entry.confirmed_until if entry else None

    def active_count(self) -> int:
        with self._lock:
            return len(self._entries)

    def keeper_thread(self) -> threading.Thread | None:
        return self._keeper

    def _hit(self, name: str) -> None:
        if self._hooks is not None:
            self._hooks.hit(name)

    # -- requests ------------------------------------------------------------

    def acquire(self, token: str, request_id: str) -> Lease:
        """Take the session's lease for this request and register it with the keeper."""
        with self._admission:
            with self._lock:
                if not self._accepting:
                    raise SessionStoreUnavailable("Session storage is unavailable.")
                self._ensure_keeper_locked()
                generation = self._generation
            lease = self.store.acquire(
                token, request_id,
                instance=self.instance, keeper_gen=generation,
                ttl=self.ttl, turn_max=self.turn_max, history_turns=self.history_turns,
            )
            with self._lock:
                entry = _Entry(lease=lease, confirmed_until=lease.expires_at)
                if self._closed:
                    # Shutdown fenced the keeper while this acquire was in flight:
                    # nothing will renew it, so it can't be saved; it expires.
                    self._set_state_locked(entry, LEASE_LOST, "shutdown")
                self._entries[request_id] = entry
            return lease

    def finish(self, lease: Lease, turn: tuple[str, str, str | None] | None = None) -> bool | None:
        """End the request's use of its lease. Always call it, in ``finally``.

        With ``turn=(question, answer, source_used)`` the turn is saved only if
        ownership is still confirmed; returns ``turn_saved``. Without a turn the
        lease is released and None is returned. The lease is always removed
        from the keeper.
        """
        saved = None
        try:
            with self._lock:
                entry = self._entries.get(lease.request_id)
                confirmed = (
                    entry is not None
                    and entry.lease.epoch == lease.epoch
                    and entry.state != LEASE_LOST
                    and self._clock() < entry.confirmed_until
                )
            if turn is not None:
                saved = False
                if confirmed:
                    # Admission to a save is serialised with the shutdown fence
                    # (see ``shutdown`` step 3): the closed check and the whole
                    # transaction run under the save gate. The wait for the
                    # gate is bounded by the store's busy timeout, so one
                    # stalled save can't hold up every other session; a timeout
                    # means the turn is not saved and the lease is released.
                    refused = None
                    if not self._save_gate.acquire(timeout=self.store.busy_timeout_seconds):
                        refused = "save_gate_timeout"
                    else:
                        try:
                            if self._closed:
                                refused = "manager_closed"
                            else:
                                try:
                                    saved = self.store.save_turn(
                                        lease, question=turn[0], answer=turn[1],
                                        source_used=turn[2], turn_max=self.turn_max,
                                        max_turns=self.max_turns, answer_chars=self.answer_chars,
                                    )
                                except SessionStoreUnavailable:
                                    saved = False
                        finally:
                            self._save_gate.release()
                    if refused is not None:
                        logger.warning(
                            "Session turn not saved: id=%s reason=%s", lease.session_id, refused
                        )
                else:
                    logger.warning(
                        "Session turn not saved: id=%s reason=%s", lease.session_id,
                        "lease_lost" if entry is not None and entry.state == LEASE_LOST
                        else "lease_unconfirmed",
                    )
            if not saved:
                self._release_quietly(lease)
            return saved
        finally:
            with self._lock:
                entry = self._entries.get(lease.request_id)
                if entry is not None and entry.lease.epoch == lease.epoch:
                    del self._entries[lease.request_id]
                self._drained.notify_all()

    def _release_quietly(self, lease: Lease) -> None:
        """Release with one retry; otherwise the lease expires within its TTL."""
        timeout = self.store.busy_timeout_seconds / 2
        for _attempt in range(2):
            try:
                self.store.release(lease, timeout=timeout)
                return
            except SessionStoreUnavailable:
                continue
        logger.warning("Session lease release failed; it will expire: id=%s", lease.session_id)

    # -- keeper ----------------------------------------------------------------

    def _ensure_keeper_locked(self) -> None:
        """Start the keeper, or restart it if it died or stopped cycling (caller holds _lock)."""
        alive = self._keeper is not None and self._keeper.is_alive()
        fresh = (
            self._last_cycle is not None
            and self._monotonic() - self._last_cycle <= 2 * self.interval
        )
        if alive and fresh:
            return
        restarting = self._keeper is not None
        self._stop.set()  # the old keeper, if any, stops at its next check
        self._generation += 1
        generation = self._generation
        self._stop = threading.Event()
        self._last_cycle = self._monotonic()
        if restarting:
            self.restarts += 1
            for entry in list(self._entries.values()):
                try:
                    expires = self.store.move_generation(
                        entry.lease, new_gen=generation,
                        timeout=min(1.0, self.store.busy_timeout_seconds),
                    )
                except SessionStoreUnavailable:
                    expires = None
                if expires is None:
                    self._set_state_locked(entry, LEASE_LOST, "keeper_restart")
                else:
                    entry.confirmed_until = expires
            logger.error("session_keeper_restarted generation=%d", generation)
        self._keeper = threading.Thread(
            target=self._keeper_main, args=(generation, self._stop),
            name=f"session-keeper-{generation}", daemon=True,
        )
        self._keeper.start()

    def _is_current(self, generation: int) -> bool:
        return generation == self._generation and not self._closed

    def _keeper_main(self, generation: int, stop: threading.Event) -> None:
        while True:
            try:
                if self._wait is not None:
                    ready = self._wait(self.interval)
                else:
                    ready = not stop.wait(self.interval)
                with self._lock:
                    if stop.is_set() or not self._is_current(generation):
                        return
                    self._last_cycle = self._monotonic()
                if ready:
                    self._cycle(generation, stop)
            except Exception:  # noqa: BLE001 - the keeper must never die silently
                self._errors.error("keeper", "session_keeper_error", exc_info=True)

    def _cycle(self, generation: int, stop: threading.Event) -> None:
        started = self._monotonic()
        with self._lock:
            entries = sorted(self._entries.values(), key=lambda entry: entry.confirmed_until)
        for entry in entries:
            with self._lock:
                if stop.is_set() or not self._is_current(generation):
                    return
            try:
                self._hit("keeper:before_lease")
                self._renew_entry(entry, generation, stop)
            except Exception:  # noqa: BLE001 - one lease never stops the others
                self._errors.error("lease", "session_keeper_lease_error", exc_info=True)
                with self._lock:
                    if self._is_current(generation) and self._entries.get(entry.lease.request_id) is entry:
                        if entry.state != LEASE_LOST:
                            self._set_state_locked(entry, LEASE_UNCONFIRMED, "keeper_error")
        self.cycles += 1
        self._slowest_cycle = max(self._slowest_cycle, self._monotonic() - started)
        now = self._monotonic()
        if now >= self._next_cleanup:
            self._next_cleanup = now + self.cleanup_interval
            try:
                self.store.cleanup(timeout=min(1.0, self.interval))
            except Exception:  # noqa: BLE001 - next cycle retries
                self._errors.error("cleanup", "session_cleanup_failed", exc_info=True)
        if now >= self._next_health:
            self._next_health = now + self.health_interval
            with self._lock:
                active = len(self._entries)
            logger.info(
                "session_keeper_health active=%d renewal_failures=%d slowest_cycle=%.3fs generation=%d",
                active, self.renewal_failures, self._slowest_cycle, generation,
            )
            self.renewal_failures = 0
            self._slowest_cycle = 0.0

    def _renew_entry(self, entry: _Entry, generation: int, stop: threading.Event) -> None:
        if entry.state == LEASE_LOST:
            return
        window_end = entry.confirmed_until - self.margin
        attempt = 0
        while not stop.is_set():
            now = self._clock()
            if now >= window_end:
                break
            try:
                expires = self.store.renew_once(
                    entry.lease, keeper_gen=generation, ttl=self.ttl,
                    turn_max=self.turn_max, timeout=min(1.0, window_end - now),
                )
            except sqlite3.Error as exc:
                self.renewal_failures += 1
                if not _is_temporary(exc):
                    self._errors.error("renew", "session_lease_renew_error (%s)", type(exc).__name__)
                else:
                    logger.debug("Session lease renewal retry: id=%s", entry.lease.session_id)
                self._apply(entry, generation, LEASE_UNCONFIRMED, None, "renew_error")
                delay = _BACKOFF_SECONDS[min(attempt, len(_BACKOFF_SECONDS) - 1)]
                delay *= 0.5 + self._jitter()
                attempt += 1
                remaining = window_end - self._clock()
                if remaining > 0:
                    self._sleep(min(delay, remaining))
                continue
            if expires is None:
                reason = self.store.lost_reason(entry.lease, keeper_gen=generation, turn_max=self.turn_max)
                self._apply(entry, generation, LEASE_LOST, None, reason)
            else:
                self._apply(entry, generation, LEASE_OK, expires, None)
            return
        if not stop.is_set():
            self._apply(entry, generation, LEASE_LOST, None, "unconfirmed_window_ended")

    def _apply(self, entry: _Entry, generation: int, state: str, expires: float | None,
               reason: str | None) -> bool:
        """Apply a renewal result, only from the current keeper and for a registered lease."""
        with self._lock:
            if not self._is_current(generation) or self._entries.get(entry.lease.request_id) is not entry:
                return False
            if entry.state == LEASE_LOST:
                return False
            if expires is not None:
                entry.confirmed_until = expires
            self._set_state_locked(entry, state, reason)
            return True

    def _set_state_locked(self, entry: _Entry, state: str, reason: str | None) -> None:
        entry.state = state
        if state != entry.reported_state:
            entry.reported_state = state
            if state == LEASE_OK:
                logger.info("Session lease confirmed again: id=%s", entry.lease.session_id)
            else:
                logger.warning(
                    "session_lease_%s id=%s reason=%s",
                    state.lower(), entry.lease.session_id, reason or "unknown",
                )

    # -- shutdown (§7.5) ---------------------------------------------------------

    def shutdown(self) -> ShutdownReport:
        """Bounded shutdown; returns by start + grace + keeper_join + release_budget.

        Blocks; call it off the event loop. A SQLite call already blocked on
        storage may keep running on its daemon thread after this returns; the
        bulk release is one transaction, so its effect is all-or-nothing.
        """
        start = self._monotonic()
        grace_end = start + self.shutdown_grace
        join_end = grace_end + self.keeper_join
        deadline = join_end + self.release_budget

        # 1. Stop accepting new leases and keeper restarts. Waiting for an
        #    in-flight acquire is bounded: one stuck on storage can't hold us.
        got_admission = self._admission.acquire(timeout=max(0.0, grace_end - self._monotonic()))
        try:
            with self._lock:
                self._accepting = False
        finally:
            if got_admission:
                self._admission.release()

        # 2. Let in-flight turns finish (the keeper keeps renewing).
        with self._lock:
            while self._entries:
                remaining = grace_end - self._monotonic()
                if remaining <= 0:
                    break
                self._drained.wait(timeout=min(0.05, remaining))

        # 3. Fence the keeper, stop new cycles and refuse new turn saves (H2).
        #    The fence waits for the save gate at most until the grace period
        #    ends, then sets ``_closed`` with or without it, so a save stuck on
        #    storage can't hold shutdown. Afterwards:
        #    - a save that takes the gate after ``_closed`` is set is refused;
        #    - a save already admitted (it holds the gate) is not cancelled:
        #      its SQLite transaction commits only if it gets the write lock
        #      and still owns the lease. If the bulk release below commits
        #      first, the save fails its lease check and saves nothing.
        #    Admission before the fence does not guarantee that a turn is saved.
        got_gate = self._save_gate.acquire(timeout=max(0.0, grace_end - self._monotonic()))
        try:
            got_admission = self._admission.acquire(timeout=max(0.0, grace_end - self._monotonic()))
            try:
                with self._lock:
                    self._generation += 1
                    self._closed = True
                    self._stop.set()
                    keeper = self._keeper
                    keys = [
                        (entry.lease.session_id, entry.lease.epoch, entry.lease.request_id)
                        for entry in self._entries.values()
                    ]
            finally:
                if got_admission:
                    self._admission.release()
            self._hit("shutdown:fenced")  # test pause point; a no-op in production
        finally:
            if got_gate:
                self._save_gate.release()

        # 4. Wait for the keeper thread (a daemon).
        keeper_timed_out = False
        if keeper is not None:
            keeper.join(max(0.0, join_end - self._monotonic()))
            keeper_timed_out = keeper.is_alive()

        # 5–6. Release what's left in bulk, on a helper thread, by the deadline.
        status, released, undetermined = "none", 0, 0
        if keys:
            outcome: dict = {}
            holder: dict = {}

            def run():
                try:
                    outcome["result"] = self.store.bulk_release(
                        self.instance, keys,
                        timeout=max(0.01, min(
                            self.release_budget,
                            deadline - self._monotonic() - _RELEASE_SAFETY_MARGIN_SECONDS,
                        )),
                        within_deadline=lambda: self._monotonic() < deadline,
                        on_connection=lambda connection: holder.setdefault("connection", connection),
                    )
                except Exception:  # noqa: BLE001 - reported as rolled back
                    logger.exception("Session bulk release failed.")
                    outcome["result"] = BulkReleaseResult("rolled_back", 0)

            helper = threading.Thread(target=run, name="session-release", daemon=True)
            helper.start()
            helper.join(max(0.0, deadline - self._monotonic()))
            if helper.is_alive():
                connection = holder.get("connection")
                if connection is not None:
                    try:
                        connection.interrupt()
                    except Exception:  # noqa: BLE001 - best effort
                        pass
                status, undetermined = "unfinished", len(keys)
                logger.error(
                    "session_release_unfinished leases=%d (outcome undetermined; "
                    "unreleased leases expire within their TTL)", len(keys),
                )
            else:
                result = outcome["result"]
                status, released = result.status, result.released
                if status == "rolled_back":
                    logger.error("Session bulk release rolled back; %d leases left to expire.", len(keys))

        left = len(keys) - released - undetermined if status != "unfinished" else 0
        report = ShutdownReport(
            released=released,
            left_to_expire=left,
            undetermined=undetermined,
            keeper_join_timed_out=keeper_timed_out,
            release_status=status,
            elapsed=self._monotonic() - start,
        )
        if keeper_timed_out:
            logger.error("Session keeper did not stop within its join timeout.")
        logger.info(
            "session_shutdown released=%d left_to_expire=%d undetermined=%d keeper_join_timed_out=%s elapsed=%.2fs",
            report.released, report.left_to_expire, report.undetermined,
            report.keeper_join_timed_out, report.elapsed,
        )
        return report


# ---------------------------------------------------------------------------
# Process-wide runtime (A2-5): one store and one lease manager per process
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionRuntime:
    store: SessionStore
    manager: SessionLeaseManager


_runtime: SessionRuntime | None = None
_runtime_lock = threading.Lock()


def get_session_runtime(settings) -> SessionRuntime:
    """The process's session store and lease manager, created on first use.

    Creating them opens no database and starts no thread: the keeper starts
    with the first lease, and the database file is created with the first
    session.
    """
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            store = SessionStore.from_settings(settings)
            _runtime = SessionRuntime(store, SessionLeaseManager.from_settings(store, settings))
        return _runtime


def peek_session_runtime() -> SessionRuntime | None:
    """The runtime if one was created; never creates one."""
    with _runtime_lock:
        return _runtime


def clear_session_runtime(runtime: SessionRuntime | None = None) -> None:
    """Forget the runtime (after shutdown), so a restarted app gets a fresh one."""
    global _runtime
    with _runtime_lock:
        if runtime is None or _runtime is runtime:
            _runtime = None
