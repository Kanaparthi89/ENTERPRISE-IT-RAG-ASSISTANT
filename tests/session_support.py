"""A2-0 test foundation for session management (design revision 8, §22).

Test utilities only; nothing here is used by the application.

* ``AdvancingClock``: a clock that moves forward on every call and counts
  calls, to show that a statement reads ``app_now()`` exactly once.
* ``ConnectionSpy``: a stand-in for ``sqlite3.connect`` that records the
  timeout and thread of every connection and statement. It can make
  statements fail with "database is locked" (``fail_next``) or block until
  released (``stall``), and records ``interrupt()`` calls.
* ``Checkpoints``: named pause points for keeper and request hooks.
* ``ManualTrigger``: lets a test run one keeper cycle on demand.
* ``find_token``: searches logs, SQLite files (including WAL) and response
  bodies for a bearer token.
* ``count_session_rows``: the number of rows in a ``sessions`` table.
"""

import logging
import sqlite3
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

GUARD = 5.0

# Security S1-5 Phase B: the store requires an explicit owner on every
# token operation (no default). Tests that don't test ownership use this one.
TEST_OWNER = "a1" * 32
OTHER_OWNER = "b2" * 32


# -------------------------------------------------------------------
# Clock
# -------------------------------------------------------------------

class AdvancingClock:
    """Returns a larger value on every call; counts calls.

    Register it as ``app_now()`` with ``connection.create_function``. If a
    statement read the time twice, ``calls`` would increase by two and the
    stored values would differ by ``step``.
    """

    def __init__(self, start: float = 1_000.0, step: float = 0.25):
        self._value = start
        self.step = step
        self.calls = 0
        self.values: list[float] = []
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            value = self._value
            self._value += self.step
            self.calls += 1
            self.values.append(value)
            return value

    def peek(self) -> float:
        with self._lock:
            return self._value

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._value += seconds

    def reset_count(self) -> None:
        with self._lock:
            self.calls = 0
            self.values.clear()


def register_app_now(connection: sqlite3.Connection, clock: Callable[[], float]) -> None:
    """Register ``app_now()`` as the session store will (non-deterministic)."""
    connection.create_function("app_now", 0, clock, deterministic=False)


# -------------------------------------------------------------------
# Connections: timeouts, threads, injected failures and stalls
# -------------------------------------------------------------------

@dataclass
class ConnectRecord:
    database: str
    timeout: float
    thread_id: int
    thread_name: str


@dataclass
class ExecuteRecord:
    sql: str
    thread_id: int
    thread_name: str


class ConnectionSpy:
    """A stand-in for ``sqlite3.connect`` that records and scripts behaviour.

    ``fail_next(n, matching=...)`` makes the next ``n`` matching statements raise
    ``sqlite3.OperationalError("database is locked")``. ``stall(matching=...)``
    makes the next matching statement block until ``release_stall()``.
    """

    def __init__(self):
        self.connects: list[ConnectRecord] = []
        self.executes: list[ExecuteRecord] = []
        self.interrupts = 0
        self._lock = threading.Lock()
        self._failures: list[tuple[Callable[[str], bool], int]] = []
        self._stall_predicate: Callable[[str], bool] | None = None
        self.stall_reached = threading.Event()
        self._stall_release = threading.Event()

    # -- scripting ---------------------------------------------------

    def fail_next(self, count: int, matching: Callable[[str], bool] = lambda sql: True) -> None:
        with self._lock:
            self._failures.append((matching, count))

    def stall(self, matching: Callable[[str], bool] = lambda sql: True) -> None:
        with self._lock:
            self._stall_predicate = matching
            self.stall_reached.clear()
            self._stall_release.clear()

    def release_stall(self) -> None:
        self._stall_release.set()

    # -- the connect() replacement -----------------------------------

    def __call__(self, database, timeout: float = 5.0, **kwargs) -> sqlite3.Connection:
        thread = threading.current_thread()
        with self._lock:
            self.connects.append(ConnectRecord(str(database), timeout, thread.ident, thread.name))
        spy = self

        class SpyConnection(sqlite3.Connection):
            def execute(self, sql, *args, **kw):
                spy._before_execute(sql)
                return super().execute(sql, *args, **kw)

            def interrupt(self):
                with spy._lock:
                    spy.interrupts += 1
                return super().interrupt()

        return sqlite3.connect(database, timeout=timeout, factory=SpyConnection, **kwargs)

    def _before_execute(self, sql: str) -> None:
        thread = threading.current_thread()
        stall = False
        with self._lock:
            self.executes.append(ExecuteRecord(sql, thread.ident, thread.name))
            for index, (predicate, remaining) in enumerate(self._failures):
                if remaining > 0 and predicate(sql):
                    self._failures[index] = (predicate, remaining - 1)
                    raise sqlite3.OperationalError("database is locked")
            if self._stall_predicate is not None and self._stall_predicate(sql):
                self._stall_predicate = None
                stall = True
        if stall:
            self.stall_reached.set()
            # Blocks like an I/O stall: interrupt() cannot end it, only the test can.
            if not self._stall_release.wait(GUARD * 6):
                raise RuntimeError("stalled statement was never released")

    # -- assertions ----------------------------------------------------

    def threads_used(self) -> set[int]:
        with self._lock:
            return {r.thread_id for r in self.connects} | {r.thread_id for r in self.executes}

    def assert_not_on_thread(self, thread_id: int) -> None:
        offenders = [
            r for r in (*self.connects, *self.executes) if r.thread_id == thread_id
        ]
        if offenders:
            raise AssertionError(f"SQLite used on a forbidden thread: {offenders[:3]}")

    def timeouts(self) -> list[float]:
        with self._lock:
            return [r.timeout for r in self.connects]


# -------------------------------------------------------------------
# Pause points and manual triggers (keeper and request hooks)
# -------------------------------------------------------------------

class Checkpoints:
    """Named pause points. Code under test calls ``hit(name)``.

    A test calls ``pause(name)`` first; ``hit`` then blocks until
    ``release(name)``. Unpaused names pass straight through.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._paused: dict[str, threading.Event] = {}
        self._reached: dict[str, threading.Event] = {}
        self.hits: list[str] = []

    def pause(self, name: str) -> None:
        with self._lock:
            self._paused[name] = threading.Event()
            self._reached[name] = threading.Event()

    def hit(self, name: str) -> None:
        with self._lock:
            self.hits.append(name)
            release = self._paused.get(name)
            reached = self._reached.get(name)
        if release is None:
            return
        reached.set()
        if not release.wait(GUARD * 6):
            raise RuntimeError(f"checkpoint {name!r} was never released")

    def wait_reached(self, name: str, timeout: float = GUARD) -> None:
        with self._lock:
            reached = self._reached[name]
        if not reached.wait(timeout):
            raise AssertionError(f"checkpoint {name!r} not reached")

    def release(self, name: str) -> None:
        with self._lock:
            event = self._paused.pop(name)
            self._reached.pop(name, None)
        event.set()


class ManualTrigger:
    """Replaces a timed loop wait: each ``fire()`` lets one cycle run."""

    def __init__(self):
        self._condition = threading.Condition()
        self._pending = 0
        self.cycles = 0

    def fire(self) -> None:
        with self._condition:
            self._pending += 1
            self._condition.notify_all()

    def wait(self, timeout: float | None = None) -> bool:
        """Called by the loop under test instead of sleeping."""
        with self._condition:
            if not self._condition.wait_for(lambda: self._pending > 0, timeout):
                return False
            self._pending -= 1
            self.cycles += 1
            return True


# -------------------------------------------------------------------
# Token scanner and row counter
# -------------------------------------------------------------------

@dataclass
class TokenHits:
    locations: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.locations)


def _record_text(record: logging.LogRecord) -> str:
    parts = [str(record.msg), repr(record.args)]
    try:
        parts.append(record.getMessage())
    except Exception:  # noqa: BLE001 - malformed records still get scanned raw
        pass
    if record.exc_info:
        parts.append(logging.Formatter().formatException(record.exc_info))
    if record.exc_text:
        parts.append(record.exc_text)
    parts.extend(repr(value) for key, value in vars(record).items() if key not in {"msg", "args"})
    return "\n".join(parts)


def find_token(
    token: str,
    *,
    records: Iterable[logging.LogRecord] = (),
    db_paths: Iterable[Path] = (),
    bodies: Iterable[bytes | str] = (),
) -> TokenHits:
    """Search logs, SQLite files (raw bytes of the db, -wal and -journal) and bodies."""
    hits = TokenHits()
    needle = token.encode("utf-8")

    for index, record in enumerate(records):
        if token in _record_text(record):
            hits.locations.append(f"log[{index}] {record.name}")

    for path in db_paths:
        for candidate in (Path(path), Path(f"{path}-wal"), Path(f"{path}-journal")):
            if candidate.exists() and needle in candidate.read_bytes():
                hits.locations.append(f"file {candidate.name}")

    for index, body in enumerate(bodies):
        data = body.encode("utf-8") if isinstance(body, str) else bytes(body)
        if needle in data:
            hits.locations.append(f"body[{index}]")

    return hits


def count_session_rows(db_path: Path) -> int:
    """Rows in ``sessions``; 0 when the database or the table does not exist."""
    path = Path(db_path)
    if not path.exists():
        return 0
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sessions'"
        ).fetchone()
        return connection.execute("SELECT count(*) FROM sessions").fetchone()[0] if exists else 0
    finally:
        connection.close()


# -------------------------------------------------------------------
# Multi-process acquisition race (A2-2)
# -------------------------------------------------------------------

def acquire_race_worker(db_path, tokens, instance, barrier, results) -> None:
    """Run in a spawned process: try to acquire every token's lease once.

    Imports the app inside the function so a spawned child only loads what
    it needs. Reports the indexes this process won.
    """
    from app.services.sessions import SessionBusy, SessionStore

    store = SessionStore(
        db_path,
        idle_timeout_seconds=1800.0,
        absolute_timeout_seconds=28800.0,
        max_active_sessions=10**6,
        busy_timeout_seconds=10.0,
    )
    barrier.wait()
    wins = []
    for index, token in enumerate(tokens):
        try:
            store.acquire(
                token, f"{instance}-{index}", instance=instance, keeper_gen=1,
                ttl=300.0, turn_max=600.0, history_turns=0, owner_ref=TEST_OWNER,
            )
            wins.append(index)
        except SessionBusy:
            pass
    results.put((instance, wins))
