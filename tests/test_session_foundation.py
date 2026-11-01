"""A2-0: checks for the session test foundation and the revision 8 SQL.

No session store exists yet. These tests check that:
* every §6 and §6.1 statement compiles on this SQLite build;
* the advancing-clock helper detects a statement that reads ``app_now()``
  more than once, and the approved statements read it exactly once;
* the connection spy, pause points, trigger, token scanner and row counter
  behave as later steps will rely on.
"""

import asyncio
import logging
import sqlite3
import threading

import pytest

from tests import session_sql
from tests.session_sql import MIN_SQLITE, STATEMENTS, open_schema, sqlite_version_tuple
from tests.session_support import (
    AdvancingClock,
    Checkpoints,
    ConnectionSpy,
    ManualTrigger,
    count_session_rows,
    find_token,
    register_app_now,
)

TTL, TURN_MAX, IDLE, ABSOLUTE = 30.0, 180.0, 1800.0, 28800.0


# -------------------------------------------------------------------
# SQLite version and compilation
# -------------------------------------------------------------------

def test_local_sqlite_meets_the_minimum():
    assert sqlite_version_tuple() >= MIN_SQLITE, sqlite3.sqlite_version


@pytest.mark.parametrize("name", list(STATEMENTS))
def test_statement_compiles(name):
    sql, params, _ = STATEMENTS[name]
    connection = open_schema()

    connection.execute("EXPLAIN " + sql, params).fetchall()


def test_compile_all_reports_every_statement():
    assert session_sql.compile_all() == {name: None for name in STATEMENTS}


# -------------------------------------------------------------------
# One time value per statement
# -------------------------------------------------------------------

def seeded(clock):
    """A schema with sessions in the states the statements expect.

    S1: valid, leased by r1 (epoch 1, instance i1, generation 1)
    S2: valid, not leased
    S3: expired, not leased (cleanup target)
    """
    connection = open_schema()
    register_app_now(connection, clock)
    now = clock.peek()
    rows = [
        ("S1", "h1", now, now, now + IDLE, now + ABSOLUTE, 1, "r1", "i1", 1, now, now + TTL),
        ("S2", "h2", now, now, now + IDLE, now + ABSOLUTE, 0, None, None, None, None, None),
        ("S3", "h3", now - 9000, now - 9000, now - 10, now + ABSOLUTE, 0, None, None, None, None, None),
    ]
    connection.executemany(
        """INSERT INTO sessions (id, token_hash, created_at, last_activity_at, idle_expires_at,
               absolute_expires_at, lease_epoch, lease_owner, lease_instance, lease_keeper_gen,
               lease_started_at, lease_expires_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    clock.reset_count()
    return connection


def row(connection, sid):
    connection.row_factory = sqlite3.Row
    return dict(connection.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone())


def test_negative_control_two_reads_in_one_statement_are_detected():
    clock = AdvancingClock()
    connection = seeded(clock)

    connection.execute(
        "UPDATE sessions SET created_at = app_now(), last_activity_at = app_now() WHERE id = 'S2'"
    )

    assert clock.calls == 2
    stored = row(connection, "S2")
    assert stored["created_at"] != stored["last_activity_at"]  # the detector sees the difference


def test_negative_control_repeated_subqueries_without_a_cte_are_detected():
    clock = AdvancingClock()
    connection = seeded(clock)

    connection.execute(
        """UPDATE sessions SET created_at = (SELECT app_now()), last_activity_at = (SELECT app_now())
           WHERE id = 'S2'"""
    )

    assert clock.calls == 2


TIME_STATEMENTS = [name for name, (_, _, reads_time) in STATEMENTS.items() if reads_time]


@pytest.mark.parametrize("name", TIME_STATEMENTS)
def test_statement_reads_the_clock_exactly_once(name):
    clock = AdvancingClock()
    connection = seeded(clock)
    if name == "acquire":
        sql, params, _ = STATEMENTS[name]
        params = {**params, "sid": "S2"}  # the unleased session
    else:
        sql, params, _ = STATEMENTS[name]

    result = connection.execute(sql, params).fetchall()

    assert clock.calls == 1, f"{name} called app_now() {clock.calls} times"
    assert result or name in {"create_session", "cleanup"}, f"{name} matched no row"


def test_acquire_stores_one_consistent_time():
    clock = AdvancingClock()
    connection = seeded(clock)
    sql, params, _ = STATEMENTS["acquire"]

    epoch, expires, started = connection.execute(sql, {**params, "sid": "S2"}).fetchone()

    stored = row(connection, "S2")
    assert (epoch, stored["lease_owner"]) == (1, "r2")
    assert stored["last_activity_at"] == started
    assert expires == started + TTL  # min(now + ttl, now + turn_max) from a single now


def test_renew_stores_one_consistent_time():
    clock = AdvancingClock()
    connection = seeded(clock)
    sql, params, _ = STATEMENTS["renew"]

    (expires,) = connection.execute(sql, params).fetchone()

    stored = row(connection, "S1")
    assert clock.calls == 1
    now = clock.values[0]
    assert expires == min(now + TTL, stored["lease_started_at"] + TURN_MAX)
    assert stored["last_activity_at"] == now
    assert stored["idle_expires_at"] == now + IDLE


def test_save_returns_the_time_used_for_the_turn():
    clock = AdvancingClock()
    connection = seeded(clock)
    sql, params, _ = STATEMENTS["save_update"]

    turn_count, last_activity = connection.execute(sql, params).fetchone()

    assert clock.calls == 1 and last_activity == clock.values[0]
    stored = row(connection, "S1")
    assert stored["lease_owner"] is None and stored["lease_expires_at"] is None
    assert turn_count == 1


def test_create_session_uses_one_time_for_all_timestamps():
    clock = AdvancingClock()
    connection = seeded(clock)
    sql, params, _ = STATEMENTS["create_session"]

    connection.execute(sql, params)

    stored = row(connection, "S9")
    now = clock.values[0]
    assert (stored["created_at"], stored["last_activity_at"]) == (now, now)
    assert (stored["idle_expires_at"], stored["absolute_expires_at"]) == (now + IDLE, now + ABSOLUTE)


def test_cleanup_removes_only_the_expired_unleased_session():
    clock = AdvancingClock()
    connection = seeded(clock)
    sql, params, _ = STATEMENTS["cleanup"]

    connection.execute(sql, params)

    remaining = {r[0] for r in connection.execute("SELECT id FROM sessions")}
    assert remaining == {"S1", "S2"}
    assert clock.calls == 1


def test_bulk_release_matches_epoch_and_owner_per_row():
    connection = seeded(AdvancingClock())
    sql, _, _ = STATEMENTS["bulk_release"]

    # S1 matches; a stale epoch for S1 and an unleased S2 must not match anything else.
    cursor = connection.execute(sql, ("i1", "S1", 1, "r1", "S2", 7, "rX"))

    assert cursor.rowcount == 1
    assert row(connection, "S1")["lease_owner"] is None


def test_materialized_is_what_makes_it_one_read():
    """Characterisation: without MATERIALIZED, SQLite may read the clock per reference."""
    clock = AdvancingClock()
    connection = seeded(clock)

    connection.execute(
        """WITH t(now) AS NOT MATERIALIZED (SELECT app_now())
           UPDATE sessions SET created_at = (SELECT now FROM t),
                               last_activity_at = (SELECT now FROM t)
           WHERE id = 'S2'"""
    )

    # Recorded either way; the approved statements rely on MATERIALIZED, not on this.
    assert clock.calls in (1, 2)


# -------------------------------------------------------------------
# Connection spy
# -------------------------------------------------------------------

def test_spy_records_timeout_and_thread(tmp_path):
    spy = ConnectionSpy()

    def worker():
        spy(tmp_path / "s.db", timeout=0.75).execute("SELECT 1").fetchall()

    thread = threading.Thread(target=worker, name="spy-worker")
    thread.start()
    thread.join(5)

    assert spy.timeouts() == [0.75]
    assert {r.thread_name for r in spy.connects} == {"spy-worker"}
    assert [r.sql for r in spy.executes] == ["SELECT 1"]


def test_spy_injects_lock_errors_then_recovers(tmp_path):
    spy = ConnectionSpy()
    spy.fail_next(2, matching=lambda sql: sql.startswith("SELECT"))
    connection = spy(tmp_path / "s.db", timeout=0.1)

    for _ in range(2):
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            connection.execute("SELECT 1")

    assert connection.execute("SELECT 1").fetchone() == (1,)


def test_spy_stall_blocks_until_released_and_records_interrupt(tmp_path):
    spy = ConnectionSpy()
    spy.stall(matching=lambda sql: "UPDATE" in sql)
    connection = spy(tmp_path / "s.db", timeout=0.1, check_same_thread=False)
    connection.execute("CREATE TABLE x (v INTEGER)")
    done = threading.Event()

    def worker():
        connection.execute("UPDATE x SET v = 1")
        done.set()

    thread = threading.Thread(target=worker)
    thread.start()
    assert spy.stall_reached.wait(5)

    connection.interrupt()  # has no effect on a stall, as with a blocked I/O call
    assert not done.wait(0.2)
    assert spy.interrupts == 1

    spy.release_stall()
    assert done.wait(5)
    thread.join(5)


@pytest.mark.anyio
async def test_spy_detects_use_on_the_event_loop_thread(tmp_path):
    spy = ConnectionSpy()
    loop_thread = threading.get_ident()

    await asyncio.to_thread(lambda: spy(tmp_path / "s.db").execute("SELECT 1").fetchall())
    spy.assert_not_on_thread(loop_thread)  # off-loop use passes

    spy(tmp_path / "s.db").execute("SELECT 1").fetchall()  # on the loop thread
    with pytest.raises(AssertionError, match="forbidden thread"):
        spy.assert_not_on_thread(loop_thread)


def test_spy_detects_two_reads_through_a_real_connection(tmp_path):
    spy = ConnectionSpy()
    clock = AdvancingClock()
    connection = spy(tmp_path / "s.db")
    register_app_now(connection, clock)

    connection.execute("SELECT app_now(), app_now()").fetchone()

    assert clock.calls == 2


# -------------------------------------------------------------------
# Pause points and trigger
# -------------------------------------------------------------------

def test_checkpoint_pauses_until_released():
    points = Checkpoints()
    points.pause("before_renew_update")
    passed = threading.Event()

    def worker():
        points.hit("before_renew_update")
        passed.set()

    thread = threading.Thread(target=worker)
    thread.start()
    points.wait_reached("before_renew_update")
    assert not passed.wait(0.1)

    points.release("before_renew_update")
    assert passed.wait(5)
    thread.join(5)
    points.hit("not_paused")  # passes straight through
    assert points.hits == ["before_renew_update", "not_paused"]


def test_manual_trigger_runs_one_cycle_per_fire():
    trigger = ManualTrigger()
    cycles = []

    def loop():
        while trigger.wait(timeout=1.0):
            cycles.append(len(cycles))
            if len(cycles) == 3:
                return

    thread = threading.Thread(target=loop)
    thread.start()
    for _ in range(3):
        trigger.fire()
    thread.join(5)

    assert cycles == [0, 1, 2] and trigger.cycles == 3
    assert trigger.wait(timeout=0.05) is False  # no pending fire


# -------------------------------------------------------------------
# Token scanner and row counter
# -------------------------------------------------------------------

TOKEN = "q3VtokenZk_test_value_0123456789abcdefABCDEF"


def test_scanner_finds_token_in_log_message_args_and_exceptions(caplog):
    logger = logging.getLogger("tests.session_foundation")
    with caplog.at_level(logging.INFO):
        logger.info("plain line")
        logger.info("header was %s", TOKEN)
        try:
            raise ValueError(f"bad {TOKEN}")
        except ValueError:
            logger.exception("failed")

    hits = find_token(TOKEN, records=caplog.records)

    assert len(hits.locations) == 2
    assert not find_token("not-present", records=caplog.records)


def test_scanner_finds_token_in_database_and_wal(tmp_path):
    path = tmp_path / "audit.db"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("CREATE TABLE t (v TEXT)")
    connection.execute("INSERT INTO t VALUES (?)", (TOKEN,))
    connection.commit()

    hits = find_token(TOKEN, db_paths=[path])
    connection.close()

    assert hits  # in the WAL or main file, before or after checkpoint
    assert not find_token(TOKEN, db_paths=[tmp_path / "missing.db"])


def test_scanner_finds_token_in_bodies():
    assert find_token(TOKEN, bodies=[b"x", f'{{"id":"{TOKEN}"}}']).locations == ["body[1]"]
    assert not find_token(TOKEN, bodies=[b'{"session":{"turn_saved":true}}'])


def test_row_counter(tmp_path):
    path = tmp_path / "sessions.db"
    assert count_session_rows(path) == 0  # no file

    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE other (v)")
    connection.commit()
    assert count_session_rows(path) == 0  # no table

    for ddl in session_sql.SCHEMA:
        connection.execute(ddl)
    connection.execute("INSERT INTO sessions (id, token_hash) VALUES ('a', 'h')")
    connection.commit()
    connection.close()
    assert count_session_rows(path) == 1
