"""A2 design revision 8: the session-store SQL (§5.1, §6, §6.1), for verification only.

This is not the session store. It holds the approved schema and statements so
tests can check, before A2-1, that each statement compiles on a given SQLite
build and reads ``app_now()`` once.

It uses only the standard library, so it also runs inside the Docker base
image without the project's dependencies:

    docker run --rm -v "$PWD/tests:/t:ro" python:3.11-slim python /t/session_sql.py
"""

import sqlite3
import sys

MIN_SQLITE = (3, 35, 0)

SCHEMA = (
    """
    CREATE TABLE sessions (
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
    CREATE TABLE turns (
        session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
        seq INTEGER, epoch INTEGER, request_id TEXT,
        question TEXT, answer TEXT, source_used TEXT, created_at REAL,
        PRIMARY KEY (session_id, seq)
    )
    """,
)

# name -> (sql, parameters, reads_time)
STATEMENTS = {
    "acquire": (
        """
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
        """,
        {"rid": "r2", "inst": "i1", "gen": 1, "ttl": 30.0, "turn_max": 180.0, "sid": "S1"},
        True,
    ),
    "renew": (
        """
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
        """,
        {"ttl": 30.0, "turn_max": 180.0, "idle": 1800.0, "sid": "S1", "epoch": 1,
         "rid": "r1", "inst": "i1", "gen": 1},
        True,
    ),
    "save_update": (
        """
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
        """,
        {"idle": 1800.0, "sid": "S1", "epoch": 1, "rid": "r1", "turn_max": 180.0},
        True,
    ),
    "save_insert_turn": (
        """
        INSERT INTO turns (session_id, seq, epoch, request_id, question, answer, source_used, created_at)
        VALUES (:sid, :turn_count, :epoch, :rid, :question, :answer, :source_used, :last_activity_at)
        """,
        {"sid": "S1", "turn_count": 1, "epoch": 1, "rid": "r1", "question": "q",
         "answer": "a", "source_used": "private_kb", "last_activity_at": 1000.0},
        False,
    ),
    "save_trim": (
        "DELETE FROM turns WHERE session_id = :sid AND seq <= :turn_count - :max_turns",
        {"sid": "S1", "turn_count": 21, "max_turns": 20},
        False,
    ),
    "release": (
        """
        UPDATE sessions
           SET lease_owner = NULL, lease_expires_at = NULL, lease_keeper_gen = NULL
         WHERE id = :sid AND lease_epoch = :epoch AND lease_owner = :rid
        """,
        {"sid": "S1", "epoch": 1, "rid": "r1"},
        False,
    ),
    "create_session": (
        """
        WITH t(now) AS MATERIALIZED (SELECT app_now())
        INSERT INTO sessions (id, token_hash, created_at, last_activity_at,
                              idle_expires_at, absolute_expires_at)
        SELECT :id, :token_hash, t.now, t.now, t.now + :idle, t.now + :absolute
          FROM t
        """,
        {"id": "S9", "token_hash": "h9", "idle": 1800.0, "absolute": 28800.0},
        True,
    ),
    "cap_check": (
        """
        WITH t(now) AS MATERIALIZED (SELECT app_now())
        SELECT count(*)
          FROM sessions AS s, t
         WHERE s.absolute_expires_at > t.now
           AND (s.idle_expires_at > t.now
                OR (s.lease_owner IS NOT NULL AND s.lease_expires_at > t.now))
        """,
        {},
        True,
    ),
    "lookup": (
        """
        WITH t(now) AS MATERIALIZED (SELECT app_now())
        SELECT s.id,
               (s.lease_owner IS NOT NULL AND s.lease_expires_at > t.now) AS live_lease,
               (s.absolute_expires_at <= t.now
                OR (s.idle_expires_at <= t.now
                    AND NOT (s.lease_owner IS NOT NULL AND s.lease_expires_at > t.now))) AS expired,
               s.lease_expires_at - t.now AS lease_remaining
          FROM sessions AS s, t
         WHERE s.token_hash = :token_hash
        """,
        {"token_hash": "h1"},
        True,
    ),
    "keeper_generation_move": (
        """
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
        """,
        {"new_gen": 2, "sid": "S1", "epoch": 1, "rid": "r1", "inst": "i1"},
        True,
    ),
    "cleanup": (
        """
        WITH t(now) AS MATERIALIZED (SELECT app_now())
        DELETE FROM sessions
         WHERE id IN (
               SELECT s.id
                 FROM sessions AS s, t
                WHERE (s.lease_owner IS NULL OR s.lease_expires_at <= t.now)
                  AND (s.idle_expires_at <= t.now OR s.absolute_expires_at <= t.now)
                LIMIT :batch)
        """,
        {"batch": 500},
        True,
    ),
    "delete_session": (
        "DELETE FROM sessions WHERE token_hash = :token_hash",
        {"token_hash": "h1"},
        False,
    ),
    "bulk_release": (
        """
        UPDATE sessions
           SET lease_owner = NULL, lease_expires_at = NULL, lease_keeper_gen = NULL
         WHERE lease_instance = ?
           AND (id, lease_epoch, lease_owner) IN (VALUES (?, ?, ?), (?, ?, ?))
        """,
        ("i1", "S1", 1, "r1", "S2", 1, "r2"),
        False,
    ),
}


def sqlite_version_tuple(module=sqlite3) -> tuple[int, int, int]:
    return tuple(int(part) for part in module.sqlite_version.split("."))[:3]


def open_schema(app_now=lambda: 0.0) -> sqlite3.Connection:
    """An in-memory database with the A2 schema and ``app_now()`` registered."""
    connection = sqlite3.connect(":memory:")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.create_function("app_now", 0, app_now)
    for ddl in SCHEMA:
        connection.execute(ddl)
    return connection


def compile_all() -> dict[str, str | None]:
    """Compile every statement with EXPLAIN; name -> None (ok) or the error text."""
    results = {}
    connection = open_schema()
    for name, (sql, params, _reads_time) in STATEMENTS.items():
        try:
            connection.execute("EXPLAIN " + sql, params).fetchall()
            results[name] = None
        except sqlite3.Error as exc:
            results[name] = f"{type(exc).__name__}: {exc}"
    connection.close()
    return results


def main() -> int:
    print(f"python {sys.version.split()[0]}  sqlite {sqlite3.sqlite_version}")
    version_ok = sqlite_version_tuple() >= MIN_SQLITE
    print(f"minimum {'.'.join(map(str, MIN_SQLITE))}: {'OK' if version_ok else 'TOO OLD'}")
    failures = 0
    for name, error in compile_all().items():
        print(f"  {'OK  ' if error is None else 'FAIL'} {name}{'' if error is None else ': ' + error}")
        failures += error is not None
    return 0 if version_ok and failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
