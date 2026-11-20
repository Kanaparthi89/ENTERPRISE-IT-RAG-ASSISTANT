"""A2-2: session leases, the lease keeper, fencing and bounded shutdown (revision 8, §6–§8).

Store-level lease SQL is tested with a FakeClock. The keeper is driven one
cycle at a time with a ManualTrigger, and paused with Checkpoints. Shutdown
deadlines use the real monotonic clock with scaled-down budgets, plus one
test with the default 3 s + 2 s + 2 s budgets.
"""

import logging
import multiprocessing
import re
import sqlite3
import threading
import time

import pytest

from app.services import sessions
from app.services.sessions import (
    LEASE_LOST,
    LEASE_OK,
    LEASE_UNCONFIRMED,
    Lease,
    SessionBusy,
    SessionExpired,
    SessionLeaseManager,
    SessionStore,
    SessionStoreUnavailable,
)
from tests import session_sql
from tests.session_support import (
    AdvancingClock,
    Checkpoints,
    ConnectionSpy,
    ManualTrigger,
    TEST_OWNER,
    acquire_race_worker,
)
from tests.streaming_support import FakeClock

IDLE, ABSOLUTE = 1800.0, 28800.0
TTL, TURN_MAX, MARGIN = 30.0, 180.0, 2.0
RENEW_MARKER = "sessions.lease_keeper_gen = :gen"  # only in the renewal statement
SAVE_MARKER = "turn_count + 1"  # only in the save statement


def normalized(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip()


class FakeMonotonic:
    def __init__(self, start: float = 100.0):
        self.value = start

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def make_store(path, clock=None, **overrides):
    options = dict(
        idle_timeout_seconds=IDLE,
        absolute_timeout_seconds=ABSOLUTE,
        max_active_sessions=10**6,
        max_active_sessions_per_owner=10**6,  # caps are irrelevant here, as above (Phase B)
        busy_timeout_seconds=0.5,
        clock=clock or FakeClock(),
    )
    options.update(overrides)
    return SessionStore(path, **options)


def acquire(store, token, rid="r1", *, gen=1, inst="i1", history_turns=4):
    return store.acquire(token, rid, instance=inst, keeper_gen=gen, ttl=TTL,
                         turn_max=TURN_MAX, history_turns=history_turns, owner_ref=TEST_OWNER)


def save(store, lease, answer="answer", question="question", **overrides):
    options = dict(question=question, answer=answer, source_used="private_kb",
                   turn_max=TURN_MAX, max_turns=20, answer_chars=2000)
    options.update(overrides)
    return store.save_turn(lease, **options)


def row(store, session_id):
    connection = sqlite3.connect(store.path)
    connection.row_factory = sqlite3.Row
    try:
        return dict(connection.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone())
    finally:
        connection.close()


def turns(store, session_id):
    connection = sqlite3.connect(store.path)
    try:
        return connection.execute(
            "SELECT seq, epoch, request_id, question, answer FROM turns WHERE session_id = ? ORDER BY seq",
            (session_id,),
        ).fetchall()
    finally:
        connection.close()


@pytest.fixture
def store(tmp_path):
    return make_store(tmp_path / "sessions.db")


# ===================================================================
# Design conformance
# ===================================================================

@pytest.mark.parametrize(
    "store_sql, name",
    [
        (sessions.SQL_ACQUIRE, "acquire"),
        (sessions.SQL_RENEW, "renew"),
        (sessions.SQL_SAVE_UPDATE, "save_update"),
        (sessions.SQL_SAVE_INSERT_TURN, "save_insert_turn"),
        (sessions.SQL_SAVE_TRIM, "save_trim"),
        (sessions.SQL_RELEASE, "release"),
        (sessions.SQL_KEEPER_MOVE, "keeper_generation_move"),
    ],
)
def test_lease_sql_is_the_verified_design_sql(store_sql, name):
    assert normalized(store_sql) == normalized(session_sql.STATEMENTS[name][0])


def test_bulk_release_sql_is_the_verified_design_sql():
    two_rows = sessions.SQL_BULK_RELEASE_PREFIX + "(?, ?, ?), (?, ?, ?))"
    assert normalized(two_rows) == normalized(session_sql.STATEMENTS["bulk_release"][0])


def test_lease_settings_defaults_and_validation():
    from pydantic import ValidationError

    from app.core.config import Settings

    fields = Settings.model_fields
    assert fields["session_lease_ttl_seconds"].default == 30.0
    assert fields["session_turn_max_seconds"].default == 180.0
    assert fields["session_lease_margin_seconds"].default == 2.0
    with pytest.raises(ValidationError, match="session_turn_max_seconds must be at least"):
        Settings(_env_file=None, session_turn_max_seconds=120)  # < 120 + 15
    with pytest.raises(ValidationError, match="smaller than one renewal interval"):
        Settings(_env_file=None, session_lease_margin_seconds=10)  # 30 / 3 = 10


# ===================================================================
# Acquisition
# ===================================================================

def test_acquire_takes_the_lease(store):
    created = store.create(owner_ref=TEST_OWNER)

    lease = acquire(store, created.token)

    stored = row(store, created.session_id)
    assert (lease.session_id, lease.epoch, lease.request_id) == (created.session_id, 1, "r1")
    assert lease.expires_at == lease.started_at + TTL
    assert (stored["lease_owner"], stored["lease_instance"], stored["lease_keeper_gen"]) == ("r1", "i1", 1)
    assert stored["last_activity_at"] == lease.started_at


def test_acquire_returns_recent_history_oldest_first(store):
    created = store.create(owner_ref=TEST_OWNER)
    for number in range(1, 6):
        lease = acquire(store, created.token, rid=f"r{number}")
        assert save(store, lease, question=f"q{number}", answer=f"a{number}")

    lease = acquire(store, created.token, rid="r6", history_turns=3)

    assert [(t.seq, t.question, t.answer) for t in lease.history] == [
        (3, "q3", "a3"), (4, "q4", "a4"), (5, "q5", "a5")]
    assert acquire(store, store.create(owner_ref=TEST_OWNER).token, rid="x", history_turns=0).history == ()


def test_second_request_gets_busy_with_retry_after(store):
    created = store.create(owner_ref=TEST_OWNER)
    acquire(store, created.token, rid="r1")
    store._clock.advance(12)

    with pytest.raises(SessionBusy) as busy:
        acquire(store, created.token, rid="r2")

    assert busy.value.retry_after == pytest.approx(TTL - 12)


@pytest.mark.parametrize("case", ["unknown", "deleted", "idle_expired", "absolute_expired"])
def test_acquire_refuses_missing_or_expired_sessions(store, case):
    created = store.create(owner_ref=TEST_OWNER)
    token = created.token
    if case == "unknown":
        token = "never-issued"
    elif case == "deleted":
        store.delete(token, owner_ref=TEST_OWNER)
    elif case == "idle_expired":
        store._clock.advance(IDLE)
    else:
        store._clock.advance(ABSOLUTE)

    with pytest.raises(SessionExpired):
        acquire(store, token)


def test_acquire_without_a_database_creates_nothing(tmp_path):
    empty = make_store(tmp_path / "none.db")

    with pytest.raises(SessionExpired):
        acquire(empty, "anything")
    assert not empty.path.exists()


def test_expired_lease_can_be_taken_at_exactly_its_expiry(store):
    """No gap at the boundary: at t == lease_expires_at the old owner is invalid."""
    created = store.create(owner_ref=TEST_OWNER)
    first = acquire(store, created.token, rid="r1")
    store._clock.advance(TTL - 0.001)
    with pytest.raises(SessionBusy):
        acquire(store, created.token, rid="r2")

    store._clock.advance(0.001)  # exactly at lease_expires_at
    second = acquire(store, created.token, rid="r2")

    assert second.epoch == first.epoch + 1  # crash recovery: a new epoch


# ===================================================================
# Renewal boundaries and fencing
# ===================================================================

def renew(store, lease, gen=1):
    return store.renew_once(lease, keeper_gen=gen, ttl=TTL, turn_max=TURN_MAX, timeout=0.5)


def test_renewal_extends_and_keeps_the_session_from_idling(store):
    created = store.create(owner_ref=TEST_OWNER)
    lease = acquire(store, created.token)
    store._clock.advance(10)

    expires = renew(store, lease)

    stored = row(store, created.session_id)
    now = lease.started_at + 10
    assert expires == stored["lease_expires_at"] == now + TTL
    assert stored["last_activity_at"] == now
    assert stored["idle_expires_at"] == max(created.created_at + IDLE, now + IDLE)


def test_renewal_is_capped_at_turn_max(store):
    created = store.create(owner_ref=TEST_OWNER)
    lease = acquire(store, created.token)
    for _ in range(int((TURN_MAX - 1) // 20)):
        store._clock.advance(20)
        assert renew(store, lease) is not None
    store._clock.advance((lease.started_at + TURN_MAX - 0.001) - store._clock())

    assert renew(store, lease) == lease.started_at + TURN_MAX  # capped exactly, not now + TTL

    store._clock.advance(0.001)  # exactly at TURN_MAX
    assert renew(store, lease) is None
    store._clock.advance(5)
    assert renew(store, lease) is None


def test_an_expired_owner_cannot_revive_its_lease(store):
    created = store.create(owner_ref=TEST_OWNER)
    lease = acquire(store, created.token)
    store._clock.advance(TTL)  # expired, nobody else took it
    before = row(store, created.session_id)

    assert renew(store, lease) is None
    assert row(store, created.session_id) == before  # unchanged


@pytest.mark.parametrize("field", ["gen", "instance", "owner", "epoch"])
def test_renewal_is_fenced(store, field):
    created = store.create(owner_ref=TEST_OWNER)
    lease = acquire(store, created.token)
    store._clock.advance(5)
    wrong = {
        "gen": dict(gen=2),
        "instance": dict(lease=Lease(**{**lease.__dict__, "instance": "other"})),
        "owner": dict(lease=Lease(**{**lease.__dict__, "request_id": "other"})),
        "epoch": dict(lease=Lease(**{**lease.__dict__, "epoch": 99})),
    }[field]

    result = renew(store, wrong.get("lease", lease), gen=wrong.get("gen", 1))

    assert result is None
    assert row(store, created.session_id)["lease_expires_at"] == lease.expires_at


# ===================================================================
# Save, release, keeper move
# ===================================================================

def test_save_writes_the_turn_and_releases(store):
    created = store.create(owner_ref=TEST_OWNER)
    lease = acquire(store, created.token)
    store._clock.advance(3)

    assert save(store, lease, answer="x" * 5000, answer_chars=2000)

    stored = row(store, created.session_id)
    assert stored["lease_owner"] is None and stored["lease_expires_at"] is None
    assert stored["lease_keeper_gen"] is None and stored["turn_count"] == 1
    assert stored["idle_expires_at"] == lease.started_at + 3 + IDLE
    [(seq, epoch, rid, question, answer)] = turns(store, created.session_id)
    assert (seq, epoch, rid, question, len(answer)) == (1, 1, "r1", "question", 2000)


def test_save_keeps_only_the_newest_turns(store):
    created = store.create(owner_ref=TEST_OWNER)
    for number in range(1, 8):
        lease = acquire(store, created.token, rid=f"r{number}")
        assert save(store, lease, question=f"q{number}", max_turns=3)

    assert [seq for seq, *_ in turns(store, created.session_id)] == [5, 6, 7]


def test_save_boundaries_at_turn_max(store):
    created = store.create(owner_ref=TEST_OWNER)
    lease = acquire(store, created.token)
    for _ in range(8):
        store._clock.advance(20)
        renew(store, lease)
    store._clock.advance((lease.started_at + TURN_MAX - 0.001) - store._clock())
    renew(store, lease)  # capped at TURN_MAX

    assert save(store, lease) is True  # TURN_MAX - ε

    other = store.create(owner_ref=TEST_OWNER)
    lease = acquire(store, other.token, rid="r2")
    for _ in range(8):
        store._clock.advance(20)
        renew(store, lease)
    store._clock.advance((lease.started_at + TURN_MAX) - store._clock())

    assert save(store, lease) is False  # exactly TURN_MAX
    assert turns(store, other.session_id) == []


@pytest.mark.parametrize("case", ["expired", "taken_over", "deleted", "released"])
def test_save_refused_when_ownership_is_gone(store, case):
    created = store.create(owner_ref=TEST_OWNER)
    lease = acquire(store, created.token)
    if case == "expired":
        store._clock.advance(TTL)
    elif case == "taken_over":
        store._clock.advance(TTL)
        acquire(store, created.token, rid="r2")
    elif case == "deleted":
        store.delete(created.token, owner_ref=TEST_OWNER)
    else:
        store.release(lease)

    assert save(store, lease) is False
    if case != "deleted":
        assert turns(store, created.session_id) == []


def test_release_is_fenced_on_epoch_and_owner(store):
    created = store.create(owner_ref=TEST_OWNER)
    old = acquire(store, created.token, rid="r1")
    store._clock.advance(TTL)
    new = acquire(store, created.token, rid="r2")

    assert store.release(old) is False  # old epoch: harmless
    assert row(store, created.session_id)["lease_owner"] == "r2"
    assert store.release(new) is True
    assert row(store, created.session_id)["lease_owner"] is None


def test_keeper_move_is_fenced_and_refuses_expired_leases(store):
    created = store.create(owner_ref=TEST_OWNER)
    lease = acquire(store, created.token)

    assert store.move_generation(lease, new_gen=7) == lease.expires_at
    assert row(store, created.session_id)["lease_keeper_gen"] == 7
    assert renew(store, lease, gen=1) is None  # old generation no longer renews
    assert renew(store, lease, gen=7) is not None

    store._clock.advance(TTL + 1)
    assert store.move_generation(lease, new_gen=8) is None


def test_lost_reason_classification(store):
    created = store.create(owner_ref=TEST_OWNER)
    lease = acquire(store, created.token)
    assert store.lost_reason(lease, keeper_gen=2, turn_max=TURN_MAX) == "generation"
    store._clock.advance(TTL)
    assert store.lost_reason(lease, keeper_gen=1, turn_max=TURN_MAX) == "expired"
    acquire(store, created.token, rid="r2")
    assert store.lost_reason(lease, keeper_gen=1, turn_max=TURN_MAX) == "taken_over"
    store.delete(created.token, owner_ref=TEST_OWNER)
    assert store.lost_reason(lease, keeper_gen=1, turn_max=TURN_MAX) == "deleted"


def test_bulk_release_fences_each_row_and_chunks(store):
    tokens = [store.create(owner_ref=TEST_OWNER) for _ in range(650)]
    leases = [acquire(store, created.token, rid=f"r{index}") for index, created in enumerate(tokens)]
    keys = [(lease.session_id, lease.epoch, lease.request_id) for lease in leases]
    keys[0] = (keys[0][0], 99, keys[0][2])  # stale epoch: must not match
    keys[1] = (keys[1][0], keys[1][1], "someone-else")  # wrong owner

    result = store.bulk_release("i1", keys, timeout=0.5, within_deadline=lambda: True)

    assert (result.status, result.released) == ("committed", 648)
    assert row(store, leases[0].session_id)["lease_owner"] == "r0"
    assert row(store, leases[2].session_id)["lease_owner"] is None


def test_bulk_release_rolls_back_entirely_when_the_deadline_passes(store):
    leases = [acquire(store, store.create(owner_ref=TEST_OWNER).token, rid=f"r{i}") for i in range(400)]
    keys = [(lease.session_id, lease.epoch, lease.request_id) for lease in leases]
    checks = iter([True, False])  # first chunk runs, then the deadline has passed

    result = store.bulk_release("i1", keys, timeout=0.5, within_deadline=lambda: next(checks))

    assert (result.status, result.released) == ("rolled_back", 0)
    assert all(row(store, lease.session_id)["lease_owner"] == lease.request_id for lease in leases)


def test_lease_statements_read_the_clock_at_most_once(tmp_path):
    clock = AdvancingClock()
    spy = ConnectionSpy()
    store = make_store(tmp_path / "s.db", clock=clock, connect=spy)
    created = store.create(owner_ref=TEST_OWNER)
    counts = {}

    def measure(name, call):
        clock.reset_count()
        before = len(spy.executes)
        result = call()
        timed = [r.sql for r in spy.executes[before:] if "app_now()" in r.sql]
        counts[name] = (clock.calls, len(timed))
        return result

    lease = measure("acquire", lambda: acquire(store, created.token))  # look-up + acquire
    measure("renew", lambda: renew(store, lease))
    measure("move", lambda: store.move_generation(lease, new_gen=1))
    measure("save", lambda: save(store, lease))

    assert counts == {"acquire": (2, 2), "renew": (1, 1), "move": (1, 1), "save": (1, 1)}
    stored = row(store, created.session_id)
    assert turns(store, created.session_id)[0][0] == 1
    assert stored["last_activity_at"] == clock.values[-1]  # the save's single time value


# ===================================================================
# Concurrent acquisition
# ===================================================================

def test_two_threads_never_both_win_1000_times(tmp_path):
    store = make_store(tmp_path / "s.db", clock=time.time, busy_timeout_seconds=10.0)
    tokens = [store.create(owner_ref=TEST_OWNER).token for _ in range(1000)]
    wins = {"a": [], "b": []}
    barrier = threading.Barrier(2)

    def contender(name):
        barrier.wait()
        for index, token in enumerate(tokens):
            try:
                acquire(store, token, rid=f"{name}-{index}", inst=name)
                wins[name].append(index)
            except SessionBusy:
                pass

    threads = [threading.Thread(target=contender, args=(name,)) for name in wins]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(120)

    assert sorted(wins["a"] + wins["b"]) == list(range(1000))  # exactly one winner each


def test_two_processes_never_both_win_1000_times(tmp_path):
    store = make_store(tmp_path / "s.db", clock=time.time)
    tokens = [store.create(owner_ref=TEST_OWNER).token for _ in range(1000)]
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(target=acquire_race_worker, args=(str(store.path), tokens, name, barrier, results))
        for name in ("p1", "p2")
    ]
    for process in processes:
        process.start()
    collected = dict(results.get(timeout=120) for _ in processes)
    for process in processes:
        process.join(30)

    assert all(process.exitcode == 0 for process in processes)
    assert sorted(collected["p1"] + collected["p2"]) == list(range(1000))


# ===================================================================
# The keeper
# ===================================================================

class Harness:
    def __init__(self, tmp_path, *, clock=None, connect=sqlite3.connect, store_hooks=None,
                 ttl=TTL, margin=MARGIN, turn_max=TURN_MAX, **manager_options):
        self.clock = clock or FakeClock()
        self.store = make_store(tmp_path / "sessions.db", clock=self.clock, connect=connect,
                                hooks=store_hooks)
        self.trigger = ManualTrigger()
        self.points = Checkpoints()
        self.monotonic = manager_options.pop("monotonic", FakeMonotonic())
        options = dict(
            ttl=ttl, margin=margin, turn_max=turn_max, max_turns=20, history_turns=4,
            answer_chars=2000, shutdown_grace=0.2, keeper_join=0.3, release_budget=1.0,
            monotonic=self.monotonic,
            sleep=lambda seconds: self.clock.advance(seconds) if isinstance(self.clock, FakeClock) else time.sleep(seconds),
            wait=lambda timeout: self.trigger.wait(0.02),
            hooks=self.points,
            instance="i1",
        )
        options.update(manager_options)
        self.manager = SessionLeaseManager(self.store, **options)

    def cycle(self, timeout=5.0):
        before = self.manager.cycles
        self.trigger.fire()
        deadline = time.monotonic() + timeout
        while self.manager.cycles == before:
            if time.monotonic() > deadline:
                raise AssertionError("keeper cycle did not run")
            time.sleep(0.002)

    def close(self):
        for name in list(self.points._paused):
            self.points.release(name)
        self.manager.monotonic = time.monotonic
        self.manager._monotonic = time.monotonic
        self.manager.shutdown()


@pytest.fixture
def harness(tmp_path):
    created = []

    def build(**options):
        h = Harness(tmp_path, **options)
        created.append(h)
        return h

    yield build
    for h in created:
        h.close()
    deadline = time.monotonic() + 5
    while any(t.name.startswith("session-keeper") and t.is_alive() for t in threading.enumerate()):
        if time.monotonic() > deadline:
            raise AssertionError("keeper thread leaked")
        time.sleep(0.01)


def test_keeper_renews_registered_leases(harness):
    h = harness()
    created = h.store.create(owner_ref=TEST_OWNER)
    lease = h.manager.acquire(created.token, "r1", owner_ref=TEST_OWNER)
    h.clock.advance(10)

    h.cycle()

    assert h.manager.state_of("r1") == LEASE_OK
    assert h.manager.confirmed_until("r1") == lease.started_at + 10 + TTL


def test_long_turn_keeps_its_lease_and_session(harness):
    h = harness()
    created = h.store.create(owner_ref=TEST_OWNER)
    h.manager.acquire(created.token, "r1", owner_ref=TEST_OWNER)
    for _ in range(12):  # 120 s, four TTLs
        h.clock.advance(10)
        h.cycle()

    with pytest.raises(SessionBusy):
        h.manager.acquire(created.token, "r2", owner_ref=TEST_OWNER)
    assert h.store.lookup(created.token, owner_ref=TEST_OWNER).expired is False
    assert h.manager.state_of("r1") == LEASE_OK


def test_turn_max_ends_renewal_and_the_turn_is_not_saved(harness):
    spy = ConnectionSpy()
    h = harness(connect=spy)
    created = h.store.create(owner_ref=TEST_OWNER)
    lease = h.manager.acquire(created.token, "r1", owner_ref=TEST_OWNER)
    for _ in range(int(TURN_MAX // 10)):
        h.clock.advance(10)
        h.cycle()

    assert h.manager.state_of("r1") == LEASE_LOST
    before = len(spy.executes)
    assert h.manager.finish(lease, ("q", "a", "private_kb")) is False
    assert not any(SAVE_MARKER in r.sql for r in spy.executes[before:])  # no save attempted
    assert row(h.store, created.session_id)["lease_owner"] is None  # released
    assert h.manager.active_count() == 0


def test_temporary_lock_errors_are_retried_within_the_window(harness):
    spy = ConnectionSpy()
    h = harness(connect=spy)
    created = h.store.create(owner_ref=TEST_OWNER)
    lease = h.manager.acquire(created.token, "r1", owner_ref=TEST_OWNER)
    h.clock.advance(10)
    spy.fail_next(3, matching=lambda sql: RENEW_MARKER in sql)

    h.cycle()

    assert h.manager.state_of("r1") == LEASE_OK
    assert h.manager.confirmed_until("r1") > lease.expires_at
    assert h.manager.renewal_failures == 3


def test_errors_until_the_window_ends_lose_the_lease_and_skip_the_save(harness, caplog):
    spy = ConnectionSpy()
    h = harness(connect=spy)
    created = h.store.create(owner_ref=TEST_OWNER)
    lease = h.manager.acquire(created.token, "r1", owner_ref=TEST_OWNER)
    h.clock.advance(10)
    spy.fail_next(10**6, matching=lambda sql: RENEW_MARKER in sql)

    with caplog.at_level(logging.WARNING, logger="app.services.sessions"):
        h.cycle()
        h.cycle()  # stays lost; no repeated warning

    assert h.manager.state_of("r1") == LEASE_LOST
    assert h.manager.confirmed_until("r1") == lease.expires_at  # never moved on a failure
    states = [r.getMessage() for r in caplog.records if "session_lease_" in r.getMessage()]
    assert [m.split()[0] for m in states] == ["session_lease_unconfirmed", "session_lease_lost"]
    before = len(spy.executes)
    assert h.manager.finish(lease, ("q", "a", None)) is False
    assert not any(SAVE_MARKER in r.sql for r in spy.executes[before:])


@pytest.mark.parametrize("hold, expected", [(0.4, LEASE_OK), (3.5, LEASE_LOST)])
def test_real_lock_contention_shorter_and_longer_than_the_window(tmp_path, hold, expected):
    """Real clock: TTL 3 s, margin 0.2 s, so the window is about 2.8 s."""
    h = Harness(tmp_path, clock=time.time, ttl=3.0, margin=0.2, turn_max=600.0)
    try:
        created = h.store.create(owner_ref=TEST_OWNER)
        h.manager.acquire(created.token, "r1", owner_ref=TEST_OWNER)
        # check_same_thread=False: the timer thread must be able to end the lock.
        blocker = sqlite3.connect(h.store.path, isolation_level=None, check_same_thread=False)
        blocker.execute("BEGIN EXCLUSIVE")
        released = threading.Event()

        def end_lock():
            blocker.execute("ROLLBACK")
            blocker.close()
            released.set()

        releaser = threading.Timer(hold, end_lock)
        releaser.start()
        h.cycle(timeout=10)
        releaser.join(10)
        assert released.is_set()  # the lock really was released by the timer thread
        assert h.manager.state_of("r1") == expected
    finally:
        h.close()


def test_an_error_in_one_lease_does_not_stop_the_others(harness, monkeypatch):
    h = harness()
    leases = [h.manager.acquire(h.store.create(owner_ref=TEST_OWNER).token, f"r{i}", owner_ref=TEST_OWNER) for i in range(3)]
    h.clock.advance(10)
    real = h.store.renew_once

    def flaky(lease, **kwargs):
        if lease.request_id == "r1":
            raise RuntimeError("unexpected bug")
        return real(lease, **kwargs)

    monkeypatch.setattr(h.store, "renew_once", flaky)
    h.cycle()

    assert [h.manager.state_of(f"r{i}") for i in range(3)] == [LEASE_OK, LEASE_UNCONFIRMED, LEASE_OK]
    assert h.manager.keeper_thread().is_alive()

    monkeypatch.setattr(h.store, "renew_once", real)
    h.cycle()
    assert h.manager.state_of("r1") == LEASE_OK
    assert leases  # all still registered


def test_stuck_keeper_is_restarted_and_its_late_renewal_is_fenced(harness):
    h = harness()
    created = h.store.create(owner_ref=TEST_OWNER)
    lease = h.manager.acquire(created.token, "r1", owner_ref=TEST_OWNER)
    old_keeper = h.manager.keeper_thread()
    h.points.pause("keeper:before_lease")
    h.trigger.fire()
    h.points.wait_reached("keeper:before_lease")  # old keeper stuck mid-cycle
    h.clock.advance(10)
    h.monotonic.advance(3 * h.manager.interval)  # no cycle for longer than 2 x interval

    h.manager.acquire(h.store.create(owner_ref=TEST_OWNER).token, "r2", owner_ref=TEST_OWNER)  # the restart check runs here

    assert h.manager.restarts == 1 and h.manager.generation == 2
    assert row(h.store, created.session_id)["lease_keeper_gen"] == 2
    confirmed = h.manager.confirmed_until("r1")
    before = row(h.store, created.session_id)

    h.points.release("keeper:before_lease")  # the stale keeper resumes its renewal
    old_keeper.join(5)

    assert not old_keeper.is_alive()
    assert row(h.store, created.session_id) == before  # its renewal matched no row
    assert h.manager.state_of("r1") == LEASE_OK  # and its result was not applied
    assert h.manager.confirmed_until("r1") == confirmed
    h.cycle()  # the new keeper renews normally
    assert h.manager.confirmed_until("r1") > confirmed
    assert lease


def test_dead_keeper_is_restarted(harness):
    h = harness()
    h.manager.acquire(h.store.create(owner_ref=TEST_OWNER).token, "r1", owner_ref=TEST_OWNER)
    keeper = h.manager.keeper_thread()
    h.manager._stop.set()  # simulate the thread ending
    keeper.join(5)

    h.manager.acquire(h.store.create(owner_ref=TEST_OWNER).token, "r2", owner_ref=TEST_OWNER)

    assert h.manager.restarts == 1
    assert h.manager.keeper_thread() is not keeper and h.manager.keeper_thread().is_alive()


def test_finish_without_a_turn_releases(harness):
    h = harness()
    created = h.store.create(owner_ref=TEST_OWNER)
    lease = h.manager.acquire(created.token, "r1", owner_ref=TEST_OWNER)

    assert h.manager.finish(lease) is None
    assert row(h.store, created.session_id)["lease_owner"] is None
    assert h.manager.active_count() == 0


def test_finish_saves_when_owned_and_reports_false_after_delete(harness):
    h = harness()
    created = h.store.create(owner_ref=TEST_OWNER)
    lease = h.manager.acquire(created.token, "r1", owner_ref=TEST_OWNER)
    assert h.manager.finish(lease, ("q1", "a1", "private_kb")) is True
    assert [t[3] for t in turns(h.store, created.session_id)] == ["q1"]

    lease = h.manager.acquire(created.token, "r2", owner_ref=TEST_OWNER)
    h.store.delete(created.token, owner_ref=TEST_OWNER)  # DELETE mid-turn
    assert h.manager.finish(lease, ("q2", "a2", None)) is False


def test_finish_reports_false_when_the_save_fails(harness, monkeypatch):
    h = harness()
    created = h.store.create(owner_ref=TEST_OWNER)
    lease = h.manager.acquire(created.token, "r1", owner_ref=TEST_OWNER)

    def broken(*args, **kwargs):
        raise SessionStoreUnavailable("down")

    monkeypatch.setattr(h.store, "save_turn", broken)

    assert h.manager.finish(lease, ("q", "a", None)) is False
    assert row(h.store, created.session_id)["lease_owner"] is None  # best-effort release
    assert h.manager.active_count() == 0


# ===================================================================
# Shutdown
# ===================================================================

def real_harness(tmp_path, **options):
    """Real monotonic clock so shutdown deadlines are wall-clock bounded."""
    defaults = dict(monotonic=time.monotonic)
    defaults.update(options)
    return Harness(tmp_path, **defaults)


def register_many(h, count):
    """Create ``count`` leased sessions quickly and register them with the manager."""
    connection = sqlite3.connect(h.store.path)
    now = h.clock()
    rows = [
        (f"S{i}", f"h{i}", now, now, now + IDLE, now + ABSOLUTE, 1, f"r{i}", "i1",
         h.manager.generation, now, now + TTL)
        for i in range(count)
    ]
    connection.executemany(
        """INSERT INTO sessions (id, token_hash, created_at, last_activity_at, idle_expires_at,
               absolute_expires_at, lease_epoch, lease_owner, lease_instance, lease_keeper_gen,
               lease_started_at, lease_expires_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    connection.commit()
    connection.close()
    with h.manager._lock:
        for i in range(count):
            lease = Lease(f"S{i}", f"r{i}", 1, "i1", h.manager.generation, now, now + TTL)
            h.manager._entries[f"r{i}"] = sessions._Entry(lease=lease, confirmed_until=now + TTL)


def leased_count(h):
    connection = sqlite3.connect(h.store.path)
    try:
        return connection.execute("SELECT count(*) FROM sessions WHERE lease_owner IS NOT NULL").fetchone()[0]
    finally:
        connection.close()


def test_shutdown_refuses_new_leases_stops_cycles_and_releases(tmp_path):
    h = real_harness(tmp_path)
    created = h.store.create(owner_ref=TEST_OWNER)
    h.manager.acquire(created.token, "r1", owner_ref=TEST_OWNER)
    h.cycle()
    cycles = h.manager.cycles

    report = h.manager.shutdown()

    with pytest.raises(SessionStoreUnavailable):
        h.manager.acquire(h.store.create(owner_ref=TEST_OWNER).token, "r2", owner_ref=TEST_OWNER)
    h.trigger.fire()
    time.sleep(0.1)
    assert h.manager.cycles == cycles  # no cycle after the fence
    assert (report.release_status, report.released, report.left_to_expire) == ("committed", 1, 0)
    assert row(h.store, created.session_id)["lease_owner"] is None
    assert not h.manager.keeper_thread().is_alive()


def test_shutdown_grace_ends_early_when_turns_finish(tmp_path):
    h = real_harness(tmp_path, shutdown_grace=2.0)
    lease = h.manager.acquire(h.store.create(owner_ref=TEST_OWNER).token, "r1", owner_ref=TEST_OWNER)
    threading.Timer(0.2, lambda: h.manager.finish(lease, ("q", "a", None))).start()

    report = h.manager.shutdown()

    assert report.elapsed < 1.0
    assert (report.release_status, report.released) == ("none", 0)


@pytest.mark.parametrize("order", ["renewal_first", "release_first"])
def test_renewal_and_bulk_release_race_never_revives(tmp_path, order):
    """500 runs per order; the renewal pauses after opening its connection, before its UPDATE."""
    points = Checkpoints()
    clock = FakeClock()
    store = make_store(tmp_path / "race.db", clock=clock, hooks=points, busy_timeout_seconds=5.0)
    for run in range(500):
        created = store.create(owner_ref=TEST_OWNER)
        lease = acquire(store, created.token, rid=f"r{run}")
        key = [(lease.session_id, lease.epoch, lease.request_id)]
        result = {}
        if order == "release_first":
            points.pause("renew:before_update")
        renewer = threading.Thread(target=lambda: result.setdefault("renew", renew(store, lease)))
        renewer.start()
        if order == "release_first":
            points.wait_reached("renew:before_update")
            store.bulk_release("i1", key, timeout=5.0, within_deadline=lambda: True)
            points.release("renew:before_update")
            renewer.join(5)
            assert result["renew"] is None
        else:
            renewer.join(5)
            store.bulk_release("i1", key, timeout=5.0, within_deadline=lambda: True)
            assert result["renew"] is not None
        stored = row(store, lease.session_id)
        assert stored["lease_owner"] is None and stored["lease_expires_at"] is None


def test_stale_keeper_result_is_discarded_after_the_fence(tmp_path):
    h = real_harness(tmp_path)
    created = h.store.create(owner_ref=TEST_OWNER)
    h.manager.acquire(created.token, "r1", owner_ref=TEST_OWNER)
    h.points.pause("keeper:before_lease")
    h.trigger.fire()
    h.points.wait_reached("keeper:before_lease")
    shutdown = threading.Thread(target=h.manager.shutdown)
    shutdown.start()
    deadline = time.monotonic() + 5
    while not h.manager._closed and time.monotonic() < deadline:
        time.sleep(0.01)
    state = h.manager.state_of("r1")

    h.points.release("keeper:before_lease")  # stale keeper continues
    shutdown.join(10)

    assert h.manager.state_of("r1") in (state, None)  # never changed by the stale keeper
    assert row(h.store, created.session_id)["lease_owner"] is None


@pytest.mark.parametrize("count", [1000, 5000])
def test_shutdown_with_many_leases_stays_within_its_bound(tmp_path, count):
    h = real_harness(tmp_path, shutdown_grace=0.2, keeper_join=0.3, release_budget=2.0)
    h.manager.acquire(h.store.create(owner_ref=TEST_OWNER).token, "starter", owner_ref=TEST_OWNER)  # starts the keeper
    register_many(h, count)

    report = h.manager.shutdown()

    assert report.elapsed <= 0.2 + 0.3 + 2.0 + 0.5
    assert (report.release_status, report.released) == ("committed", count + 1)
    assert leased_count(h) == 0


def test_default_budgets_bound_shutdown_at_seven_seconds(tmp_path):
    h = real_harness(tmp_path, shutdown_grace=3.0, keeper_join=2.0, release_budget=2.0)
    h.manager.acquire(h.store.create(owner_ref=TEST_OWNER).token, "starter", owner_ref=TEST_OWNER)
    register_many(h, 1000)

    report = h.manager.shutdown()

    assert report.elapsed <= 7.0 + 0.5
    assert report.release_status == "committed"


def test_locked_database_rolls_back_and_leases_expire(tmp_path):
    h = real_harness(tmp_path, shutdown_grace=0.2, keeper_join=0.3, release_budget=0.5)
    created = h.store.create(owner_ref=TEST_OWNER)
    h.manager.acquire(created.token, "r1", owner_ref=TEST_OWNER)
    blocker = sqlite3.connect(h.store.path, isolation_level=None)
    blocker.execute("BEGIN EXCLUSIVE")
    try:
        report = h.manager.shutdown()
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()

    assert report.elapsed <= 0.2 + 0.3 + 0.5 + 0.5
    assert report.release_status == "rolled_back" and report.left_to_expire == 1
    assert row(h.store, created.session_id)["lease_owner"] == "r1"  # untouched
    h.clock.advance(TTL)
    assert h.store.lookup(created.token, owner_ref=TEST_OWNER).live_lease is False  # recovered by TTL expiry


def test_stuck_keeper_join_times_out_within_the_bound(tmp_path):
    h = real_harness(tmp_path, shutdown_grace=0.2, keeper_join=0.3, release_budget=0.5)
    created = h.store.create(owner_ref=TEST_OWNER)
    h.manager.acquire(created.token, "r1", owner_ref=TEST_OWNER)
    h.points.pause("keeper:before_lease")
    h.trigger.fire()
    h.points.wait_reached("keeper:before_lease")

    report = h.manager.shutdown()

    assert report.keeper_join_timed_out is True
    assert report.elapsed <= 0.2 + 0.3 + 0.5 + 0.5
    assert report.release_status == "committed"
    h.points.release("keeper:before_lease")
    h.manager.keeper_thread().join(5)
    assert row(h.store, created.session_id)["lease_owner"] is None  # stale renewal didn't revive


def test_stalled_sqlite_call_does_not_hold_the_hook(tmp_path, caplog):
    spy = ConnectionSpy()
    h = real_harness(tmp_path, connect=spy, shutdown_grace=0.2, keeper_join=0.3, release_budget=0.5)
    leases = [h.manager.acquire(h.store.create(owner_ref=TEST_OWNER).token, f"r{i}", owner_ref=TEST_OWNER) for i in range(5)]
    spy.stall(matching=lambda sql: "IN (VALUES" in sql)

    with caplog.at_level(logging.ERROR, logger="app.services.sessions"):
        report = h.manager.shutdown()

    assert report.elapsed <= 0.2 + 0.3 + 0.5 + 0.5
    assert (report.release_status, report.undetermined) == ("unfinished", 5)
    assert spy.interrupts == 1
    assert any("session_release_unfinished" in r.getMessage() for r in caplog.records)

    spy.release_stall()  # the storage stall ends; the helper finishes on its own
    deadline = time.monotonic() + 5
    while any(t.name == "session-release" and t.is_alive() for t in threading.enumerate()):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    owners = [row(h.store, lease.session_id)["lease_owner"] for lease in leases]
    assert owners == [None] * 5 or owners == [lease.request_id for lease in leases]  # all or nothing


def test_late_renewal_after_a_failed_release_extends_at_most_once(tmp_path):
    points = Checkpoints()
    h = real_harness(tmp_path, store_hooks=points, shutdown_grace=0.2, keeper_join=0.3, release_budget=0.5)
    created = h.store.create(owner_ref=TEST_OWNER)
    h.manager.acquire(created.token, "r1", owner_ref=TEST_OWNER)
    h.clock.advance(10)
    points.pause("renew:before_update")
    h.trigger.fire()
    points.wait_reached("renew:before_update")  # keeper's renewal is in flight
    blocker = sqlite3.connect(h.store.path, isolation_level=None)
    blocker.execute("BEGIN EXCLUSIVE")
    report = h.manager.shutdown()
    blocker.execute("ROLLBACK")
    blocker.close()
    assert report.release_status == "rolled_back"

    points.release("renew:before_update")  # the in-flight renewal lands once
    h.manager.keeper_thread().join(5)
    expires = row(h.store, created.session_id)["lease_expires_at"]
    h.trigger.fire()
    time.sleep(0.1)

    assert not h.manager.keeper_thread().is_alive()
    assert row(h.store, created.session_id)["lease_expires_at"] == expires  # no further renewal
    h.clock.advance(TTL)
    assert h.store.lookup(created.token, owner_ref=TEST_OWNER).live_lease is False


def test_a_request_still_running_after_shutdown_cannot_save(tmp_path):
    h = real_harness(tmp_path)
    created = h.store.create(owner_ref=TEST_OWNER)
    lease = h.manager.acquire(created.token, "r1", owner_ref=TEST_OWNER)

    h.manager.shutdown()

    assert h.manager.finish(lease, ("q", "a", None)) is False
    assert turns(h.store, created.session_id) == []


def test_shutdown_without_sessions_creates_nothing(tmp_path):
    h = real_harness(tmp_path)

    report = h.manager.shutdown()

    assert report.release_status == "none"
    assert not h.store.path.exists()


# ---------------------------------------------------------------------------
# H2: no new turn save after the shutdown fence
# ---------------------------------------------------------------------------

BOUND = 0.2 + 0.3 + 0.5 + 0.5  # grace + keeper join + release budget + slack
JOIN = 5.0  # bounded test waits


def run_in_thread(target):
    result = {}

    def run():
        try:
            result["value"] = target()
        except BaseException as exc:  # noqa: BLE001 - re-raised by the test
            result["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, result


def test_finish_after_a_rolled_back_release_is_refused(tmp_path):
    """The database still shows the lease, but the fence refuses the save."""
    h = real_harness(tmp_path, shutdown_grace=0.2, keeper_join=0.3, release_budget=0.5)
    created = h.store.create(owner_ref=TEST_OWNER)
    lease = h.manager.acquire(created.token, "r1", owner_ref=TEST_OWNER)
    blocker = sqlite3.connect(h.store.path, isolation_level=None)
    blocker.execute("BEGIN EXCLUSIVE")
    try:
        report = h.manager.shutdown()
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    assert report.release_status == "rolled_back"
    assert row(h.store, created.session_id)["lease_owner"] == "r1"  # still leased in the database

    assert h.manager.finish(lease, ("q", "a", None)) is False

    assert turns(h.store, created.session_id) == []
    assert row(h.store, created.session_id)["lease_owner"] is None  # released, not saved


def test_a_save_holding_the_gate_first_completes_atomically(tmp_path):
    store_points = Checkpoints()
    h = real_harness(tmp_path, store_hooks=store_points,
                     shutdown_grace=0.2, keeper_join=0.3, release_budget=0.5)
    created = h.store.create(owner_ref=TEST_OWNER)
    lease = h.manager.acquire(created.token, "r1", owner_ref=TEST_OWNER)
    store_points.pause("save:before_commit")

    saver, saved = run_in_thread(lambda: h.manager.finish(lease, ("q", "a", None)))
    store_points.wait_reached("save:before_commit")  # inside the save transaction, gate held

    started = time.monotonic()
    report = h.manager.shutdown()  # can't take the gate; bounded all the same
    assert time.monotonic() - started <= BOUND
    assert report.release_status in ("rolled_back", "unfinished")  # the save holds the write lock

    store_points.release("save:before_commit")
    saver.join(JOIN)
    assert not saver.is_alive()
    assert saved == {"value": True}  # the transaction already in progress committed
    assert [turn[3] for turn in turns(h.store, created.session_id)] == ["q"]
    assert row(h.store, created.session_id)["lease_owner"] is None


def test_a_save_after_the_fence_took_the_gate_is_refused(tmp_path):
    h = real_harness(tmp_path, shutdown_grace=0.2, keeper_join=0.3, release_budget=0.5)
    created = h.store.create(owner_ref=TEST_OWNER)
    lease = h.manager.acquire(created.token, "r1", owner_ref=TEST_OWNER)
    h.points.pause("shutdown:fenced")

    stopper, report = run_in_thread(h.manager.shutdown)
    h.points.wait_reached("shutdown:fenced")  # closed, gate held

    saver, saved = run_in_thread(lambda: h.manager.finish(lease, ("q", "a", None)))
    time.sleep(0.1)
    assert saver.is_alive()  # waiting for the gate, not saving

    h.points.release("shutdown:fenced")
    saver.join(JOIN)
    stopper.join(JOIN)
    assert not saver.is_alive() and not stopper.is_alive()
    assert saved == {"value": False}
    assert turns(h.store, created.session_id) == []
    assert row(h.store, created.session_id)["lease_owner"] is None
    assert report["value"].elapsed <= BOUND


def test_a_save_waiting_for_the_gate_at_the_fence_is_refused(tmp_path, caplog):
    """The closed check happens under the gate, not before it.

    Another save holds the gate (the test stands in for it), so shutdown
    fences without it; the database is locked, so the bulk release rolls back
    and the row still shows the lease. A save that was waiting for the gate
    must see the manager closed once it gets the gate. The busy timeout (the
    bound on the gate wait, U3) is raised so the save is still waiting then.
    """
    h = real_harness(tmp_path, shutdown_grace=0.2, keeper_join=0.3, release_budget=0.5)
    h.store.busy_timeout_seconds = 3.0
    created = h.store.create(owner_ref=TEST_OWNER)
    lease = h.manager.acquire(created.token, "r1", owner_ref=TEST_OWNER)
    h.manager._save_gate.acquire()  # another save in progress
    try:
        saver, saved = run_in_thread(lambda: h.manager.finish(lease, ("q", "a", None)))
        time.sleep(0.1)
        assert saver.is_alive()  # waiting for the gate

        blocker = sqlite3.connect(h.store.path, isolation_level=None)
        blocker.execute("BEGIN EXCLUSIVE")
        try:
            report = h.manager.shutdown()  # fences without the gate (bounded)
        finally:
            blocker.execute("ROLLBACK")
            blocker.close()
        assert report.release_status == "rolled_back"
        assert row(h.store, created.session_id)["lease_owner"] == "r1"
    finally:
        h.manager._save_gate.release()

    saver.join(JOIN)
    assert not saver.is_alive()
    assert saved == {"value": False}
    assert turns(h.store, created.session_id) == []
    assert f"id={created.session_id} reason=manager_closed" in caplog.text  # not a gate timeout


@pytest.fixture
def gated(tmp_path):
    """A real harness whose store pauses at ``save:*`` points; two leased sessions.

    Teardown releases any pause point still held and shuts the manager down,
    so the lease keeper never outlives the test.
    """
    store_points = Checkpoints()
    h = real_harness(tmp_path, store_hooks=store_points, shutdown_grace=0.2,
                     keeper_join=0.3, release_budget=0.5)
    first, second = h.store.create(owner_ref=TEST_OWNER), h.store.create(owner_ref=TEST_OWNER)
    leases = h.manager.acquire(first.token, "r1", owner_ref=TEST_OWNER), h.manager.acquire(second.token, "r2", owner_ref=TEST_OWNER)
    yield h, store_points, (first, second), leases
    for name in list(store_points._paused):
        store_points.release(name)
    h.manager.shutdown()


def test_a_save_that_cannot_get_the_gate_in_time_is_not_saved(gated, caplog):
    """U3: the gate wait is bounded by the busy timeout; the lease is released."""
    h, points, (first, second), (lease_1, lease_2) = gated
    points.pause("save:before_begin")  # the first save holds the gate, not the write lock
    saver_1, saved_1 = run_in_thread(lambda: h.manager.finish(lease_1, ("q1", "a1", None)))
    points.wait_reached("save:before_begin")

    started = time.monotonic()
    saver_2, saved_2 = run_in_thread(lambda: h.manager.finish(lease_2, ("q2", "a2", None)))
    saver_2.join(JOIN)
    elapsed = time.monotonic() - started

    assert not saver_2.is_alive()
    assert saved_2 == {"value": False}
    busy = h.store.busy_timeout_seconds
    assert busy - 0.05 <= elapsed < busy + 1.0  # waited for the gate, but only that long
    assert turns(h.store, second.session_id) == []
    assert row(h.store, second.session_id)["lease_owner"] is None  # released, not left active
    assert f"id={second.session_id} reason=save_gate_timeout" in caplog.text

    points.release("save:before_begin")
    saver_1.join(JOIN)
    assert saved_1 == {"value": True}  # the stalled save itself still completes
    assert [turn[3] for turn in turns(h.store, first.session_id)] == ["q1"]


def test_a_gate_timeout_while_the_holder_has_the_write_lock_leaves_the_lease_to_expire(gated):
    """U3 limit: the release needs the write lock too, so the lease expires instead."""
    h, points, (first, second), (lease_1, lease_2) = gated
    points.pause("save:before_commit")  # the first save holds the gate and the write lock
    saver_1, saved_1 = run_in_thread(lambda: h.manager.finish(lease_1, ("q1", "a1", None)))
    points.wait_reached("save:before_commit")

    saver_2, saved_2 = run_in_thread(lambda: h.manager.finish(lease_2, ("q2", "a2", None)))
    saver_2.join(JOIN)
    assert not saver_2.is_alive() and saved_2 == {"value": False}
    assert turns(h.store, second.session_id) == []
    assert row(h.store, second.session_id)["lease_owner"] == "r2"  # release couldn't get the lock

    points.release("save:before_commit")
    saver_1.join(JOIN)
    assert saved_1 == {"value": True}
    h.clock.advance(TTL)
    assert h.store.lookup(second.token, owner_ref=TEST_OWNER).live_lease is False  # recovered by TTL expiry


def test_bulk_release_winning_the_write_lock_makes_an_admitted_save_fail(gated, caplog):
    """U1, lock order 2: the save holds the gate but the bulk release commits first."""
    h, points, (first, second), (lease_1, lease_2) = gated
    points.pause("save:before_begin")  # admitted (gate held), not yet in a transaction
    saver_1, saved_1 = run_in_thread(lambda: h.manager.finish(lease_1, ("q1", "a1", None)))
    points.wait_reached("save:before_begin")

    started = time.monotonic()
    stopper, report = run_in_thread(h.manager.shutdown)
    stopper.join(JOIN)
    assert not stopper.is_alive()
    assert time.monotonic() - started <= BOUND  # the gate-wait fallback stays bounded
    assert report["value"].release_status == "committed"  # the bulk release got the write lock
    assert row(h.store, first.session_id)["lease_owner"] is None

    points.release("save:before_begin")
    saver_1.join(JOIN)
    assert saved_1 == {"value": False}  # failed its lease check
    assert turns(h.store, first.session_id) == []

    # The fallback set _closed without the gate; a new save is refused by the fence.
    assert h.manager.finish(lease_2, ("q2", "a2", None)) is False
    assert turns(h.store, second.session_id) == []
    assert f"id={second.session_id} reason=manager_closed" in caplog.text


def test_save_winning_the_write_lock_commits_after_the_bounded_fallback(gated):
    """U1, lock order 1: the admitted save already holds the write lock."""
    h, points, (first, second), (lease_1, lease_2) = gated
    points.pause("save:before_commit")  # gate and write lock held, before COMMIT
    saver_1, saved_1 = run_in_thread(lambda: h.manager.finish(lease_1, ("q1", "a1", None)))
    points.wait_reached("save:before_commit")

    started = time.monotonic()
    stopper, report = run_in_thread(h.manager.shutdown)
    stopper.join(JOIN)
    assert not stopper.is_alive()
    assert time.monotonic() - started <= BOUND
    assert report["value"].release_status in ("rolled_back", "unfinished")  # lost the write lock

    points.release("save:before_commit")
    saver_1.join(JOIN)
    assert saved_1 == {"value": True}
    assert [turn[3] for turn in turns(h.store, first.session_id)] == ["q1"]
    assert row(h.store, first.session_id)["lease_owner"] is None  # released by the save itself

    assert h.manager.finish(lease_2, ("q2", "a2", None)) is False  # new saves refused
    assert turns(h.store, second.session_id) == []


def test_a_turn_finishing_within_the_grace_is_still_saved(tmp_path):
    h = real_harness(tmp_path, shutdown_grace=1.0, keeper_join=0.3, release_budget=0.5)
    created = h.store.create(owner_ref=TEST_OWNER)
    lease = h.manager.acquire(created.token, "r1", owner_ref=TEST_OWNER)

    stopper, report = run_in_thread(h.manager.shutdown)
    time.sleep(0.05)  # shutdown is waiting for in-flight turns

    assert h.manager.finish(lease, ("q", "a", None)) is True
    stopper.join(JOIN)
    assert [turn[3] for turn in turns(h.store, created.session_id)] == ["q"]
    assert report["value"].release_status == "none"  # nothing left to release
    assert report["value"].elapsed < 1.0  # the grace ended early


def test_default_keeper_loop_renews_on_its_own(tmp_path):
    """No injected trigger: the keeper's own timed loop renews every TTL/3."""
    store = make_store(tmp_path / "sessions.db", clock=time.time)
    manager = SessionLeaseManager(
        store, ttl=0.6, margin=0.05, turn_max=60.0, max_turns=20, history_turns=4,
        answer_chars=2000, shutdown_grace=0.1, keeper_join=0.5, release_budget=0.5,
    )
    try:
        created = store.create(owner_ref=TEST_OWNER)
        lease = manager.acquire(created.token, "r1", owner_ref=TEST_OWNER)
        time.sleep(1.5)  # 2.5 TTLs: without renewal the lease would have expired

        assert manager.cycles >= 4
        assert manager.state_of("r1") == LEASE_OK
        assert manager.confirmed_until("r1") > lease.expires_at
        assert store.lookup(created.token, owner_ref=TEST_OWNER).live_lease is True
        assert manager.finish(lease, ("q", "a", None)) is True
    finally:
        report = manager.shutdown()
    assert not manager.keeper_thread().is_alive()
    assert report.elapsed < 1.5
