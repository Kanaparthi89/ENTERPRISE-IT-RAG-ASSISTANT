"""Security S1-4a: the standalone quota engine (no routes, no network).

Every test builds its own engine with controlled clocks, so nothing is
shared between tests and no global state needs resetting.
"""

import inspect
import logging
import math
import sys
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from app.services import quota
from app.services.quota import (
    ANONYMOUS,
    BUSY,
    CAPACITY,
    CHAT,
    DAILY_LIMIT,
    HOURLY_LIMIT,
    MINUTES_LIMIT,
    REQUEST_TOO_LONG,
    TRANSCRIPTION,
    UNAVAILABLE,
    ChatLimits,
    QuotaConfig,
    QuotaConfigError,
    QuotaEngine,
    Refusal,
    Ticket,
    TranscriptionLimits,
)

NOON = datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc).timestamp()
MIDNIGHT = datetime(2026, 11, 20, 0, 0, tzinfo=timezone.utc).timestamp()


class Clock:
    """Separately controllable monotonic and wall clocks."""

    def __init__(self, wall=NOON, mono=10_000.0):
        self.wall = wall
        self.mono = mono

    def advance(self, seconds):
        self.wall += seconds
        self.mono += seconds


@pytest.fixture
def clock():
    return Clock()


def engine_for(clock, config=None):
    return QuotaEngine(config, monotonic=lambda: clock.mono, wall=lambda: clock.wall)


@pytest.fixture
def engine(clock):
    return engine_for(clock)


def chat(engine, identity="alice"):
    """Admit a chat and release it at once (a finished request)."""
    result = engine.admit(CHAT, identity)
    if isinstance(result, Ticket):
        result.release()
    return result


def admitted(result):
    return isinstance(result, Ticket)


def snapshot(engine, identities=("alice", "bob")):
    return engine.usage(), {name: engine.usage(name) for name in identities}, engine.tracked_identities()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_defaults_are_the_approved_values():
    config = QuotaConfig()
    assert config.chat == ChatLimits(
        per_code_hourly=10, per_code_daily=25, global_hourly=60, global_daily=150,
        per_code_concurrent=1, global_concurrent=2,
    )
    assert config.transcription == TranscriptionLimits(
        per_code_daily=10, per_code_daily_seconds=300, global_daily_seconds=1800,
        per_code_concurrent=1, global_concurrent=1, max_request_seconds=60, max_settle_seconds=300,
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda: ChatLimits(per_code_hourly=0),
        lambda: ChatLimits(per_code_hourly=-1),
        lambda: ChatLimits(per_code_hourly=True),
        lambda: ChatLimits(per_code_hourly=1.5),
        lambda: ChatLimits(per_code_hourly="10"),
        lambda: ChatLimits(global_daily=100_001),
        lambda: ChatLimits(per_code_hourly=30, per_code_daily=25),  # hourly > daily
        lambda: ChatLimits(global_hourly=200, global_daily=150),
        lambda: ChatLimits(per_code_daily=200, global_daily=150),  # per code > global
        lambda: ChatLimits(per_code_hourly=70, per_code_daily=70, global_hourly=60),
        lambda: ChatLimits(per_code_concurrent=3, global_concurrent=2),
        lambda: TranscriptionLimits(per_code_daily=0),
        lambda: TranscriptionLimits(max_request_seconds=601, per_code_daily_seconds=700, global_daily_seconds=800),
        lambda: TranscriptionLimits(max_request_seconds=400),  # longer than the per-code budget
        lambda: TranscriptionLimits(per_code_daily_seconds=2000),  # per code > global
        lambda: TranscriptionLimits(per_code_concurrent=2),
        lambda: TranscriptionLimits(max_settle_seconds=59),  # below one recording
        lambda: TranscriptionLimits(max_settle_seconds=301),  # above the per-code budget
        lambda: QuotaConfig(max_identities=0),
        lambda: QuotaConfig(max_identities=True),
        lambda: QuotaConfig(busy_retry_after_seconds=0),
        lambda: QuotaConfig(busy_retry_after_seconds=301),
        lambda: QuotaConfig(chat="not limits"),
        lambda: QuotaEngine("not a config"),
    ],
)
def test_invalid_configurations_are_refused(build):
    with pytest.raises(QuotaConfigError):
        build()


@pytest.mark.parametrize(
    "call",
    [
        lambda e: e.admit("speech", "alice"),
        lambda e: e.admit(CHAT, "Alice"),  # not a code id
        lambda e: e.admit(CHAT, ""),
        lambda e: e.admit(CHAT, None),
        lambda e: e.admit(CHAT, "a" * 33),
        lambda e: e.admit(CHAT, "alice", seconds=10),
        lambda e: e.admit(TRANSCRIPTION, "alice"),
        lambda e: e.admit(TRANSCRIPTION, "alice", seconds=0),
        lambda e: e.admit(TRANSCRIPTION, "alice", seconds=1.5),
        lambda e: e.admit(TRANSCRIPTION, "alice", seconds=True),
    ],
)
def test_invalid_requests_are_programming_errors_and_reserve_nothing(engine, call):
    before = snapshot(engine)
    with pytest.raises(ValueError):
        call(engine)
    assert snapshot(engine) == before


# ---------------------------------------------------------------------------
# Chat: per-code and global boundaries (N-1, N, N+1)
# ---------------------------------------------------------------------------

def test_per_code_hourly_boundary(engine, clock):
    for count in range(1, 10):  # N-1 = 9
        assert admitted(chat(engine)), count
        clock.advance(60)
    assert admitted(chat(engine))  # N = 10
    refused = chat(engine)  # N + 1
    assert refused == Refusal(CHAT, HOURLY_LIMIT, "code", 3600 - 9 * 60)
    assert admitted(chat(engine, "bob"))  # another code is unaffected

    clock.advance(3600 - 9 * 60 - 1)
    assert chat(engine).reason == HOURLY_LIMIT  # one second early
    clock.advance(1)
    assert admitted(chat(engine))  # the oldest left the window


def test_per_code_daily_boundary(engine, clock):
    for count in range(24):  # N-1, spread over hours to stay under the hourly limit
        assert admitted(chat(engine)), count
        clock.advance(400)
    assert admitted(chat(engine))  # N = 25
    clock.advance(3600)
    refused = chat(engine)
    assert (refused.reason, refused.scope) == (DAILY_LIMIT, "code")
    assert refused.retry_after == math.ceil(MIDNIGHT - clock.wall)


def test_global_hourly_boundary(engine):
    codes = [f"code-{index}" for index in range(7)]
    for index in range(59):  # N-1 = 59, at most 10 per code
        assert admitted(chat(engine, codes[index // 10]))
    assert admitted(chat(engine, "code-5"))  # N = 60
    refused = chat(engine, "code-6")  # N + 1, a fresh code
    assert (refused.reason, refused.scope) == (HOURLY_LIMIT, "global")
    assert refused.retry_after == 3600


def test_global_daily_boundary(clock):
    engine = engine_for(clock, QuotaConfig(chat=ChatLimits(
        per_code_hourly=5, per_code_daily=5, global_hourly=10, global_daily=12)))
    for index in range(11):  # N-1
        assert admitted(chat(engine, f"c{index}"))
        clock.advance(400)
    assert admitted(chat(engine, "c11"))  # N
    refused = chat(engine, "c12")  # N + 1
    assert (refused.reason, refused.scope) == (DAILY_LIMIT, "global")
    assert refused.retry_after == math.ceil(MIDNIGHT - clock.wall)


def test_global_budget_exhausted_by_many_codes_with_default_limits(engine, clock):
    used = 0
    for _ in range(25):  # 25 rounds of 6 codes = 150
        for code in range(6):
            assert admitted(chat(engine, f"code-{code}"))
            used += 1
        clock.advance(400)  # 9 per code and 54 in total per hour: under both hourly limits
    assert used == 150 and engine.usage()["daily"][CHAT] == 150
    assert chat(engine, "fresh").reason == DAILY_LIMIT
    assert chat(engine, "fresh").scope == "global"


# ---------------------------------------------------------------------------
# Concurrency, release and failures
# ---------------------------------------------------------------------------

def test_one_active_chat_per_code_and_two_globally(engine):
    first = engine.admit(CHAT, "alice")
    assert engine.admit(CHAT, "alice") == Refusal(CHAT, BUSY, "code", 5)
    second = engine.admit(CHAT, "bob")
    assert engine.admit(CHAT, "carol") == Refusal(CHAT, BUSY, "global", 5)
    first.release()
    third = engine.admit(CHAT, "carol")
    assert admitted(third)
    assert engine.usage()["active"][CHAT] == 2
    second.release(), third.release()
    assert engine.usage()["active"][CHAT] == 0


def test_a_refusal_reserves_nothing(engine):
    held = engine.admit(CHAT, "alice")
    before = snapshot(engine)
    assert engine.admit(CHAT, "alice").reason == BUSY
    assert snapshot(engine) == before  # no partial reservation
    held.release()


def test_a_counted_limit_is_reported_before_busy(engine, clock):
    for _ in range(10):
        chat(engine)
    held = engine.admit(CHAT, "bob")
    held2 = engine.admit(CHAT, "carol")
    refused = engine.admit(CHAT, "alice")  # over the hourly limit and globally busy
    assert refused.reason == HOURLY_LIMIT  # waiting 5 s would not help
    held.release(), held2.release()


def test_the_longest_wait_is_reported_when_several_limits_block(clock):
    engine = engine_for(clock, QuotaConfig(chat=ChatLimits(
        per_code_hourly=2, per_code_daily=2, global_hourly=60, global_daily=150)))
    chat(engine), chat(engine)
    refused = chat(engine)  # hourly (3600 s) and daily (12 h) both block
    assert (refused.reason, refused.retry_after) == (DAILY_LIMIT, math.ceil(MIDNIGHT - clock.wall))


def test_release_is_idempotent_and_never_goes_negative(engine):
    ticket = engine.admit(CHAT, "alice")
    other = engine.admit(CHAT, "bob")
    assert ticket.release() is True
    assert ticket.release() is False
    assert ticket.release() is False
    assert ticket.released
    assert engine.usage()["active"][CHAT] == 1  # the double release didn't free bob's slot
    assert engine.usage("alice")["active"][CHAT] == 0
    other.release()
    assert engine.usage()["active"][CHAT] == 0


def test_a_failed_or_cancelled_request_still_counts(engine):
    with pytest.raises(RuntimeError):
        with engine.admit(CHAT, "alice"):
            raise RuntimeError("provider failed")  # the context manager releases
    assert engine.usage("alice")["active"][CHAT] == 0
    assert engine.usage("alice")["daily"][CHAT] == 1  # but the request counted
    assert engine.usage("alice")["hourly_chat"] == 1

    ticket = engine.admit(CHAT, "alice")
    ticket.release()  # cancelled, for example a client that disconnected
    assert engine.usage("alice")["daily"][CHAT] == 2


def test_chat_and_transcription_have_separate_slots_but_shared_identities(engine):
    chat_ticket = engine.admit(CHAT, "alice")
    voice = engine.admit(TRANSCRIPTION, "alice", seconds=30)
    assert admitted(chat_ticket) and admitted(voice)  # one of each at once
    assert engine.admit(TRANSCRIPTION, "bob", seconds=30) == Refusal(TRANSCRIPTION, BUSY, "global", 5)
    assert admitted(chat(engine, "bob"))  # chat is unaffected by the transcription slot
    assert engine.tracked_identities() == 2
    chat_ticket.release(), voice.release()


# ---------------------------------------------------------------------------
# 50 simultaneous threads
# ---------------------------------------------------------------------------

def _together(count, work):
    barrier = threading.Barrier(count)
    results, lock = [], threading.Lock()

    def run():
        barrier.wait(5)
        value = work()
        with lock:
            results.append(value)

    threads = [threading.Thread(target=run) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert len(results) == count
    return results


def test_admission_is_atomic_under_forced_thread_switching(clock):
    # A tiny switch interval makes the interpreter swap threads between
    # almost every bytecode, so without the lock threads would interleave
    # between checking a limit and reserving it.
    import sys

    config = QuotaConfig(chat=ChatLimits(per_code_hourly=5, per_code_daily=5, per_code_concurrent=2,
                                         global_concurrent=50, global_hourly=60, global_daily=150))
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for _ in range(20):
            engine = engine_for(clock, config)
            results = _together(50, lambda: engine.admit(CHAT, "alice"))
            assert sum(admitted(result) for result in results) == 2  # the per-code concurrency limit
            assert engine.usage("alice")["active"][CHAT] == 2
            assert engine.usage("alice")["daily"][CHAT] == 2
    finally:
        sys.setswitchinterval(previous)


def test_50_threads_one_code_get_exactly_one_slot(engine):
    results = _together(50, lambda: engine.admit(CHAT, "alice"))
    tickets = [result for result in results if admitted(result)]
    assert len(tickets) == 1
    assert all(result.reason == BUSY for result in results if not admitted(result))
    assert engine.usage("alice")["daily"][CHAT] == 1  # refusals counted nothing


def test_50_threads_many_codes_get_exactly_the_global_slots(engine):
    names = iter(f"code-{index}" for index in range(50))
    name_lock = threading.Lock()

    def work():
        with name_lock:
            name = next(names)
        return engine.admit(CHAT, name)

    results = _together(50, work)
    assert sum(admitted(result) for result in results) == 2
    assert engine.usage()["daily"][CHAT] == 2


def test_50_threads_admit_and_release_without_leaks_or_overcounting(clock):
    engine = engine_for(clock, QuotaConfig(chat=ChatLimits(
        per_code_hourly=40, per_code_daily=40, global_hourly=1000, global_daily=1000,
        per_code_concurrent=3, global_concurrent=5)))
    peak, lock = [0], threading.Lock()

    def work():
        outcomes = []
        for _ in range(20):
            result = engine.admit(CHAT, "alice")
            if admitted(result):
                with lock:
                    peak[0] = max(peak[0], engine.usage("alice")["active"][CHAT])
                result.release()
                result.release()  # a double release mixed in
            outcomes.append(admitted(result))
        return sum(outcomes)

    total = sum(_together(50, work))
    assert total == 40  # exactly the per-code limit, never more
    assert peak[0] <= 3
    assert engine.usage()["active"] == {CHAT: 0, TRANSCRIPTION: 0}
    assert engine.usage("alice")["daily"][CHAT] == 40


def test_50_threads_transcription_minutes_are_never_overbooked(clock):
    engine = engine_for(clock, QuotaConfig(
        transcription=TranscriptionLimits(per_code_daily=1, global_concurrent=50, per_code_concurrent=1)))
    names = iter(f"code-{index}" for index in range(50))
    name_lock = threading.Lock()

    def work():
        with name_lock:
            name = next(names)
        return engine.admit(TRANSCRIPTION, name, seconds=60)

    results = _together(50, work)
    assert sum(admitted(result) for result in results) == 30  # 1800 s / 60 s
    assert engine.usage()["seconds"] == 1800


# ---------------------------------------------------------------------------
# Transcription: requests, minutes, reservation and settlement
# ---------------------------------------------------------------------------

def transcribe(engine, identity="alice", seconds=20):
    result = engine.admit(TRANSCRIPTION, identity, seconds=seconds)
    if admitted(result):
        result.release()
    return result


def test_transcription_daily_request_boundary(engine):
    for count in range(9):  # N-1, 20 s each: under the 300 s budget
        assert admitted(transcribe(engine, seconds=20)), count
    assert admitted(transcribe(engine, seconds=20))  # N = 10
    refused = transcribe(engine, seconds=1)  # N + 1
    assert (refused.reason, refused.scope) == (DAILY_LIMIT, "code")


def test_transcription_per_code_minutes_boundary(engine):
    for _ in range(4):
        assert admitted(transcribe(engine, seconds=60))  # 240 s
    assert admitted(transcribe(engine, seconds=59))  # 299 s: N-1
    assert admitted(transcribe(engine, seconds=1))  # 300 s: N
    refused = transcribe(engine, seconds=1)  # N + 1
    assert (refused.reason, refused.scope) == (MINUTES_LIMIT, "code")
    assert engine.usage("alice")["seconds"] == 300


def test_transcription_global_minutes_boundary(engine):
    for index in range(29):
        assert admitted(transcribe(engine, f"code-{index}", seconds=60))  # 1740 s
    assert admitted(transcribe(engine, "code-29", seconds=60))  # 1800 s: N
    assert engine.usage()["seconds"] == 1800
    refused = transcribe(engine, "code-30", seconds=1)
    assert (refused.reason, refused.scope) == (MINUTES_LIMIT, "global")


def test_a_reservation_that_would_exceed_the_budget_is_refused_whole(engine):
    for _ in range(4):
        transcribe(engine, seconds=60)
    transcribe(engine, seconds=50)  # 290 s used
    before = snapshot(engine)
    assert transcribe(engine, seconds=11).reason == MINUTES_LIMIT  # 301 > 300
    assert snapshot(engine) == before
    assert admitted(transcribe(engine, seconds=10))  # exactly 300 fits


def test_one_recording_longer_than_allowed_is_refused_and_not_retryable(engine):
    before = snapshot(engine)
    assert engine.admit(TRANSCRIPTION, "alice", seconds=61) == Refusal(TRANSCRIPTION, REQUEST_TOO_LONG, "request", None)
    assert snapshot(engine) == before
    assert admitted(transcribe(engine, seconds=60))


def test_settlement_never_refunds_a_shorter_recording(engine):
    ticket = engine.admit(TRANSCRIPTION, "alice", seconds=60)
    assert ticket.settle(12.3) == 0
    ticket.release()
    assert engine.usage("alice")["seconds"] == 60  # the reservation stands
    assert engine.usage()["seconds"] == 60


def test_settlement_charges_a_longer_recording_and_blocks_further_requests(engine):
    for _ in range(4):
        transcribe(engine, seconds=60)  # 240 s
    ticket = engine.admit(TRANSCRIPTION, "alice", seconds=30)  # 270 s reserved
    assert ticket.settle(90.2) == 61  # actually 91 s: 61 s more than reserved
    ticket.release()
    assert engine.usage("alice")["seconds"] == 331  # past the 300 s budget: recorded, not hidden
    assert transcribe(engine, seconds=1).reason == MINUTES_LIMIT
    assert engine.usage()["seconds"] == 331


class _ExplodingInt(int):
    def __ge__(self, other):
        raise RuntimeError("comparison failed")


@pytest.mark.parametrize(
    "actual",
    [math.nan, math.inf, -math.inf, -1, -0.5, True, False, "30", None, [30], Decimal("30"), object(),
     _ExplodingInt(30)],
)
def test_a_malformed_duration_is_not_trusted_and_charged_the_cap(engine, caplog, actual):
    ticket = engine.admit(TRANSCRIPTION, "alice", seconds=10)
    assert ticket.settle(actual) == 290  # never raises; 300 s charged in all
    ticket.release()
    assert engine.usage("alice")["seconds"] == 300
    assert engine.anomalies() == {"malformed": 1, "implausible": 0, "overrun": 0}
    assert "not trusted: category=malformed charged_seconds=300" in caplog.text


@pytest.mark.parametrize("actual", [10**400, 1e9, 1e308, 42_000, 301, 300.5])
def test_an_implausible_duration_is_capped_and_recorded(engine, caplog, actual):
    # 42_000: 42 s accidentally passed as milliseconds.
    ticket = engine.admit(TRANSCRIPTION, "alice", seconds=10)
    assert ticket.settle(actual) == 290
    ticket.release()
    assert engine.usage("alice")["seconds"] == 300  # bounded, exact integers
    assert engine.usage()["seconds"] == 300
    assert engine.anomalies()["implausible"] == 1
    assert "not trusted: category=implausible charged_seconds=300" in caplog.text
    assert str(actual) not in caplog.text and "42000" not in caplog.text  # the value isn't logged


def test_one_bad_report_blocks_only_its_own_code(engine):
    # Milliseconds passed as seconds: before the fix this blocked everyone for the day.
    ticket = engine.admit(TRANSCRIPTION, "alice", seconds=42)
    ticket.release()
    ticket.settle(42_000)
    assert transcribe(engine, "alice", seconds=1).reason == MINUTES_LIMIT
    for index in range(5):
        assert admitted(transcribe(engine, f"code-{index}", seconds=60))  # 300 + 5 x 60 of 1800
    assert engine.usage()["seconds"] == 600


def test_the_cap_itself_is_a_plausible_overrun(engine):
    ticket = engine.admit(TRANSCRIPTION, "alice", seconds=10)
    assert ticket.settle(300) == 290
    assert engine.anomalies() == {"malformed": 0, "implausible": 0, "overrun": 1}
    ticket.release()


def test_settlement_never_refunds_or_goes_negative(engine):
    for actual in (0, 0.0, 9.0, 10):
        ticket = engine.admit(TRANSCRIPTION, "alice", seconds=10)
        assert ticket.settle(actual) == 0
        ticket.release()
    assert engine.usage("alice")["seconds"] == 40
    assert engine.anomalies() == {"malformed": 0, "implausible": 0, "overrun": 0}


def test_counters_stay_bounded_under_repeated_bad_reports(engine):
    for index in range(6):
        ticket = engine.admit(TRANSCRIPTION, f"code-{index}", seconds=1)
        ticket.release()
        ticket.settle(10**400)
    assert engine.usage()["seconds"] == 1800  # 6 x the cap: the global budget, no more
    assert engine.anomalies()["implausible"] == 6


def test_settlement_happens_once_and_only_for_transcription(engine):
    ticket = engine.admit(TRANSCRIPTION, "alice", seconds=10)
    assert ticket.settle(40) == 30
    assert ticket.settle(10**400) == 0  # a second settlement is ignored
    assert ticket.settle(math.nan) == 0
    assert engine.usage("alice")["seconds"] == 40
    assert engine.anomalies() == {"malformed": 0, "implausible": 0, "overrun": 1}
    ticket.release()
    with pytest.raises(ValueError):
        engine.admit(CHAT, "alice").settle(10)


def test_settlement_after_midnight_charges_the_new_day(engine, clock):
    clock.wall = MIDNIGHT - 10
    ticket = engine.admit(TRANSCRIPTION, "alice", seconds=10)
    clock.advance(20)  # the call finished after midnight
    assert ticket.settle(70) == 60
    ticket.release()
    assert engine.usage("alice")["seconds"] == 60  # the new day carries the overrun
    assert engine.usage()["seconds"] == 60


def test_an_implausible_settlement_after_midnight_charges_the_new_day_at_most_the_cap(engine, clock):
    clock.wall = MIDNIGHT - 10
    ticket = engine.admit(TRANSCRIPTION, "alice", seconds=10)
    clock.advance(20)
    assert ticket.settle(1e9) == 290
    ticket.release()
    assert engine.usage("alice")["seconds"] == 290  # the new day: bounded by the cap
    assert admitted(transcribe(engine, "bob", seconds=60))  # others keep their budget


# ---------------------------------------------------------------------------
# Clocks: UTC days, rollback and forward jumps
# ---------------------------------------------------------------------------

def use_daily_chats(engine, clock, identity="alice", count=25):
    """Use (the rest of) the per-code daily chat limit, 400 s apart (under the hourly limit)."""
    for _ in range(count):
        assert admitted(chat(engine, identity))
        clock.advance(400)


def test_the_day_rolls_over_at_utc_midnight(engine, clock):
    clock.wall = MIDNIGHT - 3 * 3600  # before the first reading: the anchor
    use_daily_chats(engine, clock)  # 10,000 s later: 800 s before midnight
    refused = chat(engine)
    assert (refused.reason, refused.retry_after) == (DAILY_LIMIT, 800)
    clock.advance(799)
    assert chat(engine).retry_after == 1
    clock.advance(1)  # 00:00:00 UTC
    assert admitted(chat(engine))
    assert engine.usage("alice")["daily"][CHAT] == 1


def test_days_are_utc_with_the_real_clocks_whatever_the_local_time_zone(monkeypatch):
    # The default clocks (time.time, time.monotonic) and a far-from-UTC zone;
    # local-time functions are made to fail, so the engine must not use them.
    monkeypatch.setenv("TZ", "Pacific/Kiritimati")  # UTC+14
    time.tzset()
    try:
        def no_local_time(*args, **kwargs):
            raise AssertionError("local time used")

        for name in ("localtime", "mktime", "strftime"):
            monkeypatch.setattr(time, name, no_local_time)
        before = math.floor(time.time() / 86400)
        engine = QuotaEngine()
        assert admitted(chat(engine))
        after = math.floor(time.time() / 86400)
        assert engine._day in (before, after)  # UTC days since the epoch, not the local date
    finally:
        monkeypatch.undo()
        time.tzset()


def test_a_wall_clock_jump_forward_neither_resets_nor_freezes_budgets(engine, clock):
    use_daily_chats(engine, clock)  # from noon: about 14:47 UTC
    expected = math.ceil(MIDNIGHT - clock.wall)
    clock.wall += 365 * 86400  # the wall clock jumps a year ahead
    refused = chat(engine)
    assert (refused.reason, refused.retry_after) == (DAILY_LIMIT, expected)  # no fresh budget
    clock.wall -= 365 * 86400  # NTP corrects it
    assert chat(engine).retry_after == expected

    clock.advance(expected)  # the real midnight
    assert admitted(chat(engine))  # the next day opens on time
    use_daily_chats(engine, clock, count=24)  # the rest of that day is used up too
    assert chat(engine).reason == DAILY_LIMIT
    clock.advance(86400)
    assert admitted(chat(engine))  # and the day after opens: nothing frozen


def test_a_one_day_forward_glitch_does_not_merge_two_days(engine, clock):
    assert admitted(chat(engine, "bob"))  # the anchor: noon
    clock.wall += 86400  # a one-day glitch
    assert admitted(chat(engine))
    clock.wall -= 86400  # corrected
    clock.advance(400)
    for _ in range(24):  # the rest of today's 25
        assert admitted(chat(engine))
        clock.advance(400)
    assert chat(engine).reason == DAILY_LIMIT  # the glitch gave no fresh budget
    clock.advance(math.ceil(MIDNIGHT - clock.wall))  # the real midnight
    assert admitted(chat(engine))  # the real next day has its own budget


def test_a_wall_clock_jump_back_neither_reopens_nor_freezes_budgets(engine, clock):
    use_daily_chats(engine, clock)
    expected = math.ceil(MIDNIGHT - clock.wall)
    clock.wall -= 2 * 86400  # the clock is set back two days
    refused = chat(engine)
    assert (refused.reason, refused.retry_after) == (DAILY_LIMIT, expected)  # not reopened, not longer
    clock.mono += expected  # only real (monotonic) time passes; the wall clock stays wrong
    assert admitted(chat(engine))  # the next day opens on time
    clock.mono += 86400
    assert admitted(chat(engine))  # and the one after


def test_retry_after_follows_the_anchored_clock_after_jumps(engine, clock):
    use_daily_chats(engine, clock)
    for jump in (3600, -7200, 86400 * 10):
        clock.wall += jump
        refused = chat(engine)
        clock.mono += refused.retry_after - 1
        assert chat(engine).reason == DAILY_LIMIT  # one second early: still refused
        clock.mono += 1
        assert admitted(chat(engine))  # exactly when promised
        use_daily_chats(engine, clock, count=24)


def test_a_monotonic_clock_going_backwards_does_not_reopen_the_hourly_window(engine, clock):
    for _ in range(10):
        chat(engine)
    clock.mono -= 5000
    refused = chat(engine)
    assert refused.reason == HOURLY_LIMIT
    assert refused.retry_after == 3600  # measured from the latest time seen, not overstated
    clock.mono += 5000 + 3600
    assert admitted(chat(engine))


def test_the_wall_clock_is_read_once(clock):
    reads = []

    def wall():
        reads.append(1)
        return clock.wall

    engine = QuotaEngine(monotonic=lambda: clock.mono, wall=wall)
    for _ in range(5):
        chat(engine)
        clock.advance(10)
    assert len(reads) == 1


# ---------------------------------------------------------------------------
# Bounded memory, restart and failing closed
# ---------------------------------------------------------------------------

def test_identities_are_bounded_and_full_capacity_fails_closed(clock):
    engine = engine_for(clock, QuotaConfig(max_identities=3))
    for name in ("a", "b", "c"):
        assert admitted(chat(engine, name))
    refused = chat(engine, "d")
    assert (refused.reason, refused.scope) == (CAPACITY, "global")
    assert refused.retry_after == math.ceil(MIDNIGHT - clock.wall)  # today's usage is kept until midnight
    assert engine.tracked_identities() == 3
    assert admitted(chat(engine, "a"))  # known identities still work

    clock.advance(refused.retry_after - 1)
    assert chat(engine, "d").reason == CAPACITY  # one second early
    clock.advance(1)
    assert admitted(chat(engine, "d"))  # exactly when promised
    assert engine.tracked_identities() == 1


def test_capacity_retry_after_accounts_for_usage_crossing_midnight(clock):
    clock.wall = MIDNIGHT - 600
    engine = engine_for(clock, QuotaConfig(max_identities=2))
    held = engine.admit(CHAT, "a")
    assert admitted(chat(engine, "b"))
    clock.advance(700)  # a new day; both are still in their hourly windows, and a is active
    refused = chat(engine, "c")
    assert refused.reason == CAPACITY
    assert refused.retry_after == 3600 - 700  # when the hourly windows end, not 5 s
    held.release()
    clock.advance(refused.retry_after - 1)
    assert chat(engine, "c").reason == CAPACITY
    clock.advance(1)
    assert admitted(chat(engine, "c"))


def test_capacity_with_an_idle_cross_midnight_identity_reports_its_hour(clock):
    clock.wall = MIDNIGHT - 100
    engine = engine_for(clock, QuotaConfig(max_identities=1))
    assert admitted(chat(engine, "a"))
    clock.advance(200)  # the next day: "a" has no daily usage, one recent timestamp
    refused = chat(engine, "b")
    assert (refused.reason, refused.retry_after) == (CAPACITY, 3600 - 200)


def test_timestamp_storage_is_bounded_by_the_limits(engine):
    for _ in range(500):
        chat(engine, "alice")
    assert len(engine._identities["alice"].hourly_chat) == 10
    for index in range(100):
        chat(engine, f"code-{index}")
    assert len(engine._global.hourly_chat) <= 60
    assert engine.tracked_identities() <= 1000


def test_a_restart_starts_from_zero(clock):
    first = engine_for(clock)
    for _ in range(25):
        chat(first)
        clock.mono += 400
    assert chat(first).reason == DAILY_LIMIT
    restarted = engine_for(clock)  # the documented limitation: counters are in memory
    assert admitted(chat(restarted))


@pytest.mark.parametrize("bad", [math.nan, math.inf, "noon"])
def test_a_broken_clock_fails_closed_and_reserves_nothing(clock, caplog, bad):
    engine = QuotaEngine(monotonic=lambda: clock.mono, wall=lambda: bad)
    result = engine.admit(CHAT, "alice")
    assert result == Refusal(CHAT, UNAVAILABLE, "global", 5)
    assert engine.usage("alice")["daily"][CHAT] == 0
    assert engine.usage()["active"][CHAT] == 0
    assert "Quota check failed" in caplog.text


def test_a_raising_clock_fails_closed(clock):
    def broken():
        raise OSError("clock")

    engine = QuotaEngine(monotonic=broken, wall=lambda: clock.wall)
    assert engine.admit(TRANSCRIPTION, "alice", seconds=5).reason == UNAVAILABLE
    assert engine.usage()["seconds"] == 0


def test_anonymous_requests_share_one_bucket_distinct_from_any_code(engine):
    for _ in range(10):
        assert admitted(chat(engine, ANONYMOUS))
    assert chat(engine, ANONYMOUS).reason == HOURLY_LIMIT
    assert admitted(chat(engine, "anonymous"))  # a code id that looks alike is separate


def test_engines_are_isolated_from_each_other(clock):
    one, two = engine_for(clock), engine_for(clock)
    ticket = one.admit(CHAT, "alice")
    assert admitted(two.admit(CHAT, "alice"))
    ticket.release()


# ---------------------------------------------------------------------------
# Retry-After
# ---------------------------------------------------------------------------

def test_retry_after_is_whole_seconds_and_at_least_one(engine, clock):
    for _ in range(10):
        chat(engine)
    clock.mono += 3599.4
    refused = chat(engine)
    assert refused.retry_after == 1  # 0.6 s rounds up, never 0
    assert isinstance(refused.retry_after, int)


def test_retry_after_for_the_daily_limit_counts_to_utc_midnight(engine, clock):
    clock.wall = MIDNIGHT - 3 * 3600 - 0.5
    use_daily_chats(engine, clock)  # 10,000 s later: 800.5 s before midnight
    refused = chat(engine)
    assert refused.retry_after == 801
    clock.advance(800)
    assert chat(engine).reason == DAILY_LIMIT  # 0.5 s early
    clock.advance(1)
    assert admitted(chat(engine))


def test_waiting_retry_after_is_enough(engine, clock):
    for _ in range(10):
        chat(engine)
        clock.advance(17)
    refused = chat(engine)
    clock.advance(refused.retry_after)
    assert admitted(chat(engine))


def test_a_configured_busy_delay_is_used(clock):
    engine = engine_for(clock, QuotaConfig(busy_retry_after_seconds=7))
    held = engine.admit(CHAT, "alice")
    assert engine.admit(CHAT, "alice").retry_after == 7
    held.release()


# ---------------------------------------------------------------------------
# Nothing sensitive is logged
# ---------------------------------------------------------------------------

def test_logs_never_contain_identities(engine, clock, caplog):
    caplog.set_level(logging.DEBUG, logger="app.services.quota")
    for _ in range(12):
        chat(engine, "secret-recruiter")  # the last two are over the hourly limit
    held = engine.admit(CHAT, "secret-other")
    assert engine.admit(CHAT, "secret-other").reason == BUSY
    held.release()
    ticket = engine.admit(TRANSCRIPTION, "secret-recruiter", seconds=10)
    ticket.settle(math.nan)
    ticket.release()
    engine.admit(TRANSCRIPTION, "secret-recruiter", seconds=61)

    assert "Quota refused: kind=chat reason=hourly_limit scope=code" in caplog.text
    assert "Quota refused: kind=transcription reason=request_too_long scope=request" in caplog.text
    assert "Transcription duration report not trusted: category=malformed" in caplog.text
    assert "Quota refused: kind=chat reason=busy scope=code" in caplog.text
    for name in ("secret-recruiter", "secret-other"):
        assert name not in caplog.text
    assert "secret-other" not in repr(held)


def test_the_module_imports_nothing_from_the_application():
    source = open(quota.__file__, encoding="utf-8").read()
    assert "from app" not in source and "import app" not in source
    for network in ("requests", "httpx", "urllib", "socket", "openai"):
        assert f"import {network}" not in source


# ---------------------------------------------------------------------------
# All-or-nothing reservation: deterministic fault injection
# ---------------------------------------------------------------------------

class _FaultyDeque(quota.deque):
    """Fails on its first append."""

    def __init__(self, *args, error=MemoryError):
        super().__init__(*args)
        self.error = error

    def append(self, item):
        if self.error is not None:
            error, self.error = self.error, None
            raise error("injected")
        super().append(item)


class _FaultyDict(dict):
    """Fails on its first assignment."""

    armed = False

    def __setitem__(self, key, value):
        if self.armed:
            self.armed = False
            raise MemoryError("injected")
        super().__setitem__(key, value)


class _FaultyUsage(quota._Usage):
    """Global usage whose seconds counter fails on its next assignment."""

    __slots__ = ("_value", "_fail")

    @property
    def seconds(self):
        return self._value

    @seconds.setter
    def seconds(self, value):
        if getattr(self, "_fail", False):
            self._fail = False
            raise MemoryError("injected")
        self._value = value


def _state(engine):
    return (engine.usage(), engine.usage("alice"), engine.usage("bob"), engine.tracked_identities(),
            sorted(engine._identities))


def _assert_rolled_back(engine, result, before, caplog):
    assert result == Refusal(result.kind, UNAVAILABLE, "global", 5)
    assert _state(engine) == before  # nothing reserved, per code or globally
    assert "Quota check failed (MemoryError)" in caplog.text


def test_a_fault_appending_the_global_timestamp_rolls_everything_back(engine, clock, caplog):
    assert admitted(chat(engine, "alice"))  # alice already has usage to restore exactly
    clock.advance(400)
    before = _state(engine)
    engine._global.hourly_chat = _FaultyDeque(engine._global.hourly_chat)
    _assert_rolled_back(engine, engine.admit(CHAT, "alice"), before, caplog)
    assert admitted(chat(engine, "alice"))  # no leaked slot: alice isn't "busy"


def test_a_fault_on_the_global_daily_counter_rolls_back_a_new_identity(engine, caplog):
    assert admitted(chat(engine, "alice"))  # the day has started (a new day replaces the counters)
    before = _state(engine)
    faulty = _FaultyDict(engine._global.daily)
    faulty.armed = True
    engine._global.daily = faulty
    _assert_rolled_back(engine, engine.admit(CHAT, "bob"), before, caplog)
    assert "bob" not in engine._identities  # the new identity was removed again
    assert admitted(chat(engine, "bob"))


def test_a_fault_on_the_global_seconds_rolls_back_a_transcription(engine, caplog):
    faulty = _FaultyUsage()
    engine._global = faulty
    assert admitted(transcribe(engine, "alice", seconds=30))
    before = _state(engine)
    faulty._fail = True
    _assert_rolled_back(engine, engine.admit(TRANSCRIPTION, "alice", seconds=40), before, caplog)
    assert engine.usage()["seconds"] == 30 and engine.usage("alice")["seconds"] == 30
    assert admitted(transcribe(engine, "alice", seconds=40))  # the slot wasn't leaked


def test_a_fault_creating_the_ticket_reserves_nothing(engine, monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise MemoryError("injected")

    before = _state(engine)
    monkeypatch.setattr(quota, "Ticket", broken)
    _assert_rolled_back(engine, engine.admit(CHAT, "alice"), before, caplog)


def test_an_interrupt_during_reservation_is_rolled_back_and_propagates(engine):
    before = _state(engine)
    engine._global.hourly_chat = _FaultyDeque(error=KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        engine.admit(CHAT, "alice")  # not swallowed
    assert _state(engine) == before
    assert admitted(chat(engine, "alice"))  # the lock was released


# ---------------------------------------------------------------------------
# Deterministic interleaving: one thread is paused inside the critical
# section (with sys.settrace) while a second thread tries to act. With the
# lock, the second must wait; without it, the invariants below break.
# ---------------------------------------------------------------------------

def _line_in(function, marker):
    lines, start = inspect.getsourcelines(function)
    found = [start + index for index, line in enumerate(lines) if marker in line]
    assert len(found) == 1, marker
    return found[0]


def interleave(function, marker, first, second, pause=0.3):
    """Run ``first`` paused at ``marker`` in ``function`` while ``second`` runs in another thread.

    Returns both results and the order in which they got past the pause.
    """
    code = function.__code__
    line = _line_in(function, marker)
    reached, resume = threading.Event(), threading.Event()
    order, results = [], {}

    def local(frame, event, arg):
        if event == "line" and frame.f_lineno == line and not reached.is_set():
            reached.set()
            resume.wait(pause)  # the second thread can finish now only if nothing excludes it
            order.append("first")
        return local

    def tracer(frame, event, arg):
        return local if frame.f_code is code else None

    def run_first():
        sys.settrace(tracer)
        try:
            results["first"] = first()
        finally:
            sys.settrace(None)

    def run_second():
        results["second"] = second()
        order.append("second")
        resume.set()

    one = threading.Thread(target=run_first)
    one.start()
    assert reached.wait(5), "the first thread never reached the marker"
    two = threading.Thread(target=run_second)
    two.start()
    two.join(5)
    one.join(5)
    assert not one.is_alive() and not two.is_alive()
    return results["first"], results["second"], order


RELEASE_MARK = "ticket._released = True"
SETTLE_MARK = "ticket._settled = True"
ADMIT_MARK = "ticket = Ticket(self, kind, identity, seconds)"


def test_two_threads_releasing_the_same_ticket_free_exactly_one_slot(clock):
    engine = engine_for(clock, QuotaConfig(chat=ChatLimits(global_concurrent=3)))
    ticket = engine.admit(CHAT, "alice")
    others = [engine.admit(CHAT, name) for name in ("bob", "carol")]  # the pool is full: 3 of 3
    first, second, order = interleave(QuotaEngine._release, RELEASE_MARK, ticket.release, ticket.release)
    assert order == ["first", "second"]  # the second release waited
    assert sorted([first, second]) == [False, True]
    assert engine.usage()["active"][CHAT] == 2  # bob's and carol's slots are still counted
    dave = engine.admit(CHAT, "dave")
    assert admitted(dave)  # the one freed slot
    assert engine.admit(CHAT, "eve").reason == BUSY  # and no second one
    for held in (*others, dave):
        held.release()


def test_two_threads_settling_the_same_ticket_charge_once(engine):
    ticket = engine.admit(TRANSCRIPTION, "alice", seconds=10)
    ticket.release()
    first, second, order = interleave(
        QuotaEngine._settle, SETTLE_MARK, lambda: ticket.settle(70), lambda: ticket.settle(70))
    assert order == ["first", "second"]
    assert sorted([first, second]) == [0, 60]
    assert engine.usage()["seconds"] == 70 and engine.usage("alice")["seconds"] == 70
    assert engine.anomalies()["overrun"] == 1


def test_admission_racing_with_release(engine):
    held = engine.admit(CHAT, "alice")
    first, second, order = interleave(
        QuotaEngine._admit_locked, ADMIT_MARK, lambda: engine.admit(CHAT, "bob"), held.release)
    assert order == ["first", "second"]  # the release waited for the decision to finish
    assert admitted(first) and second is True
    assert engine.usage()["active"][CHAT] == 1
    assert engine.usage("bob")["active"][CHAT] == 1 and engine.usage("alice")["active"][CHAT] == 0
    first.release()


def test_admission_racing_with_settlement(clock):
    engine = engine_for(clock, QuotaConfig(transcription=TranscriptionLimits(global_concurrent=2)))
    held = engine.admit(TRANSCRIPTION, "alice", seconds=10)
    first, second, order = interleave(
        QuotaEngine._admit_locked, ADMIT_MARK,
        lambda: engine.admit(TRANSCRIPTION, "bob", seconds=60), lambda: held.settle(70))
    assert order == ["first", "second"]  # the settlement waited for the decision to finish
    assert admitted(first) and second == 60
    assert engine.usage()["seconds"] == 10 + 60 + 60
    first.release(), held.release()


@pytest.mark.parametrize("kind, seconds, held", [(CHAT, None, 1), (TRANSCRIPTION, 30, 0)])
def test_concurrent_admissions_at_global_capacity(engine, kind, seconds, held):
    extra = {} if seconds is None else {"seconds": seconds}
    holders = [engine.admit(kind, "alice", **extra) for _ in range(held)]  # chat: 1 of 2 held
    first, second, order = interleave(
        QuotaEngine._admit_locked, ADMIT_MARK,
        lambda: engine.admit(kind, "bob", **extra), lambda: engine.admit(kind, "carol", **extra))
    assert order == ["first", "second"]
    assert admitted(first)
    assert second == Refusal(kind, BUSY, "global", 5)  # the last slot went to exactly one
    limit = engine.config.chat.global_concurrent if kind == CHAT else engine.config.transcription.global_concurrent
    assert engine.usage()["active"][kind] == limit
    first.release()
    for ticket in holders:
        ticket.release()
