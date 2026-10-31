"""Tests for EventChannel (A1-2): capacity, terminal slot, wake-ups, abort, loop closure.

Ordering is driven by threading events and barriers; time limits are only
failure guards. Every channel created through ``make_channel`` replaces its
``asyncio.Event`` with one that fails if touched off the event-loop thread.
"""

import asyncio
import random
import threading
import time

import pytest

from app.api.streaming import ABORTED, END, EventChannel
from tests.streaming_support import LoopClosingOnSchedule

GUARD = 5.0  # seconds; failure guard only


class LoopThreadEvent(asyncio.Event):
    """asyncio.Event that raises when used off the owning loop thread."""

    def __init__(self, thread_id):
        super().__init__()
        self._thread_id = thread_id
        self.violations = []

    def _check(self, name):
        if threading.get_ident() != self._thread_id:
            self.violations.append(name)
            raise AssertionError(f"asyncio.Event.{name} called off the loop thread")

    def set(self):
        self._check("set")
        super().set()

    def clear(self):
        self._check("clear")
        super().clear()

    async def wait(self):
        self._check("wait")
        return await super().wait()


class CountingLoop:
    """Proxy for an event loop that counts and tracks wake callbacks."""

    def __init__(self, loop):
        self._loop = loop
        self.scheduled = 0
        self.executed = 0
        self.max_outstanding = 0
        self._lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self._loop, name)

    def call_soon_threadsafe(self, callback, *args):
        def run():
            with self._lock:
                self.executed += 1
            callback(*args)

        handle = self._loop.call_soon_threadsafe(run)
        with self._lock:
            self.scheduled += 1
            self.max_outstanding = max(self.max_outstanding, self.scheduled - self.executed)
        return handle


def make_channel(loop=None, *, capacity=256, should_stop=lambda: None, poll_seconds=0.01):
    loop = CountingLoop(loop or asyncio.get_running_loop())
    channel = EventChannel(loop, capacity=capacity, should_stop=should_stop, poll_seconds=poll_seconds)
    channel._wake = LoopThreadEvent(threading.get_ident())
    return channel, loop


def start_thread(target, *args):
    thread = threading.Thread(target=target, args=args, daemon=True)
    thread.start()
    return thread


async def wait_until(predicate, timeout=GUARD):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.001)


async def join(thread, timeout=GUARD):
    await wait_until(lambda: not thread.is_alive(), timeout)


async def drain(channel, *, max_wait=GUARD):
    """Receive until END/ABORTED; an idle timeout fails as a lost wake-up."""
    received = []
    while True:
        batch = await channel.receive_batch(max_wait)
        if batch is END or batch is ABORTED:
            return received, batch
        if batch == []:
            raise AssertionError("consumer idled: lost wake-up")
        received.extend(batch)


# -------------------------------------------------------------------
# Capacity and backpressure
# -------------------------------------------------------------------

@pytest.mark.anyio
async def test_publish_blocks_at_capacity_and_resumes_after_drain():
    channel, _ = make_channel(capacity=4)
    published = []

    def produce():
        for index in range(6):
            published.append(channel.publish(index))

    thread = start_thread(produce)
    await wait_until(lambda: len(published) == 4)
    await asyncio.sleep(0.05)
    assert len(published) == 4 and thread.is_alive()  # blocked on the fifth

    assert await channel.receive_batch(GUARD) == [0, 1, 2, 3]
    await join(thread)
    assert await channel.receive_batch(GUARD) == [4, 5]
    assert published == [True] * 6


@pytest.mark.anyio
async def test_batches_never_exceed_capacity():
    channel, _ = make_channel(capacity=16)

    def produce():
        for index in range(2_000):
            assert channel.publish(index)
        channel.set_terminal("terminal")
        channel.finish()

    thread = start_thread(produce)
    received = []
    while True:
        batch = await channel.receive_batch(GUARD)
        if batch is END:
            break
        assert batch != [], "lost wake-up"
        assert len(batch) <= 16
        received.extend(batch)
        await asyncio.sleep(0)
    await join(thread)

    assert received == [*range(2_000), "terminal"]


# -------------------------------------------------------------------
# Terminal slot
# -------------------------------------------------------------------

@pytest.mark.anyio
async def test_terminal_slot_ignores_capacity_and_is_delivered_last_once():
    channel, _ = make_channel(capacity=3)
    for index in range(3):
        assert channel.publish(index)

    assert channel.set_terminal("terminal")  # full channel: still does not block
    assert not channel.set_terminal("second")  # set once
    assert channel.finish()

    assert await channel.receive_batch(GUARD) == [0, 1, 2]
    assert await channel.receive_batch(GUARD) == ["terminal"]
    assert await channel.receive_batch(GUARD) is END


@pytest.mark.anyio
async def test_publish_refuses_to_block_the_event_loop():
    channel, _ = make_channel(capacity=1)
    channel.publish("full")  # does not need to wait: allowed on the loop thread

    with pytest.raises(RuntimeError, match="block the event loop"):
        channel.publish("would block")


@pytest.mark.anyio
async def test_terminal_comes_after_all_queued_items():
    channel, _ = make_channel(capacity=10)
    channel.publish("a")
    channel.set_terminal("terminal")
    channel.publish("b")  # published after the terminal, still delivered before it
    assert channel.finish()

    received, end = await drain(channel)

    assert received == ["a", "b", "terminal"]
    assert end is END


@pytest.mark.anyio
async def test_finish_requires_a_terminal_and_rejects_further_publishing():
    channel, _ = make_channel()
    assert not channel.finish()

    channel.set_terminal("terminal")
    assert channel.finish()
    assert not channel.publish("late")
    assert not channel.set_terminal("late")

    received, end = await drain(channel)
    assert (received, end) == (["terminal"], END)


@pytest.mark.anyio
async def test_abort_discards_queued_items_and_terminal():
    channel, _ = make_channel()
    channel.publish("a")
    channel.set_terminal("terminal")

    assert channel.abort("disconnected") is True
    assert channel.abort("again") is False  # idempotent; first reason kept

    assert await channel.receive_batch(GUARD) is ABORTED
    assert channel.abort_reason == "disconnected"
    assert not channel.publish("b") and not channel.set_terminal("t") and not channel.finish()


# -------------------------------------------------------------------
# Concurrency and wake-ups
# -------------------------------------------------------------------

@pytest.mark.anyio
async def test_concurrent_publishers_lose_nothing_and_coalesce_wakeups():
    channel, loop = make_channel(capacity=256)
    producers, per_producer = 4, 25_000

    def produce(pid):
        for index in range(per_producer):
            assert channel.publish((pid, index))

    threads = [start_thread(produce, pid) for pid in range(producers)]

    def finish_when_done():
        for thread in threads:
            thread.join()
        channel.set_terminal("terminal")
        channel.finish()

    closer = start_thread(finish_when_done)
    received, end = await drain(channel)
    await join(closer)

    assert end is END and received[-1] == "terminal"
    items = received[:-1]
    assert len(items) == producers * per_producer
    for pid in range(producers):
        assert [index for owner, index in items if owner == pid] == list(range(per_producer))
    assert loop.max_outstanding <= 1  # at most one wake callback pending
    assert channel._wake.violations == []


@pytest.mark.anyio
async def test_publish_between_check_and_wait_is_not_lost():
    """A publish landing after the locked check but before the await wakes the consumer."""
    channel, _ = make_channel()
    triggered = []

    def publish_now():
        if not triggered:
            triggered.append(True)
            thread = start_thread(channel.publish, "late")
            thread.join(GUARD)  # loop is held here: the wake callback cannot run yet

    channel._before_wait = publish_now

    batch = await channel.receive_batch(max_wait=GUARD)

    assert batch == ["late"]


@pytest.mark.anyio
async def test_finish_racing_receive(  # 500 interleavings
):
    rng = random.Random(1234)
    for _ in range(500):
        channel, _ = make_channel(capacity=8)
        count = rng.randint(0, 20)

        def produce():
            for index in range(count):
                channel.publish(index)
            channel.set_terminal("terminal")
            channel.finish()

        thread = start_thread(produce)
        received, end = await drain(channel)
        await join(thread)
        assert received == [*range(count), "terminal"] and end is END


@pytest.mark.anyio
async def test_abort_racing_receive_and_publish():
    rng = random.Random(4321)
    for _ in range(500):
        channel, _ = make_channel(capacity=4)
        aborted = threading.Event()
        attempted = []
        false_while_open = []

        def produce():
            for index in range(50):
                attempted.append(index)
                # False means "stop producing"; the item may already have been
                # delivered (revision 10 contract), so compare with attempts.
                if not channel.publish(index):
                    if channel.state != "aborted":
                        false_while_open.append(index)
                    return

        def abort_later(delay):
            time.sleep(delay)
            aborted.set()
            channel.abort("timeout")

        producer = start_thread(produce)
        aborter = start_thread(abort_later, rng.random() * 0.002)
        received, end = await drain(channel)
        await join(producer)
        await join(aborter)

        assert end in (END, ABORTED)
        assert false_while_open == []
        assert received == sorted(set(received))  # in order, no duplicates
        assert set(received) <= set(attempted)


# -------------------------------------------------------------------
# Abort semantics
# -------------------------------------------------------------------

@pytest.mark.anyio
async def test_abort_from_worker_thread_wakes_consumer_and_publisher():
    channel, _ = make_channel(capacity=2, poll_seconds=5.0)  # long poll: only notify_all can wake
    channel.publish(1)
    channel.publish(2)
    result = {}

    def blocked_publish():
        start = time.monotonic()
        result["ok"] = channel.publish(3)
        result["elapsed"] = time.monotonic() - start

    publisher = start_thread(blocked_publish)
    assert await channel.receive_batch(GUARD) == [1, 2]
    await wait_until(lambda: len(channel) == 1)  # item 3 now queued
    channel.publish(4)
    blocked = start_thread(blocked_publish)  # blocks: channel full again (3, 4)
    await asyncio.sleep(0.05)

    started = time.monotonic()
    aborter = start_thread(channel.abort, "timeout")
    assert await channel.receive_batch(max_wait=GUARD) is ABORTED
    assert time.monotonic() - started < 1.0  # woken, not timed out
    await join(blocked)
    await join(aborter)
    await join(publisher)

    assert result["ok"] is False and result["elapsed"] < 1.0  # released by notify_all


@pytest.mark.anyio
async def test_abort_from_loop_thread_releases_blocked_publisher():
    channel, _ = make_channel(capacity=1, poll_seconds=5.0)
    channel.publish("full")
    result = {}
    thread = start_thread(lambda: result.setdefault("ok", channel.publish("blocked")))
    await asyncio.sleep(0.05)

    channel.abort("disconnected")  # on the loop thread
    await join(thread, timeout=1.0)

    assert result["ok"] is False
    assert await channel.receive_batch(GUARD) is ABORTED
    assert channel._wake.violations == []


@pytest.mark.anyio
async def test_abort_with_pending_wake_callback_is_coalesced():
    channel, loop = make_channel()

    def publish_then_abort():
        channel.publish("a")  # schedules the one wake callback
        channel.abort("timeout")  # must not schedule another

    thread = threading.Thread(target=publish_then_abort)
    thread.start()
    thread.join(GUARD)  # the loop is held: the pending callback cannot run yet

    assert loop.scheduled == 1
    assert await channel.receive_batch(max_wait=GUARD) is ABORTED  # pending callback suffices


@pytest.mark.anyio
async def test_backstop_abort_inside_publish_releases_publisher_and_schedules_wake():
    stop = threading.Event()
    channel, loop = make_channel(
        capacity=1, should_stop=lambda: "consumer_unresponsive" if stop.is_set() else None
    )
    channel.publish("full")  # fits: does not wait
    await asyncio.sleep(0)  # let the wake callback for "full" run
    scheduled_before = loop.scheduled
    result = {}
    thread = start_thread(lambda: result.setdefault("ok", channel.publish("blocked")))
    await asyncio.sleep(0.05)
    assert thread.is_alive()  # blocked: the consumer is not draining

    stop.set()  # the publisher's poll sees the backstop and aborts
    await join(thread)

    assert result["ok"] is False
    assert channel.abort_reason == "consumer_unresponsive"
    assert loop.scheduled == scheduled_before + 1  # wake scheduled outside the lock
    assert await channel.receive_batch(max_wait=GUARD) is ABORTED


# -------------------------------------------------------------------
# Event-loop closure and stopped loops (no consumer involved)
# -------------------------------------------------------------------

def test_publish_after_loop_closed_returns_false():
    loop = asyncio.new_event_loop()
    channel = EventChannel(loop)
    loop.close()

    assert channel.publish("a") is False
    assert channel.abort_reason == "loop_closed"


def test_closed_loop_detected_despite_pending_wake_callback():
    loop = asyncio.new_event_loop()  # never runs: callbacks stay pending
    counting = CountingLoop(loop)
    channel = EventChannel(counting)

    assert channel.publish("a") is True  # wake callback now pending
    loop.close()

    assert channel.publish("b") is False  # is_closed() checked before coalescing
    assert channel.abort_reason == "loop_closed"
    assert counting.scheduled == 1


def test_blocked_publisher_exits_when_loop_closes():
    loop = asyncio.new_event_loop()
    channel = EventChannel(loop, capacity=1, poll_seconds=0.02)
    channel.publish("full")
    result = {}

    def blocked():
        start = time.monotonic()
        result["ok"] = channel.publish("blocked")
        result["elapsed"] = time.monotonic() - start

    thread = start_thread(blocked)
    time.sleep(0.05)
    loop.close()
    thread.join(GUARD)

    assert result["ok"] is False and result["elapsed"] < 1.0
    assert channel.abort_reason == "loop_closed"


def test_terminal_and_finish_after_loop_closed_return_false():
    loop = asyncio.new_event_loop()
    channel = EventChannel(loop)
    loop.close()

    assert channel.set_terminal("terminal") is False
    assert channel.finish() is False


def test_abort_after_loop_closed_is_safe_and_releases_publishers(monkeypatch):
    loop = asyncio.new_event_loop()
    counting = CountingLoop(loop)
    channel = EventChannel(counting, capacity=1, poll_seconds=5.0)
    channel.publish("full")
    calls = {"notify": 0}
    real_notify = channel._notify

    def counted_notify():
        calls["notify"] += 1
        return real_notify()

    monkeypatch.setattr(channel, "_notify", counted_notify)
    result = {}
    thread = start_thread(lambda: result.setdefault("ok", channel.publish("blocked")))
    time.sleep(0.05)
    loop.close()
    scheduled_before = counting.scheduled

    assert channel.abort("timeout") is True  # no exception, no recursion
    thread.join(1.0)

    assert result["ok"] is False
    assert calls["notify"] == 0  # abort never calls _notify
    assert counting.scheduled == scheduled_before  # no scheduling on a closed loop
    assert not channel.publish("x") and not channel.set_terminal("t") and not channel.finish()


def test_stopped_but_not_closed_loop_relies_on_backstop():
    loop = asyncio.new_event_loop()  # stopped (never run), not closed
    stop = threading.Event()
    channel = EventChannel(
        loop, capacity=1, poll_seconds=0.02, should_stop=lambda: "abandon_due" if stop.is_set() else None
    )
    assert channel.publish("accepted") is True  # undetectable: accepted
    result = {}
    thread = start_thread(lambda: result.setdefault("ok", channel.publish("blocked")))
    time.sleep(0.05)
    assert thread.is_alive()

    stop.set()
    thread.join(GUARD)

    assert result["ok"] is False
    assert channel.abort_reason == "abandon_due"
    loop.close()


@pytest.mark.anyio
async def test_receive_batch_off_loop_thread_raises():
    channel, _ = make_channel()
    errors = []

    def call_from_thread():
        try:
            asyncio.run(channel.receive_batch(0.01))
        except RuntimeError as exc:
            errors.append(str(exc))

    await join(start_thread(call_from_thread))

    assert errors and "event-loop thread" in errors[0]


@pytest.mark.anyio
async def test_loop_closing_between_check_and_schedule(monkeypatch):
    """is_closed() is False, then call_soon_threadsafe raises: the race window."""
    proxy = LoopClosingOnSchedule(asyncio.get_running_loop())
    channel = EventChannel(proxy, capacity=1, poll_seconds=5.0)  # long poll: only notify_all releases
    channel._wake = LoopThreadEvent(threading.get_ident())
    channel.publish("full")  # fits; schedules a normal wake callback
    await asyncio.sleep(0)  # the callback runs, so no wake is pending
    assert channel._wake_pending is False

    calls = {"abort": 0, "_notify": 0, "_schedule_wake": 0}
    transitions = []
    for name in calls:
        real = getattr(channel, name)

        def counted(*args, _name=name, _real=real):
            calls[_name] += 1
            return _real(*args)

        monkeypatch.setattr(channel, name, counted)
    real_mark = channel._mark_aborted_locked
    monkeypatch.setattr(
        channel, "_mark_aborted_locked", lambda reason: transitions.append(real_mark(reason)) or transitions[-1]
    )

    result = {}
    blocked = start_thread(lambda: result.setdefault("blocked", channel.publish("blocked")))
    await asyncio.sleep(0.05)
    assert blocked.is_alive()  # waiting on the full channel

    proxy.arm()
    stored = await asyncio.to_thread(channel.set_terminal, "terminal")  # a producer thread notifies

    assert stored is False  # the channel became unusable while notifying
    assert proxy.attempts == 1  # the check passed, scheduling raised
    assert channel._wake_pending is False  # reset after the failed schedule
    assert channel.state == "aborted" and channel.abort_reason == "loop_closed"
    assert calls == {"abort": 0, "_notify": 1, "_schedule_wake": 1}  # no recursion
    assert transitions.count(True) == 1  # exactly one abort transition

    await join(blocked, timeout=1.0)  # released by notify_all, not the 5 s poll
    assert result["blocked"] is False

    assert channel.abort("timeout") is False  # already aborted; no exception
    assert proxy.attempts == 1  # closed loop: no further scheduling attempt
    assert channel._wake.violations == []
