"""Usage quotas and concurrency limits for paid features (Security S1-4a).

A standalone engine: nothing calls it yet (routes are wired in a later
phase), and it imports nothing from the application.

* ``QuotaEngine.admit(kind, identity)`` decides atomically, under one lock:
  every applicable limit is checked first, and only if all pass are they
  all reserved together. A refusal reserves nothing, and so does an
  internal error during the reservation (every change is rolled back).
* ``kind`` is ``CHAT`` (one pool shared by every chat route) or
  ``TRANSCRIPTION``. ``identity`` is an access-code id, or ``ANONYMOUS``
  for every request without one (they share one bucket).
* An admitted request counts against its quotas whatever happens next
  (a failed provider call may still be billed). Only its concurrency slot
  is given back, by ``Ticket.release()``, which is idempotent.
* Time: the wall clock is read once, as an anchor; afterwards time advances
  only with the monotonic clock. Days are UTC calendar days of that
  anchored time, so a wall-clock change while the process runs (forward or
  back) can neither open nor freeze a day. Hourly windows slide on the
  monotonic clock. A restart re-anchors.
* Storage is bounded: at most ``max_identities`` identities (beyond that,
  new identities are refused until one can be forgotten: fail closed) and
  at most *limit* timestamps per hourly window. Daily counters exist for
  the current day only, and settlement charges are capped (below).
* Any unexpected error refuses the request (``unavailable``).

Transcription minutes (accounted in whole seconds):

* ``admit(TRANSCRIPTION, identity, seconds=n)`` reserves ``n`` seconds,
  the duration the caller established *before* the paid call (1 to
  ``max_request_seconds``). It is refused if the reservation would exceed
  the per-code or global daily budget.
* After the call, ``Ticket.settle(actual)`` records the duration the
  provider reports. It never raises for a bad value, and it only ever
  adds (never a refund, never a negative counter):

  - a duration up to the reservation: nothing more is charged;
  - a plausible overrun, up to ``max_settle_seconds``: charged in full;
  - an implausible duration (above ``max_settle_seconds``, for example
    milliseconds passed as seconds) or a malformed one (not a finite,
    non-negative number): not trusted; charged as ``max_settle_seconds``
    and recorded as an anomaly (``anomalies()``) and in the log.

* ``max_settle_seconds`` (default 300, the per-code daily budget) is the
  most one request can be charged. An admitted request was checked to be
  at most ``max_request_seconds`` long, so anything beyond that is already
  abnormal; charging up to a whole per-code day refuses the responsible
  code for the rest of the day (fail closed for that code) while one bad
  report can't use up everyone's budget (the global budget is several
  times larger). It also keeps every counter bounded.
* If the actual duration exceeds the reservation, that money has already
  been spent: settlement can't prevent it, only record it, which may take
  usage past the limit. A settlement after midnight charges the new day.
  Bounding the overrun itself is the job of the duration check before the
  call (a later phase) and the upload size limit.

Nothing is persisted: a restart or redeploy starts every counter at zero.
Logs carry the kind, reason, scope and anomaly category only, never an
identity or a reported value, and are written after the lock is released.
"""

import logging
import math
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field, fields

logger = logging.getLogger(__name__)

CHAT = "chat"
TRANSCRIPTION = "transcription"
KINDS = (CHAT, TRANSCRIPTION)

ANONYMOUS = "*anonymous"  # not a valid code id, so it can't collide with one
IDENTITY_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")  # access-code ids (S1-3)

HOUR_SECONDS = 3600
DAY_SECONDS = 86400
MAX_LIMIT = 100_000

# Refusal reasons
BUSY = "busy"  # a concurrency limit; retry shortly
HOURLY_LIMIT = "hourly_limit"
DAILY_LIMIT = "daily_limit"
MINUTES_LIMIT = "minutes_limit"
REQUEST_TOO_LONG = "request_too_long"  # one recording longer than allowed; retrying won't help
CAPACITY = "capacity"  # can't track another identity yet (fail closed)
UNAVAILABLE = "unavailable"  # an internal error (fail closed)

CODE_SCOPE = "code"
GLOBAL_SCOPE = "global"
REQUEST_SCOPE = "request"

# Settlement anomaly categories
MALFORMED = "malformed"  # not a finite, non-negative number
IMPLAUSIBLE = "implausible"  # longer than max_settle_seconds
OVERRUN = "overrun"  # plausible, but longer than reserved


class QuotaConfigError(ValueError):
    """Quota settings that must not be used."""


def _check_positive_ints(limits, maximum: int = MAX_LIMIT) -> None:
    for item in fields(limits):
        value = getattr(limits, item.name)
        if type(value) is not int or not 1 <= value <= maximum:
            raise QuotaConfigError(f"{item.name} must be a whole number from 1 to {maximum}.")


@dataclass(frozen=True)
class ChatLimits:
    per_code_hourly: int = 10
    per_code_daily: int = 25
    global_hourly: int = 60
    global_daily: int = 150
    per_code_concurrent: int = 1
    global_concurrent: int = 2

    def __post_init__(self):
        _check_positive_ints(self)
        if self.per_code_hourly > self.per_code_daily or self.global_hourly > self.global_daily:
            raise QuotaConfigError("An hourly chat limit must not exceed its daily limit.")
        if (
            self.per_code_hourly > self.global_hourly
            or self.per_code_daily > self.global_daily
            or self.per_code_concurrent > self.global_concurrent
        ):
            raise QuotaConfigError("A per-code chat limit must not exceed its global limit.")


@dataclass(frozen=True)
class TranscriptionLimits:
    per_code_daily: int = 10  # requests
    per_code_daily_seconds: int = 5 * 60
    global_daily_seconds: int = 30 * 60
    per_code_concurrent: int = 1
    global_concurrent: int = 1
    max_request_seconds: int = 60  # one recording, checked before the paid call
    max_settle_seconds: int = 5 * 60  # the most one request can be charged (see the module docstring)

    def __post_init__(self):
        _check_positive_ints(self)
        if self.max_request_seconds > 600:
            raise QuotaConfigError("max_request_seconds must be at most 600.")
        if not self.max_request_seconds <= self.per_code_daily_seconds <= self.global_daily_seconds:
            raise QuotaConfigError(
                "Transcription budgets must satisfy max_request_seconds <= "
                "per_code_daily_seconds <= global_daily_seconds."
            )
        if not self.max_request_seconds <= self.max_settle_seconds <= self.per_code_daily_seconds:
            raise QuotaConfigError(
                "max_settle_seconds must be from max_request_seconds to per_code_daily_seconds."
            )
        if self.per_code_concurrent > self.global_concurrent:
            raise QuotaConfigError("A per-code transcription limit must not exceed its global limit.")


@dataclass(frozen=True)
class QuotaConfig:
    chat: ChatLimits = field(default_factory=ChatLimits)
    transcription: TranscriptionLimits = field(default_factory=TranscriptionLimits)
    max_identities: int = 1000
    busy_retry_after_seconds: int = 5

    def __post_init__(self):
        if not isinstance(self.chat, ChatLimits) or not isinstance(self.transcription, TranscriptionLimits):
            raise QuotaConfigError("chat and transcription must be ChatLimits and TranscriptionLimits.")
        if type(self.max_identities) is not int or not 1 <= self.max_identities <= MAX_LIMIT:
            raise QuotaConfigError(f"max_identities must be a whole number from 1 to {MAX_LIMIT}.")
        if type(self.busy_retry_after_seconds) is not int or not 1 <= self.busy_retry_after_seconds <= 300:
            raise QuotaConfigError("busy_retry_after_seconds must be a whole number from 1 to 300.")


@dataclass(frozen=True)
class Refusal:
    """Why a request was refused. ``retry_after`` is whole seconds, or None if retrying won't help."""

    kind: str
    reason: str
    scope: str
    retry_after: int | None


class _Usage:
    """Counters for one identity, or for everyone (the global usage)."""

    __slots__ = ("hourly_chat", "daily", "seconds", "active")

    def __init__(self):
        self.hourly_chat: deque = deque()  # monotonic times of chat admissions in the last hour
        self.daily = {CHAT: 0, TRANSCRIPTION: 0}  # admissions today (UTC)
        self.seconds = 0  # transcription seconds today (UTC)
        self.active = {CHAT: 0, TRANSCRIPTION: 0}  # admitted and not yet released

    def idle(self) -> bool:
        return (
            not self.hourly_chat
            and not any(self.daily.values())
            and self.seconds == 0
            and not any(self.active.values())
        )


class Ticket:
    """One admitted request. ``release()`` gives back its concurrency slot (once)."""

    __slots__ = ("_engine", "kind", "_identity", "reserved_seconds", "_released", "_settled")

    def __init__(self, engine: "QuotaEngine", kind: str, identity: str, reserved_seconds: int):
        self._engine = engine
        self.kind = kind
        self._identity = identity
        self.reserved_seconds = reserved_seconds
        self._released = False
        self._settled = False

    def __repr__(self) -> str:  # no identity: it is a code id
        return f"Ticket(kind={self.kind!r}, reserved_seconds={self.reserved_seconds})"

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> bool:
        """Give back the concurrency slot; True the first time, False afterwards."""
        return self._engine._release(self)

    def settle(self, actual_seconds) -> int:
        """Record a transcription's reported duration (never raises for a bad value).

        Returns the seconds charged beyond the reservation (0 on a second call).
        """
        return self._engine._settle(self, actual_seconds)

    def __enter__(self) -> "Ticket":
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()


def _check_identity(identity) -> None:
    if identity != ANONYMOUS and (not isinstance(identity, str) or not IDENTITY_PATTERN.fullmatch(identity)):
        raise ValueError("identity must be an access-code id or ANONYMOUS.")


def _reported_seconds(value) -> tuple[int | None, str | None]:
    """A reported duration as whole seconds, or (None, MALFORMED). Never raises.

    Large integers are kept exact (no float conversion, so no overflow).
    """
    try:
        if isinstance(value, bool):
            return None, MALFORMED
        if isinstance(value, int):
            return (int(value), None) if value >= 0 else (None, MALFORMED)
        if isinstance(value, float):
            if math.isfinite(value) and value >= 0:
                return math.ceil(value), None
            return None, MALFORMED
    except Exception:  # noqa: BLE001 - an exotic subclass: not trusted
        pass
    return None, MALFORMED


class QuotaEngine:
    """In-memory quotas for one process. Thread-safe; every decision is atomic."""

    def __init__(self, config: QuotaConfig | None = None, *, monotonic=None, wall=None):
        self.config = config or QuotaConfig()
        if not isinstance(self.config, QuotaConfig):
            raise QuotaConfigError("config must be a QuotaConfig.")
        self._monotonic = monotonic or (lambda: time.monotonic())  # looked up per call (a seam for tests)
        self._wall = wall or (lambda: time.time())  # read once, as the anchor
        self._lock = threading.Lock()
        self._identities: dict[str, _Usage] = {}
        self._global = _Usage()
        self._anchor: tuple[float, float] | None = None  # (wall, monotonic) at the first reading
        self._day: int | None = None  # UTC days since the epoch, of the anchored time
        self._last_monotonic = -math.inf
        self._anomalies = {MALFORMED: 0, IMPLAUSIBLE: 0, OVERRUN: 0}

    # -- clocks -----------------------------------------------------------

    def _now(self) -> tuple[float, float]:
        """(monotonic, anchored UTC time) now; rolls the day forward. Called with the lock held."""
        monotonic = float(self._monotonic())
        if not math.isfinite(monotonic):
            raise ValueError("monotonic clock returned a non-finite time")
        monotonic = max(monotonic, self._last_monotonic)  # never backwards
        if self._anchor is None:
            wall = float(self._wall())
            if not math.isfinite(wall):
                raise ValueError("wall clock returned a non-finite time")
            self._anchor = (wall, monotonic)
        self._last_monotonic = monotonic
        anchor_wall, anchor_monotonic = self._anchor
        now = anchor_wall + (monotonic - anchor_monotonic)  # wall-clock changes after the anchor don't matter
        day = math.floor(now / DAY_SECONDS)
        if self._day is None or day > self._day:  # monotonic time only moves forward
            self._start_day(day, monotonic)
        return monotonic, now

    def _start_day(self, day: int, monotonic: float) -> None:
        self._day = day
        for usage in (self._global, *self._identities.values()):
            usage.daily = {CHAT: 0, TRANSCRIPTION: 0}
            usage.seconds = 0
        self._prune(monotonic)

    def _seconds_to_next_day(self, now: float) -> int:
        return max(1, math.ceil((self._day + 1) * DAY_SECONDS - now))

    @staticmethod
    def _drop_old(times: deque, monotonic: float) -> None:
        cutoff = monotonic - HOUR_SECONDS
        while times and times[0] <= cutoff:
            times.popleft()

    @staticmethod
    def _seconds_to_free(times: deque, monotonic: float) -> int:
        return max(1, math.ceil(times[0] + HOUR_SECONDS - monotonic))

    def _prune(self, monotonic: float) -> None:
        """Forget identities with nothing left to enforce (expired timestamps dropped first)."""
        for key in list(self._identities):
            usage = self._identities[key]
            self._drop_old(usage.hourly_chat, monotonic)
            if usage.idle():
                del self._identities[key]

    def _seconds_until_capacity(self, monotonic: float, now: float) -> int:
        """The earliest time a tracked identity can be forgotten (so a new one admitted)."""
        to_next_day = self._seconds_to_next_day(now)
        earliest = None
        for usage in self._identities.values():
            wait = 0
            if any(usage.daily.values()) or usage.seconds:
                wait = to_next_day  # today's usage is kept until the day ends
            if usage.hourly_chat:
                wait = max(wait, math.ceil(usage.hourly_chat[-1] + HOUR_SECONDS - monotonic))
            if any(usage.active.values()):
                wait = max(wait, self.config.busy_retry_after_seconds)  # a release time can't be known
            earliest = wait if earliest is None else min(earliest, wait)
        return max(1, earliest if earliest is not None else 1)

    # -- admission --------------------------------------------------------

    def admit(self, kind: str, identity: str, *, seconds: int | None = None) -> Ticket | Refusal:
        """A ``Ticket`` with every limit reserved, or a ``Refusal`` that reserved nothing."""
        if kind not in KINDS:
            raise ValueError("kind must be CHAT or TRANSCRIPTION.")
        _check_identity(identity)
        if kind == CHAT:
            if seconds is not None:
                raise ValueError("seconds applies to transcription only.")
            seconds = 0
        else:
            if type(seconds) is not int or seconds < 1:
                raise ValueError("A transcription reservation needs whole seconds (at least 1).")
            if seconds > self.config.transcription.max_request_seconds:
                return self._log_refusal(Refusal(kind, REQUEST_TOO_LONG, REQUEST_SCOPE, None))

        failure = None
        with self._lock:
            try:
                result = self._admit_locked(kind, identity, seconds)
            except Exception as exc:  # noqa: BLE001 - fail closed; everything was rolled back
                failure = type(exc).__name__
                result = Refusal(kind, UNAVAILABLE, GLOBAL_SCOPE, self.config.busy_retry_after_seconds)
        # Logging happens outside the lock, so a slow handler can't hold up other requests.
        if failure is not None:
            logger.error("Quota check failed (%s); request refused.", failure)
        if isinstance(result, Refusal):
            return self._log_refusal(result)
        return result

    def _admit_locked(self, kind: str, identity: str, seconds: int) -> Ticket | Refusal:
        monotonic, now = self._now()
        usage = self._identities.get(identity)
        everyone = self._global
        to_next_day = self._seconds_to_next_day(now)

        # 1. Counted limits. If several block, report the one that frees last,
        #    so a client that waits Retry-After isn't refused again by another.
        blocked: list[tuple[int, str, str]] = []
        if kind == CHAT:
            limits = self.config.chat
            self._drop_old(everyone.hourly_chat, monotonic)
            if usage is not None:
                self._drop_old(usage.hourly_chat, monotonic)
                if len(usage.hourly_chat) >= limits.per_code_hourly:
                    blocked.append((self._seconds_to_free(usage.hourly_chat, monotonic), HOURLY_LIMIT, CODE_SCOPE))
                if usage.daily[CHAT] >= limits.per_code_daily:
                    blocked.append((to_next_day, DAILY_LIMIT, CODE_SCOPE))
            if len(everyone.hourly_chat) >= limits.global_hourly:
                blocked.append((self._seconds_to_free(everyone.hourly_chat, monotonic), HOURLY_LIMIT, GLOBAL_SCOPE))
            if everyone.daily[CHAT] >= limits.global_daily:
                blocked.append((to_next_day, DAILY_LIMIT, GLOBAL_SCOPE))
        else:
            limits = self.config.transcription
            if usage is not None:
                if usage.daily[TRANSCRIPTION] >= limits.per_code_daily:
                    blocked.append((to_next_day, DAILY_LIMIT, CODE_SCOPE))
                if usage.seconds + seconds > limits.per_code_daily_seconds:
                    blocked.append((to_next_day, MINUTES_LIMIT, CODE_SCOPE))
            if everyone.seconds + seconds > limits.global_daily_seconds:
                blocked.append((to_next_day, MINUTES_LIMIT, GLOBAL_SCOPE))
        if blocked:
            retry_after, reason, scope = max(blocked, key=lambda item: item[0])
            return Refusal(kind, reason, scope, retry_after)

        # 2. Concurrency: only worth reporting when nothing longer blocks.
        busy = self.config.busy_retry_after_seconds
        if usage is not None and usage.active[kind] >= limits.per_code_concurrent:
            return Refusal(kind, BUSY, CODE_SCOPE, busy)
        if everyone.active[kind] >= limits.global_concurrent:
            return Refusal(kind, BUSY, GLOBAL_SCOPE, busy)

        # 3. Room to track a new identity (fail closed when full).
        if usage is None and len(self._identities) >= self.config.max_identities:
            self._prune(monotonic)
            if len(self._identities) >= self.config.max_identities:
                return Refusal(kind, CAPACITY, GLOBAL_SCOPE, self._seconds_until_capacity(monotonic, now))

        # 4. Everything passed: reserve all of it together, or none of it.
        ticket = Ticket(self, kind, identity, seconds)  # created first: nothing to undo if this fails
        self._reserve(kind, identity, usage, seconds, monotonic)
        return ticket

    def _reserve(self, kind: str, identity: str, usage: "_Usage | None", seconds: int, monotonic: float) -> None:
        """Apply a reservation to the identity and to everyone; on any error, restore both exactly."""
        created = usage is None
        targets: tuple = ()
        saved: list = []
        try:
            if created:
                usage = self._identities[identity] = _Usage()
            targets = (usage, self._global)
            saved = [(target.daily[kind], target.active[kind], target.seconds, len(target.hourly_chat))
                     for target in targets]
            for target in targets:
                target.daily[kind] += 1
                target.active[kind] += 1
                target.seconds += seconds
                if kind == CHAT:
                    target.hourly_chat.append(monotonic)
        except BaseException:
            for target, (daily, active, used, length) in zip(targets, saved):
                target.daily[kind] = daily
                target.active[kind] = active
                target.seconds = used
                while len(target.hourly_chat) > length:
                    target.hourly_chat.pop()
            if created:
                self._identities.pop(identity, None)
            raise

    @staticmethod
    def _log_refusal(refusal: Refusal) -> Refusal:
        logger.info(
            "Quota refused: kind=%s reason=%s scope=%s", refusal.kind, refusal.reason, refusal.scope
        )
        return refusal

    # -- release and settlement --------------------------------------------

    def _release(self, ticket: Ticket) -> bool:
        with self._lock:
            if ticket._released:
                return False
            ticket._released = True
            usage = self._identities.get(ticket._identity)
            for target in (usage, self._global):
                if target is not None and target.active[ticket.kind] > 0:
                    target.active[ticket.kind] -= 1
            return True

    def _settle(self, ticket: Ticket, actual_seconds) -> int:
        if ticket.kind != TRANSCRIPTION:
            raise ValueError("Only a transcription can be settled.")  # a programming error, not a value
        cap = self.config.transcription.max_settle_seconds
        reported, category = _reported_seconds(actual_seconds)
        if reported is not None and reported > cap:
            category = IMPLAUSIBLE
        charged = cap if category is not None else reported  # an untrusted report is charged the cap
        extra = max(0, min(charged, cap) - ticket.reserved_seconds)  # never a refund, never above the cap
        if category is None and extra:
            category = OVERRUN

        clock_failure = None
        with self._lock:
            if ticket._settled:
                return 0
            ticket._settled = True
            if category is not None:
                self._anomalies[category] += 1
            if extra:
                try:
                    self._now()  # the current day is charged if the day rolled since admission
                except Exception as exc:  # noqa: BLE001 - still charge below
                    clock_failure = type(exc).__name__
                usage = self._identities.get(ticket._identity)
                if usage is None and len(self._identities) < self.config.max_identities:
                    usage = self._identities[ticket._identity] = _Usage()
                for target in (usage, self._global):
                    if target is not None:
                        target.seconds += extra
        if clock_failure is not None:
            logger.error("Quota clock failed during settlement (%s).", clock_failure)
        if category in (MALFORMED, IMPLAUSIBLE):
            logger.warning(
                "Transcription duration report not trusted: category=%s charged_seconds=%d", category, cap
            )
        elif category == OVERRUN:
            logger.warning("Transcription ran past its reservation: extra_seconds=%d", extra)
        return extra

    # -- inspection (tests and, later, status reporting) ---------------------

    def usage(self, identity: str | None = None) -> dict:
        """A copy of the current counters, for one identity or (None) everyone."""
        with self._lock:
            target = self._global if identity is None else self._identities.get(identity, _Usage())
            return {
                "hourly_chat": len(target.hourly_chat),
                "daily": dict(target.daily),
                "seconds": target.seconds,
                "active": dict(target.active),
            }

    def anomalies(self) -> dict:
        """How many settlements were malformed, implausible or overran, since the process started."""
        with self._lock:
            return dict(self._anomalies)

    def tracked_identities(self) -> int:
        with self._lock:
            return len(self._identities)
