"""The process-wide chat quota engine (Security S1-4b).

* ``check_settings(settings)`` validates the quota limits and the deadlines
  that must be bounded while quotas are on; the settings call it at startup
  when ``QUOTA_ENABLED`` is true, so an invalid configuration refuses to
  start.
* ``get_quota_engine(settings)`` returns the one ``QuotaEngine`` of this
  process, created from the settings on first use and never rebuilt: its
  active tickets and counters must survive for the life of the process.
  Creating it re-runs ``check_settings`` (settings changed after startup
  must not weaken it); unsafe settings raise and no engine is created, so
  the caller fails closed. Limits changed later take effect only after a
  restart. It is never called while quotas are disabled, so no engine
  exists then.

Why the deadlines: a deadline stops a question before its *next* external
call. A provider call already in progress is not interrupted (a Python
thread can't be cancelled, and that wouldn't stop the HTTP request anyway),
so what one admitted question can still cost after its deadline is bounded
by the provider timeouts and retries, which are capped here too: OpenAI's
timeout and retries, and Pinecone's longest honoured ``Retry-After``
(without a cap the Pinecone SDK waits up to 6 hours per retry).
"""

import threading

from app.services.audio_duration import MAX_DURATION_SECONDS
from app.services.quota import ChatLimits, QuotaConfig, QuotaConfigError, QuotaEngine, TranscriptionLimits

MAX_DEADLINE_SECONDS = 90.0  # CHAT_DEADLINE_SECONDS and STREAM_TIMEOUT_SECONDS
MAX_OPENAI_TIMEOUT_SECONDS = 30.0
MAX_OPENAI_RETRIES = 1
MAX_PINECONE_RETRY_AFTER_SECONDS = 5  # per retry; the SDK's 5 retries and backoff are unchanged
MAX_TRANSCRIPTION_TIMEOUT_SECONDS = 30.0  # S1-4c: with quotas and transcription on
MAX_TRANSCRIPTION_RETRIES = 0


def config_from_settings(settings) -> QuotaConfig:
    """The engine configuration from the settings; raises ``QuotaConfigError``."""
    return QuotaConfig(
        chat=ChatLimits(
            per_code_hourly=settings.quota_chat_per_code_hourly,
            per_code_daily=settings.quota_chat_per_code_daily,
            global_hourly=settings.quota_chat_global_hourly,
            global_daily=settings.quota_chat_global_daily,
            per_code_concurrent=settings.quota_chat_per_code_concurrent,
            global_concurrent=settings.quota_chat_global_concurrent,
        ),
        transcription=TranscriptionLimits(
            per_code_daily=settings.quota_transcription_per_code_daily,
            per_code_daily_seconds=settings.quota_transcription_per_code_daily_seconds,
            global_daily_seconds=settings.quota_transcription_global_daily_seconds,
            per_code_concurrent=settings.quota_transcription_per_code_concurrent,
            global_concurrent=settings.quota_transcription_global_concurrent,
            max_request_seconds=MAX_DURATION_SECONDS,  # the measured-duration limit (S1-4c)
            # The most one recording can be charged: the per-code daily budget
            # (see app/services/quota.py), so one bad provider report refuses
            # only that code for the day.
            max_settle_seconds=settings.quota_transcription_per_code_daily_seconds,
        ),
        max_identities=settings.quota_max_identities,
        busy_retry_after_seconds=settings.quota_busy_retry_after_seconds,
    )


def check_settings(settings) -> QuotaConfig:
    """Everything quotas need to be safe; raises ``QuotaConfigError``."""
    from app.core.config import _STREAM_FINALIZE_MARGIN_SECONDS

    config = config_from_settings(settings)
    deadline = settings.chat_deadline_seconds
    if deadline is None or not 0 < deadline <= MAX_DEADLINE_SECONDS:
        raise QuotaConfigError(
            f"QUOTA_ENABLED requires CHAT_DEADLINE_SECONDS from 1 to {MAX_DEADLINE_SECONDS:g}."
        )
    if not 0 < settings.stream_timeout_seconds <= MAX_DEADLINE_SECONDS:
        raise QuotaConfigError(
            f"QUOTA_ENABLED requires STREAM_TIMEOUT_SECONDS of at most {MAX_DEADLINE_SECONDS:g}."
        )
    if not 0 < settings.openai_timeout_seconds <= MAX_OPENAI_TIMEOUT_SECONDS:
        raise QuotaConfigError(
            f"QUOTA_ENABLED requires OPENAI_TIMEOUT_SECONDS of at most {MAX_OPENAI_TIMEOUT_SECONDS:g}."
        )
    if not 0 <= settings.openai_max_retries <= MAX_OPENAI_RETRIES:
        raise QuotaConfigError(f"QUOTA_ENABLED requires OPENAI_MAX_RETRIES of at most {MAX_OPENAI_RETRIES}.")
    retry_after = settings.pinecone_retry_after_max_seconds
    if type(retry_after) is not int or not 0 <= retry_after <= MAX_PINECONE_RETRY_AFTER_SECONDS:
        raise QuotaConfigError(
            "QUOTA_ENABLED requires PINECONE_RETRY_AFTER_MAX_SECONDS from 0 to "
            f"{MAX_PINECONE_RETRY_AFTER_SECONDS} (unset, Pinecone may wait hours per retry)."
        )
    # A session question's deadline is its turn limit, which must already be at
    # least the stream timeout plus the finalize margin; bound it from above too.
    longest_turn = MAX_DEADLINE_SECONDS + _STREAM_FINALIZE_MARGIN_SECONDS
    if settings.session_turn_max_seconds > longest_turn:
        raise QuotaConfigError(
            f"QUOTA_ENABLED requires SESSION_TURN_MAX_SECONDS of at most {longest_turn:g}."
        )
    if settings.voice_transcription_enabled:  # S1-4c
        if not 0 < settings.voice_transcription_timeout_seconds <= MAX_TRANSCRIPTION_TIMEOUT_SECONDS:
            raise QuotaConfigError(
                "QUOTA_ENABLED with transcription requires VOICE_TRANSCRIPTION_TIMEOUT_SECONDS "
                f"of at most {MAX_TRANSCRIPTION_TIMEOUT_SECONDS:g}."
            )
        if settings.voice_transcription_max_retries != MAX_TRANSCRIPTION_RETRIES:
            raise QuotaConfigError(
                "QUOTA_ENABLED with transcription requires VOICE_TRANSCRIPTION_MAX_RETRIES=0."
            )
    return config


def check_transcription_deployment(settings) -> None:
    """Production transcription is refused (S1-4c); raises ``QuotaConfigError``.

    ACCESS_ENABLED does not yet make the transcription route require a
    verified access code (that is S1-5), so in production anonymous users
    could still transcribe. Until S1-5 enforces verified access on this
    route, production transcription never starts, whatever the settings:
    there is deliberately no setting to bypass this. S1-5 must replace the
    final ``raise`` below with a check that verified access is enforced.

    Development and testing may enable transcription alone, so local work
    (everything disabled by default) is unaffected.
    """
    if not settings.voice_transcription_enabled or settings.app_env != "production":
        return
    if not (settings.quota_enabled and settings.access_enabled):
        raise QuotaConfigError(
            "VOICE_TRANSCRIPTION_ENABLED in production requires QUOTA_ENABLED=true and ACCESS_ENABLED=true."
        )
    raise QuotaConfigError(
        "VOICE_TRANSCRIPTION_ENABLED is not available in production until verified access "
        "enforcement (S1-5) is implemented."
    )


_engine: QuotaEngine | None = None
_engine_lock = threading.Lock()


def get_quota_engine(settings) -> QuotaEngine:
    """The one engine of this process (created on first use, never rebuilt).

    Raises ``QuotaConfigError`` (and creates nothing) if the settings are
    unsafe when the engine would be created.
    """
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = QuotaEngine(check_settings(settings))  # the complete safety check again
        return _engine


def peek_quota_engine() -> QuotaEngine | None:
    """The engine if one was created; never creates one."""
    with _engine_lock:
        return _engine


def reset_quota_engine() -> None:
    """Forget the engine. Tests only: a running process must keep its engine."""
    global _engine
    with _engine_lock:
        _engine = None
