"""Security S1-4c: transcription quotas, upload slots and provider-task ownership.

The provider is always faked (the real client is blocked by an autouse
fixture in tests/test_voice_transcribe.py, imported below), so nothing here
can reach a paid API.
"""

import asyncio
import gc
import json
import logging
from types import SimpleNamespace

import openai
import pytest
from pydantic import SecretStr, ValidationError

import app.main
from app.api import routes
from app.services import access, quota, quota_runtime, transcription
from tests.conftest import IsolatedSettings
from tests.streaming_support import ASGIHarness
from tests.test_quota_routes import CODE_ALICE, CODE_BOB, access_codes, quotas_on, release_calls  # noqa: F401
from tests.test_voice_transcribe import (  # noqa: F401
    MULTIPART,
    TRANSCRIPT,
    URL,
    _REQUEST,
    _status_error,
    canonical_wav,
    real_provider_blocked,
    upload,
)

GUARD = 5.0


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

class ControlledProvider:
    """A fake AsyncOpenAI whose calls can be held open, and that records what happened."""

    def __init__(self):
        self.started = 0
        self.finished = 0
        self.cancelled = 0
        self.gate: asyncio.Event | None = None
        self.started_event: asyncio.Event | None = None
        self.result = SimpleNamespace(text=TRANSCRIPT)
        self.error = None
        self.sleep = 0.0

    def __call__(self, **options):
        provider = self

        class Client:
            def __init__(self):
                self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self.create))

            async def create(self, **request):
                provider.started += 1
                if provider.started_event is not None:
                    provider.started_event.set()
                try:
                    if provider.gate is not None:
                        await provider.gate.wait()
                    if provider.sleep:
                        await asyncio.sleep(provider.sleep)
                except asyncio.CancelledError:
                    provider.cancelled += 1
                    raise
                if provider.error is not None:
                    raise provider.error
                provider.finished += 1
                return provider.result

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc_info):
                return False

        return Client()


@pytest.fixture
def provider(monkeypatch):
    fake = ControlledProvider()
    monkeypatch.setattr(transcription, "AsyncOpenAI", fake)
    return fake


@pytest.fixture
def stt_on(monkeypatch, settings):
    monkeypatch.setattr(settings, "voice_transcription_enabled", True)
    monkeypatch.setattr(settings, "openai_api_key", SecretStr("sk-test-not-a-real-key"))
    monkeypatch.setattr(settings, "voice_transcription_timeout_seconds", 30.0)
    monkeypatch.setattr(settings, "voice_transcription_max_retries", 0)
    return settings


@pytest.fixture
def stt_quotas(stt_on, quotas_on):  # noqa: F811 - quotas_on is the shared fixture
    return stt_on


def stt_limits(monkeypatch, settings, **values):
    for name, value in values.items():
        monkeypatch.setattr(settings, f"quota_transcription_{name}", value)


@pytest.fixture(autouse=True)
def no_provider_task_outlives_a_test():
    yield
    assert routes._transcription_tasks == set(), "a provider task outlived its test"
    assert routes._voice_upload_identities == set(), "an upload slot leaked"


def engine():
    return quota_runtime.peek_quota_engine()


def usage(identity=None):
    current = engine()
    if current is None:
        return {"daily": {"transcription": 0}, "seconds": 0, "active": {"transcription": 0}}
    return current.usage(identity)


def active():
    return usage()["active"]["transcription"]


def post(client, seconds=1.0, headers=None, data=None):
    body = upload(data=canonical_wav(seconds) if data is None else data)
    return client.post(URL, content=body, headers={"content-type": MULTIPART, **(headers or {})})


def assert_refusal(response, status, code, retry_after=True):
    assert response.status_code == status, response.text
    assert response.json()["detail"]["code"] == code
    assert response.headers["cache-control"] == "no-store"
    if retry_after:
        assert int(response.headers["retry-after"]) >= 1
    else:
        assert "retry-after" not in response.headers


async def harness_post(seconds=1.0, headers=None):
    harness = ASGIHarness(app.main.app)
    harness._request_body = upload(data=canonical_wav(seconds))
    result = await harness.request("POST", URL, headers={"content-type": MULTIPART, **(headers or {})})
    return result


def harness_json(result):
    return json.loads(result.body)


async def wait_for(condition, timeout=GUARD):
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.005)


# ---------------------------------------------------------------------------
# Admission, counting and the measured reservation
# ---------------------------------------------------------------------------

def test_an_admitted_recording_reserves_its_measured_seconds(client, stt_quotas, provider):
    response = post(client, seconds=1.5)
    assert response.status_code == 200 and response.json() == {"text": TRANSCRIPT}
    assert usage(quota.ANONYMOUS)["daily"]["transcription"] == 1
    assert usage(quota.ANONYMOUS)["seconds"] == 2  # ceil(1.5)
    assert active() == 0


@pytest.mark.parametrize(
    "data, status, code",
    [
        (canonical_wav(60.0 + 1 / 16000), 413, "audio_too_long"),
        (canonical_wav(1.0)[:-2], 400, "invalid_audio"),
        (b"RIFF" + b"\x00" * 60, 415, "unsupported_audio"),
    ],
)
def test_rejected_audio_never_reaches_the_provider_or_the_quota(client, stt_quotas, provider, data, status, code):
    assert_refusal(post(client, data=data), status, code, retry_after=False)
    assert provider.started == 0
    assert usage()["daily"]["transcription"] == 0 and usage()["seconds"] == 0


def test_exactly_60_seconds_is_admitted(client, stt_quotas, provider):
    assert post(client, seconds=60.0).status_code == 200
    assert usage(quota.ANONYMOUS)["seconds"] == 60


def test_the_daily_recording_count_is_enforced(client, stt_quotas, provider, monkeypatch):
    stt_limits(monkeypatch, stt_quotas, per_code_daily=3)
    for _ in range(3):
        assert post(client, seconds=1).status_code == 200
    response = post(client, seconds=1)
    assert_refusal(response, 429, "quota_daily_limit")
    assert response.json()["detail"]["message"] == "You've reached today's recording limit. Please try again tomorrow."
    assert provider.started == 3


def test_the_per_code_audio_budget_is_enforced(client, stt_quotas, provider, monkeypatch):
    stt_limits(monkeypatch, stt_quotas, per_code_daily_seconds=60)
    assert post(client, seconds=59).status_code == 200  # N-1
    assert post(client, seconds=1).status_code == 200  # N: 60 s
    assert_refusal(post(client, seconds=0.5), 429, "quota_minutes_limit")  # N+1
    assert provider.started == 2 and usage(quota.ANONYMOUS)["seconds"] == 60


def test_the_global_audio_budget_spans_codes(client, stt_quotas, provider, access_codes, monkeypatch):  # noqa: F811
    stt_limits(monkeypatch, stt_quotas, per_code_daily_seconds=60, global_daily_seconds=60)
    assert post(client, seconds=40, headers=access_codes["alice"]).status_code == 200
    assert post(client, seconds=20, headers=access_codes["bob"]).status_code == 200
    response = post(client, seconds=1)  # anonymous: its own per-code budget, but the global one is used up
    assert_refusal(response, 429, "quota_minutes_limit")
    assert "The service has used" in response.json()["detail"]["message"]


def test_verified_and_anonymous_identities_are_separate(client, stt_quotas, provider, access_codes, monkeypatch):  # noqa: F811
    stt_limits(monkeypatch, stt_quotas, per_code_daily=1)
    assert post(client, headers=access_codes["alice"]).status_code == 200
    assert_refusal(post(client, headers=access_codes["alice"]), 429, "quota_daily_limit")
    assert post(client, headers=access_codes["bob"]).status_code == 200
    assert post(client).status_code == 200  # anonymous
    forged = {"cookie": access_codes["alice"]["cookie"][:-2] + "AA"}
    assert_refusal(post(client, headers=forged), 429, "quota_daily_limit")  # a forged cookie is anonymous
    assert usage("alice")["daily"]["transcription"] == 1
    assert usage(quota.ANONYMOUS)["daily"]["transcription"] == 1


def test_with_quotas_off_transcription_still_measures_and_creates_no_engine(client, stt_on, provider):
    assert post(client).status_code == 200
    assert_refusal(post(client, data=canonical_wav(61)), 413, "audio_too_long", retry_after=False)
    assert engine() is None


# ---------------------------------------------------------------------------
# Settlement: the provider's duration, never a refund
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "reported, charged, anomaly",
    [
        (None, 2, None),  # missing: the measurement settles (adds nothing)
        (1.5, 2, None),  # as measured
        (0.2, 2, None),  # less: never a refund
        (7.0, 7, "overrun"),  # more: charged in full
        (1e9, 300, "implausible"),  # charged the cap (the per-code budget)
        ("abc", 300, "malformed"),
        (float("nan"), 300, "malformed"),
    ],
)
def test_settlement_uses_the_providers_duration_conservatively(client, stt_quotas, provider, reported, charged, anomaly):
    provider.result = SimpleNamespace(text=TRANSCRIPT) if reported is None else SimpleNamespace(text=TRANSCRIPT, duration=reported)
    assert post(client, seconds=1.5).status_code == 200
    assert usage(quota.ANONYMOUS)["seconds"] == charged
    anomalies = engine().anomalies()
    assert anomalies == {key: (1 if key == anomaly else 0) for key in anomalies}


@pytest.mark.parametrize(
    "error, status, code",
    [
        (openai.APITimeoutError(request=_REQUEST), 504, "transcription_timeout"),
        (_status_error(openai.RateLimitError, 429), 503, "transcription_unavailable"),
        (_status_error(openai.BadRequestError, 400), 422, "audio_not_transcribable"),
        (ValueError("provider-internal"), 500, "transcription_failed"),
    ],
)
def test_provider_failures_keep_the_reservation_and_release_once(client, stt_quotas, provider, release_calls, error, status, code):  # noqa: F811
    provider.error = error
    response = post(client, seconds=3)
    assert response.status_code == status and response.json()["detail"]["code"] == code
    assert usage(quota.ANONYMOUS)["seconds"] == 3 and usage(quota.ANONYMOUS)["daily"]["transcription"] == 1
    assert active() == 0 and len(release_calls) == 1


def test_a_settlement_failure_still_releases(client, stt_quotas, provider, monkeypatch, release_calls, caplog):  # noqa: F811
    def broken(self, actual):
        raise RuntimeError("settle failed")

    monkeypatch.setattr(quota.Ticket, "settle", broken)
    assert post(client).status_code == 200
    assert active() == 0 and len(release_calls) == 1
    assert "Transcription quota settlement failed (RuntimeError)." in caplog.text


# ---------------------------------------------------------------------------
# Concurrency: quota slots and upload slots
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_one_transcription_at_a_time(stt_quotas, provider, access_codes):  # noqa: F811
    # Two different codes (each with its own upload slot): the quota's global
    # limit of one transcription refuses the second.
    provider.gate, provider.started_event = asyncio.Event(), asyncio.Event()
    first = asyncio.create_task(harness_post(headers=access_codes["alice"]))
    await asyncio.wait_for(provider.started_event.wait(), GUARD)
    second = await harness_post(headers=access_codes["bob"])
    assert second.status == 429 and harness_json(second)["detail"]["code"] == "quota_busy"
    assert second.headers["retry-after"] and second.headers["cache-control"] == "no-store"
    # The same identity again is refused earlier, at its upload slot.
    again = await harness_post(headers=access_codes["alice"])
    assert again.status == 503 and harness_json(again)["detail"]["code"] == "voice_busy"
    assert provider.started == 1
    provider.gate.set()
    assert (await first).status == 200
    await wait_for(lambda: active() == 0)


@pytest.mark.anyio
async def test_a_third_identity_is_voice_busy_before_reading(stt_on, provider, access_codes, monkeypatch):  # noqa: F811
    reads = []
    real = transcription.read_bounded_body

    async def spy(stream):
        reads.append(1)
        return await real(stream)

    monkeypatch.setattr(transcription, "read_bounded_body", spy)
    provider.gate = asyncio.Event()
    held = [asyncio.create_task(harness_post(headers=access_codes[name])) for name in ("alice", "bob")]
    await wait_for(lambda: provider.started == 2)  # two different codes use both slots
    third = await harness_post()  # anonymous: a third identity
    assert third.status == 503
    assert harness_json(third)["detail"]["code"] == "voice_busy"
    assert third.headers["retry-after"] == "5" and third.headers["cache-control"] == "no-store"
    assert len(reads) == 2  # the third body was never read
    provider.gate.set()
    for task in held:
        assert (await task).status == 200
    assert routes._voice_upload_identities == set()


# ---------------------------------------------------------------------------
# Ownership: disconnects, cancellation, hard timeout, strong references
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_a_client_disconnect_keeps_the_ticket_until_the_provider_finishes(stt_quotas, provider):
    provider.gate, provider.started_event = asyncio.Event(), asyncio.Event()
    harness = ASGIHarness(app.main.app)
    harness._request_body = upload(data=canonical_wav(1.0))
    request = asyncio.create_task(harness.request("POST", URL, headers={"content-type": MULTIPART}))
    await asyncio.wait_for(provider.started_event.wait(), GUARD)
    harness.disconnect()  # the client goes away mid-transcription
    await asyncio.sleep(0.05)
    assert active() == 1 and provider.finished == 0  # still in flight: still held
    provider.gate.set()
    await request
    await wait_for(lambda: active() == 0)
    assert provider.finished == 1 and provider.cancelled == 0  # the call was never interrupted


@pytest.mark.anyio
async def test_a_cancelled_handler_leaves_the_task_owning_the_ticket(stt_quotas, provider, release_calls):  # noqa: F811
    provider.gate, provider.started_event = asyncio.Event(), asyncio.Event()
    request = asyncio.create_task(harness_post())
    await asyncio.wait_for(provider.started_event.wait(), GUARD)
    request.cancel()  # as at shutdown
    with pytest.raises(asyncio.CancelledError):
        await request
    [task] = list(routes._transcription_tasks)
    assert not task.done() and active() == 1 and release_calls == []  # never released while in flight
    gc.collect()
    assert task in routes._transcription_tasks and not task.done()  # a strong reference keeps it alive
    provider.gate.set()
    await wait_for(lambda: active() == 0)
    assert provider.finished == 1 and provider.cancelled == 0
    assert len(release_calls) == 1  # exactly once, from the task's completion
    assert usage(quota.ANONYMOUS)["daily"]["transcription"] == 1  # counted, settled, not refunded


@pytest.mark.anyio
async def test_the_hard_timeout_bounds_the_provider_and_then_releases(stt_quotas, provider, monkeypatch, release_calls):  # noqa: F811
    monkeypatch.setattr(stt_quotas, "voice_transcription_timeout_seconds", 1.0)
    monkeypatch.setattr(routes, "PROVIDER_HARD_TIMEOUT_MARGIN_SECONDS", -0.95)  # a hard bound of 0.05 s
    provider.sleep = 10.0
    result = await harness_post(seconds=2)
    assert result.status == 504 and harness_json(result)["detail"]["code"] == "transcription_timeout"
    assert provider.cancelled == 1 and provider.finished == 0
    await wait_for(lambda: active() == 0)
    assert len(release_calls) == 1
    assert usage(quota.ANONYMOUS)["seconds"] == 2  # the reservation stands


@pytest.mark.anyio
async def test_a_task_is_referenced_while_running_and_forgotten_after(stt_quotas, provider):
    provider.gate, provider.started_event = asyncio.Event(), asyncio.Event()
    request = asyncio.create_task(harness_post())
    await asyncio.wait_for(provider.started_event.wait(), GUARD)
    assert len(routes._transcription_tasks) == 1
    provider.gate.set()
    assert (await request).status == 200
    await wait_for(lambda: routes._transcription_tasks == set())


def test_a_failure_before_the_hand_off_is_released_by_the_handler(client, stt_quotas, provider, monkeypatch, release_calls):  # noqa: F811
    def broken(ticket):
        raise RuntimeError("callback could not be built")

    monkeypatch.setattr(routes, "_transcription_finished", broken)
    response = post(client)
    assert response.status_code == 500 and response.json()["detail"]["code"] == "transcription_failed"
    assert provider.started == 0  # the task never ran a step
    assert active() == 0 and len(release_calls) == 1
    assert routes._transcription_tasks == set()


def test_a_failure_registering_the_task_cancels_it_before_it_runs(client, stt_quotas, provider, monkeypatch, release_calls):  # noqa: F811
    class Refusing(set):
        def add(self, item):
            raise RuntimeError("cannot track")

    monkeypatch.setattr(routes, "_transcription_tasks", Refusing())
    response = post(client)
    assert response.status_code == 500
    assert provider.started == 0 and active() == 0 and len(release_calls) == 1
    monkeypatch.undo()


# ---------------------------------------------------------------------------
# Configuration: startup matrix and runtime checks
# ---------------------------------------------------------------------------

QUOTA_VALID = dict(quota_enabled=True, chat_deadline_seconds=90, stream_timeout_seconds=90,
                   session_turn_max_seconds=105, openai_timeout_seconds=30, openai_max_retries=1,
                   pinecone_retry_after_max_seconds=5)
ACCESS_VALID = dict(access_enabled=True, access_code_hashes=f"alice:{access.hash_code(CODE_ALICE)}",
                    access_cookie_secret="Q7v-test-secret-9fK2pX4mZ8rT1yW6nB3cJ5hL0dSxE4uA",
                    access_trust_socket_peer=True)


@pytest.mark.parametrize(
    "values, ok",
    [
        (dict(app_env="development", voice_transcription_enabled=True), True),
        (dict(app_env="testing", voice_transcription_enabled=True), True),
        (dict(app_env="production", voice_transcription_enabled=False), True),
        (dict(app_env="production", voice_transcription_enabled=True), False),
        (dict(app_env="production", voice_transcription_enabled=True, **QUOTA_VALID), False),
        (dict(app_env="production", voice_transcription_enabled=True, **ACCESS_VALID), False),
        (dict(app_env="production", voice_transcription_enabled=True, **QUOTA_VALID, **ACCESS_VALID), False),
    ],
)
def test_production_transcription_cannot_start_before_s1_5(values, ok):
    if ok:
        IsolatedSettings(**values)
    else:
        with pytest.raises(ValidationError, match="VOICE_TRANSCRIPTION_ENABLED"):
            IsolatedSettings(**values)


def test_quotas_and_access_together_still_do_not_allow_production_transcription():
    # ACCESS_ENABLED doesn't enforce verified access on the route (S1-5), so
    # even the complete configuration is refused, with a message saying why.
    with pytest.raises(ValidationError, match="not available in production until verified access enforcement"):
        IsolatedSettings(app_env="production", voice_transcription_enabled=True, **QUOTA_VALID, **ACCESS_VALID)


def test_there_is_no_setting_that_bypasses_the_production_guard():
    from app.core import config

    names = set(config.Settings.model_fields)
    assert not [name for name in names if "s1_5" in name or ("bypass" in name) or name.endswith("access_required")]
    settings = IsolatedSettings(app_env="production", **QUOTA_VALID, **ACCESS_VALID)  # transcription off: fine
    with pytest.raises(quota.QuotaConfigError, match="S1-5"):
        quota_runtime.check_transcription_deployment(settings.model_copy(update={"voice_transcription_enabled": True}))


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(voice_transcription_timeout_seconds=31), "VOICE_TRANSCRIPTION_TIMEOUT_SECONDS"),
        (dict(voice_transcription_max_retries=1), "VOICE_TRANSCRIPTION_MAX_RETRIES=0"),
        (dict(quota_transcription_per_code_daily_seconds=2000), "per_code_daily_seconds <= global_daily_seconds"),
    ],
)
def test_unsafe_transcription_quota_settings_refuse_to_start(overrides, message):
    with pytest.raises(ValidationError, match=message):
        IsolatedSettings(**QUOTA_VALID, voice_transcription_enabled=True, **overrides)


def test_transcription_checks_apply_only_with_transcription_on():
    assert IsolatedSettings(**QUOTA_VALID, voice_transcription_max_retries=2).quota_enabled


def test_the_engine_gets_the_approved_transcription_limits():
    limits = quota_runtime.config_from_settings(IsolatedSettings(**QUOTA_VALID)).transcription
    assert (limits.per_code_daily, limits.per_code_daily_seconds, limits.global_daily_seconds) == (10, 300, 1800)
    assert (limits.per_code_concurrent, limits.global_concurrent) == (1, 1)
    assert (limits.max_request_seconds, limits.max_settle_seconds) == (60, 300)


@pytest.mark.parametrize("complete", [False, True])
def test_production_is_rechecked_per_request_before_reading(client, stt_quotas, provider, access_codes, monkeypatch, caplog, complete):  # noqa: F811
    reads = []
    monkeypatch.setattr(transcription, "read_bounded_body", lambda stream: reads.append(1))
    if not complete:
        monkeypatch.setattr(stt_quotas, "quota_enabled", False)
    monkeypatch.setattr(stt_quotas, "app_env", "production")  # changed after startup
    for headers in ({}, access_codes["alice"]):  # anonymous or verified: refused before S1-5
        assert_refusal(post(client, headers=headers), 503, "transcription_unavailable", retry_after=False)
    assert reads == [] and provider.started == 0 and engine() is None
    assert "production guard" in caplog.text


def test_unsafe_runtime_settings_fail_closed_when_the_engine_is_created(client, stt_quotas, provider, monkeypatch):
    monkeypatch.setattr(stt_quotas, "voice_transcription_max_retries", 1)
    assert_refusal(post(client), 503, "quota_unavailable")
    assert provider.started == 0 and engine() is None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def test_no_audio_transcript_or_identity_is_logged(client, stt_quotas, provider, access_codes, monkeypatch, caplog):  # noqa: F811
    caplog.set_level(logging.DEBUG)
    stt_limits(monkeypatch, stt_quotas, per_code_daily=1)
    audio = canonical_wav(1.0, sample=b"\x5a\x5a")
    assert post(client, headers=access_codes["alice"], data=audio).status_code == 200
    refused = post(client, headers=access_codes["alice"], data=audio)
    assert refused.status_code == 429
    text = caplog.text + refused.text
    for secret in (TRANSCRIPT, "ZZZZ", repr(audio[44:52]), CODE_ALICE, "alice", access_codes["alice"]["cookie"]):
        assert secret not in text, secret[:12]
    assert "Transcription refused by quota for request_id=" in caplog.text and "code_ref=" in caplog.text


def test_the_upload_limits_are_as_approved():
    assert transcription.UPLOAD_READ_TIMEOUT_SECONDS == 30.0  # about 1.92 MB at under 1 Mbit/s
    assert transcription.MAX_AUDIO_BYTES == 2 * 1024 * 1024  # the 2 MiB limit is unchanged
    assert routes.VOICE_UPLOAD_SLOTS == 2
    assert list(transcription.FORMATS_BY_EXTENSION) == [".wav"]  # WAV only



# ---------------------------------------------------------------------------
# S1-4c-1 remediation: upload slots per identity (stalled-upload exhaustion)
# ---------------------------------------------------------------------------

BOUNDARY_RAW = MULTIPART.split("boundary=", 1)[1]


async def raw_request(headers=None, *, stall=False):
    """Call the app directly; with ``stall``, the body never arrives (a stalled upload)."""
    body = upload(data=canonical_wav(1.0))
    sent, given = [], {"done": False}

    async def receive():
        if stall or given["done"]:
            await asyncio.sleep(3600)
        given["done"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        sent.append(message)

    raw_headers = [(b"host", b"testserver"), (b"content-type", MULTIPART.encode()),
                   (b"content-length", str(len(body)).encode())]
    raw_headers += [(key.lower().encode(), value.encode()) for key, value in (headers or {}).items()]
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
             "scheme": "http", "path": URL, "raw_path": URL.encode(), "query_string": b"", "root_path": "",
             "headers": raw_headers, "client": ("127.0.0.1", 4321), "server": ("testserver", 80), "state": {}}
    await app.main.app(scope, receive, send)
    start = next(message for message in sent if message["type"] == "http.response.start")
    payload = b"".join(message.get("body", b"") for message in sent if message["type"] == "http.response.body")
    return start["status"], json.loads(payload) if payload else None


def held_identities():
    return set(routes._voice_upload_identities)


async def stall(headers=None):
    task = asyncio.create_task(raw_request(headers, stall=True))
    await asyncio.sleep(0.02)
    return task


async def end(*tasks):
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.anyio
async def test_two_stalled_anonymous_uploads_cannot_hold_both_slots(stt_quotas, provider):
    first = await stall()
    status, body = await asyncio.wait_for(raw_request(), GUARD)
    assert (status, body["detail"]["code"]) == (503, "voice_busy")  # the second anonymous upload is refused
    assert held_identities() == {quota.ANONYMOUS}  # one slot held, one free
    await end(first)


@pytest.mark.anyio
async def test_a_stalled_anonymous_upload_leaves_room_for_a_verified_code(stt_quotas, provider, access_codes):  # noqa: F811
    attacker = await stall()
    status, body = await asyncio.wait_for(raw_request(access_codes["alice"]), GUARD)
    assert status == 200 and body == {"text": TRANSCRIPT}
    assert provider.started == 1
    await end(attacker)


@pytest.mark.anyio
async def test_one_verified_code_cannot_hold_two_slots(stt_quotas, provider, access_codes):  # noqa: F811
    first = await stall(access_codes["alice"])
    status, body = await asyncio.wait_for(raw_request(access_codes["alice"]), GUARD)
    assert (status, body["detail"]["code"]) == (503, "voice_busy")
    assert held_identities() == {"alice"}
    status, _ = await asyncio.wait_for(raw_request(access_codes["bob"]), GUARD)  # another code still can
    assert status == 200
    await end(first)


@pytest.mark.anyio
async def test_two_codes_use_both_slots_and_a_third_identity_is_busy(stt_quotas, provider, access_codes):  # noqa: F811
    stalls = [await stall(access_codes["alice"]), await stall(access_codes["bob"])]
    assert held_identities() == {"alice", "bob"}
    status, body = await asyncio.wait_for(raw_request(), GUARD)
    assert (status, body["detail"]["code"]) == (503, "voice_busy")
    await end(*stalls)


@pytest.mark.anyio
async def test_forged_identities_share_the_anonymous_slot(stt_quotas, provider, access_codes):  # noqa: F811
    first = await stall()
    forged = {"cookie": access_codes["alice"]["cookie"][:-2] + "AA"}
    for headers in (forged, {"x-quota-identity": "alice"}, {"x-forwarded-for": "203.0.113.9"}):
        status, body = await asyncio.wait_for(raw_request(headers), GUARD)
        assert (status, body["detail"]["code"]) == (503, "voice_busy"), headers
    assert held_identities() == {quota.ANONYMOUS}
    await end(first)


@pytest.mark.anyio
async def test_slots_recover_after_a_read_timeout(stt_quotas, provider, monkeypatch):
    monkeypatch.setattr(transcription, "UPLOAD_READ_TIMEOUT_SECONDS", 0.05)
    status, body = await asyncio.wait_for(raw_request(stall=True), GUARD)
    assert (status, body["detail"]["code"]) == (408, "upload_timeout")
    assert held_identities() == set()
    monkeypatch.setattr(transcription, "UPLOAD_READ_TIMEOUT_SECONDS", 30.0)
    assert (await asyncio.wait_for(raw_request(), GUARD))[0] == 200


@pytest.mark.anyio
async def test_slots_recover_after_cancellation(stt_quotas, provider):
    stalled = await stall()
    assert held_identities() == {quota.ANONYMOUS}
    await end(stalled)
    assert held_identities() == set()
    assert (await asyncio.wait_for(raw_request(), GUARD))[0] == 200


@pytest.mark.anyio
async def test_refused_uploads_use_no_quota_and_call_no_provider(stt_quotas, provider, access_codes):  # noqa: F811
    stalls = [await stall(access_codes["alice"]), await stall(access_codes["bob"])]
    for _ in range(5):
        assert (await asyncio.wait_for(raw_request(), GUARD))[0] == 503
    assert provider.started == 0
    assert usage()["daily"]["transcription"] == 0 and usage()["seconds"] == 0
    await end(*stalls)


@pytest.mark.anyio
async def test_50_concurrent_uploads_never_exceed_one_per_identity_or_two_in_all(stt_on, provider, access_codes):  # noqa: F811
    provider.gate = asyncio.Event()
    peak = {"total": 0}
    seen_duplicate = []

    async def watch():
        while True:
            held = list(routes._voice_upload_identities)
            peak["total"] = max(peak["total"], len(held))
            if len(held) != len(set(held)):
                seen_duplicate.append(held)
            await asyncio.sleep(0)

    watcher = asyncio.create_task(watch())
    identities = [{}, access_codes["alice"], access_codes["bob"]]
    tasks = [asyncio.create_task(raw_request(identities[index % 3])) for index in range(50)]
    await wait_for(lambda: provider.started == 2)
    await asyncio.sleep(0.05)
    assert provider.started == 2 and len(held_identities()) == 2
    provider.gate.set()
    results = await asyncio.gather(*tasks)
    watcher.cancel()
    statuses = [status for status, _ in results]
    assert statuses.count(200) == 2 and statuses.count(503) == 48
    assert all(body["detail"]["code"] == "voice_busy" for status, body in results if status == 503)
    assert peak["total"] <= 2 and seen_duplicate == []
    assert held_identities() == set()


@pytest.mark.anyio
async def test_slot_refusals_log_no_identity(stt_quotas, provider, access_codes, caplog):  # noqa: F811
    caplog.set_level(logging.DEBUG)
    first = await stall(access_codes["alice"])
    await asyncio.wait_for(raw_request(access_codes["alice"]), GUARD)
    await end(first)
    for secret in ("alice", access_codes["alice"]["cookie"], CODE_ALICE):
        assert secret not in caplog.text
