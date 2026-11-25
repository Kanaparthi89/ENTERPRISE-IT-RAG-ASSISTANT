"""Voice V2: POST /api/voice/transcribe (no network; the provider is faked).

The real OpenAI client is blocked in this module by an autouse fixture, so a
test can only reach the fake provider installed by ``provider``.
"""

import logging
import struct
from types import SimpleNamespace

import anyio
import httpx2
import openai
import pytest
from pydantic import SecretStr, ValidationError

from app.api import routes, streaming
from app.core import config
from app.services import transcription
from app.services.audit import get_recent_audits
from tests.conftest import IsolatedSettings

URL = "/api/voice/transcribe"
NOT_FOUND = {"detail": "Not Found"}
BOUNDARY = "BluePeakTestBoundary7MA4YWxk"
MULTIPART = f"multipart/form-data; boundary={BOUNDARY}"
TRANSCRIPT = "SECRET-TRANSCRIPT-how do I reset my VPN token"
SECRET_FILENAME = "patient-jane-doe-recording.wav"
PROVIDER_DETAIL = "provider-internal-detail-must-not-leak"

def canonical_wav(seconds=1.0, sample=b"\x01\x00"):
    """A canonical 16 kHz mono 16-bit PCM WAV, the only accepted format (S1-4c)."""
    data = sample * int(16000 * seconds)
    return (
        b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
        + b"data" + struct.pack("<I", len(data)) + data
    )


AUDIO = canonical_wav(1.0)  # 32,044 bytes

# Real container signatures of formats that are now refused (S1-4c), padded.
WEBM = b"\x1a\x45\xdf\xa3" + b"\x9f" * 60
OGG = b"OggS" + b"\x00" * 60
MP4 = b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 52
MP3_ID3 = b"ID3\x04\x00\x00" + b"\x00" * 58
MP3_FRAME = b"\xff\xfb\x90\x64" + b"\x00" * 60
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 48


def multipart(*parts, boundary=BOUNDARY, close=True) -> bytes:
    """Build a multipart body from (headers_text, data) pairs."""
    body = b""
    for headers, data in parts:
        body += f"--{boundary}\r\n{headers}\r\n\r\n".encode("latin-1") + data + b"\r\n"
    if close:
        body += f"--{boundary}--\r\n".encode()
    return body


def audio_part(data=AUDIO, filename="clip.wav", content_type="audio/wav", name="audio"):
    headers = f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\nContent-Type: {content_type}'
    return (headers, data)


def upload(data=AUDIO, filename="clip.wav", content_type="audio/wav"):
    return multipart(audio_part(data, filename, content_type))


# ---------------------------------------------------------------------------
# Fakes and fixtures
# ---------------------------------------------------------------------------

class FakeProvider:
    """Stands in for openai.AsyncOpenAI: records the client options and the request."""

    def __init__(self, result=None, error=None):
        self.result = SimpleNamespace(text=TRANSCRIPT) if result is None else result
        self.error = error
        self.clients = []
        self.requests = []

    def __call__(self, **options):
        provider = self

        class Client:
            def __init__(self):
                self.options = options
                self.closed = False
                self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self.create))

            async def create(self, **request):
                provider.requests.append(request)
                if provider.error is not None:
                    raise provider.error
                return provider.result

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc_info):
                self.closed = True

        client = Client()
        self.clients.append(client)
        return client


@pytest.fixture(autouse=True)
def real_provider_blocked(monkeypatch):
    def blocked(**options):
        raise AssertionError("the real OpenAI client must never be constructed in tests")

    monkeypatch.setattr(transcription, "AsyncOpenAI", blocked)


@pytest.fixture
def provider(monkeypatch):
    def install(**kwargs):
        fake = FakeProvider(**kwargs)
        monkeypatch.setattr(transcription, "AsyncOpenAI", fake)
        return fake

    return install


@pytest.fixture
def stt_on(monkeypatch, settings):
    monkeypatch.setattr(settings, "voice_transcription_enabled", True)
    monkeypatch.setattr(settings, "openai_api_key", SecretStr("sk-test-not-a-real-key"))
    return settings


@pytest.fixture
def body_reads(monkeypatch):
    """Counts reads of the request body (to prove early rejections read nothing)."""
    calls = []
    real = transcription.read_bounded_body

    async def spy(stream):
        calls.append(True)
        return await real(stream)

    monkeypatch.setattr(transcription, "read_bounded_body", spy)
    return calls


def post(client, body=None, content_type=MULTIPART, headers=None, **kwargs):
    all_headers = {"content-type": content_type, **(headers or {})}
    return client.post(URL, content=upload() if body is None else body, headers=all_headers, **kwargs)


def error_code(response):
    return response.json()["detail"]["code"]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def test_transcription_is_disabled_by_default_and_separate_from_voice(settings):
    fields = config.Settings.model_fields
    assert fields["voice_transcription_enabled"].default is False
    assert IsolatedSettings().voice_transcription_enabled is False
    assert settings.voice_transcription_enabled is False
    defaults = IsolatedSettings()
    assert defaults.voice_transcription_model == "whisper-1"
    assert defaults.voice_transcription_timeout_seconds == 30.0
    assert defaults.voice_transcription_max_retries == 0  # no retries unless configured


@pytest.mark.parametrize("value, expected", [(True, True), ("true", True), ("FALSE", False), (" false ", False)])
def test_transcription_flag_accepts_only_true_or_false(value, expected):
    assert IsolatedSettings(voice_transcription_enabled=value).voice_transcription_enabled is expected


@pytest.mark.parametrize("value", ["1", "0", "yes", "on", "", None, 1])
def test_transcription_flag_rejects_ambiguous_values(value):
    with pytest.raises(ValidationError, match="must be true or false"):
        IsolatedSettings(voice_transcription_enabled=value)


@pytest.mark.parametrize(
    "name, value",
    [
        ("voice_transcription_timeout_seconds", 0.5),
        ("voice_transcription_timeout_seconds", 121),
        ("voice_transcription_max_retries", -1),
        ("voice_transcription_max_retries", 3),  # bounded: at most 2
        ("voice_transcription_model", "   "),
    ],
)
def test_transcription_limits_are_bounded(name, value):
    with pytest.raises(ValidationError):
        IsolatedSettings(**{name: value})


# ---------------------------------------------------------------------------
# Disabled, and the order of the early checks (nothing read before them)
# ---------------------------------------------------------------------------

def test_disabled_is_404_like_an_unknown_route_and_reads_nothing(client, provider, body_reads):
    fake = provider()
    response = post(client)
    unknown = client.post("/api/voice/no-such-route", content=upload(), headers={"content-type": MULTIPART})

    assert response.status_code == 404
    assert response.json() == NOT_FOUND
    assert (response.status_code, response.json()) == (unknown.status_code, unknown.json())
    assert body_reads == []
    assert fake.clients == []


def test_voice_status_is_unchanged_and_independent(client, monkeypatch, settings, stt_on, provider):
    provider()
    # Transcription on, spoken answers off: the status endpoint stays 404.
    assert client.get("/api/voice/status").status_code == 404
    monkeypatch.setattr(settings, "voice_enabled", True)
    assert client.get("/api/voice/status").json() == {"tts": True}  # contract unchanged
    # Spoken answers on, transcription off: 404.
    monkeypatch.setattr(settings, "voice_transcription_enabled", False)
    assert post(client).status_code == 404


@pytest.mark.parametrize(
    "headers",
    [
        {"sec-fetch-site": "cross-site"},
        {"sec-fetch-site": "same-site"},
        {"sec-fetch-site": "none"},
        {"origin": "https://evil.example"},
        {"origin": "null"},
        {"origin": "http://testserver:9999"},
        {"sec-fetch-site": "same-origin", "origin": "https://evil.example"},
    ],
)
def test_cross_origin_requests_are_refused_before_reading(client, stt_on, provider, body_reads, headers):
    fake = provider()
    response = post(client, headers=headers)

    assert response.status_code == 403
    assert error_code(response) == "origin_not_allowed"
    assert response.headers["cache-control"] == "no-store"
    assert body_reads == []
    assert fake.requests == []


@pytest.mark.parametrize(
    "headers",
    [{"sec-fetch-site": "same-origin"}, {"origin": "http://testserver"}, {"sec-fetch-site": "same-origin", "origin": "http://testserver"}],
)
def test_same_origin_requests_are_accepted(client, stt_on, provider, headers):
    provider()
    assert post(client, headers=headers).status_code == 200


def test_legacy_shim_only_a_request_without_origin_headers_is_transcribed(client, stt_on, provider):
    """SHIM-ONLY behaviour, not production: see tests/conftest.py (legacy_access_compat).

    The shim lets this module's header-less test clients through. In
    production, a request with neither Origin nor Sec-Fetch-Site is refused
    with 403 before the upload is read:
    tests/test_access_enforcement.py::test_transcription_without_origin_headers_is_refused_in_production.
    """
    from app.services import access_policy

    assert access_policy.origin_allowed.__module__ == "tests.conftest"  # the shim is what allows it
    provider()
    assert post(client).status_code == 200


@pytest.mark.parametrize(
    "content_type",
    ["application/json", "text/plain", "application/x-www-form-urlencoded", "multipart/form-data", "multipart/mixed; boundary=x", ""],
)
def test_other_content_types_are_415_before_reading(client, stt_on, provider, body_reads, content_type):
    provider()
    response = post(client, content_type=content_type)

    assert response.status_code == 415
    assert error_code(response) == "unsupported_media_type"
    assert body_reads == []


def test_a_declared_length_over_the_limit_is_413_before_reading(client, stt_on, provider, body_reads):
    fake = provider()
    too_big = upload(data=WEBM + b"\x00" * transcription.MAX_REQUEST_BYTES)

    response = post(client, body=too_big)

    assert response.status_code == 413
    assert error_code(response) == "audio_too_large"
    assert body_reads == []  # refused on Content-Length alone
    assert fake.requests == []


def test_a_missing_api_key_is_503_before_reading(client, monkeypatch, settings, stt_on, provider, body_reads, caplog):
    fake = provider()
    monkeypatch.setattr(settings, "openai_api_key", SecretStr(""))

    response = post(client)

    assert response.status_code == 503
    assert error_code(response) == "transcription_unavailable"
    assert body_reads == []
    assert fake.clients == []
    assert "OPENAI_API_KEY is not set" in caplog.text


@pytest.mark.parametrize("value", ["abc", "-1", "1.5"])
def test_an_invalid_declared_length_is_rejected(value):
    with pytest.raises(transcription.VoiceUploadError) as raised:
        transcription.check_declared_length(value)
    assert raised.value.code == "invalid_upload"


# ---------------------------------------------------------------------------
# Bounded reading
# ---------------------------------------------------------------------------

def test_reading_stops_as_soon_as_the_limit_is_passed():
    consumed = []
    chunk = b"\x00" * (256 * 1024)
    total_chunks = 64  # 16 MiB offered

    async def stream():
        for _ in range(total_chunks):
            consumed.append(True)
            yield chunk

    async def run():
        with pytest.raises(transcription.VoiceUploadError) as raised:
            await transcription.read_bounded_body(stream())
        return raised.value.code

    assert anyio.run(run) == "audio_too_large"
    # 2 MiB + 16 KiB fits in 9 chunks of 256 KiB; the 9th passes the limit.
    assert len(consumed) == 9
    assert len(consumed) < total_chunks


def test_a_body_exactly_at_the_limit_is_read():
    async def stream():
        yield b"\x00" * transcription.MAX_REQUEST_BYTES

    assert len(anyio.run(transcription.read_bounded_body, stream())) == transcription.MAX_REQUEST_BYTES


def test_a_stalled_upload_times_out(monkeypatch):
    monkeypatch.setattr(transcription, "UPLOAD_READ_TIMEOUT_SECONDS", 0.05)

    async def stream():
        yield b"\x00" * 10
        await anyio.sleep(5)
        yield b"never"

    async def run():
        with pytest.raises(transcription.VoiceUploadError) as raised:
            await transcription.read_bounded_body(stream())
        return raised.value.code

    assert anyio.run(run) == "upload_timeout"


def test_a_chunked_upload_without_a_length_is_still_capped(client, stt_on, provider, body_reads, monkeypatch):
    fake = provider()
    big = upload(data=WEBM + b"\x00" * transcription.MAX_REQUEST_BYTES)
    seen_lengths = []
    real_check = transcription.check_declared_length
    monkeypatch.setattr(
        transcription, "check_declared_length", lambda value: (seen_lengths.append(value), real_check(value))[1]
    )

    def chunks():
        for start in range(0, len(big), 64 * 1024):
            yield big[start : start + 64 * 1024]

    response = post(client, body=chunks())  # an iterator: sent chunked, no Content-Length

    assert seen_lengths == [None]  # really no declared length
    assert body_reads == [True]  # so the cap was enforced while streaming
    assert response.status_code == 413
    assert error_code(response) == "audio_too_large"
    assert fake.requests == []


def test_the_2_mib_limit_still_applies_before_the_duration(client, stt_on, provider):
    # Exactly 2 MiB passes the size limit (then 65.5 s of audio is too long);
    # one byte more is refused by size. Neither reaches the provider.
    fake = provider()
    limit = transcription.MAX_AUDIO_BYTES
    exact = canonical_wav((limit - 44) / 32000)
    assert len(exact) == limit

    response = post(client, body=upload(data=exact))
    assert (response.status_code, error_code(response)) == (413, "audio_too_long")

    response = post(client, body=upload(data=exact + b"\x00"))
    assert (response.status_code, error_code(response)) == (413, "audio_too_large")
    assert fake.requests == []


# ---------------------------------------------------------------------------
# Multipart structure (400)
# ---------------------------------------------------------------------------

def _two_parts():
    return multipart(audio_part(), ('Content-Disposition: form-data; name="note"', b"hello"))


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(b"not multipart at all", id="garbage"),
        pytest.param(upload()[:-30], id="truncated, no closing boundary"),
        pytest.param(f"--{BOUNDARY}--\r\n".encode(), id="no parts"),
        pytest.param(_two_parts(), id="an extra field"),
        pytest.param(multipart(audio_part(), audio_part()), id="two audio files"),
        pytest.param(multipart(audio_part(name="file")), id="wrong field name"),
        pytest.param(multipart(('Content-Disposition: form-data; name="audio"', WEBM)), id="a plain field, not a file"),
        pytest.param(upload(data=b""), id="empty audio"),
        pytest.param(upload(data=WEBM[:8]), id="too short to identify"),
        pytest.param(
            multipart((f'Content-Disposition: form-data; name="audio"; filename="a.webm"\r\nX-Pad: {"x" * 5000}\r\nContent-Type: audio/webm', WEBM)),
            id="oversized part headers",
        ),
        pytest.param(
            multipart(('Content-Disposition: form-data; name="audio"; filename="a.webm"\r\nContent-Type: audio/webm\r\nContent-Type: audio/ogg', WEBM)),
            id="repeated header",
        ),
        pytest.param(
            multipart(('Content-Disposition: attachment; name="audio"; filename="a.webm"\r\nContent-Type: audio/webm', WEBM)),
            id="not form-data",
        ),
    ],
)
def test_malformed_uploads_are_400(client, stt_on, provider, body):
    fake = provider()
    response = post(client, body=body)

    assert response.status_code == 400, response.text
    assert error_code(response) == "invalid_upload"
    assert response.headers["cache-control"] == "no-store"
    assert fake.requests == []


# ---------------------------------------------------------------------------
# Formats: extension, declared type and content must agree (415)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "data, filename, content_type, provider_name, provider_mime",
    [
        (AUDIO, "clip.wav", "audio/wav", "audio.wav", "audio/wav"),
        (AUDIO, "CLIP.WAV", "audio/x-wav", "audio.wav", "audio/wav"),
        (AUDIO, "clip.wav", "audio/wave", "audio.wav", "audio/wav"),
        (AUDIO, "clip.wav", "audio/vnd.wave", "audio.wav", "audio/wav"),
    ],
)
def test_allowed_formats_are_transcribed(client, stt_on, provider, data, filename, content_type, provider_name, provider_mime):
    fake = provider()
    response = post(client, body=upload(data, filename, content_type))

    assert response.status_code == 200, response.text
    [request] = fake.requests
    assert request["file"] == (provider_name, data, provider_mime)


@pytest.mark.parametrize(
    "data, filename, content_type",
    [
        pytest.param(WEBM, "clip.flac", "audio/flac", id="extension not allowed"),
        pytest.param(WEBM, "clip.txt", "text/plain", id="text file"),
        pytest.param(WEBM, "clip", "audio/webm", id="no extension"),
        pytest.param(WEBM, "clip.webm.exe", "audio/webm", id="disguised executable"),
        pytest.param(WEBM, "clip.webm", "audio/ogg", id="declared type does not match the extension"),
        pytest.param(WEBM, "clip.webm", "application/octet-stream", id="generic type"),
        pytest.param(WEBM, "clip.webm", "", id="no declared type"),
        pytest.param(OGG, "clip.webm", "audio/webm", id="content does not match"),
        pytest.param(b"#!/bin/sh\necho owned " + b"x" * 60, "clip.webm", "audio/webm", id="script renamed as audio"),
        pytest.param(b"<html><script>x()</script>" + b"x" * 60, "clip.mp3", "audio/mpeg", id="html renamed as audio"),
        # S1-4c: compressed formats are refused (their duration can't be bounded).
        pytest.param(WEBM, "clip.webm", "audio/webm", id="webm"),
        pytest.param(WEBM, "clip.webm", "audio/webm;codecs=opus", id="webm opus (Chrome)"),
        pytest.param(OGG, "clip.ogg", "audio/ogg; codecs=opus", id="ogg opus (Firefox)"),
        pytest.param(MP4, "clip.m4a", "audio/mp4", id="m4a (Safari)"),
        pytest.param(MP3_ID3, "clip.mp3", "audio/mpeg", id="mp3"),
        pytest.param(WAV, "clip.wav", "audio/wav", id="non-canonical WAV header"),
        pytest.param(OGG, "clip.wav", "audio/wav", id="ogg renamed as wav"),
    ],
)
def test_inconsistent_or_disallowed_audio_is_415(client, stt_on, provider, data, filename, content_type):
    fake = provider()
    response = post(client, body=upload(data, filename, content_type))

    assert response.status_code == 415
    assert error_code(response) == "unsupported_audio"
    assert fake.requests == []


def test_the_client_filename_is_never_sent_or_trusted(client, stt_on, provider):
    fake = provider()
    response = post(client, body=upload(filename="../../etc/passwd/../" + SECRET_FILENAME))

    assert response.status_code == 200
    [request] = fake.requests
    assert request["file"][0] == "audio.wav"


# ---------------------------------------------------------------------------
# The provider call
# ---------------------------------------------------------------------------

def test_success_returns_only_the_transcript_with_no_store(client, stt_on, provider):
    fake = provider(result=SimpleNamespace(text=f"  {TRANSCRIPT}\n"))
    response = post(client)

    assert response.status_code == 200
    assert response.json() == {"text": TRANSCRIPT}  # trimmed; nothing else
    assert response.headers["cache-control"] == "no-store"

    [client_used] = fake.clients
    assert client_used.options == {"api_key": "sk-test-not-a-real-key", "timeout": 30.0, "max_retries": 0}
    assert client_used.closed is True
    [request] = fake.requests
    assert request["model"] == "whisper-1"
    assert request["response_format"] == "verbose_json"
    assert set(request) == {"model", "file", "response_format"}


def test_configured_model_timeout_and_retries_are_passed(client, monkeypatch, settings, stt_on, provider):
    fake = provider()
    monkeypatch.setattr(settings, "voice_transcription_model", "gpt-4o-mini-transcribe")
    monkeypatch.setattr(settings, "voice_transcription_timeout_seconds", 12.5)
    monkeypatch.setattr(settings, "voice_transcription_max_retries", 2)

    assert post(client).status_code == 200
    assert fake.clients[0].options["timeout"] == 12.5
    assert fake.clients[0].options["max_retries"] == 2
    assert fake.requests[0]["model"] == "gpt-4o-mini-transcribe"


def test_an_empty_transcript_is_returned_as_empty_text(client, stt_on, provider):
    provider(result=SimpleNamespace(text="   "))
    response = post(client)
    assert response.status_code == 200
    assert response.json() == {"text": ""}


_REQUEST = httpx2.Request("POST", "https://api.openai.com/v1/audio/transcriptions")


def _status_error(cls, status):
    return cls(PROVIDER_DETAIL, response=httpx2.Response(status, request=_REQUEST), body={"error": PROVIDER_DETAIL})


@pytest.mark.parametrize(
    "error, status, code",
    [
        pytest.param(openai.APITimeoutError(request=_REQUEST), 504, "transcription_timeout", id="timeout"),
        pytest.param(openai.APIConnectionError(message=PROVIDER_DETAIL, request=_REQUEST), 503, "transcription_unavailable", id="connection"),
        pytest.param(_status_error(openai.BadRequestError, 400), 422, "audio_not_transcribable", id="400"),
        pytest.param(_status_error(openai.UnprocessableEntityError, 422), 422, "audio_not_transcribable", id="422"),
        pytest.param(_status_error(openai.AuthenticationError, 401), 503, "transcription_unavailable", id="401"),
        pytest.param(_status_error(openai.PermissionDeniedError, 403), 503, "transcription_unavailable", id="403"),
        pytest.param(_status_error(openai.RateLimitError, 429), 503, "transcription_unavailable", id="429"),
        pytest.param(_status_error(openai.InternalServerError, 500), 503, "transcription_unavailable", id="500"),
        pytest.param(ValueError(PROVIDER_DETAIL), 500, "transcription_failed", id="unexpected"),
    ],
)
def test_provider_errors_are_generic(client, stt_on, provider, caplog, error, status, code):
    provider(error=error)
    caplog.set_level(logging.DEBUG)

    response = post(client, body=upload(filename=SECRET_FILENAME))

    assert response.status_code == status
    assert error_code(response) == code
    assert response.headers["cache-control"] == "no-store"
    assert PROVIDER_DETAIL not in response.text
    assert PROVIDER_DETAIL not in caplog.text  # provider messages are never logged
    assert SECRET_FILENAME not in caplog.text
    assert "sk-test-not-a-real-key" not in caplog.text


@pytest.mark.parametrize(
    "error",
    [
        openai.APITimeoutError(request=_REQUEST),
        _status_error(openai.RateLimitError, 429),
        ValueError("unexpected"),
    ],
)
def test_the_provider_client_is_closed_after_an_error(client, stt_on, provider, error):
    fake = provider(error=error)
    post(client)
    [client_used] = fake.clients
    assert client_used.closed is True


def test_a_cancelled_transcription_closes_the_provider_client(monkeypatch):
    # For example a server shutdown that cancels the request mid-call.
    fake = FakeProvider()
    entered = []

    def make_client(**options):
        made = fake(**options)

        async def never_answers(**request):
            entered.append(True)
            await anyio.sleep(30)

        made.audio.transcriptions.create = never_answers
        return made

    monkeypatch.setattr(transcription, "AsyncOpenAI", make_client)
    audio = transcription.AudioUpload(data=AUDIO, format=transcription.FORMATS_BY_EXTENSION[".wav"], seconds=1.0)

    async def run():
        with anyio.move_on_after(0.05) as scope:
            await transcription.transcribe(audio, api_key="k", model="m", timeout_seconds=30, max_retries=0)
        return scope.cancelled_caught

    assert anyio.run(run) is True
    assert entered == [True]
    [client_used] = fake.clients
    assert client_used.closed is True  # async with released it on cancellation


def test_a_malformed_upload_logs_no_request_bytes(client, stt_on, provider, caplog):
    # Regression: python-multipart logs a byte of the request and its offset
    # (for example "got 103 at index 2") at WARNING on a parse error.
    provider()
    caplog.set_level(logging.DEBUG)

    for body in (b"garbage-not-multipart", multipart(("Bad\x01Header: x", WEBM))):
        response = post(client, body=body)
        assert error_code(response) == "invalid_upload"

    assert [r for r in caplog.records if r.name.startswith("python_multipart")] == []


def test_the_log_filter_only_applies_while_parsing_a_voice_upload(caplog):
    # Other users of python-multipart keep their diagnostics.
    from python_multipart.exceptions import MultipartParseError
    from python_multipart.multipart import MultipartParser

    caplog.set_level(logging.WARNING, logger="python_multipart.multipart")
    with pytest.raises(MultipartParseError):
        MultipartParser(b"XB", {}).write(b"garbage")

    assert any(r.name == "python_multipart.multipart" for r in caplog.records)


def test_a_malformed_provider_response_is_503(client, stt_on, provider):
    provider(result=SimpleNamespace(text=None))
    response = post(client)
    assert response.status_code == 503
    assert error_code(response) == "transcription_unavailable"


# ---------------------------------------------------------------------------
# Privacy and isolation
# ---------------------------------------------------------------------------

def test_nothing_is_logged_stored_audited_or_sent_to_the_chat(
    client, monkeypatch, settings, stt_on, provider, blocked_network, caplog, tmp_path
):
    provider()
    caplog.set_level(logging.DEBUG)

    def forbidden(*args, **kwargs):
        raise AssertionError("transcription must never submit a chat question")

    monkeypatch.setattr(routes, "ask", forbidden)
    files_before = {p for p in tmp_path.rglob("*")}
    audits_before = len(get_recent_audits(500))

    response = post(client, body=upload(filename=SECRET_FILENAME))

    assert response.status_code == 200
    assert response.json() == {"text": TRANSCRIPT}
    # Logs: size and format only; never the transcript, the audio, the name or the key.
    assert TRANSCRIPT not in caplog.text
    assert SECRET_FILENAME not in caplog.text
    assert repr(AUDIO[:8]) not in caplog.text and repr(AUDIO[44:52]) not in caplog.text
    assert "sk-test-not-a-real-key" not in caplog.text
    assert "Voice transcription completed: format=wav bytes=32044 audio_seconds=1.00" in caplog.text
    # Nothing written, audited, or started.
    assert {p for p in tmp_path.rglob("*")} == files_before
    assert not settings.upload_dir.exists() or not any(settings.upload_dir.iterdir())
    assert len(get_recent_audits(500)) == audits_before
    assert not settings.session_db_path.exists()
    assert streaming.peek_stream_service() is None
    assert blocked_network() == []


@pytest.mark.parametrize("enabled", [False, True])
def test_other_methods_get_the_standard_405(client, monkeypatch, settings, enabled):
    monkeypatch.setattr(settings, "voice_transcription_enabled", enabled)
    response = client.get(URL)
    assert response.status_code == 405
    assert response.headers["allow"] == "POST"


@pytest.mark.parametrize("enabled", [False, True])
def test_chat_and_sessions_are_unchanged(client, monkeypatch, settings, enabled):
    from tests.test_chat_api import workflow_result

    monkeypatch.setattr(settings, "voice_transcription_enabled", enabled)
    monkeypatch.setattr(routes, "ask", lambda question: workflow_result())

    assert client.post("/api/chat", json={"question": "How do I connect to the VPN?"}).status_code == 200
    assert client.post("/api/chat", json={"question": "x"}).status_code == 422
    assert client.post("/api/chat/stream", json={"question": "VPN?"}).json() == NOT_FOUND
    assert client.post("/api/sessions", json={}).status_code == 404
