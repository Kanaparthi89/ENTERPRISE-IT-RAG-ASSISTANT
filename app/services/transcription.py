"""Voice V2: bounded audio upload parsing and OpenAI speech-to-text.

Used only by ``POST /api/voice/transcribe``. The route passes every setting
in; this module reads no configuration.

Upload handling never buffers more than ``MAX_REQUEST_BYTES``:

* A declared ``Content-Length`` above the limit is refused before any byte of
  the body is read.
* The body is read from the ASGI stream chunk by chunk and reading stops as
  soon as the count passes the limit (chunked uploads without a length are
  covered too). The server only reads from the socket when the application
  asks for more, so an oversized upload is never accumulated.
* Only then is the bounded body parsed, in memory, with python-multipart's
  low-level parser. Starlette's form parser is not used: it spools file parts
  to temporary files, and audio must never touch the disk.

Audio (Security S1-4c): only canonical 16 kHz mono 16-bit PCM WAV is
accepted, and its exact duration is measured from the bytes received
(``app/services/audio_duration.py``) before any provider call: at most 60
seconds. Other formats are refused (their length can't be bounded reliably
without decoding).

Privacy: audio and transcripts are held in memory for one request and never
written, logged or audited. Log lines carry sizes, formats, durations,
timings and error classes only.
"""

import logging
import time
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import PurePath

import anyio

from app.services import audio_duration
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import MultipartParser, parse_options_header

logger = logging.getLogger(__name__)

# python-multipart logs parse problems at WARNING with a byte of the request
# and its offset (for example "got 103 at index 2"). While this module parses
# an upload, those records are dropped: a voice upload's bytes never reach the
# logs. The flag is context-local, so other users of the library (such as
# /api/ingest) keep their messages.
_parsing_voice_upload: ContextVar[bool] = ContextVar("parsing_voice_upload", default=False)


class _DropDuringVoiceParse(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _parsing_voice_upload.get()


_MULTIPART_LOG_FILTER = _DropDuringVoiceParse()
logging.getLogger("python_multipart.multipart").addFilter(_MULTIPART_LOG_FILTER)

MAX_AUDIO_BYTES = 2 * 1024 * 1024  # 2 MiB of audio
# Room for the multipart boundaries and part headers around the audio.
MULTIPART_OVERHEAD_BYTES = 16 * 1024
MAX_REQUEST_BYTES = MAX_AUDIO_BYTES + MULTIPART_OVERHEAD_BYTES
# A slow (or stalled) upload is abandoned after this long. 60 s of the
# accepted WAV is about 1.92 MB, about 15 s at 1 Mbit/s (S1-4c).
UPLOAD_READ_TIMEOUT_SECONDS = 30.0
MIN_AUDIO_BYTES = 16  # enough to recognise the container's signature
AUDIO_FIELD = "audio"
MAX_PARTS = 1
MAX_PART_HEADER_BYTES = 4096


class VoiceUploadError(Exception):
    """A rejected upload or transcription, with a stable public code."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


# ---------------------------------------------------------------------------
# Formats: extension, declared MIME type and content signature must agree
# ---------------------------------------------------------------------------

def _is_webm(data: bytes) -> bool:
    return data[:4] == b"\x1a\x45\xdf\xa3"  # EBML header


def _is_ogg(data: bytes) -> bool:
    return data[:4] == b"OggS"


def _is_mp4(data: bytes) -> bool:
    return data[4:8] == b"ftyp"  # ISO base media file


def _is_mp3(data: bytes) -> bool:
    return data[:3] == b"ID3" or (data[0] == 0xFF and (data[1] & 0xE0) == 0xE0)


def _is_wav(data: bytes) -> bool:
    return data[:4] == b"RIFF" and data[8:12] == b"WAVE"


@dataclass(frozen=True)
class AudioFormat:
    name: str
    extension: str  # sent to the provider; the client's filename never is
    provider_mime: str  # sent to the provider
    mime_types: frozenset[str]  # accepted from the client
    matches: object  # Callable[[bytes], bool]


_WEBM = AudioFormat("webm", ".webm", "audio/webm", frozenset({"audio/webm", "video/webm"}), _is_webm)
_OGG = AudioFormat("ogg", ".ogg", "audio/ogg", frozenset({"audio/ogg", "application/ogg"}), _is_ogg)
_MP4 = AudioFormat(
    "mp4", ".m4a", "audio/mp4", frozenset({"audio/mp4", "audio/m4a", "audio/x-m4a", "video/mp4"}), _is_mp4
)
_MP3 = AudioFormat("mp3", ".mp3", "audio/mpeg", frozenset({"audio/mpeg", "audio/mp3"}), _is_mp3)
_WAV = AudioFormat(
    "wav", ".wav", "audio/wav", frozenset({"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}), _is_wav
)

# Accepted uploads (S1-4c): canonical PCM WAV only. The compressed formats
# above are still recognised by their signatures, so they are refused as
# unsupported rather than as malformed.
FORMATS_BY_EXTENSION = {
    ".wav": _WAV,
}


@dataclass(frozen=True)
class AudioUpload:
    data: bytes
    format: AudioFormat
    seconds: float = 0.0  # measured exactly from the samples (S1-4c)
    reserved_seconds: int = 0  # whole seconds rounded up: what the quota reserves


@dataclass(frozen=True)
class Transcription:
    text: str
    duration: object = None  # the provider's reported duration (verbose_json), not trusted


# ---------------------------------------------------------------------------
# Bounded reading
# ---------------------------------------------------------------------------

def check_declared_length(content_length: str | None) -> None:
    """Refuse a declared body over the limit before reading any of it."""
    if content_length is None:
        return  # chunked upload: counted while reading
    try:
        declared = int(content_length.strip())
    except ValueError:
        raise VoiceUploadError("invalid_upload") from None
    if declared < 0:
        raise VoiceUploadError("invalid_upload")
    if declared > MAX_REQUEST_BYTES:
        raise VoiceUploadError("audio_too_large")


async def read_bounded_body(stream) -> bytes:
    """Read an async byte stream, stopping as soon as it passes the limit.

    Raises ``audio_too_large`` without reading further, and
    ``upload_timeout`` if the whole body does not arrive in
    ``UPLOAD_READ_TIMEOUT_SECONDS``.
    """
    chunks: list[bytes] = []
    size = 0
    try:
        with anyio.fail_after(UPLOAD_READ_TIMEOUT_SECONDS):
            async for chunk in stream:
                size += len(chunk)
                if size > MAX_REQUEST_BYTES:
                    raise VoiceUploadError("audio_too_large")
                chunks.append(chunk)
    except TimeoutError:
        raise VoiceUploadError("upload_timeout") from None
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# Multipart parsing (in memory, one part)
# ---------------------------------------------------------------------------

def multipart_boundary(content_type: str | None) -> bytes:
    """The boundary of a multipart/form-data request, or ``unsupported_media_type``."""
    media_type, options = parse_options_header(content_type or "")
    boundary = options.get(b"boundary", b"")
    if media_type != b"multipart/form-data" or not boundary or len(boundary) > 70:
        raise VoiceUploadError("unsupported_media_type")
    return boundary


def _parse_parts(body: bytes, boundary: bytes) -> list[dict]:
    parts: list[dict] = []
    state = {"field": b"", "value": b"", "ended": False}

    def on_part_begin():
        if len(parts) >= MAX_PARTS:
            raise VoiceUploadError("invalid_upload")
        parts.append({"headers": {}, "data": [], "header_bytes": 0})

    def on_header_field(data, start, end):
        state["field"] += data[start:end]

    def on_header_value(data, start, end):
        state["value"] += data[start:end]

    def on_header_end():
        part = parts[-1]
        part["header_bytes"] += len(state["field"]) + len(state["value"])
        if part["header_bytes"] > MAX_PART_HEADER_BYTES:
            raise VoiceUploadError("invalid_upload")
        name = state["field"].decode("latin-1").strip().lower()
        if name in part["headers"]:
            raise VoiceUploadError("invalid_upload")  # a repeated header is ambiguous
        part["headers"][name] = state["value"].decode("latin-1").strip()
        state["field"], state["value"] = b"", b""

    def on_part_data(data, start, end):
        parts[-1]["data"].append(data[start:end])

    def on_end():
        state["ended"] = True

    parser = MultipartParser(
        boundary,
        {
            "on_part_begin": on_part_begin,
            "on_header_field": on_header_field,
            "on_header_value": on_header_value,
            "on_header_end": on_header_end,
            "on_part_data": on_part_data,
            "on_end": on_end,
        },
        max_size=MAX_REQUEST_BYTES,
    )
    token = _parsing_voice_upload.set(True)
    try:
        parser.write(body)
        parser.finalize()
    except MultipartParseError:
        raise VoiceUploadError("invalid_upload") from None
    finally:
        _parsing_voice_upload.reset(token)
    if not state["ended"]:
        raise VoiceUploadError("invalid_upload")  # truncated: no closing boundary
    return parts


def parse_audio_upload(body: bytes, content_type: str | None) -> AudioUpload:
    """Exactly one file part named ``audio`` in an allowed, consistent format."""
    boundary = multipart_boundary(content_type)
    parts = _parse_parts(body, boundary)
    if len(parts) != 1:
        raise VoiceUploadError("invalid_upload")

    [part] = parts
    disposition, params = parse_options_header(part["headers"].get("content-disposition", ""))
    if disposition != b"form-data" or params.get(b"name") != AUDIO_FIELD.encode():
        raise VoiceUploadError("invalid_upload")
    raw_filename = params.get(b"filename")
    if not raw_filename:
        raise VoiceUploadError("invalid_upload")  # a plain form field, not a file

    data = b"".join(part["data"])
    if len(data) > MAX_AUDIO_BYTES:
        raise VoiceUploadError("audio_too_large")
    if len(data) < MIN_AUDIO_BYTES:
        raise VoiceUploadError("invalid_upload")

    # The filename is untrusted: only its extension is used, never the name.
    extension = PurePath(raw_filename.decode("latin-1").replace("\\", "/")).suffix.lower()
    audio_format = FORMATS_BY_EXTENSION.get(extension)
    declared_type, _ = parse_options_header(part["headers"].get("content-type", ""))
    if (
        audio_format is None
        or declared_type.decode("latin-1") not in audio_format.mime_types
        or not audio_format.matches(data)
    ):
        raise VoiceUploadError("unsupported_audio")

    # The duration is measured from the bytes themselves; nothing declared is trusted.
    try:
        measured = audio_duration.measure_canonical_wav(data)
    except audio_duration.AudioDurationError as exc:
        raise VoiceUploadError(exc.code) from None
    return AudioUpload(
        data=data, format=audio_format, seconds=measured.seconds, reserved_seconds=measured.reserved_seconds
    )


# ---------------------------------------------------------------------------
# OpenAI transcription
# ---------------------------------------------------------------------------

async def transcribe(
    upload: AudioUpload,
    *,
    api_key: str,
    model: str,
    timeout_seconds: float,
    max_retries: int,
) -> Transcription:
    """The transcript of ``upload`` and the provider's reported duration.

    Raises ``VoiceUploadError`` with a public code.

    ``timeout_seconds`` bounds each provider attempt; ``max_retries`` (0-2)
    bounds the attempts. Provider messages and response bodies are never
    passed on or logged.
    """
    if not api_key:
        logger.error("Voice transcription is enabled but OPENAI_API_KEY is not set.")
        raise VoiceUploadError("transcription_unavailable")

    started = time.monotonic()
    try:
        async with AsyncOpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=max_retries) as client:
            result = await client.audio.transcriptions.create(
                model=model,
                file=(f"audio{upload.format.extension}", upload.data, upload.format.provider_mime),
                response_format="verbose_json",  # includes the provider's duration (S1-4c)
            )
    except APITimeoutError:
        _log_failure("timeout", upload, started)
        raise VoiceUploadError("transcription_timeout") from None
    except (BadRequestError, UnprocessableEntityError) as exc:
        _log_failure(f"rejected status={exc.status_code}", upload, started)
        raise VoiceUploadError("audio_not_transcribable") from None
    except (AuthenticationError, PermissionDeniedError) as exc:
        logger.error("Voice transcription provider refused the credentials (status=%s).", exc.status_code)
        raise VoiceUploadError("transcription_unavailable") from None
    except (RateLimitError, APIStatusError) as exc:
        _log_failure(f"status={exc.status_code}", upload, started)
        raise VoiceUploadError("transcription_unavailable") from None
    except APIConnectionError:
        _log_failure("connection", upload, started)
        raise VoiceUploadError("transcription_unavailable") from None

    text = getattr(result, "text", None)
    if not isinstance(text, str):
        _log_failure("unexpected response", upload, started)
        raise VoiceUploadError("transcription_unavailable")

    logger.info(
        "Voice transcription completed: format=%s bytes=%d audio_seconds=%.2f duration_ms=%d",
        upload.format.name,
        len(upload.data),
        upload.seconds,
        int((time.monotonic() - started) * 1000),
    )
    return Transcription(text=text.strip(), duration=getattr(result, "duration", None))


def _log_failure(reason: str, upload: AudioUpload, started: float) -> None:
    logger.warning(
        "Voice transcription failed (%s): format=%s bytes=%d duration_ms=%d",
        reason,
        upload.format.name,
        len(upload.data),
        int((time.monotonic() - started) * 1000),
    )
