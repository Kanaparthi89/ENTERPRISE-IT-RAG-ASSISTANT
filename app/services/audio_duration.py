"""Server-side audio duration for transcription (Security S1-4c).

Only one format is accepted: canonical PCM WAV, 16 kHz, mono, 16-bit, with
the standard 44-byte header (``RIFF``/``WAVE``, a 16-byte ``fmt `` chunk,
then one ``data`` chunk holding every remaining byte). For uncompressed PCM
the duration *is* the number of sample bytes divided by the byte rate, so
it is measured exactly from the bytes received, without decoding.

Nothing the client declares is trusted. The header's sizes are checked
against the real byte count: a ``RIFF`` or ``data`` size that disagrees,
trailing bytes, extra chunks (``LIST`` and the like), a second ``data``
chunk, any other encoding, rate, channel count or sample size are refused.
Compressed formats are refused too: their size says nothing reliable about
their length (silence can be encoded in almost no bytes), and their
metadata can claim any duration.

Pure standard library; O(1) work on the header and no copy of the audio.
"""

import struct
from dataclasses import dataclass

SAMPLE_RATE = 16_000
CHANNELS = 1
BITS_PER_SAMPLE = 16
BLOCK_ALIGN = CHANNELS * BITS_PER_SAMPLE // 8  # 2
BYTE_RATE = SAMPLE_RATE * BLOCK_ALIGN  # 32,000 bytes per second
HEADER_BYTES = 44
PCM_FORMAT = 1
MAX_DURATION_SECONDS = 60
MAX_DATA_BYTES = MAX_DURATION_SECONDS * BYTE_RATE  # 1,920,000
MAX_WAV_BYTES = HEADER_BYTES + MAX_DATA_BYTES  # 1,920,044

_HEADER = struct.Struct("<4sI4s4sIHHIIHH4sI")  # exactly 44 bytes


class AudioDurationError(ValueError):
    """Audio that must not be sent to the provider, with a stable public code."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code  # unsupported_audio | invalid_audio | audio_too_long


@dataclass(frozen=True)
class PcmDuration:
    data_bytes: int
    seconds: float  # exact: data_bytes / BYTE_RATE
    reserved_seconds: int  # whole seconds, rounded up: what the quota reserves


def measure_canonical_wav(data) -> PcmDuration:
    """The exact duration of a canonical 16 kHz mono 16-bit PCM WAV, or ``AudioDurationError``."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise AudioDurationError("invalid_audio")
    total = len(data)
    if total < HEADER_BYTES:
        raise AudioDurationError("invalid_audio")
    (
        riff, riff_size, wave, fmt_id, fmt_size, audio_format, channels, sample_rate,
        byte_rate, block_align, bits, data_id, data_size,
    ) = _HEADER.unpack_from(data, 0)

    if riff != b"RIFF" or wave != b"WAVE":
        raise AudioDurationError("unsupported_audio")
    if fmt_id != b"fmt " or fmt_size != 16 or data_id != b"data":
        raise AudioDurationError("unsupported_audio")  # extensible formats, extra or reordered chunks
    if (audio_format, channels, sample_rate, byte_rate, block_align, bits) != (
        PCM_FORMAT, CHANNELS, SAMPLE_RATE, BYTE_RATE, BLOCK_ALIGN, BITS_PER_SAMPLE
    ):
        raise AudioDurationError("unsupported_audio")

    actual = total - HEADER_BYTES
    if riff_size != total - 8 or data_size != actual:
        raise AudioDurationError("invalid_audio")  # sizes must match the bytes received exactly
    if actual == 0 or actual % BLOCK_ALIGN:
        raise AudioDurationError("invalid_audio")
    if actual > MAX_DATA_BYTES:
        raise AudioDurationError("audio_too_long")
    return PcmDuration(
        data_bytes=actual,
        seconds=actual / BYTE_RATE,
        reserved_seconds=-(-actual // BYTE_RATE),  # ceil without floats
    )
