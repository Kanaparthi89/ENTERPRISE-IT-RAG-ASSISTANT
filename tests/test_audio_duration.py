"""Security S1-4c: canonical PCM WAV duration, measured from the bytes."""

import struct

import pytest

from app.services import audio_duration
from app.services.audio_duration import AudioDurationError, measure_canonical_wav


def wav(samples: int, *, rate=16000, channels=1, bits=16, audio_format=1, byte_rate=None, block_align=None,
        sample=b"\x01\x00"):
    data = sample * samples
    byte_rate = rate * channels * bits // 8 if byte_rate is None else byte_rate
    block_align = channels * bits // 8 if block_align is None else block_align
    return (
        b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, audio_format, channels, rate, byte_rate, block_align, bits)
        + b"data" + struct.pack("<I", len(data)) + data
    )


def with_field(data: bytes, offset: int, fmt: str, value) -> bytes:
    changed = bytearray(data)
    struct.pack_into(fmt, changed, offset, value)
    return bytes(changed)


ONE_SECOND = 16000
SIXTY_SECONDS = 60 * ONE_SECOND


def code(data) -> str:
    with pytest.raises(AudioDurationError) as caught:
        measure_canonical_wav(data)
    return caught.value.code


# ---------------------------------------------------------------------------
# Accepted: exact durations and boundaries
# ---------------------------------------------------------------------------

def test_the_constants_describe_the_canonical_format():
    assert audio_duration.BYTE_RATE == 32000
    assert audio_duration.MAX_DATA_BYTES == 1_920_000
    assert audio_duration.MAX_WAV_BYTES == 1_920_044
    assert audio_duration.MAX_WAV_BYTES < 2 * 1024 * 1024  # fits the 2 MiB upload limit


@pytest.mark.parametrize(
    "samples, seconds, reserved",
    [
        (1, 1 / 16000, 1),  # one sample: reserved as a whole second
        (ONE_SECOND, 1.0, 1),
        (ONE_SECOND + 1, 1 + 1 / 16000, 2),  # rounded up, never down
        (SIXTY_SECONDS - 1, 60 - 1 / 16000, 60),
        (SIXTY_SECONDS, 60.0, 60),  # exactly 60 s: accepted
    ],
)
def test_the_duration_is_exact_and_reserved_rounded_up(samples, seconds, reserved):
    measured = measure_canonical_wav(wav(samples))
    assert measured.data_bytes == samples * 2
    assert measured.seconds == pytest.approx(seconds, abs=1e-12)
    assert measured.reserved_seconds == reserved
    assert type(measured.reserved_seconds) is int


def test_one_sample_beyond_60_seconds_is_too_long():
    assert code(wav(SIXTY_SECONDS + 1)) == "audio_too_long"


def test_the_content_never_changes_the_measured_duration():
    # Whatever the sample bytes hold (silence, noise, an MP3 or Opus stream
    # pasted in), the provider decodes them as PCM: the duration is the length.
    for sample in (b"\x00\x00", b"\xff\xfb", b"OS", b"\x1a\x45"):
        assert measure_canonical_wav(wav(ONE_SECOND, sample=sample)).seconds == 1.0


@pytest.mark.parametrize("kind", [bytes, bytearray, memoryview])
def test_bytes_like_inputs_are_accepted(kind):
    assert measure_canonical_wav(kind(wav(ONE_SECOND))).reserved_seconds == 1


# ---------------------------------------------------------------------------
# Refused: other formats and header manipulation
# ---------------------------------------------------------------------------

BASE = wav(ONE_SECOND)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(b"RIFX" + BASE[4:], id="big-endian RIFX"),
        pytest.param(BASE[:8] + b"AVI " + BASE[12:], id="not WAVE"),
        pytest.param(BASE[:12] + b"junk" + BASE[16:], id="no fmt chunk first"),
        pytest.param(with_field(BASE, 16, "<I", 18), id="fmt chunk of 18 bytes"),
        pytest.param(wav(ONE_SECOND, audio_format=3, bits=32), id="float samples"),
        pytest.param(wav(ONE_SECOND, audio_format=0xFFFE), id="WAVE_FORMAT_EXTENSIBLE"),
        pytest.param(wav(ONE_SECOND, audio_format=6, bits=8), id="A-law"),
        pytest.param(wav(ONE_SECOND, channels=2), id="stereo"),
        pytest.param(wav(ONE_SECOND, rate=8000), id="8 kHz"),
        pytest.param(wav(ONE_SECOND, rate=44100), id="44.1 kHz"),
        pytest.param(wav(ONE_SECOND, rate=48000), id="48 kHz"),
        pytest.param(wav(ONE_SECOND, bits=8, sample=b"\x01"), id="8-bit"),
        pytest.param(wav(ONE_SECOND, bits=24, sample=b"\x01\x00\x00"), id="24-bit"),
        pytest.param(wav(ONE_SECOND, byte_rate=16000), id="byte rate that disagrees"),
        pytest.param(wav(ONE_SECOND, block_align=4), id="block align that disagrees"),
        pytest.param(BASE[:36] + b"LIST" + BASE[40:], id="a LIST chunk where data belongs"),
        pytest.param(b"\x1a\x45\xdf\xa3" + b"\x00" * 60, id="WebM"),
        pytest.param(b"OggS" + b"\x00" * 60, id="Ogg"),
        pytest.param(b"ID3\x04" + b"\x00" * 60, id="MP3"),
    ],
)
def test_other_formats_are_unsupported(data):
    assert code(data) == "unsupported_audio"


def _resize(data: bytes, extra: bytes) -> bytes:
    """Append bytes and fix both sizes, so only the content is in question."""
    grown = data + extra
    grown = with_field(grown, 4, "<I", len(grown) - 8)
    return with_field(grown, 40, "<I", len(grown) - 44)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(b"", id="empty"),
        pytest.param(BASE[:43], id="truncated header"),
        pytest.param(wav(0), id="no samples"),
        pytest.param(_resize(BASE, b"\x01"), id="odd number of bytes"),
        pytest.param(with_field(BASE, 4, "<I", len(BASE) - 7), id="RIFF size too large"),
        pytest.param(with_field(BASE, 4, "<I", len(BASE) - 9), id="RIFF size too small"),
        pytest.param(with_field(BASE, 40, "<I", len(BASE) - 44 + 2), id="data size claims more than sent"),
        pytest.param(with_field(BASE, 40, "<I", 2), id="data size claims almost nothing"),
        pytest.param(BASE + b"\x00\x00", id="trailing bytes"),
        pytest.param(
            with_field(BASE + b"data" + struct.pack("<I", 4) + b"\x00" * 4, 4, "<I", len(BASE) + 12 - 8),
            id="a second data chunk",
        ),
        pytest.param(
            with_field(BASE + b"LIST" + struct.pack("<I", 4) + b"INFO", 4, "<I", len(BASE) + 12 - 8),
            id="a LIST chunk after the data",
        ),
        pytest.param(None, id="not bytes"),
        pytest.param("RIFF", id="a string"),
    ],
)
def test_malformed_wavs_are_invalid(data):
    assert code(data) == "invalid_audio"


def test_a_short_data_size_cannot_hide_longer_audio():
    # 61 s of samples behind a header that claims 1 s: refused, never measured as 1 s.
    long = wav(SIXTY_SECONDS + ONE_SECOND)
    assert code(with_field(long, 40, "<I", ONE_SECOND * 2)) == "invalid_audio"


def test_a_second_chunk_counted_as_samples_is_measured_in_full():
    # If the data size covers everything, every byte is audio to the provider
    # too, and it is all measured: nothing can sit outside the measurement.
    tail = b"data" + struct.pack("<I", 4) + b"\x00" * 4
    data = _resize(BASE, tail)
    assert measure_canonical_wav(data).data_bytes == ONE_SECOND * 2 + len(tail)


def test_the_browser_recorders_header_is_accepted_by_the_server():
    # tests/js/fixtures/canonical-wav-header.hex is checked byte for byte
    # against recorder.js's wavHeader(16000) in tests/js/recorder.test.mjs.
    from pathlib import Path

    fixture = Path(__file__).parent / "js" / "fixtures" / "canonical-wav-header.hex"
    header = bytes.fromhex(fixture.read_text(encoding="ascii").strip())
    assert header == wav(ONE_SECOND)[:44]  # the same canonical layout the server builds
    measured = measure_canonical_wav(header + b"\x00\x00" * ONE_SECOND)
    assert (measured.seconds, measured.reserved_seconds) == (1.0, 1)
