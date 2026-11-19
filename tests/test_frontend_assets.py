"""Frontend wiring for streaming chat (A1-5).

The streaming client's behaviour is tested with node:test in ``tests/js``
(``npm run test:js``). These tests check that the page loads it and that the
SSE fixture those tests use matches what the backend actually sends.
"""

import re
from pathlib import Path

from app.api.streaming import KEEPALIVE, format_sse

FIXTURE = Path(__file__).parent / "js" / "fixtures" / "stream-transcript.sse"

DONE = {
    "request_id": "req-1",
    "answer": "Install GlobalProtect — then sign in.\nGrüße ✓",
    "source_used": "private_kb",
    "trace": ["Router → KB", "Answer generation → PRIVATE KB"],
    "citations": [{"title": "handbook.md", "url": "", "type": "private_kb"}],
    "rewritten_query": "How do I connect to the VPN?",
}


def test_page_loads_the_stream_client_before_the_app(client):
    html = client.get("/").text

    stream_script = html.index('<script src="/static/js/chat-stream.js"></script>')
    app_script = html.index('<script src="/static/js/app.js"></script>')
    assert stream_script < app_script


def test_stream_client_is_served(client):
    response = client.get("/static/js/chat-stream.js")

    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert "createChatController" in response.text


def test_js_fixture_matches_the_backend_sse_encoding():
    expected = b"".join(
        [
            format_sse("start", {"request_id": "req-1", "version": 1}),
            format_sse("trace", {"index": 0, "step": "Router → KB"}),
            format_sse("delta", {"text": "\n Install GlobalProtect — "}),
            KEEPALIVE,
            format_sse("delta", {"text": "then sign in.\nGrüße ✓"}),
            format_sse("delta", {"text": "  \n"}),
            format_sse("trace", {"index": 1, "step": "Answer generation → PRIVATE KB"}),
            format_sse("done", DONE),
        ]
    )

    assert FIXTURE.read_bytes() == expected


def test_fixture_answer_follows_the_delta_rule():
    """The JS tests rely on this: joined deltas, stripped, equal done.answer."""
    deltas = ["\n Install GlobalProtect — ", "then sign in.\nGrüße ✓", "  \n"]

    assert "".join(deltas).strip() == DONE["answer"]


def test_streaming_stays_disabled_by_default(settings):
    assert type(settings).model_fields["stream_enabled"].default is False


# ---------------------------------------------------------------------------
# A2-7: frontend session integration
# ---------------------------------------------------------------------------

SESSION_FIXTURE = Path(__file__).parent / "js" / "fixtures" / "session-event.sse"
STATIC_JS = Path(__file__).parent.parent / "static" / "js"


def test_js_session_fixture_matches_the_backend_sse_encoding():
    expected = format_sse(
        "session", {"request_id": "req-1", "turn_saved": True, "history_turns_used": 1}
    )

    assert SESSION_FIXTURE.read_bytes() == expected


def test_page_wires_the_session_client_into_the_controller():
    app_js = (STATIC_JS / "app.js").read_text(encoding="utf-8")

    assert "new window.BluePeakChat.SessionClient(" in app_js
    assert "sessions: sessionClient" in app_js


def test_session_token_is_never_persisted_by_the_frontend():
    """The token lives in JavaScript memory only: no storage, cookies or URLs."""
    for name in ("app.js", "chat-stream.js"):
        source = (STATIC_JS / name).read_text(encoding="utf-8")
        for forbidden in ("localStorage", "sessionStorage", "document.cookie", "indexedDB"):
            assert forbidden not in source, f"{forbidden} in {name}"


def test_sessions_stay_disabled_by_default(settings):
    assert type(settings).model_fields["sessions_enabled"].default is False


# ---------------------------------------------------------------------------
# Voice V1c: read answers aloud (offered only when VOICE_ENABLED=true)
# ---------------------------------------------------------------------------

def _opening_tag(html: str, element_id: str) -> str:
    match = re.search(rf'<[a-z]+\b[^>]*\bid="{element_id}"[^>]*>', html)
    assert match, f"#{element_id} is missing"
    return match.group(0)


def _has_boolean_attribute(tag: str, name: str) -> bool:
    return re.search(rf"\s{name}(?=[\s>])", tag) is not None


def test_page_loads_voice_between_the_stream_client_and_the_app(client):
    html = client.get("/").text

    stream_script = html.index('<script src="/static/js/chat-stream.js"></script>')
    voice_script = html.index('<script src="/static/js/voice.js"></script>')
    app_script = html.index('<script src="/static/js/app.js"></script>')
    assert stream_script < voice_script < app_script


def test_voice_module_is_served(client):
    response = client.get("/static/js/voice.js")

    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert "createSpeaker" in response.text
    assert "function prime()" in response.text


def test_voice_controls_ship_hidden_off_and_labelled(client):
    html = client.get("/").text

    toggle = _opening_tag(html, "voiceToggle")
    assert _has_boolean_attribute(toggle, "hidden")
    assert 'type="button"' in toggle
    assert 'aria-pressed="false"' in toggle
    assert 'aria-describedby="voiceHint"' in toggle

    stop = _opening_tag(html, "voiceStop")
    assert _has_boolean_attribute(stop, "hidden")
    assert 'type="button"' in stop

    # The status line is a live region that is always rendered (never hidden),
    # so screen readers announce the messages written into it.
    status = _opening_tag(html, "voiceStatus")
    assert 'role="status"' in status
    assert 'aria-live="polite"' in status
    assert not _has_boolean_attribute(status, "hidden")

    assert 'class="sr-only"' in _opening_tag(html, "voiceHint")
    assert "Read answers aloud" in html
    assert "Stop speaking" in html


def test_hidden_voice_controls_are_not_shown_by_css():
    """.text-button sets display, which would override the hidden attribute."""
    css = (STATIC_JS.parent / "css" / "style.css").read_text(encoding="utf-8")

    assert re.search(r"\.voice-toggle\[hidden\],\s*\.voice-stop\[hidden\]\s*\{\s*display:\s*none;", css)


def test_voice_frontend_records_stores_and_uploads_nothing():
    # Since Voice V3b, app.js passes window.MediaRecorder to recorder.js (which
    # alone records); test_app_js_never_captures_the_microphone_itself pins that
    # app.js never constructs one. voice.js still never touches the microphone.
    for name in ("app.js", "voice.js"):
        source = (STATIC_JS / name).read_text(encoding="utf-8")
        for forbidden in ("localStorage", "sessionStorage", "document.cookie", "indexedDB", "getUserMedia"):
            assert forbidden not in source, f"{forbidden} in {name}"
    assert "MediaRecorder" not in (STATIC_JS / "voice.js").read_text(encoding="utf-8")


def test_voice_stays_disabled_by_default(client, settings):
    assert settings.voice_enabled is False
    assert client.get("/api/voice/status").status_code == 404


# ---------------------------------------------------------------------------
# Voice V3b: dictation markup, styles and script loading (behaviour: tests/js/app-dictation-view.test.mjs)
# ---------------------------------------------------------------------------

def _css() -> str:
    return (STATIC_JS.parent / "css" / "style.css").read_text(encoding="utf-8")


def test_page_loads_the_recorder_between_voice_and_the_app(client):
    html = client.get("/").text

    order = [
        '<script src="/static/js/chat-stream.js"></script>',
        '<script src="/static/js/voice.js"></script>',
        '<script src="/static/js/recorder.js"></script>',
        '<script src="/static/js/app.js"></script>',
    ]
    for tag in order:
        assert html.count(tag) == 1, tag
    positions = [html.index(tag) for tag in order]
    assert positions == sorted(positions)


def test_recorder_module_is_served(client):
    response = client.get("/static/js/recorder.js")

    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert "createRecorder" in response.text
    assert "BluePeakRecorder" in response.text


def test_dictation_button_ships_hidden_inside_the_composer(client):
    html = client.get("/").text

    button = _opening_tag(html, "dictateBtn")
    assert _has_boolean_attribute(button, "hidden")
    assert 'type="button"' in button  # never submits the question form
    assert 'aria-describedby="dictationNote"' in button
    assert 'data-state="idle"' in button
    assert "aria-pressed" not in button  # its label changes with its state instead
    assert not _has_boolean_attribute(button, "disabled")

    # Inside the question form, after the question box and before the send button.
    form_start = html.index('id="chatForm"')
    form_end = html.index("</form>", form_start)
    position = html.index('id="dictateBtn"')
    assert form_start < html.index('id="question"') < position < html.index('id="sendBtn"') < form_end

    # Its accessible name is its visible text.
    body = html[position: html.index("</button>", position)]
    assert '<span class="dictate-label">Dictate</span>' in body
    assert re.search(r'<svg[^>]*aria-hidden="true"', body)


def test_dictation_privacy_note_is_disclosed_with_the_button(client):
    html = client.get("/").text

    note_tag = _opening_tag(html, "dictationNote")
    # Hidden only together with the button: both are shown at the same time,
    # before any recording can start.
    assert _has_boolean_attribute(note_tag, "hidden")
    start = html.index('id="dictationNote"')
    note = re.sub(r"\s+", " ", html[html.index(">", start) + 1: html.index("</p>", start)]).strip()

    assert "sends your recording to OpenAI for transcription" in note
    assert "This app doesn't store the audio" in note  # verified: V2 and the recorder keep it in memory only
    assert "isn't sent automatically" in note
    # No unverified promises about what the provider retains or deletes.
    for claim in ("delete", "retain", "retention", "never stored", "not stored by OpenAI", "OpenAI doesn't", "OpenAI does not"):
        assert claim.lower() not in note.lower(), claim


def test_dictation_status_is_a_live_region_that_is_always_rendered(client):
    html = client.get("/").text

    status = _opening_tag(html, "dictationStatus")
    assert 'role="status"' in status
    assert 'aria-live="polite"' in status
    assert not _has_boolean_attribute(status, "hidden")
    start = html.index('id="dictationStatus"')
    assert html[html.index(">", start) + 1: html.index("</p>", start)].strip() == ""  # empty until used
    assert html.index('id="dictationStatus"') != html.index('id="voiceStatus"')  # separate from read-aloud


def test_hidden_dictation_controls_are_not_shown_by_css():
    """.dictate-btn sets display, which would override the hidden attribute."""
    assert re.search(
        r"\.dictate-btn\[hidden\],\s*\.dictation-note\[hidden\]\s*\{\s*display:\s*none;", _css()
    )


def test_the_recording_indicator_respects_reduced_motion():
    css = _css()

    assert re.search(r'\.dictate-btn\[data-state="recording"\]::before\s*\{[^}]*animation:\s*dictation-pulse', css)
    reduced = re.findall(r"@media \(prefers-reduced-motion: reduce\)\s*\{(.*?)\n\}", css, re.DOTALL)
    assert any(
        re.search(r'\.dictate-btn\[data-state="recording"\]::before\s*\{\s*animation:\s*none;', block)
        for block in reduced
    )


def test_the_dictation_button_fits_small_screens():
    small = re.findall(r"@media \(max-width: 640px\)\s*\{(.*?)\n\}", _css(), re.DOTALL)
    assert any(re.search(r"\.dictate-btn\s*\{[^}]*min-height:\s*35px", block) for block in small)


def test_app_js_never_captures_the_microphone_itself():
    """Microphone capture belongs to recorder.js; the page may only pass browser objects in."""
    app_js = (STATIC_JS / "app.js").read_text(encoding="utf-8")

    for forbidden in ("getUserMedia", "new MediaRecorder", "new window.MediaRecorder", "AudioContext"):
        assert forbidden not in app_js, forbidden


def test_the_recorder_stores_nothing_in_the_browser():
    source = (STATIC_JS / "recorder.js").read_text(encoding="utf-8")

    for forbidden in ("localStorage", "sessionStorage", "document.cookie", "indexedDB"):
        assert forbidden not in source, forbidden


def test_dictation_stays_disabled_by_default(client, settings):
    assert settings.voice_transcription_enabled is False
    assert client.get("/api/voice/transcription/status").status_code == 404


def test_the_privacy_note_is_readable():
    match = re.search(r"\.dictation-note\s*\{[^}]*font-size:\s*(\d+)px", _css())
    assert match and 12 <= int(match.group(1)) <= 13
