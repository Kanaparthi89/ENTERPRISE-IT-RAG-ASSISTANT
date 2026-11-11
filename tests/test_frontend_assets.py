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
    for name in ("app.js", "voice.js"):
        source = (STATIC_JS / name).read_text(encoding="utf-8")
        for forbidden in (
            "localStorage",
            "sessionStorage",
            "document.cookie",
            "indexedDB",
            "getUserMedia",
            "MediaRecorder",
        ):
            assert forbidden not in source, f"{forbidden} in {name}"


def test_voice_stays_disabled_by_default(client, settings):
    assert settings.voice_enabled is False
    assert client.get("/api/voice/status").status_code == 404
