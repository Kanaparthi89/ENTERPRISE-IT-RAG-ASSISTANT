"""Frontend wiring for streaming chat (A1-5).

The streaming client's behaviour is tested with node:test in ``tests/js``
(``npm run test:js``). These tests check that the page loads it and that the
SSE fixture those tests use matches what the backend actually sends.
"""

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
