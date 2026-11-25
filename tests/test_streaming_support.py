"""Tests for the A1-0 streaming test foundation (tests/streaming_support.py)."""

import asyncio
import json
import threading

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from starlette.applications import Starlette
from starlette.requests import ClientDisconnect
from starlette.responses import StreamingResponse
from starlette.routing import Route

import app.main
from app.rag import workflow
from app.rag.state import EvidenceGrade, RouteDecision
from tests.conftest import (
    TEST_SETTINGS,
    FakeLLM,
    FakeRetriever,
    kb_doc,
)
from tests.streaming_support import (
    ASGIHarness,
    ChunkScript,
    FakeClock,
    LoopbackServer,
    ScriptedChatModel,
    SSEProtocolError,
    assert_stream_grammar,
    body_contains,
    is_response_start,
    parse_sse,
)

ANSWER_NODES = {"generate_from_kb", "generate_from_web", "direct_answer"}


def initial_state(question):
    return {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }


class TokenRecorder(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []
        self.chunks = []

    def on_llm_new_token(self, token, *, chunk=None, **kwargs):
        self.tokens.append(token)
        self.chunks.append(chunk)


# -------------------------------------------------------------------
# Scripted chunk models
# -------------------------------------------------------------------

def test_from_text_split_is_lossless():
    text = "\n\n  Install GlobalProtect.\n\n- Step one\n- Step two  \n"
    script = ChunkScript.from_text(text)

    assert script.full_text() == text
    assert script.count > 1


def test_streaming_fake_emits_chunks_through_the_real_graph(fake_rag):
    raw = "\n Install GlobalProtect and\n sign in with MFA.  \n"
    fake_rag(
        llm=FakeLLM(
            structured={RouteDecision: [{"route": "kb"}], EvidenceGrade: [{"grade": "good"}]},
            text=[ChunkScript.from_text(raw)],
        ),
        retriever=FakeRetriever([kb_doc("VPN uses GlobalProtect.")]),
    )

    chunks_by_node, final = {}, None
    for mode, payload in workflow.agent_graph.stream(
        initial_state("How do I connect to the VPN?"),
        stream_mode=["messages", "values"],
    ):
        if mode == "messages":
            message, metadata = payload
            chunks_by_node.setdefault(metadata["langgraph_node"], []).append(message.content)
        else:
            final = payload

    answer_chunks = chunks_by_node.pop("generate_from_kb")
    assert chunks_by_node == {}, "only the answer node may emit chunks"
    assert "".join(answer_chunks) == raw
    assert answer_chunks[-1] == ""  # final empty chunk added by langchain-core
    assert final["answer"] == raw.strip()


def test_streaming_fake_also_works_without_streaming(fake_rag):
    script = ChunkScript.from_text("  Use the VPN client.  ")
    fake_rag(
        llm=FakeLLM(
            structured={RouteDecision: [{"route": "kb"}], EvidenceGrade: [{"grade": "good"}]},
            text=[script],
        ),
        retriever=FakeRetriever([kb_doc("VPN policy.")]),
    )

    result = workflow.ask("VPN?")

    assert result["answer"] == "Use the VPN client."
    assert script.produced == script.count
    assert script.closed


def test_stream_true_converts_text_responses(fake_rag):
    fake_rag(
        llm=FakeLLM(
            structured={RouteDecision: [{"route": "direct"}]},
            text=["Hello there!"],
            stream=True,
        )
    )

    tokens = [
        message.content
        for mode, (message, metadata) in workflow.agent_graph.stream(
            initial_state("hi"), stream_mode=["messages"]
        )
    ]

    assert "".join(tokens) == "Hello there!"
    assert len(tokens) > 2


def test_pause_after_blocks_the_model_thread_until_resumed(fake_rag):
    script = ChunkScript.from_text("one two three four five", pause_after=3)
    fake_rag(
        llm=FakeLLM(
            structured={RouteDecision: [{"route": "direct"}]},
            text=[script],
        )
    )
    outcome = {}

    def run():
        outcome["result"] = list(
            workflow.agent_graph.stream(initial_state("hi"), stream_mode="values")
        )[-1]

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    assert script.paused.wait(5), "script never paused"
    assert script.produced == 3
    assert not script.closed

    script.resume.set()
    thread.join(5)

    assert not thread.is_alive()
    assert outcome["result"]["answer"] == "one two three four five"
    assert script.closed


def test_fail_after_raises_after_the_scripted_chunks():
    script = ChunkScript(["a", "b", "c", "d"], fail_after=2, error=ValueError("boom"))
    recorder = TokenRecorder()

    with pytest.raises(ValueError, match="boom"):
        ScriptedChatModel(script=script, streaming=True).invoke(
            "x", config={"callbacks": [recorder]}
        )

    assert recorder.tokens == ["a", "b"]
    assert script.produced == 2
    assert script.closed


def test_list_content_chunks_aggregate_to_text():
    script = ChunkScript(["Use ", "the ", "VPN."], list_content=True)
    recorder = TokenRecorder()

    message = ScriptedChatModel(script=script, streaming=True).invoke(
        "x", config={"callbacks": [recorder]}
    )

    non_empty = [chunk for chunk in recorder.chunks if chunk.text]
    assert [chunk.text for chunk in non_empty] == ["Use ", "the ", "VPN."]
    assert all(isinstance(chunk.message.content, list) for chunk in non_empty)
    # langchain-core passes the chunk *content* as the token argument, so for
    # list content the "token" is a list: forwarders must read chunk text.
    assert isinstance(recorder.tokens[0], list)
    assert workflow._message_text(message) == "Use the VPN."


def test_large_output_is_generated_lazily():
    script = ChunkScript(count=20_000)
    recorder = TokenRecorder()

    ScriptedChatModel(script=script, streaming=True).invoke(
        "x", config={"callbacks": [recorder]}
    )

    assert script.produced == 20_000
    assert len(recorder.tokens) == 20_001  # plus langchain-core's final empty chunk
    assert recorder.tokens[-1] == ""
    assert script._chunks is None


def test_fake_llm_records_tags_from_config():
    llm = FakeLLM(structured={RouteDecision: [{"route": "kb"}]}, text=["ok"])

    llm.invoke("plain", config={"tags": ["nostream"]})
    llm.with_structured_output(RouteDecision).invoke("route", config={"tags": ["nostream"]})

    assert [call["tags"] for call in llm.calls] == [["nostream"], ["nostream"]]
    assert [call["kind"] for call in llm.calls] == ["text", "structured"]
    assert llm.calls[1]["schema"] is RouteDecision


def test_fake_llm_calls_without_config_have_no_tags():
    llm = FakeLLM(text=["ok"])

    llm.invoke("plain")

    assert llm.calls[0]["tags"] == []


# -------------------------------------------------------------------
# SSE parser
# -------------------------------------------------------------------

def frame(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def test_parse_valid_stream_with_keepalive():
    raw = (
        frame("start", {"request_id": "r1", "version": 1})
        + b": keepalive\n\n"
        + frame("delta", {"text": "line one\nline é two"})
        + frame("done", {"answer": "x"})
    )

    parsed = parse_sse(raw)

    assert parsed.names == ["start", "delta", "done"]
    assert parsed.keepalives == 1
    assert parsed.events[1].data["text"] == "line one\nline é two"
    assert_stream_grammar(parsed, connected=True)


def test_disconnected_stream_grammar():
    parsed = parse_sse(frame("start", {}) + frame("delta", {"text": "a"}))

    assert_stream_grammar(parsed, connected=False)
    with pytest.raises(SSEProtocolError):
        assert_stream_grammar(parsed, connected=True)


@pytest.mark.parametrize(
    "names, connected",
    [
        ([], True),
        (["delta", "done"], True),
        (["start", "done", "error"], True),
        (["start", "done", "delta"], True),
        (["start", "start", "done"], True),
        (["start", "done"], False),
        (["delta", "start"], False),
    ],
)
def test_grammar_violations_are_rejected(names, connected):
    parsed = parse_sse(b"".join(frame(name, {}) for name in names))

    with pytest.raises(SSEProtocolError):
        assert_stream_grammar(parsed, connected=connected)


@pytest.mark.parametrize(
    "raw",
    [
        b"event: start\ndata: {}\n",  # incomplete final frame
        b"event: start\r\ndata: {}\r\n\r\n",  # CR characters
        b"event: unknown\ndata: {}\n\n",  # unknown event name
        b"event: start\ndata: {not json}\n\n",  # invalid JSON
        b"data: {}\nevent: start\n\n",  # wrong line order
        b"event: start\ndata: {}\ndata: {}\n\n",  # extra line
        b"event: start\ndata: {}\n\n\n\n",  # empty frame
        b"event: start\ndata: \"\xff\"\n\n",  # invalid UTF-8
    ],
)
def test_malformed_streams_are_rejected(raw):
    with pytest.raises(SSEProtocolError):
        parse_sse(raw)


def test_empty_body_parses_to_no_events():
    assert parse_sse(b"").events == []


# -------------------------------------------------------------------
# Fake clock
# -------------------------------------------------------------------

def test_fake_clock_moves_only_when_advanced():
    clock = FakeClock(start=10.0)

    assert clock() == 10.0
    clock.advance(2.5)
    assert clock() == 12.5
    with pytest.raises(ValueError):
        clock.advance(-1)


# -------------------------------------------------------------------
# ASGI harness
# -------------------------------------------------------------------

def chunk_stream_app(chunks, finished, *, endless=False):
    async def body():
        try:
            index = 0
            while True:
                if index < len(chunks):
                    yield chunks[index]
                elif not endless:
                    return
                else:
                    yield f"chunk-{index}\n".encode()
                index += 1
                await asyncio.sleep(0)
                if index > 10_000:
                    raise AssertionError("stream was never cancelled")
        finally:
            finished.append(True)

    return StreamingResponse(body(), media_type="text/event-stream")


@pytest.mark.anyio
async def test_harness_collects_a_complete_response():
    finished = []
    harness = ASGIHarness(chunk_stream_app([b"a\n", b"b\n"], finished))

    result = await harness.request("GET", "/")

    assert result.exception is None
    assert result.status == 200
    assert result.headers["content-type"].startswith("text/event-stream")
    assert result.body == b"a\nb\n"
    assert result.complete
    assert finished == [True]


@pytest.mark.anyio
async def test_spec_2_3_disconnect_cancels_the_stream():
    finished = []
    harness = ASGIHarness(chunk_stream_app([], finished, endless=True), spec_version="2.3")
    harness.disconnect_on(body_contains(b"chunk-3\n"))

    with anyio_guard():
        result = await harness.request("GET", "/")

    assert result.exception is None
    assert not result.complete
    assert finished == [True]
    assert len(harness.attempted) < 50


@pytest.mark.anyio
async def test_spec_2_4_send_failure_raises_client_disconnect():
    finished = []
    harness = ASGIHarness(
        chunk_stream_app([b"one\n", b"two\n", b"three\n"], finished),
        spec_version="2.4",
    )
    harness.fail_send(body_contains(b"two\n"))

    result = await harness.request("GET", "/")

    assert isinstance(result.exception, ClientDisconnect)
    assert result.body == b"one\n"
    assert not result.complete


@pytest.mark.anyio
async def test_send_gate_pauses_the_app_until_released():
    finished = []
    harness = ASGIHarness(chunk_stream_app([b"one\n", b"two\n", b"three\n"], finished))
    gate = harness.gate(body_contains(b"two\n"))

    task = asyncio.ensure_future(harness.request("GET", "/"))
    with anyio_guard():
        await gate.reached.wait()

    assert harness.attempted[-1]["body"] == b"two\n"
    assert b"two" not in b"".join(item.message.get("body", b"") for item in harness.sent)
    assert not task.done()

    gate.release.set()
    result = await task

    assert result.body == b"one\ntwo\nthree\n"
    assert result.complete


@pytest.mark.anyio
async def test_uvicorn_faithful_disconnect_releases_a_blocked_send_silently():
    finished = []
    harness = ASGIHarness(chunk_stream_app([b"one\n"], finished), uvicorn_faithful=True)
    gate = harness.gate(is_response_start)

    task = asyncio.ensure_future(harness.request("GET", "/"))
    with anyio_guard():
        await gate.reached.wait()
    harness.disconnect()
    with anyio_guard():
        result = await task

    assert result.delivered == []
    assert result.status is None
    assert not result.complete
    assert all(item.dropped for item in harness.sent)


@pytest.mark.anyio
async def test_non_faithful_mode_raises_after_disconnect():
    finished = []
    harness = ASGIHarness(
        chunk_stream_app([b"one\n", b"two\n"], finished),
        spec_version="2.4",
        uvicorn_faithful=False,
    )
    harness.disconnect_on(body_contains(b"one\n"))

    result = await harness.request("GET", "/")

    assert isinstance(result.exception, ClientDisconnect)


@pytest.mark.anyio
async def test_harness_runs_the_real_application():
    harness = ASGIHarness(app.main.app)

    result = await harness.request("GET", "/api/health")

    assert result.status == 200
    assert json.loads(result.body) == {"status": "ok", "service": TEST_SETTINGS.app_name}


class anyio_guard:
    """Fail instead of hanging if an awaited condition never happens."""

    def __enter__(self):
        import anyio

        self._scope = anyio.fail_after(5)
        return self._scope.__enter__()

    def __exit__(self, *exc_info):
        return self._scope.__exit__(*exc_info)


# -------------------------------------------------------------------
# Loopback server
# -------------------------------------------------------------------

def test_loopback_server_serves_the_app_on_standard_asyncio(blocked_network):
    with LoopbackServer(app.main.app) as server:
        with httpx.Client(base_url=server.base_url, trust_env=False, timeout=5) as client:
            response = client.get("/api/health")

    assert response.status_code == 200
    assert server.loop_class.__module__.startswith("asyncio.")
    assert blocked_network() == []


def test_loopback_server_delivers_chunks_incrementally():
    release = threading.Event()

    async def endpoint(request):
        async def body():
            yield b"first\n"
            for _ in range(500):
                if release.is_set():
                    break
                await asyncio.sleep(0.01)
            yield b"second\n"

        return StreamingResponse(body(), media_type="text/event-stream")

    streaming_app = Starlette(routes=[Route("/", endpoint)])

    with LoopbackServer(streaming_app) as server:
        with httpx.Client(base_url=server.base_url, trust_env=False, timeout=5) as client:
            with client.stream("GET", "/") as response:
                chunks = response.iter_raw()
                first = next(chunks)
                # Received over real TCP while the server is still holding "second".
                assert first == b"first\n"
                assert not release.is_set()
                release.set()
                rest = b"".join(chunks)

    assert rest == b"second\n"
