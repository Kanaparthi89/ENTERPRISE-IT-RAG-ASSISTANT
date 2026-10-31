"""Tests for A1-1 streaming support in the workflow (no HTTP layer yet)."""

import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tracers._streaming import _StreamingCallbackHandler
from langchain_openai import ChatOpenAI
from langgraph.constants import TAG_NOSTREAM
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.rag import vectorstore, workflow
from app.rag.state import EvidenceGrade, RouteDecision
from tests.conftest import (
    TEST_SETTINGS,
    FakeLLM,
    FakeRetriever,
    FakeWebSearch,
    kb_doc,
    web_result,
)
from tests.streaming_support import ChunkScript, ScriptedChatModel

GOOD = {"grade": "good"}
WEAK = {"grade": "weak"}


@pytest.fixture(autouse=True)
def two_retries(settings, monkeypatch):
    monkeypatch.setattr(settings, "max_retries", 2)


def run_stream(question):
    """Collect deltas and yielded events in arrival order."""
    deltas, timeline, final = [], [], None

    def on_chunk(text):
        deltas.append(text)
        timeline.append(("delta", text))

    for kind, payload in workflow.stream_ask(question, on_chunk):
        if kind == "trace":
            timeline.append(("trace", payload))
        else:
            final = payload
    return deltas, timeline, final


def kb_script(text, *, grade=GOOD, **llm_kwargs):
    return FakeLLM(
        structured={RouteDecision: [{"route": "kb"}], EvidenceGrade: [grade]},
        text=[text],
        **llm_kwargs,
    )


# -------------------------------------------------------------------
# String chunks, empty final chunk, ordering, parity with ask()
# -------------------------------------------------------------------

def test_kb_answer_streams_string_chunks_in_order(fake_rag):
    raw = "\n\n  Install GlobalProtect.\n\n- Step one\n- Step two  \n"
    fake_rag(
        llm=kb_script(ChunkScript.from_text(raw)),
        retriever=FakeRetriever([kb_doc("VPN uses GlobalProtect.")]),
    )

    deltas, timeline, final = run_stream("How do I connect to the VPN?")

    assert "".join(deltas) == raw
    assert "".join(deltas).strip() == final["answer"]
    assert "" not in deltas  # langchain-core's final empty chunk is not forwarded
    assert [step for kind, step in timeline if kind == "trace"] == final["trace"]

    kinds = [kind if kind == "delta" else step for kind, step in timeline]
    first_delta = kinds.index("delta")
    last_delta = len(kinds) - 1 - kinds[::-1].index("delta")
    assert kinds.index("KB evidence grade → GOOD") < first_delta
    assert last_delta < kinds.index("Answer generation → PRIVATE KB")


def test_stream_final_state_matches_ask(fake_rag):
    retriever_docs = [kb_doc("VPN uses GlobalProtect.", source="/kb/policy.pdf", page=0)]

    fake_rag(llm=kb_script("Install GlobalProtect."), retriever=FakeRetriever(retriever_docs))
    expected = workflow.ask("How do I connect to the VPN?")

    fake_rag(
        llm=kb_script("Install GlobalProtect.", stream=True),
        retriever=FakeRetriever(retriever_docs),
    )
    _, _, final = run_stream("How do I connect to the VPN?")

    for key in ("answer", "source_used", "trace", "citations", "current_query"):
        assert final[key] == expected[key], key


def test_ask_does_not_stream(fake_rag):
    script = ChunkScript.from_text("Use the VPN.")
    fake_rag(llm=kb_script(script), retriever=FakeRetriever([kb_doc("VPN.")]))

    result = workflow.ask("VPN?")

    # Without a streaming handler the model's non-streaming path is used.
    assert result["answer"] == "Use the VPN."
    assert script.produced == script.count


# -------------------------------------------------------------------
# Content blocks
# -------------------------------------------------------------------

def test_indexed_list_content_streams_plain_text(fake_rag):
    fake_rag(
        llm=kb_script(ChunkScript(["Use ", "the ", "VPN ", "client."], list_content=True)),
        retriever=FakeRetriever([kb_doc("VPN.")]),
    )

    deltas, _, final = run_stream("VPN?")

    assert deltas == ["Use ", "the ", "VPN ", "client."]
    assert "".join(deltas).strip() == final["answer"] == "Use the VPN client."


def forward_chunks(contents, *, node="generate_from_kb", tags=None):
    """Feed chunk contents through a forwarder; return forwarded texts and the aggregate."""
    forwarded = []
    forwarder = workflow.AnswerChunkForwarder(forwarded.append)
    forwarder.on_chat_model_start(
        {}, [[]], run_id="run-1", tags=tags or [], metadata={"langgraph_node": node}
    )
    aggregate = None
    for content in contents:
        message = AIMessageChunk(content=content)
        aggregate = message if aggregate is None else aggregate + message
        forwarder.on_llm_new_token(content, chunk=ChatGenerationChunk(message=message), run_id="run-1")
    forwarder.on_llm_end(None, run_id="run-1")
    return forwarded, aggregate


def text_before_strip(aggregate):
    """What ``_message_text`` returns for the aggregate, before ``.strip()``.

    Computed with the real function: non-whitespace sentinels around the
    content survive the strip, so slicing them off leaves the joined text.
    """
    content = aggregate.content
    if isinstance(content, str):
        return workflow._message_text("\x00" + content + "\x00")[1:-1]
    return workflow._message_text(["\x00", *content, "\x00"])[2:-2]


def text_block(text, index=None):
    block = {"type": "text", "text": text}
    if index is not None:
        block["index"] = index
    return block


@pytest.mark.parametrize(
    "contents",
    [
        pytest.param(
            [
                [text_block("Para one ", 0)],
                [text_block("continues.", 0)],
                [{"type": "reasoning", "reasoning": "hidden", "index": 1}],
                [text_block("Para two.", 2)],
                [],
            ],
            id="indexed-blocks",
        ),
        pytest.param([[text_block("A")], [text_block("B")], []], id="unindexed-blocks"),
        pytest.param([["first"], ["second"]], id="string-items"),
        pytest.param(["plain ", "string ", "chunks", ""], id="string-content"),
        pytest.param(["\n  ", "Answer", "  \n", ""], id="string-content-surrounding-whitespace"),
        # Empty text blocks are parts too: _message_text joins them with "\n".
        pytest.param([[text_block("", 0)], [text_block("A", 1)], []], id="empty-indexed-begin"),
        pytest.param(
            [[text_block("A", 0)], [text_block("", 1)], [text_block("B", 2)], []],
            id="empty-indexed-middle",
        ),
        pytest.param([[text_block("A", 0)], [text_block("", 1)], []], id="empty-indexed-end"),
        pytest.param([[text_block("")], [text_block("A")]], id="empty-unindexed-begin"),
        pytest.param(
            [[text_block("A")], [text_block("")], [text_block("B")]],
            id="empty-unindexed-middle",
        ),
        pytest.param([[text_block("A")], [text_block("")]], id="empty-unindexed-end"),
        pytest.param([[""], ["A"], [""]], id="empty-string-items"),
        pytest.param(
            [[text_block("", 0)], [text_block("A", 0)], [text_block("B", 0)]],
            id="empty-block-later-filled",
        ),
        pytest.param(
            [[text_block("A", 0), text_block("", 1), text_block("B", 2)], []],
            id="empty-block-within-one-chunk",
        ),
        pytest.param(
            [
                ["lead"],
                [text_block("", 0)],
                [text_block("A", 0)],
                [{"type": "reasoning", "reasoning": "x", "index": 1}],
                [text_block("")],
                [text_block("B", 2)],
                [""],
                [],
            ],
            id="mixed-items",
        ),
        # String/list transitions follow langchain-core's merge_content.
        pytest.param(["Hello ", [text_block("world", 0)]], id="string-then-blocks"),
        pytest.param(["", [text_block("A", 0)]], id="empty-string-then-blocks"),
        pytest.param([[text_block("A", 0)], "tail"], id="blocks-then-string"),
        pytest.param([[text_block("A", 0)], ""], id="blocks-then-empty-string"),
        pytest.param([["x"], "y"], id="string-item-then-string"),
        # Index forms: only int and "lc_" indexes merge; the first chunk is kept as-is.
        pytest.param(
            [[text_block("A", "lc_1")], [text_block("B", "lc_1")]], id="lc-string-index"
        ),
        pytest.param([[text_block("A", "x")], [text_block("B", "x")]], id="other-string-index"),
        pytest.param([[text_block("A", 0), text_block("B", 0)]], id="first-chunk-same-index"),
    ],
)
def test_streamed_text_equals_message_text_before_stripping(contents):
    forwarded, aggregate = forward_chunks(contents)

    assert "" not in forwarded
    assert "".join(forwarded) == text_before_strip(aggregate)
    assert "".join(forwarded).strip() == workflow._message_text(aggregate)


def test_writing_back_into_an_earlier_block_is_detected(caplog):
    """Append-only deltas cannot express text inserted into an earlier block."""
    contents = [[text_block("A", 0)], [text_block("B", 1)], [text_block("C", 0)], [text_block("D", 1)]]

    with caplog.at_level("WARNING", logger="app.rag.workflow"):
        forwarded, aggregate = forward_chunks(contents)

    assert text_before_strip(aggregate) == "AC\nBD"
    assert forwarded == ["A", "\nB", "D"]  # continues after the unrepresentable change
    warnings = [record for record in caplog.records if "changed already-sent text" in record.message]
    assert len(warnings) == 1  # logged once
    assert warnings[0].args == ()  # no answer content interpolated into the log


def test_string_content_fast_path_does_not_aggregate():
    tracker = workflow._AnswerTextTracker()

    assert [tracker.add(text) for text in ["a", "", "b "]] == ["a", "", "b "]
    assert tracker._aggregate is None
    assert not tracker.diverged


def test_text_before_strip_helper_keeps_surrounding_whitespace():
    aggregate = AIMessageChunk(content=[text_block("", 0), text_block("A", 1), text_block("", 2)])

    assert text_before_strip(aggregate) == "\nA\n"
    assert workflow._message_text(aggregate) == "A"


def test_non_text_blocks_and_empty_chunks_are_not_forwarded():
    forwarded, _ = forward_chunks(
        [
            [{"type": "reasoning", "reasoning": "x", "index": 0}],
            [{"type": "text", "text": "", "index": 1}],
            [],
            "",
        ]
    )

    assert forwarded == []


# -------------------------------------------------------------------
# Answer-node filtering, nostream, internal calls
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    "node, tags, forwarded",
    [
        ("generate_from_kb", [], True),
        ("generate_from_web", [], True),
        ("direct_answer", [], True),
        ("route_question", [], False),
        ("grade_kb", [], False),
        ("rewrite_query", [], False),
        ("generate_from_kb", [TAG_NOSTREAM], False),
        (None, [], False),
    ],
)
def test_forwarder_only_forwards_untagged_answer_node_runs(node, tags, forwarded):
    texts, _ = forward_chunks(["hello"], node=node, tags=tags)

    assert texts == (["hello"] if forwarded else [])


def test_internal_calls_are_tagged_and_rewrites_never_stream(fake_rag):
    llm, _, _ = fake_rag(
        llm=FakeLLM(
            structured={RouteDecision: [{"route": "kb"}], EvidenceGrade: [WEAK]},
            text=["rewrite one", "rewrite two"],
            stream=True,  # rewrites would stream if they were not tagged
        ),
        retriever=FakeRetriever([kb_doc("Unrelated.")]),
        web=FakeWebSearch({"results": []}),
    )

    deltas, _, final = run_stream("How do I request a second monitor?")

    assert final["source_used"] == "insufficient_evidence"
    assert deltas == [final["answer"]]  # only the fixed-text answer
    assert not any("rewrite" in delta for delta in deltas)
    assert [call["tags"] for call in llm.calls] == [[TAG_NOSTREAM]] * len(llm.calls)
    assert {call["kind"] for call in llm.calls} == {"structured", "text"}


def test_answer_call_is_not_tagged(fake_rag):
    llm, _, _ = fake_rag(llm=kb_script("Answer."), retriever=FakeRetriever([kb_doc("Doc.")]))

    workflow.ask("Question?")

    answer_calls = [call for call in llm.calls if call["kind"] == "text"]
    internal_calls = [call for call in llm.calls if call["kind"] == "structured"]
    assert [call["tags"] for call in answer_calls] == [[]]
    assert all(call["tags"] == [TAG_NOSTREAM] for call in internal_calls)


def test_nostream_tag_propagates_through_chains():
    """A tag passed to a chain reaches the nested chat model, as for json_mode."""

    class S(TypedDict):
        out: str

    script = ChunkScript.from_text("internal json output")

    def generate_from_kb(state):  # an answer-named node, to prove the tag alone filters
        chain = ScriptedChatModel(script=script) | StrOutputParser()
        return {"out": chain.invoke("x", config={"tags": [TAG_NOSTREAM]})}

    graph = StateGraph(S)
    graph.add_node("generate_from_kb", generate_from_kb)
    graph.add_edge(START, "generate_from_kb")
    graph.add_edge("generate_from_kb", END)

    forwarded = []
    events = list(
        graph.compile().stream(
            {"out": ""},
            config={"callbacks": [workflow.AnswerChunkForwarder(forwarded.append)]},
            stream_mode=["messages", "values"],
        )
    )

    assert forwarded == []
    assert [mode for mode, _ in events if mode == "messages"] == []
    assert events[-1][1]["out"] == "internal json output"


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

def test_direct_answer_streams(fake_rag):
    fake_rag(
        llm=FakeLLM(structured={RouteDecision: [{"route": "direct"}]}, text=["Hello there!"], stream=True)
    )

    deltas, _, final = run_stream("hi")

    assert len(deltas) > 1
    assert "".join(deltas).strip() == final["answer"] == "Hello there!"


def test_web_answer_streams_and_keeps_markers(fake_rag):
    fake_rag(
        llm=FakeLLM(structured={EvidenceGrade: [GOOD]}, text=["Use MFA [1] and [9]."], stream=True),
        web=FakeWebSearch({"results": [web_result("NIST", "https://www.nist.gov/a")]}),
    )

    deltas, _, final = run_stream("According to NIST, how do I secure accounts?")

    assert "".join(deltas) == final["answer"] == "Use MFA [1] and [9]."
    assert len(final["citations"]) == 1


def test_fixed_text_answer_is_delivered_as_one_chunk(fake_rag, settings, monkeypatch):
    monkeypatch.setattr(settings, "max_retries", 0)
    fake_rag(web=FakeWebSearch({"results": []}))

    deltas, timeline, final = run_stream("Cite official guidance on backups.")

    assert final["source_used"] == "insufficient_evidence"
    assert deltas == [final["answer"]]
    assert timeline[-1] == ("delta", final["answer"])  # after every trace step


# -------------------------------------------------------------------
# Abort and early close
# -------------------------------------------------------------------

class StopStream(Exception):
    pass


def test_sink_exception_aborts_the_model_call(fake_rag):
    script = ChunkScript(count=1_000)
    fake_rag(llm=kb_script(script), retriever=FakeRetriever([kb_doc("Doc.")]))
    received = []

    def on_chunk(text):
        received.append(text)
        if len(received) == 3:
            raise StopStream()

    with pytest.raises(StopStream):
        list(workflow.stream_ask("Question?", on_chunk))

    assert len(received) == 3
    assert script.produced <= 4
    assert script.closed


def test_closing_the_stream_early_stops_later_nodes(fake_rag):
    _, retriever, _ = fake_rag(
        llm=kb_script("Answer."),
        retriever=FakeRetriever([kb_doc("Doc.")]),
    )

    stream = workflow.stream_ask("Question?", lambda text: None)
    assert next(stream) == ("trace", "Router → KB")
    stream.close()

    assert retriever.queries == []


def test_blank_question_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        next(workflow.stream_ask("   ", lambda text: None))


# -------------------------------------------------------------------
# Graph guard and client configuration
# -------------------------------------------------------------------

def test_answer_nodes_are_terminal():
    builder = workflow.agent_graph.builder

    for node in workflow.ANSWER_NODES:
        assert {end for start, end in builder.edges if start == node} == {END}, node
        assert node not in builder.branches, node


class RecordingClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        RecordingClient.instances.append(self)


@pytest.fixture
def recording_clients(monkeypatch):
    RecordingClient.instances = []
    monkeypatch.setattr(workflow, "ChatOpenAI", RecordingClient)
    monkeypatch.setattr(vectorstore, "OpenAIEmbeddings", RecordingClient)
    return RecordingClient.instances


def test_chat_models_use_configured_timeouts(recording_clients):
    answer = workflow.llm()
    internal = workflow.internal_llm()

    assert workflow.llm() is answer and workflow.internal_llm() is internal
    assert len(recording_clients) == 2
    for client in (answer, internal):
        assert client.kwargs["timeout"] == TEST_SETTINGS.openai_timeout_seconds
        assert client.kwargs["max_retries"] == TEST_SETTINGS.openai_max_retries
        assert client.kwargs["temperature"] == 0
    assert "disable_streaming" not in answer.kwargs
    assert internal.kwargs["disable_streaming"] is True


def test_embeddings_use_configured_timeouts(recording_clients):
    embeddings = vectorstore.get_embeddings()

    assert embeddings.kwargs["timeout"] == TEST_SETTINGS.openai_timeout_seconds
    assert embeddings.kwargs["max_retries"] == TEST_SETTINGS.openai_max_retries
    assert embeddings.kwargs["model"] == TEST_SETTINGS.embedding_model


def test_timeout_defaults_are_finite():
    assert TEST_SETTINGS.openai_timeout_seconds == 60.0
    assert TEST_SETTINGS.openai_max_retries == 2


class _AttachedStreamingHandler(_StreamingCallbackHandler):
    """Stands in for LangGraph's messages handler when checking _should_stream."""

    def tap_output_iter(self, run_id, output):
        return output

    def tap_output_aiter(self, run_id, output):
        return output


class _RunManager:
    def __init__(self, handlers):
        self.handlers = handlers


def test_real_chatopenai_internal_model_never_streams():
    """Verified on the real class, without any network call."""
    kwargs = dict(model="gpt-4o-mini", api_key="test-key", timeout=60.0, max_retries=2)
    run_manager = _RunManager([_AttachedStreamingHandler()])

    answer_model = ChatOpenAI(**kwargs)
    internal_model = ChatOpenAI(**kwargs, disable_streaming=True)

    assert answer_model._should_stream(async_api=False, run_manager=run_manager) is True
    assert internal_model._should_stream(async_api=False, run_manager=run_manager) is False
    assert answer_model._should_stream(async_api=False, run_manager=_RunManager([])) is False

    structured = internal_model.with_structured_output(RouteDecision, method="json_mode")
    assert structured.first.bound is internal_model
