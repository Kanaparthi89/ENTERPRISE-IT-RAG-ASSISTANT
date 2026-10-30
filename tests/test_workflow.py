"""Tests for the LangGraph workflow, run end to end against fakes."""

import pytest
from langchain_core.messages import AIMessage

from app.rag import workflow
from app.rag.state import EvidenceGrade, RouteDecision
from tests.conftest import (
    FakeLLM,
    FakeRetriever,
    FakeWebSearch,
    KnownBugStillPresent,
    kb_doc,
    require,
    web_result,
)

GOOD = {"grade": "good"}
WEAK = {"grade": "weak"}


@pytest.fixture(autouse=True)
def two_retries(settings, monkeypatch):
    monkeypatch.setattr(settings, "max_retries", 2)


# -------------------------------------------------------------------
# Routing
# -------------------------------------------------------------------

def test_direct_route_skips_retrieval(fake_rag):
    llm, retriever, web = fake_rag(
        llm=FakeLLM(
            structured={RouteDecision: [{"route": "direct"}]},
            text=["Hello! How can I help?"],
        )
    )

    result = workflow.ask("hi there")

    assert result["answer"] == "Hello! How can I help?"
    assert result["source_used"] == "direct"
    assert result["citations"] == []
    assert result["trace"] == [
        "Router → DIRECT",
        "Direct response → no retrieval",
    ]
    assert retriever.queries == []
    assert web.queries == []


def test_official_source_keywords_bypass_llm_router(fake_rag):
    # No RouteDecision is scripted: calling the LLM router would fail.
    llm, retriever, web = fake_rag(
        llm=FakeLLM(
            structured={EvidenceGrade: [GOOD]},
            text=["Prepare, detect, contain [1][2][7]."],
        ),
        web=FakeWebSearch(
            {
                "results": [
                    web_result("NIST IR", "https://www.nist.gov/ir"),
                    web_result("Blog", "https://example.com/nist"),
                    web_result("CSRC", "https://csrc.nist.gov/ir"),
                    web_result("Dup", "https://www.nist.gov/ir/"),
                    web_result("FTP", "ftp://nist.gov/file"),
                    web_result("Empty", "https://nist.gov/empty", content=""),
                ]
            }
        ),
    )

    result = workflow.ask("According to NIST, what are the incident response phases?")

    assert result["trace"][0] == "Router → OFFICIAL EXTERNAL SOURCES"
    assert web.include_domains == [["nist.gov"]]
    assert retriever.queries == []
    assert result["source_used"] == "web_search"
    # Off-domain, duplicate, non-HTTP and empty results are dropped, and
    # the out-of-range [7] reference is ignored.
    assert [citation["url"] for citation in result["citations"]] == [
        "https://www.nist.gov/ir",
        "https://csrc.nist.gov/ir",
    ]


# -------------------------------------------------------------------
# Private knowledge base
# -------------------------------------------------------------------

def test_good_kb_evidence_answers_from_kb(fake_rag):
    llm, retriever, web = fake_rag(
        llm=FakeLLM(
            structured={RouteDecision: [{"route": "kb"}], EvidenceGrade: [GOOD]},
            text=["Install GlobalProtect and sign in with MFA."],
        ),
        retriever=FakeRetriever(
            [
                kb_doc("VPN uses GlobalProtect.", source="/kb/handbook.md"),
                kb_doc("Sign in with MFA.", source="/kb/handbook.md"),
                kb_doc("Remote access policy.", source="/kb/policy.pdf", page=0),
            ]
        ),
    )

    result = workflow.ask("How do I connect to the VPN?")

    assert result["answer"] == "Install GlobalProtect and sign in with MFA."
    assert result["source_used"] == "private_kb"
    assert result["citations"] == [
        {"title": "handbook.md", "url": "", "type": "private_kb"},
        {"title": "policy.pdf", "url": "", "type": "private_kb", "page": 1},
    ]
    assert result["trace"] == [
        "Router → KB",
        "Private KB retrieval → 3 chunks",
        "KB evidence grade → GOOD",
        "Answer generation → PRIVATE KB",
    ]
    assert retriever.queries == ["How do I connect to the VPN?"]
    assert web.queries == []
    assert "VPN uses GlobalProtect." in llm.prompts[-1]


@pytest.mark.characterizes_bug("B5")
def test_weak_kb_evidence_falls_back_to_web(fake_rag):
    llm, retriever, web = fake_rag(
        llm=FakeLLM(
            structured={
                RouteDecision: [{"route": "kb"}],
                EvidenceGrade: [WEAK, GOOD],
            },
            text=["Restart the Teams client [2]."],
        ),
        retriever=FakeRetriever([kb_doc("Unrelated printer policy.")]),
        web=FakeWebSearch(
            {
                "results": [
                    web_result("Status", "https://status.example.com"),
                    web_result("Teams outage", "https://learn.microsoft.com/teams"),
                ]
            }
        ),
    )

    result = workflow.ask("What is the latest Microsoft Teams outage guidance?")

    assert result["source_used"] == "web_search"
    assert result["citations"] == [
        {
            "title": "Teams outage",
            "url": "https://learn.microsoft.com/teams",
            "type": "web",
        }
    ]
    assert result["trace"] == [
        "Router → KB",
        "Private KB retrieval → 1 chunks",
        "KB evidence grade → WEAK",
        "Web fallback → Tavily search",
        "Web evidence grade → GOOD",
        "Answer generation → WEB SEARCH",
    ]
    assert web.include_domains == [[]]


@pytest.mark.characterizes_bug("B5")
def test_empty_kb_results_are_weak_without_grading_call(fake_rag):
    fake_rag(
        llm=FakeLLM(
            # Only the web grade is scripted; a KB grading call would fail.
            structured={RouteDecision: [{"route": "kb"}], EvidenceGrade: [GOOD]},
            text=["Answer [1]."],
        ),
        retriever=FakeRetriever([]),
        web=FakeWebSearch({"results": [web_result("Doc", "https://docs.example.com")]}),
    )

    result = workflow.ask("How do I map a network drive?")

    assert "KB evidence grade → WEAK (no results)" in result["trace"]
    assert result["source_used"] == "web_search"


# -------------------------------------------------------------------
# Retries and fallbacks
# -------------------------------------------------------------------

@pytest.mark.characterizes_bug("B5")
def test_weak_web_evidence_rewrites_until_retries_exhausted(fake_rag):
    llm, retriever, web = fake_rag(
        llm=FakeLLM(text=["rewrite one", "rewrite two"]),
        web=FakeWebSearch({"results": []}),
    )

    question = "Cite official guidance on password length."
    result = workflow.ask(question)

    assert web.queries == [question, "rewrite one", "rewrite two"]
    assert result["retry_count"] == 2
    assert result["current_query"] == "rewrite two"
    assert result["source_used"] == "insufficient_evidence"
    assert result["citations"] == []
    assert result["trace"][-1] == "Stopped → insufficient reliable evidence"
    assert "Query rewrite → retry 2" in result["trace"]


@pytest.mark.characterizes_bug("B5")
def test_zero_retries_stops_immediately(fake_rag, settings, monkeypatch):
    monkeypatch.setattr(settings, "max_retries", 0)
    fake_rag(web=FakeWebSearch({"results": []}))

    result = workflow.ask("Cite official guidance on backups.")

    assert result["retry_count"] == 0
    assert result["source_used"] == "insufficient_evidence"


def test_web_search_failure_is_handled(fake_rag, settings, monkeypatch):
    monkeypatch.setattr(settings, "max_retries", 0)
    fake_rag(web=FakeWebSearch(error=RuntimeError("Tavily 500")))

    result = workflow.ask("Cite official guidance on backups.")

    assert "Web search failed" in result["trace"]
    assert result["source_used"] == "insufficient_evidence"


@pytest.mark.characterizes_bug("B5")
def test_kb_path_never_retries_kb_today(fake_rag):
    """Characterizes current behaviour: the KB is searched only once."""
    llm, retriever, web = fake_rag(
        llm=FakeLLM(
            structured={RouteDecision: [{"route": "kb"}], EvidenceGrade: [WEAK]},
            text=["rewrite one", "rewrite two"],
        ),
        retriever=FakeRetriever([kb_doc("Unrelated.")]),
        web=FakeWebSearch({"results": []}),
    )

    workflow.ask("How do I request a second monitor?")

    assert len(retriever.queries) == 1
    assert len(web.queries) == 3


@pytest.mark.characterizes_bug("B5")
@pytest.mark.xfail(
    raises=KnownBugStillPresent,
    reason="Known issue: after_rewrite can never route back to retrieve_kb "
    "because search_web always sets source_used='web'. Fixed in Phase B5.",
)
def test_weak_kb_evidence_retries_kb_with_rewritten_query(fake_rag):
    llm, retriever, web = fake_rag(
        llm=FakeLLM(
            structured={RouteDecision: [{"route": "kb"}], EvidenceGrade: [WEAK]},
            text=["rewrite one", "rewrite two"],
        ),
        retriever=FakeRetriever([kb_doc("Unrelated.")]),
        web=FakeWebSearch({"results": []}),
    )

    result = workflow.ask("How do I request a second monitor?")
    require(
        result["source_used"] == "insufficient_evidence",
        f"workflow ended with {result['source_used']!r}",
    )
    require(retriever.queries, "the KB retriever was never called")
    require(len(llm.prompts) >= 3, "the query was never rewritten")

    if "rewrite one" not in retriever.queries:
        raise KnownBugStillPresent(
            f"KB was queried only with {retriever.queries!r}"
        )


# -------------------------------------------------------------------
# Node and helper behaviour
# -------------------------------------------------------------------

def test_generate_from_web_without_evidence_returns_safe_answer():
    result = workflow.generate_from_web(
        {"question": "q", "web_results": "", "citations": [], "trace": []}
    )

    assert result["source_used"] == "web_search"
    assert result["citations"] == []
    assert "couldn't find enough relevant external evidence" in result["answer"]


def test_ask_rejects_blank_question():
    with pytest.raises(ValueError):
        workflow.ask("   ")


@pytest.mark.parametrize(
    "message, expected",
    [
        (AIMessage(content="  plain  "), "plain"),
        (
            AIMessage(content=[{"type": "text", "text": "a"}, "b", {"type": "image"}]),
            "a\nb",
        ),
        ("raw string", "raw string"),
    ],
)
def test_message_text_normalizes_content(message, expected):
    assert workflow._message_text(message) == expected


def test_add_trace_does_not_mutate_state():
    state = {"trace": ["one"]}

    assert workflow.add_trace(state, "two") == ["one", "two"]
    assert state["trace"] == ["one"]
