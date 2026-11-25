"""Tests for POST /api/chat with the RAG workflow mocked out."""

from uuid import UUID

import pytest
from pydantic import ValidationError

from app.api import routes
from app.services.audit import get_recent_audits


def workflow_result(**overrides):
    result = {
        "answer": "Use the GlobalProtect client.",
        "source_used": "private_kb",
        "trace": ["Router → KB", "Answer generation → PRIVATE KB"],
        "citations": [{"title": "handbook.md", "url": "", "type": "private_kb"}],
        "current_query": "How do I connect to the VPN?",
    }
    result.update(overrides)
    return result


@pytest.fixture
def fake_ask(monkeypatch):
    calls = []

    def install(result=None, error=None):
        def ask(question):
            calls.append(question)
            if error:
                raise error
            return result if result is not None else workflow_result()

        monkeypatch.setattr(routes, "ask", ask)
        return calls

    return install


def test_chat_returns_workflow_result(client, fake_ask):
    calls = fake_ask()

    response = client.post(
        "/api/chat", json={"question": "  How do I connect to the VPN?  "}
    )

    assert response.status_code == 200
    body = response.json()
    UUID(body["request_id"])
    assert body["answer"] == "Use the GlobalProtect client."
    assert body["source_used"] == "private_kb"
    assert body["trace"] == ["Router → KB", "Answer generation → PRIVATE KB"]
    assert body["citations"][0]["title"] == "handbook.md"
    assert body["rewritten_query"] == "How do I connect to the VPN?"
    assert calls == ["How do I connect to the VPN?"]


def test_chat_writes_success_audit(client, fake_ask):
    fake_ask()

    response = client.post("/api/chat", json={"question": "VPN help"})

    [audit] = get_recent_audits(limit=5)
    assert audit["request_id"] == response.json()["request_id"]
    assert audit["question"] == "VPN help"
    assert audit["status"] == "success"
    assert audit["source_used"] == "private_kb"
    assert audit["citations"][0]["title"] == "handbook.md"


def test_chat_audits_insufficient_evidence(client, fake_ask):
    fake_ask(workflow_result(source_used="insufficient_evidence", citations=[]))

    client.post("/api/chat", json={"question": "Unknown topic"})

    [audit] = get_recent_audits(limit=5)
    assert audit["status"] == "insufficient_evidence"


def test_rewritten_query_defaults_to_question(client, fake_ask):
    result = workflow_result()
    del result["current_query"]
    fake_ask(result)

    response = client.post("/api/chat", json={"question": "VPN help"})

    assert response.json()["rewritten_query"] == "VPN help"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": "a"},
        {"question": "x" * 3001},
        {"question": 123},
    ],
)
def test_chat_rejects_invalid_payload(client, fake_ask, payload):
    calls = fake_ask()

    response = client.post("/api/chat", json=payload)

    assert response.status_code == 422
    [error] = response.json()["detail"]
    assert error["loc"] == ["body", "question"]
    assert calls == []


# -------------------------------------------------------------------
# Question normalization and length boundaries
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    "question",
    ["  ", "   ", "\t\n ", "  ", " a ", "\n\na\n\n"],
)
def test_whitespace_padded_short_questions_return_422(client, fake_ask, question):
    calls = fake_ask()

    response = client.post("/api/chat", json={"question": question})

    assert response.status_code == 422
    [error] = response.json()["detail"]
    assert error["loc"] == ["body", "question"]
    assert error["type"] == "string_too_short"
    assert calls == []
    assert get_recent_audits(limit=5) == []


# U+001C..U+001F are removed by str.strip() but not by Pydantic's built-in
# strip_whitespace; they previously passed validation and caused HTTP 500.
@pytest.mark.parametrize(
    "question",
    [
        "\x1c\x1c",
        "\x1d\x1d",
        "\x1e\x1e",
        "\x1f\x1f",
        "\x1c\x1d\x1e\x1f",
        " \x1f\t\x1c ",
        "\x1ca\x1f",
    ],
    ids=["U+001C", "U+001D", "U+001E", "U+001F", "all-four", "mixed", "one-char-left"],
)
def test_separator_only_questions_return_422_with_real_workflow(client, question):
    # The real ask() is used: validation must reject before it is reached.
    response = client.post("/api/chat", json={"question": question})

    assert response.status_code == 422
    [error] = response.json()["detail"]
    assert error["loc"] == ["body", "question"]
    assert error["type"] == "string_too_short"
    assert get_recent_audits(limit=5) == []


def test_separators_are_stripped_like_str_strip(client, fake_ask):
    calls = fake_ask()

    response = client.post("/api/chat", json={"question": "\x1c\x1dVPN help\x1e\x1f"})

    assert response.status_code == 200
    assert calls == ["VPN help"]


def test_surrounding_whitespace_is_stripped(client, fake_ask):
    calls = fake_ask()

    response = client.post("/api/chat", json={"question": " \t VPN help \n"})

    assert response.status_code == 200
    assert calls == ["VPN help"]
    [audit] = get_recent_audits(limit=5)
    assert audit["question"] == "VPN help"


def test_internal_whitespace_is_preserved(client, fake_ask):
    calls = fake_ask()
    question = "Error log:\n\n  line 1\n\tline 2"

    response = client.post("/api/chat", json={"question": f"  {question}  "})

    assert response.status_code == 200
    assert calls == [question]


@pytest.mark.parametrize(
    "question, status, error_type",
    [
        ("x", 422, "string_too_short"),
        ("xy", 200, None),
        ("x" * 3000, 200, None),
        ("x" * 3001, 422, "string_too_long"),
        # Length limits apply after stripping.
        ("  " + "x" * 3000 + "  ", 200, None),
        ("  " + "x" * 3001 + "  ", 422, "string_too_long"),
    ],
    ids=["1", "2", "3000", "3001", "3000-padded", "3001-padded"],
)
def test_question_length_boundaries(client, fake_ask, question, status, error_type):
    calls = fake_ask()

    response = client.post("/api/chat", json={"question": question})

    assert response.status_code == status
    if error_type:
        assert response.json()["detail"][0]["type"] == error_type
        assert calls == []
    else:
        assert calls == [question.strip()]


def test_chat_request_model_normalizes_question():
    assert routes.ChatRequest(question="  a\n b  ").question == "a\n b"

    with pytest.raises(ValidationError):
        routes.ChatRequest(question="    ")


def test_workflow_failure_returns_safe_500_and_audits(client, fake_ask):
    fake_ask(error=RuntimeError("pinecone key sk-secret leaked"))

    response = client.post("/api/chat", json={"question": "VPN help"})

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["message"] == "Unable to process the question."
    UUID(detail["request_id"])
    assert "sk-secret" not in response.text

    [audit] = get_recent_audits(limit=5)
    assert audit["status"] == "failed"
    assert audit["error_type"] == "RuntimeError"
    assert audit["request_id"] == detail["request_id"]


def test_empty_answer_is_treated_as_failure(client, fake_ask):
    fake_ask(workflow_result(answer=""))

    response = client.post("/api/chat", json={"question": "VPN help"})

    assert response.status_code == 500
    [audit] = get_recent_audits(limit=5)
    assert audit["status"] == "failed"


def test_audit_failure_does_not_break_chat(client, fake_ask, monkeypatch):
    fake_ask()

    def failing_audit(**kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(routes, "write_audit", failing_audit)

    response = client.post("/api/chat", json={"question": "VPN help"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Use the GlobalProtect client."
