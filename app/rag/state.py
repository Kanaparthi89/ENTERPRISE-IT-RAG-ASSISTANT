
"""Typed state and structured models for the BluePeak Agentic RAG workflow."""

from typing import Any, Literal
from typing_extensions import NotRequired, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel, Field


# -------------------------------------------------------------------
# Structured outputs used by LangGraph routing and grading nodes
# -------------------------------------------------------------------

class RouteDecision(BaseModel):
    """Choose whether a question needs knowledge-base retrieval."""

    route: Literal["kb", "direct"] = Field(
        description=(
            "'kb' for company IT support or handbook questions; "
            "'direct' for greetings and simple conversational questions."
        )
    )


class EvidenceGrade(BaseModel):
    """Evaluate whether retrieved evidence supports an answer."""

    grade: Literal["good", "weak"] = Field(
        description=(
            "'good' when the evidence directly supports the answer; "
            "'weak' when evidence is missing, irrelevant, or insufficient."
        )
    )
    reason: str = Field(
        default="",
        description="Brief explanation for the evidence assessment.",
    )


class QueryRewrite(BaseModel):
    """Represent a clarified query for improved retrieval."""

    query: str = Field(
        min_length=1,
        description="A clear, self-contained search query.",
    )


class Citation(BaseModel):
    """Identify a source used to support an answer."""

    source: str = Field(
        description="Source document name or source identifier."
    )
    page: int | None = Field(
        default=None,
        description="One-based page number when available.",
    )
    chunk_id: str | None = Field(
        default=None,
        description="Identifier of the supporting chunk, if available.",
    )
    quote: str | None = Field(
        default=None,
        description="Short supporting excerpt from the source, if available.",
    )


# -------------------------------------------------------------------
# Shared workflow state
# -------------------------------------------------------------------

class AgentState(TypedDict):
    """Data passed between nodes in the LangGraph agent workflow."""

    # Original request and retrieval query
    question: str
    current_query: str

    # Knowledge-base retrieval
    kb_docs: list[Document]
    kb_grade: str

    # Optional external search results
    web_results: str
    web_grade: str

    # Final response and provenance
    answer: str
    source_used: str
    citations: list[dict[str, Any]]

    # Workflow control
    retry_count: int
    trace: list[str]


class AgentStateUpdate(TypedDict):
    """Optional partial state updates returned by individual nodes."""

    question: NotRequired[str]
    current_query: NotRequired[str]
    kb_docs: NotRequired[list[Document]]
    kb_grade: NotRequired[str]
    web_results: NotRequired[str]
    web_grade: NotRequired[str]
    answer: NotRequired[str]
    source_used: NotRequired[str]
    citations: NotRequired[list[dict[str, Any]]]
    retry_count: NotRequired[int]
    trace: NotRequired[list[str]]
