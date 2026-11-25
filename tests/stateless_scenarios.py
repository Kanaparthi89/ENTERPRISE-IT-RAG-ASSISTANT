"""Stateless workflow scenarios used to prove prompts don't change (A2-3).

``tests/fixtures/stateless_prompts_golden.json`` was recorded from the
workflow at commit 5d1e45a, *before* A2-3 changed any prompt, by running
these scenarios. ``tests/test_workflow_history.py`` reruns them and requires
every prompt, retrieval query and web query to be byte-for-byte identical.

Between them the scenarios reach every node that builds a prompt: the
router, both graders, query rewriting, and the KB, web and direct answers.
"""

import json
from pathlib import Path

from app.rag.state import EvidenceGrade, RouteDecision
from tests.conftest import FakeLLM, FakeRetriever, FakeWebSearch, kb_doc

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "stateless_prompts_golden.json"

NIST_RESULT = {
    "results": [
        {
            "title": "Computer Security Incident Handling Guide",
            "url": "https://csrc.nist.gov/pubs/sp/800/61/r2/final",
            "content": "Preparation; detection and analysis; containment, eradication and recovery; post-incident activity.",
        }
    ]
}
VENDOR_RESULT = {
    "results": [
        {
            "title": "Printer driver support",
            "url": "https://support.example.com/printers",
            "content": "Download the latest driver and restart the print spooler.",
        }
    ]
}


def _scenarios():
    return {
        "kb_good": lambda: (
            "How do I connect to the company VPN?",
            FakeLLM(
                structured={RouteDecision: [{"route": "kb"}], EvidenceGrade: [{"grade": "good"}]},
                text=["Open GlobalProtect and sign in with your SSO account."],
            ),
            FakeRetriever([
                kb_doc("VPN access uses GlobalProtect with SSO.", source="/kb/vpn.md", page=1),
                kb_doc("MFA is required for remote access.", source="/kb/mfa.pdf", page=3),
            ]),
            FakeWebSearch(),
        ),
        "direct": lambda: (
            "Thanks, that was helpful!",
            FakeLLM(structured={RouteDecision: [{"route": "direct"}]}, text=["You're welcome!"]),
            FakeRetriever(),
            FakeWebSearch(),
        ),
        "official_web": lambda: (
            "According to NIST, what are the main steps in incident response? Cite official NIST sources.",
            FakeLLM(
                structured={EvidenceGrade: [{"grade": "good"}]},
                text=["The NIST guide lists four phases [1]."],
            ),
            FakeRetriever(),
            FakeWebSearch(NIST_RESULT),
        ),
        "kb_empty_then_web": lambda: (
            "Where can I download the printer driver for the 4th floor?",
            FakeLLM(
                structured={RouteDecision: [{"route": "kb"}], EvidenceGrade: [{"grade": "good"}]},
                text=["Download the driver from the vendor site [1]."],
            ),
            FakeRetriever([]),
            FakeWebSearch(VENDOR_RESULT),
        ),
        "rewrites_then_insufficient": lambda: (
            "Why does my badge not open the server room?",
            FakeLLM(
                structured={
                    RouteDecision: [{"route": "kb"}],
                    EvidenceGrade: [{"grade": "weak"}, {"grade": "weak"}, {"grade": "weak"}, {"grade": "weak"}],
                },
                text=["badge access server room permissions", "server room badge reader access request"],
            ),
            FakeRetriever([kb_doc("Badges are issued by facilities.", source="/kb/facilities.md")]),
            FakeWebSearch(VENDOR_RESULT),
        ),
    }


SCENARIO_NAMES = list(_scenarios())


def _record(llm, retriever, web, result) -> dict:
    return {
        "calls": [
            {
                "kind": call["kind"],
                "schema": call["schema"].__name__ if call["schema"] else None,
                "tags": call["tags"],
                "prompt": call["prompt"],
            }
            for call in llm.calls
        ],
        "retrieval_queries": list(retriever.queries),
        "web_queries": list(web.queries),
        "web_include_domains": [list(d) if d else d for d in web.include_domains],
        "result": {
            "answer": result.get("answer"),
            "source_used": result.get("source_used"),
            "trace": result.get("trace"),
            "citations": result.get("citations"),
            "current_query": result.get("current_query"),
        },
    }


def run_scenario(fake_rag, name, runner) -> dict:
    """Run one scenario through ``runner(question) -> final state`` and record it."""
    question, llm, retriever, web = _scenarios()[name]()
    fake_rag(llm=llm, retriever=retriever, web=web)
    result = runner(question)
    return _record(llm, retriever, web, result)


def load_golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
