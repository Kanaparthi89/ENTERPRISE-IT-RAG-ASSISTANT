# Enterprise IT Support Agentic RAG Copilot

An end-to-end **Forward Deployed Engineer (FDE) project** that transforms an Agentic Retrieval-Augmented Generation (RAG) workflow into a deployable enterprise IT support application.

The system combines private knowledge retrieval, evidence grading, conditional web search, query rewriting, grounded answer generation, source citations, workflow tracing, and a FastAPI backend. It is designed to help employees find reliable IT support information while giving engineers visibility into how answers are produced.

The project uses a fictional enterprise environment to demonstrate how AI engineering, application development, retrieval systems, and security-conscious design can address practical business problems.

**Technology stack:** Python, FastAPI, LangGraph, OpenAI, Pinecone, Tavily, SQLite, HTML, CSS, JavaScript, and Pydantic Settings.

**Project status:** Core workflow and API behavior have been exercised locally. Production deployment, comprehensive automated evaluation, and enterprise security certification remain separate milestones.

---

## 1. Business Problem

### Customer: BluePeak Technologies

BluePeak Technologies is a fictional enterprise with approximately 5,000 employees working across offices, warehouses, and remote locations.

Its IT department maintains documentation covering VPN access, authentication, approved software, account troubleshooting, network connectivity, security procedures, and service-desk processes.

As the organization grows, employees repeatedly ask support teams questions whose answers may already exist in internal documentation.

### Challenges

- **Scattered knowledge:** Employees struggle to locate the correct procedure across multiple documents.
- **Weak search relevance:** Keyword searches may return documents that mention a topic without answering the actual question.
- **Unsupported AI answers:** A general-purpose LLM can generate plausible but unverified technical instructions.
- **Changing external guidance:** Public cybersecurity recommendations and vendor documentation can change over time.
- **Repeated support requests:** IT staff spend time answering routine questions instead of investigating complex incidents.
- **Limited transparency:** Engineers need to understand why a source was selected, why retrieval failed, and when the system used external information.

### Proposed solution

Build an AI-powered IT support assistant that:

1. Prioritizes internal knowledge for company-specific questions.
2. Evaluates retrieved evidence before generating an answer.
3. Uses public search when appropriate and when internal evidence is insufficient.
4. Rewrites unsuccessful queries and retries retrieval within controlled limits.
5. Generates answers grounded in available evidence.
6. Returns source citations and a workflow trace where supported.
7. Provides REST API endpoints for application integration.
8. Supports a path toward controlled document ingestion and ongoing knowledge-base maintenance.

### Example scenario

An employee asks:

> My company laptop connects to Wi-Fi, but I cannot access internal applications from home. How should I troubleshoot it?

The assistant searches the private IT knowledge base first. If the retrieved VPN and connectivity instructions are sufficient, it generates an answer from that evidence.

For another question:

> According to NIST, what are the main steps in incident response?

The system can route the request to official public sources, evaluate the returned evidence, and produce an answer with references to relevant NIST publications.

The objective is not to make every answer depend on a web search. It is to choose an appropriate information source and respond transparently when reliable evidence is unavailable.

### Business objective

By combining semantic retrieval, evidence evaluation, and source-aware generation, the project aims to reduce the effort required to find IT support information and improve answer traceability. Any reduction in support tickets, resolution time, or operating costs must be measured during a controlled pilot rather than assumed.

---

## 2. Why This Is a Forward Deployed Engineer Project

A Forward Deployed Engineer connects customer problems with working software. The role involves understanding operational requirements, integrating data sources, developing AI-enabled applications, validating behavior, and iterating based on real-world results.

This project demonstrates that approach by translating a fictional enterprise's IT support challenges into an application architecture and an agentic workflow.

### FDE delivery lifecycle

```text
Customer Discovery
        |
        v
Identify Operational Bottlenecks
        |
        v
Define Requirements and Test Scenarios
        |
        v
Integrate Private Knowledge
        |
        v
Build the Agentic RAG Workflow
        |
        v
Expose Capabilities Through FastAPI
        |
        v
Evaluate Retrieval and Answer Quality
        |
        v
Improve Source Attribution and Reliability
        |
        v
Prepare Deployment and Monitoring
```

### Engineering responsibilities demonstrated

- **Requirements engineering:** Translate repetitive IT questions into retrieval, routing, and response requirements.
- **AI workflow engineering:** Coordinate classification, retrieval, evidence grading, query rewriting, and generation.
- **Backend development:** Expose the AI workflow through a FastAPI application.
- **Knowledge integration:** Connect private document retrieval with a vector database.
- **External data integration:** Retrieve public information through Tavily when required.
- **Reliability engineering:** Exercise routing, evidence sufficiency, and fallback behavior.
- **Debugging and observability:** Use workflow traces and source metadata to understand execution decisions.
- **Security-conscious development:** Identify controls needed for credentials, document uploads, access permissions, and audit records.
- **Iterative validation:** Test realistic scenarios and use observed weaknesses to guide improvements.

**STAR summary**

- **Situation:** Employees need reliable answers from internal IT documentation and public technical guidance.
- **Task:** Build an assistant that selects appropriate sources, evaluates evidence, and avoids treating unsupported responses as verified facts.
- **Action:** Implement a LangGraph-based workflow, connect retrieval and search services, expose API endpoints, and test representative security and troubleshooting questions.
- **Result:** A working local application that has returned responses and source metadata for tested scenarios, with additional evaluation and production hardening identified as next steps.

---

## 3. System Architecture

The application separates API handling, workflow orchestration, knowledge retrieval, external search, answer generation, and audit storage.

```text
                 Employee / API Client
                          |
                          v
                 FastAPI Application
                          |
                          v
                 LangGraph Workflow
                          |
              +-----------+-----------+
              |           |           |
              v           v           v
         Private KB   Public Web   Direct Reply
         Retrieval     Search      (when needed)
              |           |
              v           v
          Pinecone      Tavily
              |           |
              +-----+-----+
                    |
                    v
             Evidence Evaluation
                    |
                    v
             Query Rewrite / Retry
                    |
                    v
             Grounded Answer
                    |
                    v
         Answer + Source Metadata
                    |
                    v
          API Response / Trace
                    |
                    v
            SQLite Audit Store
```

The diagram describes the logical design. The exact execution path depends on the implemented graph and request type.

### Main components

| Component             | Responsibility                                                     |
| --------------------- | ------------------------------------------------------------------ |
| FastAPI               | Receives requests, validates input, and returns API responses      |
| LangGraph             | Coordinates workflow state and conditional transitions             |
| OpenAI                | Provides configured LLM and embedding capabilities                 |
| Pinecone              | Stores and retrieves document vectors                              |
| Tavily                | Searches publicly available information                            |
| SQLite                | Supports lightweight local audit persistence                       |
| Pydantic Settings     | Loads and validates application configuration                      |
| HTML, CSS, JavaScript | Provide the user-facing interface where enabled by the application |

### Current provider configuration

The project configuration uses OpenAI's `gpt-4o-mini` for chat and `text-embedding-3-small` for embeddings, along with Pinecone and Tavily. Confirm the active environment settings before deploying, because model availability and provider configuration can change.

The embedding model used for indexing must match the model and vector dimensions used during retrieval.

---

## 4. Agentic RAG Workflow

### Traditional RAG versus Agentic RAG

A basic RAG pipeline usually follows:

```text
Question → Retrieve Documents → Generate Answer
```

This is useful when retrieval consistently returns relevant evidence. Enterprise questions, however, can be ambiguous, documentation can be incomplete, and some questions require public information.

Agentic RAG adds explicit decision-making and evaluation steps:

```text
Question
   |
   v
Route Request
   |
   +---- Simple Conversation ----> Direct Response
   |
   v
Retrieve Private Knowledge
   |
   v
Grade Retrieved Evidence
   |
   +---- Sufficient Evidence ---> Generate Answer
   |
   v
Evaluate Need for Web Search
   |
   v
Search Public Sources
   |
   v
Grade Web Evidence
   |
   +---- Sufficient Evidence ---> Generate Answer
   |
   v
Rewrite Query / Retry
   |
   v
Evidence Check
   |
   +---- Sufficient Evidence ---> Grounded Answer
   |
   +---- Insufficient Evidence -> Safe Fallback
```

### Workflow stages

**1. Request routing**

The workflow classifies incoming questions and selects an appropriate path. Simple conversational requests can bypass unnecessary retrieval, while technical questions continue through the knowledge workflow.

**2. Private knowledge retrieval**

The system embeds the question and searches Pinecone for semantically related document chunks. Semantic retrieval helps identify relevant content even when the employee's wording differs from the wording in the original document.

**3. Evidence grading**

The retrieved content is evaluated for relevance and sufficiency. A document mentioning a subject is not automatically sufficient evidence for a detailed technical answer.

**4. Conditional web search**

When internal evidence is insufficient and external information is appropriate, the workflow can use Tavily to retrieve public sources. Official-source requests, such as questions explicitly asking for CISA or NIST guidance, can be routed to the external-search path.

**5. Web evidence evaluation**

The workflow evaluates retrieved web evidence before using it to generate an answer. An official domain is useful for source selection, but the content must still be relevant to the actual question.

**6. Query rewriting**

If evidence is insufficient, the workflow can rewrite the question into a more targeted search query. For example:

- Original: "My login stopped working after the update."
- Rewritten: "Troubleshooting enterprise application sign-in failures after a client software update."

The rewritten query can be used in another retrieval attempt according to the configured graph logic.

**7. Grounded answer generation**

The model generates a response from the selected evidence. For external answers, the workflow is designed to associate inline numbered references with the source list returned by the API.

**8. Safe fallback**

When reliable evidence is unavailable, the application should explain the limitation rather than inventing a procedure, policy, or approval code.

**9. Workflow tracing**

The API response can include trace entries such as:

```text
Router → OFFICIAL EXTERNAL SOURCES
Web fallback → Tavily search
Web evidence grade → GOOD
Answer generation → WEB SEARCH
```

These traces help developers inspect the application's decision path. They do not, by themselves, prove that the final answer is correct.

---

## 5. Source Attribution and Citation Handling

Source attribution is important because enterprise users need to distinguish verified evidence from unsupported generated text.

The external-answer workflow has been updated to:

- Filter out search results that lack usable HTTP(S) URLs or content.
- Restrict results to explicitly requested official domains when those domains are detected.
- Deduplicate identical URLs after basic URL normalization.
- Limit the number of sources included in the evidence context.
- Present numbered source evidence to the answer-generation model.
- Ask the model to use inline references such as `[1]` and `[2]`.
- Return only source entries whose reference numbers appear in the generated answer.

### Citation validation

The application has been exercised with official-source questions about:

- Remote-access security using CISA guidance.
- Incident response using NIST resources.
- Phishing protection using CISA resources.

The observed responses included official source URLs and inline numbered references.

These tests demonstrate that source retrieval and citation generation are functioning for the tested examples. They do not establish that every individual claim is fully supported by its cited source.

### Known citation-quality improvements

Further work should include:

1. More robust URL canonicalization to eliminate duplicate pages with different URL forms.
2. Claim-level citation evaluation to confirm that each recommendation is supported by the source attached to it.
3. Validation that every inline reference maps to an existing source entry.
4. Tests for fabricated, invalid, or unused reference numbers.
5. More precise source filtering when a question explicitly requests one authority, such as NIST.
6. Automated evaluation of source relevance and answer faithfulness.

**Important distinction:** Citation presence is not the same as citation correctness. The goal is to establish both a valid reference and a clear relationship between the cited source and the statement it supports.

---

## 6. Technology Stack

| Technology                      | Purpose                  | Why it is used                                                    |
| ------------------------------- | ------------------------ | ----------------------------------------------------------------- |
| Python                          | Application language     | Supports AI integration, retrieval, and backend development       |
| FastAPI                         | REST API framework       | Provides request validation and API endpoints                     |
| LangGraph                       | Workflow orchestration   | Supports stateful routing, conditional transitions, and retries   |
| OpenAI `gpt-4o-mini`            | LLM inference            | Supports configured workflow tasks such as grading and generation |
| OpenAI `text-embedding-3-small` | Embeddings               | Converts text into vectors for semantic retrieval                 |
| Pinecone                        | Vector database          | Stores and searches embedded document chunks                      |
| Tavily                          | External web search      | Retrieves public evidence when needed                             |
| SQLite                          | Local audit storage      | Provides lightweight persistence for development                  |
| Pydantic Settings               | Configuration management | Loads typed application settings                                  |
| HTML / CSS / JavaScript         | Frontend                 | Supports employee interaction through a web interface             |
| Uvicorn                         | ASGI server              | Runs the FastAPI application locally                              |
| Pytest                          | Testing framework        | Supports automated tests where implemented                        |

### Alternatives

- **Vector databases:** Qdrant, Weaviate, pgvector, and Milvus.
- **LLM providers:** Other hosted providers or self-hosted inference, depending on customer requirements.
- **Workflow orchestration:** Custom Python state machines or other workflow frameworks.
- **Audit storage:** PostgreSQL or a centralized logging platform for larger deployments.
- **Frontend:** React, Vue, or Next.js for a richer user experience.

Technology selection should reflect customer requirements for privacy, data residency, cost, latency, availability, and operational complexity.

---

## 7. Repository Structure

The project is organized around a modular application architecture. The tree below shows the principal files.

```text
ENTERPRISE-IT-RAG-ASSISTANT/
├── app/
│   ├── main.py                  # FastAPI application entry point
│   ├── api/
│   │   ├── main.py              # Alternative application entry point
│   │   ├── routes.py            # API routes: chat, streaming, sessions, ingestion
│   │   └── streaming.py         # Streaming chat (SSE) service
│   ├── core/
│   │   ├── config.py            # Settings (Pydantic Settings)
│   │   └── logging.py           # Logging configuration
│   ├── rag/
│   │   ├── workflow.py          # Agentic RAG workflow
│   │   ├── state.py             # Workflow state and schemas
│   │   └── vectorstore.py       # Embeddings and Pinecone integration
│   └── services/
│       ├── audit.py             # Audit persistence (SQLite)
│       ├── ingestion.py         # Document ingestion pipeline
│       └── sessions.py          # Conversation session storage and leases (SQLite)
├── templates/
│   └── index.html               # Web interface template
├── static/
│   ├── css/style.css            # Web interface styles
│   └── js/
│       ├── app.js               # Web interface behavior
│       └── chat-stream.js       # Chat client: streaming, fallback, sessions
├── data/
│   └── sample_kb/               # Sample knowledge-base documents
├── tests/                       # Python tests (pytest)
│   └── js/                      # JavaScript tests (Node.js test runner)
├── Dockerfile
├── compose.yaml
├── .env.example                 # Example configuration
├── .gitignore
├── package.json                 # JavaScript test script
├── requirements.txt
├── requirements-dev.txt         # Test dependencies
├── ingest_sample_kb.py          # Sample knowledge-base ingestion script
├── run.py                       # Optional application entry point
└── README.md
```

At runtime the application creates its SQLite databases in `data/`: `audit.db` for audit records and, once conversation sessions are used, `sessions.db`. Neither is tracked in Git.

The workflow module coordinates routing, retrieval, evidence grading, external search, query rewriting, and response generation. The API layer exposes application functionality, while configuration and service modules keep infrastructure details separate from the main workflow.

---

## 8. Local Setup

### Prerequisites

- Python version compatible with the pinned dependencies.
- Git.
- OpenAI API access for the configured model and embedding provider.
- Pinecone credentials and a correctly configured vector index.
- Tavily credentials if external search is enabled.

### Step 1: Navigate to the repository

```bash
cd /Users/Downloads/projects/enterprise-ai-projects/ENTERPRISE-IT-RAG-ASSISTANT
```

### Step 2: Activate the virtual environment

```bash
source .venv/bin/activate
```

If the virtual environment does not exist, create it:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure environment variables

Create a local `.env` file based on `.env.example`. Use the actual variable names expected by the application's settings module.

Typical configuration categories include:

```dotenv
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=fde-it-support-rag
PINECONE_NAMESPACE=company-it-kb

TAVILY_API_KEY=your_tavily_api_key
```

These are representative names, not a substitute for checking the project's actual settings definitions. Add other variables only when the implementation uses them.

Never commit real credentials to version control.

### Step 5: Configure the vector index

Ensure the Pinecone index exists and its vector dimension matches the configured embedding model. Use the same index and namespace during ingestion and retrieval.

If the application has an ingestion script, follow its instructions to index the sample knowledge-base documents. Verify successful ingestion before evaluating retrieval quality.

### Step 6: Start the API

The project can be started with Uvicorn when `app.main` exposes the FastAPI application:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Alternatively, use the repository's `run.py` if that is the configured entry point.

**Port note:** the Uvicorn command above listens on port **8000**, which the URLs in this README use. `python run.py` listens on port **8080** (`http://127.0.0.1:8080`) instead; replace `8000` with `8080` in the URLs and examples when using it.

Streaming chat is disabled by default. To try it locally, see [Streaming chat](#streaming-chat-controlled-prototype-disabled-by-default).

### Step 7: Open the application

- API documentation: `http://127.0.0.1:8000/docs`
- Health endpoint: `http://127.0.0.1:8000/health`
- API health endpoint: `http://127.0.0.1:8000/api/health`
- Chat endpoint: `http://127.0.0.1:8000/api/chat`

The homepage and frontend availability depend on the current application configuration.

---

## 9. API Usage

### Health check

```bash
curl -s http://127.0.0.1:8000/api/health
```

The local health endpoint has returned:

```json
{
  "status": "ok",
  "service": "Enterprise IT Support Agentic RAG Copilot"
}
```

### Submit a chat question

```bash
curl -s --max-time 60 -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"According to NIST, what are the main steps in incident response? Cite official NIST sources."}'
```

The tested API response contains fields such as:

- `request_id`: Identifier for the request.
- `answer`: Generated response.
- `source_used`: Selected source path.
- `trace`: Workflow execution summary.
- `citations`: Returned source titles and URLs.
- `rewritten_query`: Search query metadata when available.

The exact fields can vary with the API schema and execution path.

### Example source-aware response

A tested NIST incident-response question returned official NIST resources, including the *Computer Security Incident Handling Guide* and the NIST Computer Security Resource Center incident-response page.

A tested CISA phishing question returned CISA-related source URLs and inline numbered references.

These examples confirm that the application can retrieve public security information and return source metadata for the tested scenarios. They are not a substitute for comprehensive citation-faithfulness testing.

### Streaming chat (controlled prototype, disabled by default)

`POST /api/chat/stream` returns the same answer as `/api/chat`, delivered progressively as Server-Sent Events (SSE). It is a **controlled prototype**: it is disabled by default and must not be enabled in production until the blockers listed under [Production-readiness blockers](#production-readiness-blockers) are resolved.

#### Enabling streaming

| Setting | Default | Allowed | Purpose |
| ------- | ------- | ------- | ------- |
| `STREAM_ENABLED` | `false` | boolean | Enables the streaming endpoint. While disabled, it returns `404 Not Found` for any request body, including malformed JSON. |
| `STREAM_HEARTBEAT_SECONDS` | `15` | > 0, ≤ 300 | Idle interval after which the server sends a `: keepalive` comment. |
| `STREAM_TIMEOUT_SECONDS` | `120` | > 0, ≤ 3600 | Deadline for one streamed request, measured from admission. |
| `MAX_CONCURRENT_STREAMS` | `8` | 1–64 | Streams admitted at once, **per server process**. |
| `STREAM_RETRY_AFTER_SECONDS` | `5` | 1–300 | `Retry-After` value sent with `503` when capacity is full. |
| `OPENAI_TIMEOUT_SECONDS` | `60` | > 0, ≤ 600 | OpenAI timeout per HTTP operation (not per request). Also applies to `/api/chat` and ingestion. |
| `OPENAI_MAX_RETRIES` | `2` | 0–10 | OpenAI retries: each call makes up to 1 + this value attempts (see [Timeout and retry budget](#timeout-and-retry-budget)). Also applies to `/api/chat` and ingestion. |
| `PINECONE_TIMEOUT_SECONDS` | `10` | > 0, ≤ 15 | Timeout for each HTTP attempt of a Pinecone query or upsert. The Pinecone SDK retries a failed attempt up to 5 times, so one call can take about 6 × this value plus retry backoff (about 8 seconds, measured locally), and much longer if Pinecone sends a `Retry-After` header (see [Timeout and retry budget](#timeout-and-retry-budget)). Also applies to `/api/chat` and ingestion. Also applies to each attempt of the index list, describe, and create calls in Pinecone setup (`ensure_index()`). |
| `TAVILY_TIMEOUT_SECONDS` | `15` | > 0, ≤ 30 | Connect and read timeout for one Tavily search request, which is not retried. The read timeout applies to each read, not the whole response. A timed-out search is treated as a failed web search. Also applies to `/api/chat`. |
| `PINECONE_RETRY_AFTER_MAX_SECONDS` | unset | 0–21600 (whole seconds) | Optional. The longest wait honored for one Pinecone `Retry-After` header (HTTP 413, 429, 503); a longer value is shortened to it, and `0` uses the normal retry backoff instead. Unset keeps the SDK's policy unchanged (up to 6 hours per retry). Every other retry setting stays the SDK default. Applies to Pinecone setup, queries, and upserts, also for `/api/chat` and ingestion. Not a deadline (see [Timeout and retry budget](#timeout-and-retry-budget)). |
| `CHAT_DEADLINE_SECONDS` | unset | > 0, ≤ 3600 | Optional deadline for a stateless `/api/chat` question (no `X-Session-ID`), counted from when its handler starts. Past it, the question starts no further external call and fails with the generic `500` (audit error type `TimeoutError`); a call already in progress is not interrupted. Unset: no deadline, as before. Does not apply to streams (`STREAM_TIMEOUT_SECONDS`) or session questions (`SESSION_TURN_MAX_SECONDS`). |

Settings are read once at startup, so a change requires a server restart. An environment variable takes precedence over `.env`, so streaming can be enabled for a single local run without editing any file:

```bash
# Uvicorn directly, on port 8000:
STREAM_ENABLED=true uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# Or with run.py, on port 8080:
STREAM_ENABLED=true python run.py
```

After restarting, reload the browser page: the frontend remembers a `404` from the streaming endpoint for the rest of the page session.

#### Request and responses

The request body is the same as `/api/chat`, with the same validation (use port `8080` with `run.py`):

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"How do I connect to the company VPN?"}'
```

| Status | Meaning |
| ------ | ------- |
| `200` | An SSE stream (`Content-Type: text/event-stream; charset=utf-8`, `Cache-Control: no-cache, no-transform`, `X-Accel-Buffering: no`). |
| `404` | Streaming is disabled. |
| `422` | Invalid question, with the same error format as `/api/chat`. |
| `503` | No streaming capacity. The body is `{"detail": {"message": "The server is busy. Please retry shortly."}}` with a `Retry-After` header. |
| `500` (plain text) | The deadline passed before the response headers could be sent. This is sent by the server (Uvicorn), not by the application. |

With an `X-Session-ID` header (see [Conversation sessions](#conversation-sessions-controlled-prototype-disabled-by-default)), the request is admitted in this order, and every error below is returned before any SSE bytes:

1. Streaming capacity: if it is full, `503` with `{"detail": {"message": "...", "code": "stream_busy"}}` and `Retry-After`. The `code` field is present only for session requests; the session is not touched.
2. The session: `410` `session_expired`, `409` `session_busy` (with `Retry-After`), or `503` `session_unavailable`, with the same bodies as [the session errors](#session-errors).

#### SSE event contract (version 1)

Every event has one `event:` line and one single-line JSON `data:` line.

| Event | Data | When |
| ----- | ---- | ---- |
| `start` | `{"request_id": "...", "version": 1}` | First event of every stream. |
| `trace` | `{"index": 0, "step": "Router → KB"}` | Each workflow step, in order. |
| `delta` | `{"text": "..."}` | A piece of the answer text, in order. |
| `done` | The complete `/api/chat` response object (`request_id`, `answer`, `source_used`, `trace`, `citations`, `rewritten_query`). Session requests also include `session: {"history_turns_used": n}`. | The answer is complete. |
| `error` | `{"request_id": "...", "code": "internal_error" \| "timeout", "message": "..."}` | The request failed. Messages are always generic; exception details are never sent. |
| `session` | `{"request_id": "...", "turn_saved": true \| false \| null, "history_turns_used": n}` | Session requests only: once, after a completed `done`. |

Contract rules:

- A stream that stays connected ends with **exactly one** `done` or **exactly one** `error`, never both.
- The `delta` texts, joined and stripped of surrounding whitespace, equal `done.answer`. Routing, grading, and query-rewriting model calls never produce `delta` events.
- `done` is identical to the `/api/chat` response for the same result, apart from `request_id` and, for session requests, the `session` object: `turn_saved` is reported by the `session` event instead, because the turn is saved after `done` is sent.
- For session requests, one `session` event follows a completed `done` before the stream ends. `turn_saved` is `null` when the save could not be confirmed within 2 seconds. No `session` event follows an `error` event.
- `: keepalive` comment lines are sent when no event has been sent for `STREAM_HEARTBEAT_SECONDS`.
- If the client disconnects, the stream ends without a terminal event and the workflow is stopped at the next opportunity.

Each streamed request writes exactly one audit row (session requests also record the internal session ID): `success` or `insufficient_evidence` only when `done` was sent successfully; `failed` with an error type (for example `TimeoutError`) on errors and timeouts; and `cancelled` (for example `ClientDisconnected`) when the client went away.

#### Frontend behavior and fallback

The web interface tries the streaming endpoint first and renders the answer progressively as plain text. When `done` arrives, the existing Markdown, citation, source, and trace rendering replaces it, exactly as for `/api/chat`.

| Situation | Frontend behavior |
| --------- | ----------------- |
| `404` | Uses `/api/chat`, and does not try streaming again for the rest of the page session. |
| `503` | Uses `/api/chat`, and does not try streaming again until `Retry-After` has passed. |
| Network failure before response headers, or a `200` response that is not an event stream | Uses `/api/chat` once. |
| `500` or any other error status | Shows a generic error. **No fallback** (see limitations). |
| The stream has started, then is interrupted, malformed, or ends without `done`/`error` | Shows "The response was interrupted. Please try again." **No fallback**; partial text is removed. |
| Joined deltas do not match `done.answer` | Treated as interrupted. |
| `error` event | Shows a generic message for the error code; server text is never displayed. |
| No bytes received for 45 seconds (keepalive comments count) | Treated as interrupted. |
| Stream still running after 180 seconds | Treated as interrupted. |
| The user leaves the page (`pagehide`) | The stream is cancelled. Switching tabs or minimizing does not cancel it. |
| In a conversation session: `503` with code `stream_busy`, `404`, a network failure before headers, or a non-event-stream `200` | Uses `/api/chat` once, **with the same session token**. A session request is never retried without its session. |
| In a conversation session: `409`, `410`, or `503` `session_unavailable` | No fallback and no automatic retry. The question is not sent; see [Frontend behavior in a session](#frontend-behavior-in-a-session). |

Each question produces exactly one assistant message and at most one `/api/chat` request. Browsers without streaming `fetch` support use `/api/chat` directly.

Limitations of the fallback design:

- **Possible duplicate execution:** a network failure before response headers does not prove the server never received the request. The single fallback can therefore occasionally run a question twice on the server, producing two audit records.
- **No fallback on `500`:** intentional, to avoid running a question the server has already handled a second time. Do not change this without a separate review.
- **No fallback after a stream starts:** the user is asked to try again instead, so a partially processed question is not silently re-run.
- **`503` fallback moves load to `/api/chat`:** the streaming capacity limit protects streaming workers, but it does not limit total load. Plan capacity for both endpoints together.
- **While streaming is disabled,** each page load makes one extra request that returns `404` before using `/api/chat`.

#### Heartbeat and timeout coupling

The frontend's limits are fixed: 45 seconds without any received bytes, and 180 seconds overall. The server settings must stay compatible with them:

- Keep `STREAM_HEARTBEAT_SECONDS` at or below **30**, so keepalive comments arrive well within the 45-second idle limit.
- Keep `STREAM_TIMEOUT_SECONDS` at or below **170**, so the server's own timeout error (sent within about 2 seconds after the deadline) arrives before the 180-second limit.

These limits are **not enforced by configuration validation**. Values outside them make the browser cancel healthy streams or report "interrupted" instead of the server's timeout error, and the audit row then shows `cancelled`.

#### Timeout and retry budget

External calls are time-limited per HTTP attempt, not in total. This section describes how the timeout settings combine with each SDK's retries. The behavior was read from the installed SDKs (Pinecone 7.3.0 with urllib3 2.8.0, Requests 2.34.2, OpenAI 3.3.1), and an SDK upgrade can change it. Tests in `tests/test_external_timeouts.py` check the number of OpenAI and Tavily attempts and Pinecone's default retry settings, and tests in `tests/test_vectorstore.py` check Pinecone's `Retry-After` wait with and without `PINECONE_RETRY_AFTER_MAX_SECONDS` against a local server; OpenAI's `Retry-After`, DNS, and redirect behavior is not tested.

Four different limits apply:

1. **Per-attempt timeout:** limits one HTTP attempt. Read timeouts limit each wait for data, not the time to receive a whole response, so a server that keeps sending data slowly can make one attempt last longer.
2. **Retry budget:** how many attempts an SDK makes and how long it waits between them. The SDKs set it; this application configures OpenAI's retry count (`OPENAI_MAX_RETRIES`) and, optionally, the longest Pinecone `Retry-After` wait (`PINECONE_RETRY_AFTER_MAX_SECONDS`).
3. **Total operation deadline:** none. No external call is stopped after a fixed total time. The optional `Retry-After` cap makes a Pinecone call's worst case computable, but it is not a deadline.
4. **End-to-end request deadline:** streams have one (`STREAM_TIMEOUT_SECONDS`), and stateless `/api/chat` has one only when `CHAT_DEADLINE_SECONDS` is set. After a stream's deadline, or after a disconnect, the stream starts no further external call: each OpenAI, Pinecone, and Tavily call is checked just before it starts. A call already in progress is not interrupted. Session questions (streamed or `/api/chat`) also start no further external call once `SESSION_TURN_MAX_SECONDS` passes or their lease is lost. Stateless `/api/chat` has no deadline unless `CHAT_DEADLINE_SECONDS` is set; with it, the question likewise starts no further external call once that time has passed. Like every question, it also starts no further external call once the server is shutting down.

| Provider | Per-attempt timeout | Attempts | Retried on | Wait between attempts | `Retry-After` header |
| -------- | ------------------- | -------- | ---------- | --------------------- | -------------------- |
| Pinecone: queries, upserts, and the index list, describe, and create requests | `PINECONE_TIMEOUT_SECONDS` (10 s) to connect, then the time left for each read. | 6 (1 + 5 retries) | Connection errors; read errors, including timeouts, for every request method, POST included (so a create request can be sent more than once; see below); HTTP 500, 502, 503, and 504; HTTP 413 and 429 only with `Retry-After` | 0, 0.5, 1, 2, and 4 s, each plus 0–0.25 s of random jitter (7.5–8.75 s in total) | Honored for HTTP 413, 429, and 503, **up to 6 hours (21,600 s) per retry** by default; up to `PINECONE_RETRY_AFTER_MAX_SECONDS` when set |
| Tavily | `TAVILY_TIMEOUT_SECONDS` (15 s) to connect, and for each read | 1 (no HTTP retries) | — | — | Not used |
| OpenAI: chat and embeddings | `OPENAI_TIMEOUT_SECONDS` (60 s) each to connect, for each read and write, and to wait for a pooled connection | 1 + `OPENAI_MAX_RETRIES` (3 by default) | Timeouts, connection errors, HTTP 408, 409, 429, and 5xx, or when the server sends `x-should-retry: true` (never when it sends `x-should-retry: false`) | About 0.5 s, then 1 s (each shortened by up to 25% at random); at most 8 s | Honored up to 120 s; a longer value ends the retries |

**Index create request.** `ensure_index()` sends the create request (only when the configured index does not exist) through the SDK's generated API with `PINECONE_TIMEOUT_SECONDS` per attempt, and builds the request body with the SDK's own request factory, so it sends the same body as the public `create_index()`. Because the SDK retries a POST after a read timeout, a create request whose response was lost can be sent again after Pinecone already created the index (a local test with a stalled server counts 6 attempts). Pinecone then answers HTTP `409`; `ensure_index()` treats a `409` as "the index already exists", waits for it to become ready within the same 60-poll and 120-second limits, and refuses it if its dimension does not match the embedding model. The same handling covers two processes creating the index at the same time. Any other create error still fails setup.

For Tavily, the connect timeout applies to each address that DNS returns, and the DNS lookup itself has no timeout. Each redirect (up to 30) is a new request with its own timeouts. Separately, the workflow can run up to `1 + MAX_RETRIES` web searches (3 by default) for one question; that is application behavior, not an HTTP retry.

**Worst-case estimates.** These assume the default settings and a server that accepts each connection but never answers. They are estimates, **not guaranteed maximum durations**: slowly sent responses, multiple DNS addresses, redirects, and `Retry-After` headers can make a call take longer.

| Operation | Without `Retry-After` | If `Retry-After` is honored |
| --------- | --------------------- | --------------------------- |
| One Pinecone call | 6 × 10 s + up to 8.75 s ≈ 69 s (≈ 99 s at the 15-second maximum timeout) | Up to 6 hours per retry, so hours |
| Pinecone setup for an existing index (list and describe) | ≈ 2.3 minutes | Hours |
| Pinecone setup for a missing index (list, create, and the readiness wait) | ≈ 2.3 minutes, plus up to 120 s of readiness wait and one describe call past it | Hours |
| One Tavily search | ≈ 15 s after connecting; 15 s per address if it cannot connect | Not used |
| One OpenAI call | 3 × 60 s + about 1.5 s ≈ 3 minutes | 3 × 60 s + 2 × 120 s = 7 minutes |
| One question, every call on the longest path stalling | About 29–35 minutes (see below) | Hours |

The "If `Retry-After` is honored" column assumes the default, unset `PINECONE_RETRY_AFTER_MAX_SECONDS`. With a cap, each of a Pinecone call's 5 retries waits at most the larger of the configured `Retry-After` cap and that retry's normal SDK backoff (0, 0.5, 1, 2, or 4 s, plus up to 0.25 s of jitter), so one Pinecone call takes about 6 × `PINECONE_TIMEOUT_SECONDS` + 5 × max(configured `Retry-After` cap, normal SDK backoff): for example, about 60 s + 150 s ≈ 3.5 minutes with a 30-second cap at the default timeout, and about 60 s + 8.75 s ≈ 69 s with a cap of `0`, because the normal backoff then applies. That is still an estimate, not a guaranteed maximum, for the reasons above; OpenAI's `Retry-After` limit is unchanged.

With the current workflow routing, the longest path for one question makes up to 9 OpenAI chat calls (making a follow-up question standalone, when there is session history; routing; grading the knowledge-base results; grading web results three times; rewriting the query twice; and generating the answer), 1 embedding call, 1 Pinecone query, and 3 Tavily searches. That is about 29 minutes without session history and 32 minutes with it, plus about 2.3 minutes when the question also runs Pinecone setup (the first retrieval of each process): about 29–35 minutes in total. These figures describe how long stalled providers can hold a request, not typical latency.

**What the deadlines stop, and what they do not.** Stopping is cooperative: a question that should stop starts no further external call, but a call already in progress is never interrupted. No setting guarantees a maximum running time.

- **A stream stops before its next external call, not during one.** When a stream reaches `STREAM_TIMEOUT_SECONDS`, the client receives the timeout error; after a disconnect it receives nothing more. Either way the worker starts no further external call. A call already in progress is synchronous and is not cancelled: it finishes with all of its retries (about 3 minutes for one OpenAI call, longer with `Retry-After`). Answer generation also stops at its next streamed chunk, but not during a stalled read. The streaming slot (and a session's lease) stays in use until the call in progress returns and the worker exits.
- **The session turn limit stops further calls, not the call in progress.** After `SESSION_TURN_MAX_SECONDS` (180 s), or once the lease is lost, the question starts no further external call and its turn is not saved; the lease is not renewed. A call already in progress continues until it returns, so the conversation can accept another question while that call is still running.
- **Stateless `/api/chat` has no request deadline by default.** A non-streaming question without a session runs until its workflow finishes. With `CHAT_DEADLINE_SECONDS` set, it starts no further external call once that time has passed since its handler started, and fails with the generic `500` (audit error type `TimeoutError`); a call already in progress finishes first, and an answer whose last call was already running is still returned. The deadline does not count time spent waiting for a worker thread before the handler starts. During shutdown, Uvicorn cancels the response after its graceful timeout, but the worker thread continues; after the stream drain it starts no further external call, and it then attempts to record the question as failed (`ServerShutdown`). The process does not wait indefinitely for that worker thread, so the record is not guaranteed if the process exits first.
- **Some calls have no timeout at all:** DNS lookups and, the first time embeddings run in a process, tiktoken's download of its encoding file when no cached copy is available.

#### Production-readiness blockers

Streaming must remain disabled in production until these are resolved:

| ID | Blocker | Impact |
| -- | ------- | ------ |
| H1 | External calls have per-attempt timeouts, but no call has a total deadline. Pinecone queries and upserts and the index list, describe, and create calls in `ensure_index()` (`PINECONE_TIMEOUT_SECONDS`), and Tavily searches (`TAVILY_TIMEOUT_SECONDS`), have per-attempt timeouts; OpenAI calls have `OPENAI_TIMEOUT_SECONDS`. A create request that conflicts with an existing index (HTTP `409`) waits for that index instead of failing (see [Timeout and retry budget](#timeout-and-retry-budget)). The wait for a new index to become ready is limited to 60 polls and 120 seconds, checked between polls. Listing, describing, and creating reach the SDK's generated API through a private attribute (`pc.db.index._index_api`), and the create request body is built with the SDK's request factory (`pinecone.db_control.request_factory`); tests check that the installed SDK still provides both. Retries, `Retry-After` headers, and slowly sent responses can still make one call last much longer than its per-attempt timeout. Two optional settings, both off by default, narrow this: `PINECONE_RETRY_AFTER_MAX_SECONDS` caps the wait for a Pinecone `Retry-After` header (it is set on the SDK client's private OpenAPI configuration, with the SDK's other retry settings copied unchanged; a test compares them with the installed SDK's defaults), and `CHAT_DEADLINE_SECONDS` gives stateless `/api/chat` a cooperative deadline. Neither interrupts a call in progress. | After a stream's deadline, a disconnect, a shutdown, or a session's turn limit or lost lease, no further external call starts; only the call in progress continues. Setup runs on the first retrieval or ingestion of each process, and again after a failed attempt. With retries, setup for an existing index (a list and a describe call) can take about two to three minutes before failing at the default timeout, and the readiness wait can overrun 120 seconds by one describe call. Time-limited calls still hold a slot until they finish: with its retries, a Pinecone call can take more than a minute (hours if Pinecone sends `Retry-After`), and an OpenAI call about three minutes, longer than the stream deadline (see [Timeout and retry budget](#timeout-and-retry-budget)). `/api/chat` has the same exposure. |
| H2 | Shutdown is bounded but not complete. The application now drains stream workers before shutting sessions down, and the container runs Uvicorn with `--timeout-graceful-shutdown 8`, within a 20-second budget (see [Shutdown and deployment](#shutdown-and-deployment)). Running work is never forcibly interrupted, and Python's wait for worker threads at process exit has no limit. | Streams still open 8 seconds after a stop signal are interrupted. A worker that is still running when Docker sends `SIGKILL` (20 seconds after the stop signal), for example one blocked in a call without a timeout (H1), loses its turn and its audit record. |
| H3 | Partly addressed. Production Uvicorn uses `uvloop` and `httptools`. Seven real-server tests run on both the baseline runtime (`asyncio` and `h11`) and the production runtime (`uvloop` and `httptools`), and pass on both: the four streaming HTTP tests in `tests/test_stream_http.py` (deltas before the answer finishes, client disconnect, and the deadline before and after headers) and three shutdown tests in `tests/test_graceful_shutdown.py`. `tests/test_h3_runtime.py` checks that each test server really runs the requested event loop and HTTP protocol. Flow-control (slow-reader) behavior is tested only in-process, not against a real server on either runtime. The test server creates its event loop itself (`loop="none"`), so `tests/test_h3_auto_runtime.py` checks Uvicorn's own selection separately: it starts Uvicorn's real CLI (`python -m uvicorn`, which ends in `Server.run()`) in a subprocess with a probe app (`tests/h3_probe_app.py`, which imports nothing from the application) and confirms that the defaults (`loop="auto"`, `http="auto"`) select `uvloop` and `httptools`, that explicit `--loop asyncio --http h11` selects `asyncio` and `h11`, and that the `UVICORN_LOOP` and `UVICORN_HTTP` environment variables override the defaults. It also checks that the Dockerfile and `compose.yaml` set neither the flags nor those variables. | Disconnect, deadline, and shutdown behavior is verified under the production runtime, and Uvicorn's automatic selection of that runtime is verified in the local development environment. Not yet verified: flow control under a real server, and the selection inside the production Docker image (Linux, Python 3.11), which is a separate check. A `UVICORN_LOOP` or `UVICORN_HTTP` variable in the deployment environment, including the `.env` file that `compose.yaml` loads (tests never read it), would change the runtime. |

Related operational risks:

- **OpenAI timeouts** (`OPENAI_TIMEOUT_SECONDS`, `OPENAI_MAX_RETRIES`) also apply to `/api/chat` and ingestion. A non-streaming answer that takes longer than the timeout now fails. Validate the values against real latency.
- **Capacity is held beyond the deadline:** after a timeout or disconnect, a slot is released only when the external call in progress returns and the worker exits; no further call starts. A slow call with retries can keep a slot busy for minutes, and new streams then receive `503`.
- **No per-client limits:** a few slow or idle clients can occupy all streaming slots until their deadline. Apply connection and rate limits at the proxy.
- **Per-process capacity:** `MAX_CONCURRENT_STREAMS` applies to each server process separately.
- **Audit writes are attempted exactly once:** a failed SQLite write (for example, a lock held for more than 10 seconds) is logged but not retried.
- **Limited observability:** streaming failures are logged by error type only, and there is no per-request outcome log or capacity metric yet.

### Conversation sessions (controlled prototype, disabled by default)

**Status: Controlled prototype — disabled by default; not production-ready until the relevant production blockers and operational risks are resolved or explicitly accepted.** See [Session limitations and shutdown behavior](#session-limitations-and-shutdown-behavior) and [Session and database deployment requirements](#session-and-database-deployment-requirements).

A conversation session lets follow-up questions use earlier questions and answers. Without a session, every question is answered independently, exactly as before. Session memory is stored in its own SQLite database (`data/sessions.db` by default), separate from the audit database.

#### Enabling sessions

Set `SESSIONS_ENABLED=true` (for example `SESSIONS_ENABLED=true python run.py`) and restart the server. Settings are read once at startup. Reload the browser page afterwards: the web interface remembers a `404` from `POST /api/sessions` for the rest of the page session. Sessions work with or without streaming.

| Setting | Default | Enforced range | Purpose |
| ------- | ------- | -------------- | ------- |
| `SESSIONS_ENABLED` | `false` | boolean | Enables session creation and session-aware chat. |
| `SESSION_DB_PATH` | `data/sessions.db` | No range configured | Session database file. |
| `SESSION_IDLE_TIMEOUT_SECONDS` | `1800` (30 min) | 60–86400 | Idle expiry (see [Session lifecycle](#session-lifecycle-expiry-capacity-and-cleanup)). |
| `SESSION_ABSOLUTE_TIMEOUT_SECONDS` | `28800` (8 h) | 60–604800 | Absolute expiry, counted from creation. |
| `MAX_ACTIVE_SESSIONS` | `5000` | 1–100000 | Maximum unexpired sessions; creation fails with `503` at the cap. |
| `SESSION_DB_BUSY_TIMEOUT_SECONDS` | `2` | > 0, ≤ 30 | How long a session database operation waits for a lock. |
| `SESSION_CLEANUP_BATCH` | `500` | 1–10000 | Expired sessions deleted per cleanup pass. |
| `SESSION_CLEANUP_INTERVAL_SECONDS` | `300` | 10–3600 | Interval of the periodic cleanup. |
| `SESSION_LEASE_TTL_SECONDS` | `30` | 3–300 | Lifetime of a question's lease; renewed every TTL/3 while the question runs. |
| `SESSION_LEASE_MARGIN_SECONDS` | `2` | 0–60 | A lease that is not renewed at least this long before it expires is treated as lost, and the turn is not saved. |
| `SESSION_TURN_MAX_SECONDS` | `180` | 10–3600 | Longest time one question can hold its session's lease. |
| `SESSION_MAX_TURNS` | `20` | 1–200 | Turns kept per session; older turns are deleted. |
| `SESSION_HISTORY_TURNS` | `4` | 0–50 | Stored turns read for each question. `0` disables history. The prompt uses at most 4 turns regardless (see [Session history](#session-history)). |
| `SESSION_STORED_ANSWER_CHARS` | `2000` | 100–20000 | Stored answers are cut to this length. |
| `SESSION_SHUTDOWN_GRACE_SECONDS` | `3` | 0–30 | Shutdown: time allowed for in-flight questions to finish. |
| `SESSION_KEEPER_JOIN_SECONDS` | `2` | 0–30 | Shutdown: time allowed for the lease keeper thread to stop. |
| `SESSION_RELEASE_BUDGET_SECONDS` | `2` | > 0, ≤ 30 | Shutdown: time allowed for releasing remaining leases. |

Startup validation also requires:

- `SESSION_ABSOLUTE_TIMEOUT_SECONDS` ≥ `SESSION_IDLE_TIMEOUT_SECONDS`.
- `SESSION_TURN_MAX_SECONDS` ≥ `STREAM_TIMEOUT_SECONDS` + 15, so a streamed question never loses its lease before the stream's own deadline.
- `SESSION_LEASE_MARGIN_SECONDS` < `SESSION_LEASE_TTL_SECONDS` / 3.

These settings are not listed in `.env.example`.

#### Session API

Session tokens are **bearer credentials**: anyone who has a token can use and delete that conversation. Send the token only in the `X-Session-ID` header, never in a URL.

**`POST /api/sessions`** creates a session.

- The request must have `Content-Type: application/json` and an empty body or `{}` (at most 1 KiB); otherwise `415`.
- A request from another site is refused with `403`: a `Sec-Fetch-Site` header other than `same-origin` or `none`, or an `Origin` header whose host and port differ from the request's `Host`. These checks run before any database access. They protect against cross-site requests; they are **not** authentication, and session creation is otherwise unauthenticated.
- `201` returns the following, with `Cache-Control: no-store`:

  ```json
  {"session_id": "<token>", "idle_timeout_seconds": 1800, "expires_at": "2026-01-01T08:00:00Z"}
  ```

  **Despite its name, `session_id` in this response is the bearer token.** It is returned only here and never again. The server stores only its SHA-256 hash. A separate internal session ID (a UUID) is used in logs and audit records; the token is never written to either.
- `404` while sessions are disabled. `503` `session_unavailable` if the session database cannot be used or `MAX_ACTIVE_SESSIONS` is reached.

**`DELETE /api/sessions`** (token in `X-Session-ID`) deletes the conversation memory.

- `204` when the session was deleted, and also when it was unknown, expired, or already deleted. It also answers `204` while sessions are disabled, and still deletes a stored session in that case.
- `400` `invalid_session_id` when the header is missing, empty, or longer than 128 characters. `503` `session_unavailable` when deletion cannot be confirmed.
- Audit records are **not** deleted (see [Conversation session data](#conversation-session-data)).

**`POST /api/chat` and `POST /api/chat/stream` with `X-Session-ID`** answer the question with the conversation's history:

- Without the header, both endpoints behave exactly as before (stateless).
- With the header, the session must exist and be unexpired. A header sent while sessions are disabled, or an empty header, is answered with `410` (`/api/chat/stream` first answers `404` while streaming is disabled). It is never ignored, and the question is never answered without its session.
- `/api/chat` adds `"session": {"history_turns_used": n, "turn_saved": true | false}` to its response. `turn_saved` is `true` only if the turn was actually stored.
- `/api/chat/stream` reports the same information through `done` and the `session` event (see [SSE event contract](#sse-event-contract-version-1)).
- If answering fails (`500`), nothing is saved.

#### Session errors

Session errors have the body `{"detail": {"code": "...", "message": "..."}}`. Messages are generic and never include the token or exception details.

| Status | `code` | Meaning |
| ------ | ------ | ------- |
| `400` | `invalid_session_id` | `DELETE` without a usable `X-Session-ID` header. |
| `403` | `origin_not_allowed` | Session creation from another site. |
| `409` | `session_busy` | Another question in this conversation is still running. `Retry-After` gives the remaining lease time in whole seconds (at least 1). Nothing was run. |
| `410` | `session_expired` | The session is unknown, deleted, or expired, or sessions are disabled. Start a new session. |
| `415` | `unsupported_media_type` | Session creation without `Content-Type: application/json` or with a body other than empty or `{}`. |
| `503` | `session_unavailable` | The session database cannot be used, or the session cap is reached. |

#### Session lifecycle: expiry, capacity, and cleanup

- **One question at a time.** Each question takes its session's lease, which lasts `SESSION_LEASE_TTL_SECONDS` (30 s). A background keeper thread renews it every TTL/3 (10 s) while the question runs, up to `SESSION_TURN_MAX_SECONDS` (180 s) from when the question started. A second question during that time receives `409`.
- **Idle expiry** (`SESSION_IDLE_TIMEOUT_SECONDS`, 30 min) is counted from creation. It is extended when a turn is saved and when a running question's lease is renewed. A question that cannot be saved does not extend it beyond these renewals.
- **Absolute expiry** (`SESSION_ABSOLUTE_TIMEOUT_SECONDS`, 8 h) is counted from creation and never extended.
- **Capacity.** Only unexpired sessions count toward `MAX_ACTIVE_SESSIONS`. At the cap, creation returns `503` and a warning is logged.
- **Cleanup** deletes expired sessions that have no running question, in batches of `SESSION_CLEANUP_BATCH`, together with their turns. When it deletes anything, it then checkpoints the write-ahead log. It runs:
  - once at startup, only if the session database already exists (even while sessions are disabled), waiting at most 2 seconds;
  - before every session creation;
  - every `SESSION_CLEANUP_INTERVAL_SECONDS` (300 s) on the keeper thread, which starts with the first question of the process.

  Expired sessions are therefore removed only when the server has activity. After the startup cleanup pass, expired sessions remain on disk until the next session creation or a subsequent periodic cleanup pass triggered by session activity.

#### Session history

- Each saved turn stores the user's question, the answer (cut to `SESSION_STORED_ANSWER_CHARS`), and the answer source. The last `SESSION_MAX_TURNS` turns are kept.
- For each question, the prompt includes at most 4 earlier turns and 6,000 characters, with each answer cut to 2,000 characters. These limits are fixed in the workflow.
- When the session has saved turns, the new question is first rewritten into a standalone question by an extra internal model call, which is never streamed. Retrieval, web search, and routing use the standalone question. If the rewrite fails or is invalid, the original question is used.
- Earlier turns are marked in the prompts as untrusted conversation data, not instructions.

#### Frontend behavior in a session

- **Token handling.** The token is kept only in the page's JavaScript memory: not in cookies, browser storage, or URLs. Reloading or closing the page ends the conversation for that page; the old session is no longer used and expires on the server.
- **Session creation.** Before the first question, the page creates a session.
  - The request times out after 15 seconds. It is retried once, after 1 second, only after an ordinary network failure.
  - A timeout, or leaving the page, is not retried.
  - Leaving the page abandons the request without showing an error.
  - When sessions are disabled, the frontend may still send `POST /api/sessions` on the first question. The endpoint returns `404`, after which the chat continues statelessly. This adds an unnecessary request but does not prevent normal stateless chat.
  - **A `404` switches the page to stateless questions** for the rest of the page session. This is how a disabled feature is detected. A proxy that answers `404` for `/api/sessions` by mistake (for example, a misrouted path) therefore also makes the page stateless.
  - Any other failure blocks the question: nothing is sent, and the question is never sent without a session.
- **What the user sees after an answer.** A status line below the input box reports whether the answer was saved to the conversation:
  - saved: follow-ups can use it;
  - not saved: follow-ups won't include it;
  - unknown: the save could not be confirmed.
- **Recovery when a question is not sent.** An alert explains what happened, and the question stays in the input box:
  - **Retry** is offered when trying again can help: the conversation is busy (`409`), session storage is unavailable (`503`), a network failure or timeout, or another server error.
  - **Reload page** is offered for `415` and other client errors.
  - **No button** is shown for `403`; the page must be opened from its usual address.
  - **Start new chat** is offered when the conversation has expired (`410`). The page does not continue an expired conversation in a new session on its own: every later question is blocked until the user starts a new chat, which keeps the unsent question in the input box.
  - None of these is retried automatically, and none falls back to a question without the session.
- **Suggestion cards.** A suggested question that is blocked is placed in the input box only if the box is empty or contains only whitespace. Text the user has typed is never overwritten; in that case Retry sends the typed text.
- **New chat** forgets the token immediately and sends one `DELETE /api/sessions` without waiting for it. If the deletion fails, and no new session has started in the meantime, the status line says that the earlier memory will be cleared automatically after a period of inactivity. The deletion is not retried.
- **Streaming fallback** keeps the session token (see [Frontend behavior and fallback](#frontend-behavior-and-fallback)).

#### Operating sessions

- **SQLite 3.35.0 or newer** is required. With an older version the session store disables itself: an error is logged and session creation returns `503`.
- **Abandoned (orphan) sessions.** A server-side session can exist that no browser holds a token for. This happens when session creation timed out or the page was left after the server created the session, when its response was lost, or after a page reload. Abandoning the browser request does not undo the server-side creation. Such sessions are never used, count toward `MAX_ACTIVE_SESSIONS` until their idle expiry, and are then removed by cleanup.
- **Request load.** A question asked in a session that already has saved turns makes one more model call (the standalone rewrite) than the same question without a session.
- **Rate limits.** `POST /api/sessions` has no rate limit or authentication. Apply limits at the proxy.

---

## 10. Validation and Testing

Testing is necessary to establish that the workflow makes appropriate decisions and handles failures safely.

### Checks performed

| Check                           | Observed result                     | Interpretation                                                                            |
| ------------------------------- | ----------------------------------- | ----------------------------------------------------------------------------------------- |
| Python syntax compilation       | Passed for `app/rag/workflow.py`    | The tested module compiled without a syntax error                                         |
| Remote-access security question | Successful response                 | CISA-related results and inline citations were returned                                   |
| NIST incident-response question | Successful response                 | Official NIST source URLs were returned                                                   |
| CISA phishing question          | Successful response                 | CISA-related source URLs and inline citations were returned                               |
| Workflow trace                  | Present in the tested API responses | The response exposed routing, search, grading, and generation steps                       |
| Citation deduplication          | Needs improvement                   | Some CISA pages appeared under different URL forms                                        |
| Claim-to-source alignment       | Needs further testing               | Inline references were present, but claim-level support was not independently established |

### Syntax check

The following command was executed successfully:

```bash
python -m py_compile app/rag/workflow.py
```

A successful syntax compilation confirms that Python can compile the module. It does not validate external service connectivity, answer accuracy, authorization controls, or overall application correctness.

### Recommended automated test suite

The repository includes automated Python and JavaScript test suites (see [Running the automated tests](#running-the-automated-tests)). Areas where repeatable tests are recommended include:

- Routing for greetings, internal IT questions, and official-source requests.
- Retrieval relevance and expected document selection.
- Evidence grading for both relevant and irrelevant documents.
- Query rewriting and retry-limit enforcement.
- Behavior when private and public evidence are both insufficient.
- Citation-number validity and URL deduplication.
- API input validation and error handling.
- Missing credentials, timeouts, and provider failures.
- Authorized and unauthorized document ingestion.
- Audit-record completeness and protection of sensitive data.

### Running the automated tests

Python tests (install `requirements-dev.txt` first):

```bash
pytest -q
```

JavaScript tests for the web interface's chat client, run with the Node.js built-in test runner (tested with Node.js 25; no npm packages are needed):

```bash
npm run test:js
```

Tests marked `live` call real external services and are skipped unless `RUN_LIVE_TESTS=1` is set. A passing run shows that the tested behavior works as specified; it does not establish answer quality, security, or production readiness.

**Test provenance note.** `tests/test_session_api.py` and `tests/test_session_leases.py` have been tracked in Git since commit `9be3ef2`. Before that commit, for one round of H2 edits, their earlier contents were reconstructed from the edits rather than taken from saved copies, so the byte-level history of those edits is incomplete. The current files are what the suite runs.

---

## 11. Security and Integrity Considerations

Security is particularly important when an AI assistant can access private enterprise documentation or accept uploaded files.

The current project demonstrates source-aware workflow behavior and basic local validation. A formal security audit, penetration test, or production-readiness certification should not be claimed without separate evidence.

### Secrets management

- Keep `.env` out of version control.
- Use `.env.example` to document required configuration without exposing credentials.
- Rotate any credential accidentally committed to a repository.
- Use a managed secret store for production deployments.

Example `.gitignore` entries:

```gitignore
.env
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
audit.db
```

Review the existing `.gitignore` before modifying it.

### Input validation

The API should validate incoming request schemas, impose reasonable question-length limits, and handle malformed input without exposing internal exceptions.

Document-ingestion endpoints should enforce file-size limits, supported file types, content validation, and safe file handling.

### Authentication and authorization

Administrative operations such as document ingestion should require appropriate authentication and authorization. A shared API key can serve as a basic prototype control but is not equivalent to enterprise identity management.

For production use, consider an identity provider, role-based access control, and document-level authorization.

### Retrieval and tenant isolation

Pinecone namespaces help organize vectors, but they do not independently guarantee that users can only access authorized documents.

A production system must enforce access permissions at retrieval time and prevent cross-tenant data exposure.

### Prompt injection and untrusted content

Retrieved documents and web pages may contain malicious or misleading instructions. The application should treat retrieved content as untrusted evidence, not as instructions that can override system behavior or grant new permissions.

Future hardening should include adversarial tests involving prompt injection in documents and search results.

### Citation integrity

The workflow is designed to associate numbered inline references with retrieved sources. Further validation is needed to ensure that every reference is valid, every returned URL is normalized, and every cited source supports the claim it accompanies.

### Audit logging and privacy

Audit records can support debugging and operational investigation. They should capture only the information necessary for the stated purpose.

Avoid unnecessarily storing API keys, authentication tokens, sensitive document contents, or personal information in logs. Define access controls, retention periods, and deletion procedures before production use.

**Current behavior:** each audit record in `data/audit.db` stores the question text, the answer source, the workflow trace, citations, and the outcome. The application has no audit retention or deletion mechanism, so audit records are kept indefinitely until an operator removes them. This is a known limitation, separate from session expiry and deletion.

### Conversation session data

Implemented behavior (see [Conversation sessions](#conversation-sessions-controlled-prototype-disabled-by-default)):

- **Bearer tokens.** A session token is a random bearer credential (`secrets.token_urlsafe(32)`). It is not tied to a user or identity: anyone holding it can continue or delete the conversation. It is returned once, at creation, and must only be sent in the `X-Session-ID` header.
- **Hashed storage.** The server stores only the SHA-256 hash of the token, never the token itself, and never logs it. Logs and audit records use a separate internal session ID (a UUID).
- **Stored content.** `data/sessions.db` holds each saved turn's question, answer (cut to `SESSION_STORED_ANSWER_CHARS`), and answer source, for at most `SESSION_MAX_TURNS` turns per session.
- **SQLite settings.**
  - The session database uses write-ahead logging (`journal_mode=WAL`), so `data/sessions.db-wal` and `data/sessions.db-shm` files exist next to it and can also contain conversation text.
  - Each connection enables `foreign_keys`, so deleting a session also deletes its turns.
  - Each connection enables `secure_delete`, so deleted content is overwritten in the database file.
- **Deletion and expiry do not affect audit records.** `DELETE /api/sessions`, expiry, and cleanup remove only session data. Each audit record of a session request keeps its question text and the internal session ID.
- **Data sent to external services.** Earlier turns are included in later prompts sent to OpenAI. The rewritten standalone question is used for knowledge-base retrieval (embedded with OpenAI) and as the Tavily web-search query, so a web search can contain details taken from earlier turns.
- **Version control and images.** `.gitignore` excludes `*.db`, `*.db-wal`, `*.db-shm`, and `*.db-journal` files at any directory depth. `.dockerignore` excludes root-level `*.db` files and, at any depth, `**/*.db`, `**/*.db-wal`, `**/*.db-shm`, and `**/*.db-journal`. These rules keep the database files and their WAL, SHM, and rollback-journal files out of Git and out of Docker build contexts.

Known limitations:

- **Deletion is not forensic erasure.**
  - `secure_delete` applies to the database file, but copies of deleted content can remain in the write-ahead log until it is checkpointed. Cleanup checkpoints the log after deleting expired sessions; `DELETE /api/sessions` does not.
  - Copies can also remain in filesystem or storage-device remnants and in backups.
- **No encryption at rest.** Neither `sessions.db` nor `audit.db` is encrypted by the application. Disk or volume encryption is a deployment responsibility.
- **No identity binding.** Session protection relies on keeping the token secret; it is not a substitute for authentication.

Recommended production hardening: an identity-bound session model, encrypted storage, an audit retention policy, and backups that respect deletion requirements.

### Dependency and deployment security

Recommended checks include dependency vulnerability scanning, secret scanning, code review, container image scanning, and restricting network access to required services.

These are recommended hardening measures, not claims that a full security assessment has already been completed.

---

## 12. Cybersecurity Knowledge Demonstration: CISA and NIST

The project has been exercised with questions about public cybersecurity guidance, demonstrating how the external-source workflow can retrieve relevant official material.

### CISA: Remote-access security

A test asked for three important practices for securing remote access to a corporate network.

The application returned CISA resources related to securing remote-access software, with inline numbered references and source metadata.

This scenario demonstrates external search, evidence evaluation, and source-aware generation.

### NIST: Incident response

A test asked:

> According to NIST, what are the main steps in incident response?

The application returned a response discussing preparation, detection and analysis, containment, eradication, recovery, and post-incident activity, with official NIST references.

**Important versioning note:** The cited *Computer Security Incident Handling Guide*, NIST SP 800-61 Rev. 2, describes the traditional incident-handling lifecycle. NIST has since published Rev. 3, which reframes incident response within the Cybersecurity Framework 2.0. The appropriate reference depends on the question and the version of guidance being discussed.

### CISA: Phishing protection

A test asked how an organization should protect against phishing attacks.

The application returned CISA-related material discussing phishing awareness, verification of suspicious messages, education, and reporting.

The observed response also revealed an improvement opportunity: the same underlying pages appeared through multiple URL formats, and the references were not yet verified at the individual-claim level.

### What these tests demonstrate

- The application can route official-source requests to web search.
- External evidence can be evaluated before answer generation.
- The API can return public source URLs and inline references.
- Workflow traces can expose the selected search and generation path.
- Realistic cybersecurity questions can be used to identify weaknesses in citation quality and retrieval relevance.

These tests are application behavior checks. They do not certify that the system is compliant with NIST, CISA, HIPAA, SOC 2, or any other security or regulatory framework.

---

## 13. Evaluation Metrics

The following metrics can be used to evaluate the system as the test dataset grows.

| Metric                     | What it measures                                                      |
| -------------------------- | --------------------------------------------------------------------- |
| Retrieval Recall\@K        | Whether relevant documents appear among the top K retrieved results   |
| Retrieval Precision\@K     | How many of the top K retrieved results are relevant                  |
| Answer Faithfulness        | Whether generated statements are supported by the available evidence  |
| Citation Precision         | Whether cited sources directly support the claims they accompany      |
| Citation Coverage          | Whether claims that need evidence have appropriate citations          |
| Routing Accuracy           | Whether the workflow selects the correct execution path               |
| Fallback Accuracy          | Whether the system declines to answer when evidence is insufficient   |
| Retry Effectiveness        | Whether rewritten queries improve retrieval outcomes                  |
| Latency                    | Time spent in retrieval, search, generation, and the complete request |
| Provider Error Rate        | Frequency of API errors, timeouts, and unavailable services           |
| Estimated Cost per Request | Model and search usage cost per completed request                     |

These metrics should be measured using a representative, labeled test set. Do not report numerical performance improvements until they have been calculated from actual test results.

---

## 14. Known Limitations

The current implementation and observed tests identify several areas that need further validation:

1. **Citation relevance:** An inline reference may exist without fully supporting the associated claim.
2. **URL normalization:** Different URL formats can represent the same underlying page.
3. **Search completeness:** Public search may return incomplete or irrelevant evidence.
4. **Model variability:** LLM responses can differ between requests and model versions.
5. **Dependency on external providers:** Hosted model, vector database, and search failures can affect response availability.
6. **Production authorization:** Enterprise identity, fine-grained document access, and administrative permissions require dedicated validation.
7. **Evaluation coverage:** A few successful manual requests do not establish overall retrieval or answer accuracy.
8. **Operational readiness:** Deployment, monitoring, scaling, retention, and incident-management procedures need to be verified separately.
9. **Streaming prototype:** Streaming chat is disabled by default and is not production-ready. See [Production-readiness blockers](#production-readiness-blockers).
10. **Conversation sessions:** Controlled prototype — disabled by default; not production-ready until the relevant production blockers and operational risks are resolved or explicitly accepted. See [Session limitations and shutdown behavior](#session-limitations-and-shutdown-behavior).

Documenting these limitations makes the project more useful to engineers and reviewers by distinguishing working behavior from remaining engineering work.

### Session limitations and shutdown behavior

These are known limitations of the current implementation. They are separate issues: resolving one does not resolve the others.

| Issue | Current state |
| ----- | ------------- |
| Production blockers H1–H3 | Open (see [Production-readiness blockers](#production-readiness-blockers)). H1 is only partly addressed: Pinecone queries, upserts, and setup list, describe, and create calls, and Tavily searches, have per-attempt timeouts, but no call has a total deadline; the optional `PINECONE_RETRY_AFTER_MAX_SECONDS` and `CHAT_DEADLINE_SECONDS` (both off by default) bound Pinecone's `Retry-After` wait and give stateless `/api/chat` a cooperative deadline. H2 is only partly addressed: shutdown is bounded, but running work is not forcibly interrupted (see [Shutdown and deployment](#shutdown-and-deployment)). H3 is only partly addressed: disconnect, deadline, and shutdown tests pass on `uvloop` and `httptools`, and Uvicorn's `loop="auto"` and `http="auto"` selection of them is verified locally, but flow control under a real server and the selection inside the production Docker image are not verified. They also affect session requests, as described below. |
| Persistent storage | Not configured. `compose.yaml` defines no volume, so `data/sessions.db` and `data/audit.db` are lost when the container is recreated. |
| Encryption at rest | Not provided by the application (see [Conversation session data](#conversation-session-data)). |
| Audit retention | No retention or deletion mechanism; audit records are kept indefinitely (see [Audit logging and privacy](#audit-logging-and-privacy)). |

**Lease recovery.** A lease left behind by a stopped or crashed process remains valid for at most `SESSION_LEASE_TTL_SECONDS` (30 s) after its last renewal. Until then the conversation answers `409` with `Retry-After`; after that, the next question proceeds normally.

**Effect of slow external calls (blocker H1).** A slow external call keeps its question's lease, even after a stream has reported a timeout, and other questions in the same conversation receive `409` meanwhile. Pinecone queries and upserts and Tavily searches are time-limited (see [Enabling streaming](#enabling-streaming)), so a stalled one ends after its timeouts and retries (see [Timeout and retry budget](#timeout-and-retry-budget)): a Pinecone timeout fails the question, which releases the lease, and a Tavily timeout counts as a failed web search while the question continues. Slow Pinecone setup (its calls are retried, and a new index can take up to 120 s to become ready), or slow OpenAI calls, can keep the lease for up to `SESSION_TURN_MAX_SECONDS` (180 s). After that, or once the lease is lost, the question starts no further external call, the call in progress finishes, and the turn is not saved.

**Abandoned sessions.** Timed-out or abandoned session-creation requests, lost responses, and page reloads can leave server-side sessions that no browser uses. They count toward `MAX_ACTIVE_SESSIONS` until their idle expiry and are then removed by cleanup (see [Operating sessions](#operating-sessions)).

**Deployment topology.** The session store is designed for one host with local storage. Sharing the session database between replicas or hosts, and placing it on a network file system, are not supported or tested. Running several server processes against the same session database has not been tested either.

**One question at a time.** A conversation runs one question at a time. A second question sent while the first is running receives `409` and is not run.

### Shutdown and deployment

Streaming and conversation sessions both remain disabled by default (`STREAM_ENABLED=false`, `SESSIONS_ENABLED=false`). The Dockerfile's `--timeout-graceful-shutdown 8` limit applies to every deployment, including when both features are disabled. It also affects non-streaming `/api/chat` requests. The stream-drain and session-shutdown behavior described below applies only when streaming or sessions are enabled. It is bounded, but it is not a production-readiness claim: blockers H1 and H3 remain open, and H2 is only partly addressed.

**Shutdown sequence (implemented).** When the container is stopped, Docker sends `SIGTERM`:

1. Uvicorn stops accepting connections and waits for open responses. After `--timeout-graceful-shutdown 8` (set in the `Dockerfile`), it cancels the responses that are still open.
2. The application stops the stream service and waits up to 3 seconds (`STREAM_SHUTDOWN_DRAIN_SECONDS`, fixed in code) for stream workers to finish, so that their turns can be saved and their audit records written. A run that is still undecided after that is recorded as `cancelled` (`ServerShutdown`) and asked to stop; a run whose outcome is already decided is left as it is.
3. If sessions were used, the session shutdown then stops accepting new session questions, waits up to `SESSION_SHUTDOWN_GRACE_SECONDS` (3 s) for running questions, refuses new turn saves, and releases the leases that remain. It returns within `SESSION_SHUTDOWN_GRACE_SECONDS` + `SESSION_KEEPER_JOIN_SECONDS` + `SESSION_RELEASE_BUDGET_SECONDS` (7 s with the defaults).

**Timeout budget.** 8 s (Uvicorn graceful timeout) + 3 s (stream drain) + 7 s (session shutdown) + 2 s margin = 20 s, the `stop_grace_period` in `compose.yaml`, after which Docker sends `SIGKILL`. A test checks the `Dockerfile` command and the `compose.yaml` value against this budget. If any of these values changes, keep the total within the stop grace period.

**What users see when a deploy interrupts a stream.** A stream that is still open 8 seconds after the stop signal is cancelled. The connection ends with no `done` or `error` event, so the web interface shows its existing message, "The response was interrupted. Please try again.", and does not fall back to `/api/chat` automatically. The turn is not saved, the audit record is `cancelled`, and the user has to ask again.

**Outcomes for a session question that is running at shutdown:**

- **Saved:** the question finishes, and its save is admitted before new saves are refused. This includes a slow save that finishes after its response has already reported `turn_saved: null` (the response waits 2 seconds for the save result), as long as it finishes within the stream drain. Tests cover this with a real Uvicorn server.
- **Not saved:** a save that starts after new saves are refused is refused (`reason=manager_closed`), whatever the database shows.
- **Depends on SQLite lock order:** a save admitted before new saves are refused is not cancelled. It commits only if it gets SQLite's write lock and still owns the lease. If the bulk lease release commits first, the save fails its lease check and saves nothing. Admission before shutdown is therefore not a guarantee that the turn is saved. Tests cover both orders.
- **Not saved and not audited:** the process is stopped by `SIGKILL`, or crashes, before the worker finishes. Neither the turn nor its audit record is written. The lease is left to expire (see **Lease recovery** in [Session limitations and shutdown behavior](#session-limitations-and-shutdown-behavior)). This has not been reproduced in a container test.
- A streamed response that the server cancels is recorded as `cancelled`, and its turn is not saved.

**Limits of the shutdown guarantees (known limitations):**

- **Running work is not interrupted.** Neither an external call in progress nor a SQLite transaction that has already started is cancelled. After the stream drain, stream workers and `/api/chat` questions (stateless or with a session) start no further external call and stop once their current call returns. A worker blocked in a call without a timeout (blocker H1) can keep running beyond the budget.
- **Ingestion blocks shutdown (existing issue, not fixed by H2).** The `/api/ingest` route is asynchronous, but it loads the file and ingests the document synchronously, which blocks the event loop while ingestion runs. During a deployment, an ingestion in progress can delay Uvicorn's graceful-shutdown timer and the application's shutdown, so shutdown can exceed the 20-second Docker stop budget and end in `SIGKILL`. Until this is fixed separately, do not run ingestion during a deployment.
- **Process exit can wait for workers.** Python waits without a time limit for stream worker threads when the process exits, so a stuck worker can keep the process alive until Docker's `SIGKILL`, with the losses described above.
- **Non-streaming `/api/chat` requests have the same 8-second limit, even with both features disabled.** Uvicorn cancels a `/api/chat` request that is still running when the graceful-shutdown timeout expires.
  - **Client response (not verified for this application).** A probe with a minimal FastAPI app on the same Uvicorn and library versions returned `500` with a plain-text `Internal Server Error` body. That result was observed only in the minimal-app probe; it has **not** been tested against this application's `/api/chat` route.
  - **The handler keeps running for now.** The synchronous handler runs in an AnyIO worker thread, and cancelling the request task does not interrupt that thread. After the stream drain it starts no further external call; a call already in progress finishes first.
  - **But the process does not wait for it.** That worker thread does not keep the process alive. If the handler has not finished when the process exits, its remaining work stops, so its audit record and its session-turn save may be lost.
  - **If it finishes before the process exits,** its audit record can still be written, but its turn is saved only if its save is admitted before new saves are refused.
- **Connection teardown overlaps the drain.** When the graceful timeout expires, Uvicorn cancels open responses and starts the application shutdown at once, so the stream drain can begin while a cancelled connection is still closing. The session shutdown runs after the drain; in the real-Uvicorn tests the connection is closed by then.
- **Cancelling the shutdown itself.** If the application's shutdown is cancelled while it waits for stream workers (a normal Uvicorn shutdown does not do this), the stream service is still cleared, but the session-manager shutdown is skipped. Leases then expire within `SESSION_LEASE_TTL_SECONDS` (30 s), with `409` meanwhile.
- **Turn saves are admitted one at a time, with a bounded wait.** Each save waits at most `SESSION_DB_BUSY_TIMEOUT_SECONDS` (2 s) to be admitted. If another save is stalled, for example on storage, a waiting save is not saved (`turn_saved: false`, `reason=save_gate_timeout`) and its lease is released, instead of waiting indefinitely. Finishing a question can therefore take up to about twice the busy timeout before it gives up.
- **Lease-expiry fallback.** If the stalled save also holds SQLite's write lock, the timed-out question's lease cannot be released either; it expires within `SESSION_LEASE_TTL_SECONDS` (30 s), and that conversation answers `409` until then.
- **Test coverage.** The admission timeout is tested at the session-manager level, not through a stream. Shutdown is tested with a real Uvicorn server on both the `asyncio` event loop with `h11` and the production runtime, `uvloop` with `httptools` (see blocker H3). A `SIGKILL` during shutdown is not tested.

---

## 15. Future Enhancements

Potential next steps include:

- Improve citation deduplication and claim-level source validation.
- Add automated evaluation using a curated question-and-evidence dataset.
- Benchmark a reranker against the existing retrieval pipeline.
- Add retrieval and answer-quality metrics to continuous integration.
- Implement enterprise identity integration and document-level permissions.
- Add document versioning, deletion, and controlled re-indexing.
- Add observability for latency, retrieval failures, model usage, and costs.
- Introduce a production-grade audit database and retention policy.
- Containerize and deploy the application using managed secrets and health monitoring.
- Add adversarial tests for prompt injection, malformed documents, and untrusted external content.
- Collect user feedback and measure real operational outcomes during a pilot.

---

## 16. Deployment Readiness Checklist

Before publishing or deploying the project, verify the following:

- [ ] The application starts from a clean environment.
- [ ] Dependency versions are pinned and documented.
- [ ] Required settings are listed in `.env.example`.
- [ ] Real credentials and local databases are excluded from Git.
- [ ] The configured vector index matches the embedding dimensions.
- [ ] Sample document ingestion completes successfully.
- [ ] Health and chat endpoints behave as expected.
- [ ] Routing, grading, and retry behavior have automated tests.
- [ ] Citation references are valid and duplicate URLs are handled.
- [ ] Insufficient-evidence scenarios return a safe fallback.
- [ ] Upload and administrative endpoints enforce authorization.
- [ ] Logs avoid exposing credentials and unnecessary sensitive content.
- [ ] Dependency and secret scans have been reviewed.
- [ ] Production deployment and monitoring have been tested.
- [ ] Streaming remains disabled (`STREAM_ENABLED=false`) unless blockers H1–H3 are resolved.
- [ ] If streaming is enabled, heartbeat and timeout settings stay within the frontend limits, and capacity is planned for `/api/chat` and streaming together.
- [ ] Conversation sessions remain disabled (`SESSIONS_ENABLED=false`) unless the issues in [Session limitations and shutdown behavior](#session-limitations-and-shutdown-behavior) are resolved or explicitly accepted.
- [ ] If sessions are enabled, the requirements below are met.

### Session and database deployment requirements

Requirements of the current implementation:

- **SQLite 3.35.0 or newer.** Older versions disable the session store (creation returns `503`). The Dockerfile uses the `python:3.11-slim` tag, which is not pinned to a digest, so check the version in every new image:

  ```bash
  python -c "import sqlite3; print(sqlite3.sqlite_version)"
  ```

- **Local, persistent storage for `data/`.** Both `sessions.db` (with its `-wal` and `-shm` files) and `audit.db` live there by default. The provided `compose.yaml` does not define a volume yet, so data is lost when the container is recreated. Use storage local to the host, not a network file system, and restrict access to it.
- **One host.** Do not share the session database between replicas or hosts (see [Session limitations and shutdown behavior](#session-limitations-and-shutdown-behavior)).
- **Shutdown timeout budget.** The `Dockerfile` runs Uvicorn with `--timeout-graceful-shutdown 8`, and `compose.yaml` sets `stop_grace_period: 20s`. Uvicorn's timeout (8 s), the stream drain (3 s), and the session shutdown (7 s with the defaults) must stay at least 2 seconds within the stop grace period (see [Shutdown and deployment](#shutdown-and-deployment)).

Recommended production hardening (not implemented by the application):

- Encrypt the storage volume; the application does not encrypt either database.
- Define backups for `audit.db` and `sessions.db`. Backups keep copies of conversation and audit data, so align them with retention and deletion requirements.
- Define an audit retention period and a deletion procedure.
- Apply rate limits to `POST /api/sessions` at the proxy.
- Plan deploys around the shutdown behavior described in [Shutdown and deployment](#shutdown-and-deployment): streams in progress are interrupted, and work that outlives the 20-second stop grace period is lost.

---

## 17. Project Summary

The Enterprise IT Support Agentic RAG Copilot demonstrates how an AI workflow can be developed into a practical enterprise support application.

It brings together semantic retrieval, conditional routing, evidence grading, external search, query rewriting, grounded generation, API delivery, source metadata, and workflow tracing.

The project has been tested with representative remote-access security, NIST incident-response, and CISA phishing questions. Those tests confirmed that the application returned answers and official-source metadata for the tested cases. They also identified citation deduplication and claim-level source validation as areas for further improvement.

The central engineering objective is to make the assistant useful, traceable, and cautious when evidence is incomplete—not merely to generate fluent responses.

**The intended outcome:** a foundation for a more accessible and auditable IT support experience, with production readiness and business impact established through repeatable evaluation rather than assumptions.

---

## Disclaimer

BluePeak Technologies is a fictional customer used for demonstration purposes. Public cybersecurity examples are provided for educational and software-engineering demonstration purposes and should be checked against the latest official guidance before operational use.

This project is not a substitute for an organization's approved security policies, incident-response procedures, or professional security assessment.
