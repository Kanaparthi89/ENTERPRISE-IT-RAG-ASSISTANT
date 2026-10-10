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

The project is organized around a modular application architecture. The tree below illustrates the principal components; verify optional scripts and filenames against the final repository before using them.

```text
ENTERPRISE-IT-RAG-ASSISTANT/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── api/
│   │   └── routes.py           # API routes
│   ├── rag/
│   │   ├── workflow.py         # Agentic RAG workflow
│   │   └── state.py            # Workflow state and schemas
│   ├── services/
│   │   └── audit.py            # Audit persistence
│   ├── ingestion/              # Document ingestion components
│   ├── vectorstore/             # Pinecone integration
│   ├── templates/
│   │   └── index.html          # Frontend template, if enabled
│   └── static/                  # Static frontend assets
├── data/
│   └── sample_kb/               # Sample knowledge-base documents
├── tests/                       # Automated tests, if present
├── .env.example                 # Example configuration
├── .gitignore
├── requirements.txt
├── run.py                       # Optional application entry point
└── README.md
```

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
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

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

The next stage is to build repeatable tests covering:

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

Run the automated tests when the suite is present:

```bash
pytest -q
```

This command is a recommended validation step; do not interpret it as evidence that all tests have already passed.

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

Documenting these limitations makes the project more useful to engineers and reviewers by distinguishing working behavior from remaining engineering work.

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
