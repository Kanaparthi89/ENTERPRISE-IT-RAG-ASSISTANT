# ENTERPRISE-IT-RAG-ASSISTANT

# Enterprise ITL RAG Assistant

An end-to-end **Forward Deployed Engineer (FDE) project** that transforms an Agentic Retrieval-Augmented Generation (RAG) workflow into a practical enterprise IT support assistant. The application combines private knowledge retrieval, evidence evaluation, conditional web search, query rewriting, grounded answer generation, and a user-facing interface.

The project is designed around a fictional enterprise environment and demonstrates how AI engineering can solve a real operational problem through workflow orchestration, backend development, knowledge integration, security, and user experience.

**Core technologies:** Python, LangGraph, FastAPI, Pinecone, Hugging Face, Groq, Tavily, HTML, CSS, JavaScript, and SQLite.

---

## 1. Business Problem and Customer Context

### Customer: BluePeak Technologies

BluePeak Technologies is a fictional enterprise technology and retail operations company with approximately **5,000 employees** distributed across offices, warehouses, and remote locations.

The company's internal IT department maintains documentation covering account access, VPN configuration, multifactor authentication (MFA), approved software, device troubleshooting, network connectivity, and service-desk procedures.

As the organization grows, employees increasingly depend on the IT help desk for routine questions. Although answers may already exist in internal documentation, employees often struggle to find the correct procedure or determine whether an outdated document is still applicable.

### The problem we are solving

The existing support process creates several operational challenges.

- **Scattered documentation:** Employees must search across multiple documents to find the right instructions.
- **Poor search relevance:** Traditional keyword search can return documents that contain similar words but do not answer the actual question.
- **Unreliable chatbot answers:** A general-purpose LLM may generate a plausible response without sufficient supporting evidence.
- **Changing information:** Internal policies and external vendor guidance can change over time.
- **Repeated support requests:** IT staff spend time answering common questions instead of resolving more complex incidents.
- **Limited transparency:** Without a visible decision path, engineers may struggle to understand why an answer was produced or why retrieval failed.

### Example business scenario

An employee asks:

> "My company laptop connects to Wi-Fi, but I cannot access internal applications from home. How should I troubleshoot it?"

The assistant should first retrieve relevant private IT documentation, such as VPN setup and connectivity troubleshooting procedures. If the retrieved documents sufficiently answer the question, it should respond using that internal evidence without unnecessarily calling an external search service.

Now consider another question:

> "Has the vendor published any new guidance for the authentication issue affecting its latest desktop application release?"

If the internal knowledge base does not contain sufficient evidence, the assistant should evaluate whether an external search is appropriate. It can then retrieve public information, assess its relevance, and answer with a clear indication that the information came from external sources.

The important distinction is that the assistant does not automatically search the web or generate an answer simply because a question was submitted. It makes retrieval and evidence decisions before responding.

### Business goal

The goal is to build an enterprise IT support assistant that helps employees find reliable answers faster while giving engineers control over how information is retrieved and used.

The system is designed to:

1. Prioritize approved company documentation for internal IT questions.
2. Evaluate retrieved evidence before generating an answer.
3. Use external search when private documentation is insufficient and the question warrants public information.
4. Rewrite unclear or unsuccessful queries and retry retrieval within a controlled limit.
5. Avoid presenting unsupported information as established fact.
6. Display the retrieval source and decision trace for debugging and auditing.
7. Allow authorized personnel to upload additional company documents without modifying the application code.
8. Expose the workflow through a web interface and REST API.

**XYZ approach:** By combining private document retrieval, evidence grading, conditional search, and grounded generation in one workflow, the project aims to reduce repetitive information-search effort while improving answer traceability and the consistency of IT support responses. Actual reductions in ticket volume or response time would need to be measured through a pilot rather than assumed.

---

## 2. Why This Is a Forward Deployed Engineer (FDE) Project

A Forward Deployed Engineer works at the intersection of customer requirements, software engineering, AI systems, and operational delivery. The objective is not simply to demonstrate that an LLM can answer questions; it is to build a solution that addresses a customer's workflow and can be integrated into a usable application.

This project follows that approach by starting with BluePeak Technologies' IT support challenges and translating them into a working AI application design.

### FDE delivery lifecycle

```text
Customer Discovery
        |
        v
Identify Support Bottlenecks
        |
        v
Define Requirements and Success Metrics
        |
        v
Connect Internal Knowledge Sources
        |
        v
Build the Agentic RAG Workflow
        |
        v
Expose the Workflow Through an API
        |
        v
Develop the Employee Interface
        |
        v
Add Access Controls and Audit Logging
        |
        v
Test Retrieval, Routing, and Failure Cases
        |
        v
Deploy, Monitor, and Improve
```

### How the project demonstrates FDE skills

**Customer discovery and requirements:** The project translates repetitive IT support requests into specific system requirements, including private knowledge retrieval, source-aware responses, and controlled fallback behavior.

**Solution architecture:** LangGraph coordinates the retrieval and decision-making process, while FastAPI exposes the workflow through application endpoints. The frontend provides a practical way for employees to interact with the system.

**Knowledge integration:** Internal documentation is loaded, split into manageable chunks, embedded, and indexed for semantic retrieval. New documents can be added through a protected ingestion workflow.

**Application engineering:** The notebook-style AI workflow is organized into backend modules and connected to a web interface, moving the project toward a maintainable application rather than an isolated experiment.

**Reliability and security:** Evidence grading, retry limits, input validation, access controls, and audit logging help engineers understand the system's behavior and manage common failure cases.

**Customer feedback and iteration:** Demo scenarios expose retrieval errors, weak evidence, and confusing responses. These findings can guide improvements to document quality, chunking, retrieval settings, prompts, and workflow decisions.

**STAR summary:** The situation is an enterprise with repetitive IT support requests and scattered documentation. The task is to build a trustworthy assistant that uses internal knowledge first and handles missing evidence responsibly. The action is to integrate semantic retrieval, conditional agent routing, query rewriting, an API, and a user interface. The intended result is a more accessible and traceable support experience, with measurable operational improvements to be validated during testing.

---

## 3. High-Level Architecture

The application separates the employee interface, request handling, AI orchestration, retrieval services, and audit storage. This modular design makes it easier to test individual components and replace infrastructure choices as requirements change.

```text
                 ┌────────────────────────┐
                 │   BluePeak Employee    │
                 └────────────┬───────────┘
                              |
                              v
                 ┌────────────────────────┐
                 │  Support Portal        │
                 │  HTML / CSS / JS       │
                 └────────────┬───────────┘
                              |
                       HTTPS / JSON
                              |
                              v
                 ┌────────────────────────┐
                 │  FastAPI Request Layer │
                 │  Validation / Auth     │
                 └────────────┬───────────┘
                              |
                              v
                 ┌────────────────────────┐
                 │  LangGraph Controller  │
                 │  State + Routing       │
                 └────────────┬───────────┘
                              |
                ┌─────────────┼─────────────┐
                |             |             |
                v             v             v
       ┌────────────────┐ ┌─────────────┐ ┌────────────────┐
       │ Private Search │ │ Public Web  │ │ Direct Reply   │
       │ Pinecone +     │ │ Tavily      │ │ Simple Chat    │
       │ Embeddings     │ │ Search      │ │                │
       └───────┬────────┘ └──────┬──────┘ └────────────────┘
               |                 |
               └────────┬────────┘
                        |
                        v
              ┌─────────────────────┐
              │ Evidence Evaluation │
              │ and Answer Synthesis│
              │ Groq LLM             │
              └──────────┬──────────┘
                         |
                         v
              ┌─────────────────────┐
              │ Response + Sources  │
              │ + Workflow Trace    │
              └──────────┬──────────┘
                         |
                         v
              ┌─────────────────────┐
              │ SQLite Audit Store  │
              └─────────────────────┘
```

The browser sends a question to FastAPI. The backend validates the request and passes it to LangGraph, which manages the workflow state and decides what action to take next. Retrieval services return candidate evidence, the grading steps evaluate whether that evidence is useful, and the LLM generates a response from the selected context. The API returns the answer along with source information and a workflow trace, while the audit service records appropriate execution metadata.

The diagram represents the logical design. The exact execution order and persistence behavior depend on the implemented graph and API code.

---

## 4. Agentic RAG Workflow

### Why use Agentic RAG?

Traditional RAG generally retrieves documents and sends the resulting context to an LLM for answer generation.

```text
Question → Retrieve Documents → Generate Answer
```

This approach can work well when retrieval consistently returns relevant information. However, enterprise support questions can be ambiguous, the knowledge base may be incomplete, and some requests require current external information.

Agentic RAG introduces explicit decision points. Instead of assuming that the first retrieval result is sufficient, the workflow evaluates evidence and selects the next action based on the current state.

```text
Employee Question
        |
        v
Classify the Request
        |
        +------ Greeting / Simple Chat ------> Direct Response
        |
        v
Search Private Knowledge Base
        |
        v
Evaluate Retrieved Evidence
        |
        +------ Sufficient Evidence ---------> Internal Answer
        |
        v
Determine Whether External Search Is Needed
        |
        v
Search Public Sources
        |
        v
Evaluate External Evidence
        |
        +------ Sufficient Evidence ---------> External Answer
        |
        v
Rewrite the Search Query
        |
        v
Retry Private Retrieval
        |
        v
Retry Limit / Evidence Check
        |
        +------ Evidence Found --------------> Grounded Answer
        |
        +------ Still Insufficient ----------> Safe Fallback
```

### Step 1 — Route the question

The workflow first identifies whether the message is a greeting or a request that needs knowledge retrieval. A simple greeting can receive a direct response without invoking vector search or public web search.

For an IT support request, the workflow continues to private knowledge retrieval. This avoids unnecessary retrieval operations for messages that do not need them.

### Step 2 — Retrieve internal documentation

The system converts the question into an embedding and searches the Pinecone index for semantically related chunks. This helps match questions to relevant documents even when employees use different wording from the original documentation.

For example, a question about "getting into the office network from home" might retrieve a VPN setup document even if the document uses the phrase "remote access configuration."

### Step 3 — Grade the private evidence

The workflow evaluates whether the retrieved content actually addresses the question. A document that merely mentions VPNs is not necessarily sufficient to explain how to configure a VPN connection.

If the evidence is sufficient, the system generates an answer grounded in the internal documents. If the evidence is weak, the workflow determines whether external search is appropriate.

### Step 4 — Search public information when needed

Tavily searches publicly available sources for information relevant to the question. This branch is useful for questions about current vendor guidance, public service announcements, or information not covered by internal documentation.

External evidence is evaluated separately before being used for generation. A search result should not automatically be treated as correct simply because it is available online.

### Step 5 — Rewrite the query and retry

If retrieval does not provide sufficient evidence, the workflow can rewrite the question into a more specific search query.

For example:

- Original question: "My login keeps failing after the change. What now?"
- Rewritten query: "Troubleshooting enterprise SSO login failures after an identity-provider configuration change."

The rewritten query can be used to retry retrieval according to the implemented graph. Retry limits prevent the workflow from looping indefinitely.

### Step 6 — Generate a grounded response

When sufficient evidence is available, the LLM generates an answer based on the selected context. The response should distinguish internal company guidance from public information and avoid adding unsupported technical instructions.

If sufficient evidence remains unavailable, the assistant should say that it could not verify the answer and recommend an appropriate next step, such as contacting the service desk.

### Step 7 — Return the decision trace

The response can include the selected source, retrieval outcome, evidence grade, and relevant workflow steps. This makes it easier for engineers to understand why the system answered, searched externally, retried retrieval, or returned an insufficient-evidence response.

**Important:** A workflow trace is not the same as a guarantee of correctness. Evidence quality, citation accuracy, access permissions, and answer faithfulness must still be tested.

---

## 5. Technology Stack, Purpose, and Alternatives

The technologies below are selected to support the project's current architecture. Alternatives are included to explain which components could be replaced if a customer's infrastructure, cost, latency, or security requirements change.

| Layer | Selected technology | Purpose | Possible alternatives |
|---|---|---|---|
| Agent orchestration | LangGraph | Manages workflow state, conditional routing, evidence checks, and retries | LangChain chains, LlamaIndex workflows, custom Python state machine |
| LLM inference | Groq | Runs supported LLMs for classification, grading, rewriting, and answer generation | OpenAI, Anthropic, Google Gemini, self-hosted models through vLLM |
| Embeddings | Hugging Face `all-MiniLM-L6-v2` or configured embedding provider | Converts questions and document chunks into numerical vectors | OpenAI embeddings, BGE, E5, Cohere embeddings |
| Vector database | Pinecone | Stores document vectors and supports semantic similarity retrieval | Qdrant, Weaviate, Chroma, Milvus, pgvector |
| External search | Tavily | Retrieves relevant public web information when required | Bing Web Search, Brave Search, approved enterprise search APIs |
| API framework | FastAPI | Exposes chat, health, and document-ingestion endpoints | Flask, Django REST Framework |
| Frontend | HTML, CSS, JavaScript | Provides the employee chat experience and document-upload controls | React, Vue, Next.js |
| Audit storage | SQLite | Stores lightweight execution and decision metadata | PostgreSQL, MySQL, a centralized logging platform |
| Document processing | Python document loaders and chunking utilities | Extracts text from supported files and prepares searchable chunks | Unstructured, Apache Tika, cloud document-processing services |
| Packaging | Docker | Packages the application and its runtime dependencies | Podman, virtual-machine deployment, managed containers |
| Configuration | Environment variables and `.env` for local development | Separates configuration and credentials from source code | Cloud secret managers and platform-managed configuration |

### Why these technologies?

**LangGraph:** The core requirement is conditional decision-making rather than a single retrieval call. LangGraph provides explicit graph nodes, state transitions, and conditional edges, making the workflow easier to inspect and extend. A custom Python state machine could be sufficient for a smaller workflow, while LlamaIndex workflows are another option for retrieval-oriented applications.

**Groq:** The project needs an LLM for routing, relevance grading, query rewriting, and answer generation. Groq provides access to supported models through an API. OpenAI, Anthropic, Gemini, and self-hosted inference are alternatives, depending on model quality, data-handling requirements, cost, and latency.

**Embeddings:** Semantic retrieval requires a consistent method of converting documents and questions into vectors. The `all-MiniLM-L6-v2` model is a lightweight option that generates 384-dimensional embeddings and can run locally. OpenAI embeddings or BGE/E5 models are alternatives. The embedding model used during indexing must match the one used during query retrieval.

**Pinecone:** A managed vector database reduces the operational work required to host and scale semantic search infrastructure. Qdrant, Weaviate, and pgvector are alternatives when self-hosting, infrastructure consolidation, or deployment control is important.

**Tavily:** Public search is a fallback for questions that internal documentation cannot sufficiently answer. It helps the workflow access potentially current public information, which is then evaluated before use. Another approved search provider can be substituted if its API, licensing, and security requirements fit the deployment.

**FastAPI:** The project needs a lightweight API layer between the browser and the AI workflow. FastAPI supports typed request models, validation, and API documentation. Flask is a viable alternative for a smaller application.

**SQLite:** A local database is sufficient for a basic demonstration of query and decision-path logging. PostgreSQL or a centralized logging system is a better choice when multiple application instances, concurrent writes, retention policies, and production operations become important.

**Docker:** Containerization helps make development and deployment environments more consistent. It is useful when the customer wants a reproducible application package that can be deployed on a laptop, virtual machine, or container platform.

### Technology selection principle

The purpose of this stack is not to use as many AI technologies as possible. Each component addresses a specific engineering requirement: orchestration, retrieval, generation, external information access, API delivery, user interaction, or observability.

A production deployment should select providers based on the customer's security policies, data residency, cost, availability, operational requirements, and existing infrastructure.

---

## 6. Project Structure

*This section will be completed after the implementation is finalized. The directory tree, modules, tests, configuration files, and deployment assets will be documented to match the actual repository.*

```text
[Project structure to be added after implementation]
```

---

## 7. Setup and Installation

The following setup describes the intended local development workflow. Verify the final dependency versions and configuration names against the implemented code before treating these commands as a tested installation guide.

### Prerequisites

Install the following before starting:

- Python 3.11 or another Python version supported by the project's dependencies.
- Git for repository management.
- A Pinecone account and API key.
- A Groq API key.
- A Tavily API key if external search is enabled.
- An embedding provider and credentials if the configured embedding model uses a hosted API.
- Docker, optionally, if you want to run the containerized application.

If the project uses a local Hugging Face embedding model, it can download model files during initial setup. Make sure your machine has internet access for the first download and enough disk space for the model and document index.

### Step 1 — Clone the repository

Replace the placeholder repository URL with the actual GitHub repository URL.

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

If the repository already exists locally, open a terminal in its root directory instead.

### Step 2 — Create a virtual environment

Create an isolated Python environment so the project's dependencies do not interfere with other Python projects.

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Verify the active Python interpreter:

```bash
python --version
python -m pip --version
```

### Step 3 — Install dependencies

Install the packages listed in the project's dependency file.

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If a dependency fails to install, check the Python version and package compatibility before changing package versions. Once the project works, pin compatible versions in `requirements.txt` to improve reproducibility.

### Step 4 — Configure API credentials

Create a `.env` file in the project root. Use the following as a configuration template, and remove variables that the final implementation does not use.

```dotenv
# --------------------------------------------------
# Application
# --------------------------------------------------
APP_NAME="Enterprise ITL RAG Assistant"
APP_ENV=development
APP_HOST=127.0.0.1
APP_PORT=8000
LOG_LEVEL=INFO

# --------------------------------------------------
# LLM Provider: Groq
# --------------------------------------------------
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=your_supported_groq_model

# --------------------------------------------------
# Embeddings
# --------------------------------------------------
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Optional hosted embedding provider
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# --------------------------------------------------
# Pinecone Vector Database
# --------------------------------------------------
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=enterprise-itl-rag
PINECONE_NAMESPACE=bluepeak-it-support

# --------------------------------------------------
# Public Web Search
# --------------------------------------------------
TAVILY_API_KEY=your_tavily_api_key
ENABLE_WEB_SEARCH=true

# --------------------------------------------------
# Document Ingestion and Access Control
# --------------------------------------------------
ADMIN_API_KEY=replace_with_a_long_random_secret
MAX_UPLOAD_MB=10
UPLOAD_DIR=uploads

# --------------------------------------------------
# Retrieval and Workflow
# --------------------------------------------------
TOP_K=4
MAX_RETRIES=2

# --------------------------------------------------
# Audit Logging
# --------------------------------------------------
AUDIT_DB_PATH=./audit.db
```

**Configuration notes:**

- The fictional customer name appears in the namespace for demonstration purposes. Change it if the application is configured for another customer.
- `GROQ_MODEL` must contain a model identifier supported by the Groq account and API at deployment time.
- `EMBEDDING_PROVIDER` is illustrative unless the implementation explicitly supports provider selection.
- `OPENAI_API_KEY` and `OPENAI_EMBEDDING_MODEL` are only needed if the code actually uses OpenAI services. A local Hugging Face embedding model does not require an OpenAI API key.
- `EMBEDDING_DIMENSION` must match the selected embedding model and the Pinecone index dimension.
- `TOP_K` controls the number of candidate chunks retrieved, if this setting is supported by the code.
- `MAX_RETRIES` should align with the graph's actual retry logic. It should not be assumed to represent the exact number of graph transitions without inspecting the implementation.
- `ENABLE_WEB_SEARCH` should be enforced by the backend if it is intended to disable public search.
- `MAX_UPLOAD_MB` should be enforced by the API, not merely declared in configuration.

### Step 5 — Protect secrets

Never commit `.env` or real API keys to GitHub.

Add these entries to `.gitignore` if they are not already present:

```gitignore
.env
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
audit.db
uploads/*
!uploads/.gitkeep
```

Use strong, randomly generated secrets for administrative access. In production, prefer a managed secret store and proper authentication rather than a shared static API key.

### Step 6 — Create and configure the Pinecone index

Create a Pinecone index with the vector dimension and similarity configuration required by the selected embedding model.

For `sentence-transformers/all-MiniLM-L6-v2`, the usual embedding dimension is 384. If the implementation uses `text-embedding-3-small` instead, its configured output dimension must match the index and embedding request.

Use the configured index name and namespace consistently during document ingestion and retrieval. The namespace helps organize vectors, but it is not a substitute for authorization or tenant-level data isolation.

If the application creates indexes programmatically, follow the implementation's initialization logic rather than creating a conflicting index manually.

### Step 7 — Load the sample knowledge base

After placing the sample IT documents in the configured data directory, run the project's ingestion script if one is provided.

```bash
python ingest_sample_kb.py
```

The intended ingestion pipeline is:

```text
Internal IT Documents
        |
        v
File Validation and Text Extraction
        |
        v
Text Cleaning and Chunking
        |
        v
Embedding Generation
        |
        v
Pinecone Vector Indexing
        |
        v
Knowledge Available for Retrieval
```

Check the script output and Pinecone index statistics to verify that the documents were indexed successfully. The script name and directory must match the final repository.

### Step 8 — Start the application

If `run.py` is the entry point in the completed repository:

```bash
python run.py
```

Alternatively, if the application exposes `app` from `app.main`, start it with Uvicorn:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Use the command that matches the actual project entry point. Do not run both commands simultaneously unless the application is deliberately designed for that setup.

Open the local application:

- Web interface: `http://127.0.0.1:8000`
- API documentation: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/api/health`

These addresses assume that the app serves its frontend at `/` and exposes the documented health endpoint.

### Step 9 — Run tests

Run the project's available test suite:

```bash
pytest -q
```

Start with document-ingestion tests, then add tests for routing, retrieval, evidence grading, query rewriting, retry limits, source attribution, API validation, and protected document uploads.

Tests that call hosted LLMs or external search APIs may require credentials and incur usage costs. Where practical, use mocks and deterministic test fixtures for unit tests.

### Step 10 — Run with Docker (optional)

If a Dockerfile is included in the completed project:

```bash
docker build -t enterprise-itl-rag-assistant .
docker run --rm -p 8000:8000 --env-file .env \
  enterprise-itl-rag-assistant
```

This example assumes the Docker image starts the application and listens on port 8000. Verify the final `Dockerfile` before using these commands. Never bake secrets into the image.

---

## 8. Environment Variable Reference

The following table explains the purpose of the configuration variables and when they are required. It reflects the proposed configuration template; the final table should be reconciled with the variables read by the actual application.

| Variable | Purpose | Required? |
|---|---|---|
| `APP_NAME` | Identifies the application in logs and metadata | Optional |
| `APP_ENV` | Selects the runtime environment, such as development or production | Optional |
| `APP_HOST` | Configures the server bind address if supported by the entry point | Optional |
| `APP_PORT` | Configures the server port if supported by the entry point | Optional |
| `LOG_LEVEL` | Controls logging verbosity | Optional |
| `GROQ_API_KEY` | Authenticates requests to Groq | Required when Groq is used |
| `GROQ_MODEL` | Selects the model used for workflow tasks | Required by the configured provider client |
| `EMBEDDING_PROVIDER` | Selects a supported embedding implementation, if provider switching is implemented | Conditional |
| `EMBEDDING_MODEL` | Identifies the local or configured embedding model | Required |
| `EMBEDDING_DIMENSION` | Documents or configures the vector dimension expected by the application | Conditional |
| `OPENAI_API_KEY` | Authenticates OpenAI API requests | Conditional |
| `OPENAI_MODEL` | Configures the OpenAI chat model if used | Conditional |
| `OPENAI_EMBEDDING_MODEL` | Configures OpenAI embeddings if used | Conditional |
| `PINECONE_API_KEY` | Authenticates Pinecone requests | Required when Pinecone is used |
| `PINECONE_INDEX_NAME` | Identifies the vector index | Required or defaulted |
| `PINECONE_NAMESPACE` | Organizes vectors within the index | Optional or defaulted |
| `TAVILY_API_KEY` | Authenticates public search requests | Required when Tavily search is enabled |
| `ENABLE_WEB_SEARCH` | Enables or disables the external-search path if implemented | Optional |
| `ADMIN_API_KEY` | Protects administrative endpoints in a basic demonstration | Required if this authentication method is used |
| `MAX_UPLOAD_MB` | Sets an upload-size limit if enforced by the API | Optional |
| `UPLOAD_DIR` | Defines the location for temporary or stored uploads | Optional |
| `TOP_K` | Sets the candidate retrieval count if supported | Optional |
| `MAX_RETRIES` | Controls retry behavior if read by the workflow | Optional |
| `AUDIT_DB_PATH` | Defines the SQLite audit database location | Optional |

**Important implementation detail:** Environment variables do not automatically change application behavior. Every setting that is intended to be configurable must be read, validated, and used by the relevant code. Keep `.env.example` synchronized with the actual settings so developers do not configure variables that the application ignores.

---

## 9. Demo Scenarios and Expected Behavior

These scenarios are designed to demonstrate the assistant's decisions, not merely the quality of its final answers. Exact traces depend on the documents indexed, retrieval settings, grading prompts, and workflow implementation.

### Demo A — Internal VPN troubleshooting

**Question**

> "I am working from home. How do I connect my laptop to the company network?"

**Expected behavior**

The router identifies an IT support request. The workflow retrieves the company's VPN instructions, grades the retrieved evidence, and generates an answer from the private knowledge base when the evidence is sufficient.

**Illustrative trace**

```text
Question Received
→ IT Support Route
→ Private Knowledge Retrieval
→ Evidence Grade: GOOD
→ Generate Internal Answer
→ Return Private-KB Source
```

**Teaching point:** Internal company guidance is the preferred source for internal procedures. A successful private retrieval should not require a public web search.

**XYZ perspective:** By retrieving the approved procedure directly, the assistant aims to reduce the effort employees spend locating internal instructions.

### Demo B — Security policy and MFA

**Question**

> "A support technician asked me to read out my one-time MFA code. Is that allowed?"

**Expected behavior**

The assistant retrieves the internal security policy and uses its explicit instructions to answer. If the policy says that employees must never disclose MFA codes, the answer should state that rule and advise the employee to follow the company's reporting procedure if the request appears suspicious.

**Illustrative trace**

```text
Question Received
→ Private-KB Retrieval
→ Security Policy Retrieved
→ Evidence Grade: GOOD
→ Generate Policy-Based Answer
→ Return Internal Source
```

**Teaching point:** A general LLM may know common security advice, but retrieval allows the answer to reflect the fictional company's own documented policy.

**Testing note:** Verify that the expected policy actually exists in the indexed sample documents before presenting this as a successful retrieval demonstration.

### Demo C — Current vendor guidance

**Question**

> "Where can I find the latest public guidance for a Microsoft Teams connectivity incident?"

**Expected behavior**

If the internal knowledge base does not contain sufficient evidence and the question requires current public information, the workflow can use Tavily to retrieve relevant public sources. The evidence-grading step determines whether the returned material is suitable for an answer.

**Illustrative trace**

```text
Question Received
→ Private-KB Retrieval
→ Evidence Grade: WEAK
→ Tavily Search
→ External Evidence Evaluation
→ Generate Answer with Public Sources
→ Label Source as External
```

**Teaching point:** The assistant can change its information source instead of presenting incomplete internal documentation as the latest public guidance.

**Testing note:** This is an expected behavior, not a guarantee that Tavily will return relevant results every time. The demo should display actual retrieved source titles or URLs when available.

### Demo D — Ambiguous troubleshooting question

**Question**

> "My work apps stopped signing in after an update. How do I fix this?"

**Expected behavior**

The initial question may be too broad to retrieve the right instructions. The workflow evaluates the first results, rewrites the query into a more specific search phrase, and retries according to the configured graph logic.

**Illustrative trace**

```text
Question Received
→ Initial Private Retrieval
→ Evidence Grade: WEAK
→ Query Rewriting
→ Retry Retrieval
→ Grade New Evidence
→ Grounded Answer or Safe Fallback
```

A possible rewritten query is:

> "Troubleshooting corporate application sign-in failures following a client software update."

**Teaching point:** Query rewriting can improve retrieval when the employee's wording differs from the language used in the documentation. Retry limits help prevent repeated, unproductive search attempts.

**Testing note:** To demonstrate this reliably, use a controlled dataset and verify that the initial query retrieves weak evidence while the rewritten query retrieves the intended document.

### Demo E — Simple greeting

**Question**

> "Hi, can you help me?"

**Expected behavior**

The router classifies the message as a greeting or simple conversational request and returns a direct response.

**Illustrative trace**

```text
Question Received
→ Direct Conversation Route
→ Generate Short Response
→ Return to User
```

**Teaching point:** Not every message requires vector retrieval, web search, or a complex agent workflow. Routing simple requests directly can reduce unnecessary latency and API usage.

### Demo F — No sufficient evidence

**Question**

> "What is the internal approval code for a new hardware procurement exception?"

Assume that the available internal documents and permitted external sources do not contain the answer.

**Expected behavior**

The workflow should not invent an approval code. After its allowed retrieval attempts, it should explain that the answer could not be verified and recommend contacting the appropriate internal team.

**Illustrative trace**

```text
Question Received
→ Private-KB Retrieval
→ Evidence Grade: WEAK
→ Permitted Fallback / Retry
→ Evidence Still Insufficient
→ Safe Fallback Response
```

**Teaching point:** A reliable enterprise assistant must know when it lacks enough evidence to answer. A transparent failure is preferable to an unsupported answer presented with confidence.

### Demo G — Upload a new policy

**Action**

An authorized administrator uploads a new company IT policy in a supported format.

**Expected behavior**

The API validates the upload, extracts the document's text, splits it into chunks, generates embeddings, and stores the vectors in the configured Pinecone namespace. After indexing is verified, a new question can test whether the document is retrievable.

**Illustrative ingestion flow**

```text
Authorized Upload
→ Validate Type and Size
→ Extract Text
→ Split into Chunks
→ Generate Embeddings
→ Index in Pinecone
→ Verify Ingestion
→ Test Retrieval
```

**Teaching point:** An FDE solution must support the customer's data-maintenance process. Adding a policy should not require an engineer to rewrite the application.

**Security note:** The upload endpoint should authenticate the caller, enforce size and file-type restrictions, validate the document contents, and prevent unauthorized users from modifying the knowledge base. A shared API key is only a basic demonstration control, not a complete enterprise identity solution.

---

## 10. API Endpoints

The following endpoint contract is a proposed interface. Confirm the exact routes, authentication mechanism, and response fields against the final FastAPI implementation.

### `GET /api/health`

Checks whether the application is responding.

A production readiness endpoint may additionally check critical dependencies, such as Pinecone connectivity, without exposing secrets or internal infrastructure details.

### `POST /api/chat`

Accepts an employee's question and returns an answer with its source and workflow trace.

**Example request**

```json
{
  "question": "How do I troubleshoot my company VPN connection?"
}
```

**Illustrative response**

```json
{
  "answer": "Follow the approved company VPN troubleshooting steps...",
  "source_used": "private_kb",
  "trace": [
    "Question routed to IT support",
    "Private knowledge retrieval completed",
    "Evidence grade: GOOD",
    "Answer generated from internal documentation"
  ],
  "citations": []
}
```

The response is illustrative. In the completed implementation, populate `citations` with actual document identifiers, titles, page references, or other source metadata when available. Do not claim source-level citations if the retrieval layer does not return them.

### `POST /api/ingest`

Accepts a supported document upload and indexes its extracted content.

If the endpoint uses a shared administrative key, the request may include:

```http
X-Admin-Key: your_configured_admin_secret
```

The endpoint should reject unauthorized requests, unsupported file types, oversized uploads, and invalid documents. It should return an ingestion status and a useful error message without revealing credentials or sensitive implementation details.

---

## 11. Security, Reliability, and Evaluation

Enterprise AI applications need more than successful demo responses. They must handle poor retrieval results, invalid input, provider failures, and unauthorized access in a controlled way.

The project should be evaluated in the following areas:

- **Retrieval quality:** Check whether the expected document appears in the retrieved top-k results.
- **Answer faithfulness:** Verify that the generated answer is supported by retrieved evidence.
- **Routing accuracy:** Test greetings, internal policy questions, current-information requests, and ambiguous questions.
- **Retry behavior:** Confirm that query rewriting improves retrieval in selected test cases and that retries terminate at the configured limit.
- **Source attribution:** Ensure that internal and external information are identified correctly.
- **Failure handling:** Test missing API keys, provider timeouts, empty documents, invalid uploads, and insufficient evidence.
- **Access control:** Confirm that unauthorized callers cannot upload or alter company knowledge.
- **Auditability:** Record appropriate workflow metadata without unnecessarily storing sensitive questions, credentials, or confidential document content.
- **Operational performance:** Measure response latency, retrieval latency, provider errors, token usage, and estimated cost per request.

Because the sample customer and business environment are fictional, any improvement in ticket deflection, response time, or employee satisfaction should be reported only after a controlled evaluation or pilot.

---

## 12. Future Enhancements

Potential next steps include:

1. Add authenticated employee access using the customer's identity provider.
2. Implement document-level permissions and tenant-aware retrieval.
3. Add source citations with document titles and page-level references.
4. Introduce automated retrieval and answer-quality evaluation.
5. Add a reranker to improve the ordering of retrieved chunks.
6. Build a monitoring dashboard for latency, errors, and retrieval outcomes.
7. Add document versioning, deletion, re-indexing, and scheduled refresh.
8. Introduce PostgreSQL or a centralized audit platform for multi-user production deployments.
9. Deploy through a container platform with managed secrets and health monitoring.
10. Establish a feedback mechanism so employees can report incorrect or incomplete answers.

These enhancements would move the project from a local engineering demonstration toward a more complete enterprise application.

---

## 13. Project Summary

Enterprise ITL RAG Assistant demonstrates how an Agentic RAG workflow can be developed into a practical IT support application. It combines private semantic search, evidence grading, conditional public search, query rewriting, grounded generation, an API, and a web interface.

Its main FDE value is the connection between the customer's operational problem and the technical solution. The project demonstrates not only how to retrieve information with an LLM, but also how to decide which source to use, respond when evidence is insufficient, expose system behavior, and support changes to the customer's knowledge base.

The intended outcome is a more accessible and traceable IT support experience, with operational impact validated through measurable testing rather than assumed in advance.
