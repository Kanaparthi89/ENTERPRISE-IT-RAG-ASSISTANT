(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) =>
    Array.from(root.querySelectorAll(selector));

  const el = {
    chat: $("#chat"),
    chatForm: $("#chatForm"),
    question: $("#question"),
    sendBtn: $("#sendBtn"),
    characterCount: $("#characterCount"),
    clearChat: $("#clearChat"),
    trace: $("#trace"),
    traceCount: $("#traceCount"),
    runStatus: $("#runStatus"),
    runProgressBar: $("#runProgressBar"),
    runStepCount: $("#runStepCount"),
    sourceUsed: $("#sourceUsed"),
    sourceDescription: $("#sourceDescription"),
    uploadModal: $("#uploadModal"),
    openUpload: $("#openUpload"),
    closeUpload: $("#closeUpload"),
    adminKey: $("#adminKey"),
    fileInput: $("#fileInput"),
    fileName: $("#fileName"),
    uploadBtn: $("#uploadBtn"),
    uploadStatus: $("#uploadStatus"),
  };

  const MAX_FILE_SIZE = 10 * 1024 * 1024;
  const ALLOWED_EXTENSIONS = new Set([".pdf", ".txt", ".md", ".docx"]);

  let isChatting = false;
  let isUploading = false;
  let lastFocusedElement = null;

  function getText(value, fallback = "") {
    return typeof value === "string" && value.trim()
      ? value.trim()
      : fallback;
  }

  function setButtonBusy(button, busy, busyLabel, idleLabel) {
    if (!button) return;

    button.disabled = busy;
    button.setAttribute("aria-busy", String(busy));

    const label = $("span", button);

    if (label) {
      if (!button.dataset.idleLabel && idleLabel) {
        button.dataset.idleLabel = idleLabel;
      }

      if (button.dataset.idleLabel) {
        label.textContent = busy
          ? busyLabel
          : button.dataset.idleLabel;
      }
    }
  }

  function scrollChatToBottom() {
    if (el.chat) {
      el.chat.scrollTop = el.chat.scrollHeight;
    }
  }

  function createAvatar(label) {
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = label;
    avatar.setAttribute("aria-hidden", "true");
    return avatar;
  }

  function clearWelcomeState() {
    const welcome = $(".welcome-state", el.chat);
    if (welcome) welcome.remove();
  }

  function addMessage(role, content, options = {}) {
    const message = document.createElement("article");
    message.className = `message ${role}`;

    const avatar = createAvatar(role === "user" ? "You" : "AI");
    const bubble = document.createElement("div");
    bubble.className = "bubble";

    if (options.loading) {
      const heading = document.createElement("strong");
      heading.textContent = "Working on your question";

      const paragraph = document.createElement("p");
      paragraph.textContent =
        "Searching available evidence and preparing a response…";

      const indicator = document.createElement("div");
      indicator.className = "typing-indicator";
      indicator.setAttribute("aria-label", "Assistant is thinking");

      for (let i = 0; i < 3; i++) {
        const dot = document.createElement("span");
        indicator.appendChild(dot);
      }

      bubble.append(heading, paragraph, indicator);
    } else {
      const paragraph = document.createElement("p");
      paragraph.textContent = content;
      bubble.appendChild(paragraph);
    }

    message.append(avatar, bubble);
    el.chat.appendChild(message);
    scrollChatToBottom();

    return { message, bubble };
  }

  function safeHttpUrl(value) {
    if (typeof value !== "string" || !value.trim()) {
      return null;
    }

    try {
      const url = new URL(value, window.location.origin);

      if (url.protocol !== "https:" && url.protocol !== "http:") {
        return null;
      }

      return url.href;
    } catch {
      return null;
    }
  }

  function addAssistantAnswer(answer, citations) {
    const message = document.createElement("article");
    message.className = "message assistant";
    message.appendChild(createAvatar("AI"));

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    const answerText = document.createElement("p");
    answerText.className = "answer-text";

    const answerContent = getText(
      answer,
      "The agent did not return an answer."
    );

    const normalizedAnswer = answerContent.replace(
      /\\([*_`#[\]()])/g,
      "$1"
    );

    if (window.marked && window.DOMPurify) {
      answerText.innerHTML = window.DOMPurify.sanitize(
        window.marked.parse(normalizedAnswer)
      );
    } else {
      answerText.textContent = normalizedAnswer;
    }

    bubble.appendChild(answerText);
    const validCitations = Array.isArray(citations) ? citations : [];

    if (validCitations.length) {
      const section = document.createElement("div");
      section.className = "citations";

      const heading = document.createElement("strong");
      heading.textContent = "Sources";

      const list = document.createElement("ul");

      validCitations.forEach((citation, index) => {
        if (!citation || typeof citation !== "object") {
          return;
        }

        const item = document.createElement("li");

        const label = getText(
          citation.title ||
            citation.source ||
            citation.name ||
            citation.file,
          `Source ${index + 1}`
        );

        const url = safeHttpUrl(citation.url || citation.link);

        if (url) {
          const link = document.createElement("a");
          link.href = url;
          link.target = "_blank";
          link.rel = "noopener noreferrer";
          link.textContent = label;
          item.appendChild(link);
        } else {
          item.textContent = label;
        }

        const type = getText(citation.type || citation.source_type);

        if (type) {
          const metadata = document.createElement("span");
          metadata.className = "citation-type";
          metadata.textContent = ` · ${type.replaceAll("_", " ")}`;
          item.appendChild(metadata);
        }

        list.appendChild(item);
      });

      if (list.children.length > 0) {
        section.append(heading, list);
        bubble.appendChild(section);
      }
    }

    message.appendChild(bubble);
    el.chat.appendChild(message);
    scrollChatToBottom();

    return message;
  }

  function updateCharacterCount() {
    if (el.characterCount && el.question) {
      el.characterCount.textContent =
        `${el.question.value.length} / 3000`;
    }
  }

  function setRunState(status, progress, stepCount) {
    if (el.runStatus) {
      el.runStatus.textContent = status;
    }

    if (el.runProgressBar) {
      el.runProgressBar.style.width =
        `${Math.max(0, Math.min(100, progress))}%`;
    }

    if (el.runStepCount) {
      el.runStepCount.textContent =
        `${stepCount} ${stepCount === 1 ? "step" : "steps"}`;
    }
  }

  function resetTrace() {
    if (el.trace) {
      const empty = document.createElement("div");
      empty.className = "trace-empty";

      const heading = document.createElement("strong");
      heading.textContent = "Ready when you are";

      const description = document.createElement("span");
      description.textContent =
        "Your agent's execution steps will appear here.";

      empty.append(heading, description);
      el.trace.replaceChildren(empty);
    }

    if (el.traceCount) {
      el.traceCount.textContent = "0";
    }

    setRunState("Waiting for a question", 0, 0);
  }

  function normalizeTraceStep(step) {
    if (typeof step === "string") {
      return step;
    }

    if (step && typeof step === "object") {
      return getText(
        step.message ||
          step.name ||
          step.step ||
          step.node ||
          step.action,
        "Agent step"
      );
    }

    return String(step);
  }

  function renderTrace(trace, status = "Completed") {
    const steps = Array.isArray(trace)
      ? trace.map(normalizeTraceStep)
      : [];

    if (el.trace) {
      el.trace.replaceChildren();

      if (steps.length === 0) {
        const empty = document.createElement("div");
        empty.className = "trace-empty";

        const heading = document.createElement("strong");
        heading.textContent = "No trace details returned";

        const description = document.createElement("span");
        description.textContent =
          "The response was received, but no execution steps were provided.";

        empty.append(heading, description);
        el.trace.appendChild(empty);
      } else {
        steps.forEach((step, index) => {
          const item = document.createElement("div");
          item.className = "trace-item";

          const marker = document.createElement("span");
          marker.className = "trace-marker";
          marker.textContent = String(index + 1).padStart(2, "0");

          const content = document.createElement("div");
          content.className = "trace-step";

          const heading = document.createElement("strong");
          heading.textContent = step;

          const description = document.createElement("span");
          description.textContent =
            index === steps.length - 1
              ? "Final recorded step"
              : "Completed";

          content.append(heading, description);
          item.append(marker, content);
          el.trace.appendChild(item);
        });
      }
    }

    if (el.traceCount) {
      el.traceCount.textContent = String(steps.length);
    }

    setRunState(status, 100, steps.length);
  }

  function updateSource(source, description) {
    if (el.sourceUsed) {
      el.sourceUsed.textContent = getText(source, "Unknown");
    }

    if (el.sourceDescription) {
      el.sourceDescription.textContent = getText(
        description,
        "Source details were not provided by the backend."
      );
    }
  }

  function describeSource(source, citations) {
    const normalized = getText(source, "unknown").toLowerCase();
    const count = Array.isArray(citations) ? citations.length : 0;
    const sourceCount =
      `${count} source${count === 1 ? "" : "s"}`;

    if (
      normalized.includes("private") ||
      normalized.includes("kb") ||
      normalized.includes("knowledge")
    ) {
      return count
        ? `Answered using internal knowledge-base evidence (${sourceCount}).`
        : "The agent selected the private knowledge base.";
    }

    if (
      normalized.includes("web") ||
      normalized.includes("tavily")
    ) {
      return count
        ? `Answered using external web evidence (${sourceCount}). Verify vendor guidance when needed.`
        : "The agent selected external web search.";
    }

    if (normalized.includes("direct")) {
      return "The agent handled this as a general question without retrieving company documents.";
    }

    if (normalized.includes("insufficient")) {
      return "The agent could not find enough reliable evidence to answer confidently.";
    }

    return "The final source was reported by the agent workflow.";
  }

  async function submitQuestion(forcedQuestion) {
    if (isChatting) return;

    const question = getText(
      typeof forcedQuestion === "string"
        ? forcedQuestion
        : el.question.value
    );

    if (question.length < 2) {
      el.question.focus();
      return;
    }

    if (question.length > 3000) {
      addMessage(
        "assistant",
        "Please keep your question to 3,000 characters or fewer."
      );
      return;
    }

    isChatting = true;

    clearWelcomeState();
    addMessage("user", question);

    const loadingMessage = addMessage("assistant", "", {
      loading: true,
    });

    el.question.value = "";
    updateCharacterCount();

    setButtonBusy(
      el.sendBtn,
      true,
      "Thinking…",
      "Ask agent"
    );

    setRunState("Processing question", 15, 0);

    updateSource(
      "Processing",
      "The agent is routing your question and evaluating available evidence."
    );

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({ question }),
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          getText(
            data.detail || data.message,
            `Request failed (HTTP ${response.status}).`
          )
        );
      }

      loadingMessage.message.remove();

      const citations = Array.isArray(data.citations)
        ? data.citations
        : [];

      const assistantMessage = addAssistantAnswer(
        data.answer,
        citations
      );

      const source = getText(data.source_used, "unknown");

      updateSource(source, describeSource(source, citations));
      renderTrace(data.trace, "Completed");

      if (
        data.rewritten_query &&
        data.rewritten_query !== question
      ) {
        const note = document.createElement("p");
        note.className = "rewritten-query-note";
        note.textContent =
          `Refined search query: ${data.rewritten_query}`;

        const bubble = $(".bubble", assistantMessage);

        if (bubble) {
          bubble.appendChild(note);
        }
      }
    } catch (error) {
      loadingMessage.message.remove();

      addMessage(
        "assistant",
        `I couldn't complete that request. ${
          error instanceof Error
            ? error.message
            : "Please try again."
        }`
      );

      setRunState("Request failed", 100, 0);

      updateSource(
        "Error",
        "The request did not complete successfully. Check the application logs for details."
      );
    } finally {
      isChatting = false;

      setButtonBusy(
        el.sendBtn,
        false,
        "Thinking…",
        "Ask agent"
      );

      el.question.focus();
    }
  }

  function resetConversation() {
    if (isChatting) return;

    el.chat.replaceChildren();

    const welcome = document.createElement("div");
    welcome.className = "welcome-state";

    const kicker = document.createElement("span");
    kicker.className = "welcome-kicker";
    kicker.textContent = "YOUR AI-POWERED IT DESK";

    const heading = document.createElement("h2");
    heading.textContent = "What can I help you solve?";

    const description = document.createElement("p");
    description.textContent =
      "Ask about internal IT policies, troubleshooting steps, or vendor guidance. Your agent will find the best available evidence.";

    const suggestions = document.createElement("div");
    suggestions.className = "suggestion-grid";

    const prompts = [
      [
        "VPN & connectivity",
        "How do I connect to the company VPN?",
      ],
      [
        "Company policies",
        "What is our password reset policy?",
      ],
      [
        "External vendor guidance",
        "What is the latest Microsoft Teams outage guidance?",
      ],
      [
        "Troubleshooting",
        "My laptop cannot access internal resources.",
      ],
    ];

    prompts.forEach(([title, prompt]) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "suggestion-card example";

      const content = document.createElement("span");
      content.className = "suggestion-content";

      const strong = document.createElement("strong");
      strong.textContent = title;

      const detail = document.createElement("span");
      detail.textContent = prompt;

      content.append(strong, detail);
      button.appendChild(content);

      button.addEventListener("click", () => {
        submitQuestion(prompt);
      });

      suggestions.appendChild(button);
    });

    welcome.append(kicker, heading, description, suggestions);
    el.chat.appendChild(welcome);

    resetTrace();

    updateSource(
      "Awaiting response",
      "Evidence details will be shown after your question is processed."
    );

    el.question.value = "";
    updateCharacterCount();
    el.question.focus();
  }

  function openUploadModal() {
    if (!el.uploadModal) return;

    lastFocusedElement = document.activeElement;

    el.uploadModal.classList.remove("hidden");
    el.uploadModal.setAttribute("aria-hidden", "false");

    el.adminKey.focus();
  }

  function closeUploadModal() {
    if (!el.uploadModal || isUploading) return;

    el.uploadModal.classList.add("hidden");
    el.uploadModal.setAttribute("aria-hidden", "true");

    if (lastFocusedElement?.focus) {
      lastFocusedElement.focus();
    }
  }

  function setUploadStatus(message, kind = "") {
    if (!el.uploadStatus) return;

    el.uploadStatus.textContent = message;
    el.uploadStatus.className = "upload-status";

    if (kind) {
      el.uploadStatus.classList.add(kind);
    }
  }

  function getExtension(filename) {
    const dot = filename.lastIndexOf(".");
    return dot >= 0 ? filename.slice(dot).toLowerCase() : "";
  }

  async function uploadDocument() {
    if (isUploading) return;

    const adminKey = el.adminKey.value.trim();
    const file = el.fileInput.files?.[0];

    if (!adminKey) {
      setUploadStatus(
        "Enter the admin API key to continue.",
        "error"
      );
      el.adminKey.focus();
      return;
    }

    if (!file) {
      setUploadStatus(
        "Choose a document before indexing.",
        "error"
      );
      el.fileInput.focus();
      return;
    }

    if (!ALLOWED_EXTENSIONS.has(getExtension(file.name))) {
      setUploadStatus(
        "Unsupported file type. Choose PDF, TXT, Markdown, or DOCX.",
        "error"
      );
      return;
    }

    if (file.size === 0) {
      setUploadStatus("The selected file is empty.", "error");
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      setUploadStatus(
        "The file exceeds the 10 MB limit.",
        "error"
      );
      return;
    }

    isUploading = true;

    setButtonBusy(
      el.uploadBtn,
      true,
      "Indexing…",
      "Index document"
    );

    setUploadStatus(
      "Uploading and indexing document. This may take a moment."
    );

    try {
      const formData = new FormData();
      formData.append("file", file, file.name);

      const response = await fetch("/api/ingest", {
        method: "POST",
        headers: {
          "X-Admin-Key": adminKey,
          Accept: "application/json",
        },
        body: formData,
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          getText(
            data.detail || data.message,
            `Upload failed (HTTP ${response.status}).`
          )
        );
      }

      const chunks = Number.isFinite(Number(data.chunks))
        ? Number(data.chunks)
        : 0;

      const filename = getText(data.file, file.name);

      setUploadStatus(
        `Successfully indexed ${filename}. ${chunks} chunk${
          chunks === 1 ? "" : "s"
        } added to the knowledge base.`,
        "success"
      );

      el.fileInput.value = "";
      el.fileName.textContent = "Choose a file to upload";
      el.adminKey.value = "";
    } catch (error) {
      setUploadStatus(
        error instanceof Error
          ? error.message
          : "The document could not be indexed. Please try again.",
        "error"
      );
    } finally {
      isUploading = false;

      setButtonBusy(
        el.uploadBtn,
        false,
        "Indexing…",
        "Index document"
      );
    }
  }

  function bindEvents() {
    el.chatForm?.addEventListener("submit", (event) => {
      event.preventDefault();
      submitQuestion();
    });

    el.question?.addEventListener("input", updateCharacterCount);

    el.question?.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();

        if (el.chatForm.requestSubmit) {
          el.chatForm.requestSubmit();
        } else {
          submitQuestion();
        }
      }
    });

    $$(".example").forEach((button) => {
      button.addEventListener("click", () => {
        const suggestion = $(".suggestion-content span", button);
        const prompt = getText(
          suggestion?.textContent || button.textContent
        );

        if (prompt) {
          submitQuestion(prompt);
        }
      });
    });

    el.clearChat?.addEventListener("click", resetConversation);
    el.openUpload?.addEventListener("click", openUploadModal);
    el.closeUpload?.addEventListener("click", closeUploadModal);
    el.uploadBtn?.addEventListener("click", uploadDocument);

    el.fileInput?.addEventListener("change", () => {
      const file = el.fileInput.files?.[0];

      if (el.fileName) {
        el.fileName.textContent = file
          ? file.name
          : "Choose a file to upload";
      }

      if (file && file.size > MAX_FILE_SIZE) {
        setUploadStatus(
          "The file exceeds the 10 MB limit.",
          "error"
        );
      } else if (file) {
        setUploadStatus("");
      }
    });

    el.uploadModal?.addEventListener("click", (event) => {
      if (event.target === el.uploadModal) {
        closeUploadModal();
      }
    });

    document.addEventListener("keydown", (event) => {
      const modalOpen =
        el.uploadModal &&
        !el.uploadModal.classList.contains("hidden");

      if (event.key === "Escape" && modalOpen) {
        closeUploadModal();
      }
    });
  }

  bindEvents();
  updateCharacterCount();
  resetTrace();
})();
