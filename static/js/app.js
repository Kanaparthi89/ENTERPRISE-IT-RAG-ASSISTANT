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
    voiceToggle: $("#voiceToggle"),
    voiceStop: $("#voiceStop"),
    voiceStatus: $("#voiceStatus"),
    dictateBtn: $("#dictateBtn"),
    dictationNote: $("#dictationNote"),
    dictationStatus: $("#dictationStatus"),
  };

  const MAX_FILE_SIZE = 10 * 1024 * 1024;
  const ALLOWED_EXTENSIONS = new Set([".pdf", ".txt", ".md", ".docx"]);

  let isChatting = false;
  let isUploading = false;
  let lastFocusedElement = null;
  // Voice V1 controller; null unless the browser can speak and the server
  // reports {"tts": true}. Every hook below is a no-op while it is null.
  let voice = null;
  // Voice V3b dictation controller; null unless the browser can record
  // securely and the server reports {"stt": true}. Hooks are no-ops while null.
  let dictation = null;

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

  // The final answer view, shared by /api/chat and the stream's "done" event.
  function renderFinalAnswer(data, question) {
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
  }

  function renderFailure(message) {
    addMessage(
      "assistant",
      `I couldn't complete that request. ${message}`
    );

    setRunState("Request failed", 100, 0);

    updateSource(
      "Error",
      "The request did not complete successfully. Check the application logs for details."
    );
  }

  async function submitQuestion(forcedQuestion) {
    if (isChatting) return;

    hideSessionAlert();

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

    if (!chatController) {
      renderFailure("Please refresh the page and try again.");
      return;
    }

    // A new question silences the previous answer (speech only; never the chat).
    voice?.cancel();
    // ...and ends a dictation in progress; its transcript is discarded.
    dictation?.cancel({ announce: true });

    isChatting = true;

    try {
      // Streams when the backend allows it, otherwise uses /api/chat; one
      // answer per question either way (see static/js/chat-stream.js).
      await chatController.submit(question);
    } finally {
      isChatting = false;
    }
  }

  // -------------------------------------------------------------------
  // Streaming view: the DOM side of BluePeakChat.createChatController
  // -------------------------------------------------------------------

  const NEAR_BOTTOM_PX = 80;

  const scheduleFrame =
    typeof window.requestAnimationFrame === "function"
      ? (callback) => window.requestAnimationFrame(callback)
      : (callback) => window.setTimeout(callback, 16);

  const cancelFrame =
    typeof window.cancelAnimationFrame === "function"
      ? (id) => window.cancelAnimationFrame(id)
      : (id) => window.clearTimeout(id);

  function isChatNearBottom() {
    if (!el.chat) return true;

    return (
      el.chat.scrollHeight - el.chat.scrollTop - el.chat.clientHeight <=
      NEAR_BOTTOM_PX
    );
  }

  function appendTraceStep(step, count) {
    if (!el.trace) return;

    const item = document.createElement("div");
    item.className = "trace-item";

    const marker = document.createElement("span");
    marker.className = "trace-marker";
    marker.textContent = String(count).padStart(2, "0");

    const content = document.createElement("div");
    content.className = "trace-step";

    const heading = document.createElement("strong");
    heading.textContent = normalizeTraceStep(step);

    const description = document.createElement("span");
    description.textContent = "Completed";

    content.append(heading, description);
    item.append(marker, content);
    el.trace.appendChild(item);
  }

  // -------------------------------------------------------------------
  // Conversation session (A2-7): status line and "not sent" alerts.
  // Created only when sessions are in use; stateless pages never show them.
  // -------------------------------------------------------------------

  // Status line (polite): save state of the latest answer.
  const SESSION_TEXT = {
    on: "Conversation context on: follow-up questions can use earlier answers in this chat.",
    notSaved:
      "This answer wasn't saved to the conversation. Follow-up questions won't include it.",
    unknown: "Couldn't confirm this answer was saved to the conversation.",
    clearFailed:
      "Earlier conversation memory couldn't be cleared right now. It will be cleared automatically after a period of inactivity.",
  };

  // Alerts: the question was not sent. ``action`` is the recovery offered:
  // "retry" only where trying again can help; "reload" / "new_chat" / none
  // otherwise. Nothing ever falls back to a stateless question.
  const SESSION_ALERTS = {
    session_expired: {
      text: "This conversation has expired. Start a new chat to continue.",
      action: "new_chat",
    },
    session_busy: {
      text: "Another question in this conversation is still being answered. Your question wasn't sent. Wait a moment, then select Retry.",
      action: "retry",
    },
    session_unavailable: {
      text: "Conversation memory is temporarily unavailable. Your question wasn't sent. Select Retry to try again.",
      action: "retry",
    },
    network: {
      text: "Couldn't reach the server to start a conversation. Your question wasn't sent. Select Retry to try again.",
      action: "retry",
    },
    timeout: {
      text: "The server took too long to start a conversation. Your question wasn't sent. Select Retry to try again.",
      action: "retry",
    },
    origin_not_allowed: {
      text: "Conversations can't be started from this address. Your question wasn't sent. Open BluePeak AI Desk from its usual address to continue.",
      action: null,
    },
    unsupported_media_type: {
      text: "This page is out of date, so a conversation couldn't be started. Your question wasn't sent. Reload the page to continue.",
      action: "reload",
    },
  };

  // Any other 4xx won't succeed on a retry from this page.
  const SESSION_ALERT_CLIENT_ERROR = {
    text: "A conversation couldn't be started from this page. Your question wasn't sent. Reload the page to continue.",
    action: "reload",
  };

  const SESSION_ALERT_DEFAULT = {
    text: "Couldn't start a conversation. Your question wasn't sent. Select Retry to try again.",
    action: "retry",
  };

  function sessionAlertFor(code) {
    if (Object.hasOwn(SESSION_ALERTS, code)) return SESSION_ALERTS[code];
    if (/^http_4\d\d$/.test(code)) return SESSION_ALERT_CLIENT_ERROR;
    return SESSION_ALERT_DEFAULT;
  }

  let sessionNotice = null;
  let sessionAlert = null;

  function composerArea() {
    return el.chatForm?.parentElement || null;
  }

  function showSessionNotice(text) {
    const area = composerArea();
    if (!area) return;

    if (!sessionNotice) {
      sessionNotice = document.createElement("p");
      sessionNotice.className = "session-notice";
      sessionNotice.setAttribute("role", "status");
      sessionNotice.setAttribute("aria-live", "polite");
      el.chatForm.after(sessionNotice);
    }

    sessionNotice.textContent = text;
    sessionNotice.hidden = false;
  }

  function hideSessionNotice() {
    if (!sessionNotice) return;
    sessionNotice.textContent = "";
    sessionNotice.hidden = true;
  }

  function hideSessionAlert() {
    sessionAlert?.remove();
    sessionAlert = null;
  }

  function showSessionAlert(code) {
    hideSessionAlert();

    const area = composerArea();
    if (!area) return;

    const alert = document.createElement("div");
    alert.className = "session-alert";
    alert.setAttribute("role", "alert");

    const { text: message, action } = sessionAlertFor(code);

    const text = document.createElement("p");
    text.textContent = message;
    alert.appendChild(text);

    const actions = {
      retry: ["Retry", () => submitQuestion()],
      // Keep the unsent question for the new conversation.
      new_chat: ["Start new chat", () => resetConversation({ keepQuestion: true })],
      reload: ["Reload page", () => window.location.reload()],
    };

    if (action && actions[action]) {
      const [label, onClick] = actions[action];
      const button = document.createElement("button");
      button.type = "button";
      button.className = "session-alert-action";
      button.textContent = label;
      button.addEventListener("click", onClick);
      alert.appendChild(button);
    }

    area.insertBefore(alert, el.chatForm);
    sessionAlert = alert;
  }

  const chatView = {
    blocked({ reason, question }) {
      // Nothing was sent and nothing was added.
      if (reason === "aborted") return; // the page is being left

      // Keep the question in the box so Retry / Start new chat can use it
      // (a suggestion card's question isn't there yet). Never overwrite
      // text the user has typed.
      if (question && !el.question.value.trim()) {
        el.question.value = question;
        updateCharacterCount();
      }

      showSessionAlert(reason);
      // Expired is announced once, by the alert; drop any stale status line.
      if (reason === "session_expired") hideSessionNotice();

      // A Retry that is blocked again removed the focused button with its
      // alert: move focus to the question rather than leave it on the page.
      const active = document.activeElement;
      if (!active || active === document.body || !active.isConnected) {
        el.question.focus();
      }
    },

    sessionStarted() {
      showSessionNotice(SESSION_TEXT.on);
    },

    begin(question) {
      clearWelcomeState();
      const userMessage = addMessage("user", question);

      const loadingMessage = addMessage("assistant", "", {
        loading: true,
      });
      // Announce the answer once it is complete, not every streamed chunk.
      loadingMessage.message.setAttribute("aria-busy", "true");
      // Voice: this question's turn; only its completed answer can be spoken.
      const voiceTurn = voice ? voice.questionStarted() : 0;

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

      let answerText = null;
      let pendingText = "";
      let frameId = null;
      let traceSteps = 0;

      const flush = () => {
        frameId = null;
        if (!answerText || !pendingText) return;

        const stickToBottom = isChatNearBottom();
        answerText.appendData(pendingText);
        pendingText = "";

        if (stickToBottom) scrollChatToBottom();
      };

      const finish = () => {
        if (frameId !== null) cancelFrame(frameId);
        frameId = null;
        pendingText = "";
        loadingMessage.message.remove();
        el.trace?.removeAttribute("aria-busy");
      };

      return {
        trace(step) {
          if (!el.trace) return;

          if (traceSteps === 0) {
            el.trace.replaceChildren();
            el.trace.setAttribute("aria-busy", "true");
          }

          traceSteps += 1;
          appendTraceStep(step, traceSteps);

          if (el.traceCount) {
            el.traceCount.textContent = String(traceSteps);
          }

          setRunState(
            "Processing question",
            Math.min(90, 15 + traceSteps * 10),
            traceSteps
          );
        },

        delta(text) {
          if (!answerText) {
            const paragraph = document.createElement("p");
            paragraph.className = "answer-text streaming";
            // Plain text only while streaming; the sanitized Markdown
            // rendering replaces it when the answer is complete.
            answerText = document.createTextNode("");
            paragraph.appendChild(answerText);
            loadingMessage.bubble.replaceChildren(paragraph);

            setButtonBusy(
              el.sendBtn,
              true,
              "Responding…",
              "Ask agent"
            );
          }

          pendingText += text;
          if (frameId === null) frameId = scheduleFrame(flush);
        },

        complete(data) {
          finish();
          renderFinalAnswer(data, question);
          // The one completion point for streamed and /api/chat answers;
          // deltas, failures and rejections are never spoken.
          voice?.answerCompleted(voiceTurn, data.answer);
        },

        fail(message) {
          finish();
          renderFailure(message);

          if (traceSteps > 0) {
            // Keep the steps that did run consistent with the panel.
            setRunState("Request failed", 100, traceSteps);
          }
        },

        // A session error before anything ran: the question wasn't sent, so
        // undo begin() and give the question back.
        reject({ code }) {
          finish();
          userMessage.message.remove();

          el.question.value = question;
          updateCharacterCount();

          setRunState("Question not sent", 0, 0);
          updateSource(
            "Awaiting response",
            "Evidence details will be shown after your question is processed."
          );

          showSessionAlert(code);
          // Expired is announced once, by the alert; drop any stale status line.
          if (code === "session_expired") hideSessionNotice();
        },

        sessionStatus({ turnSaved }) {
          if (turnSaved === true) showSessionNotice(SESSION_TEXT.on);
          else if (turnSaved === false) showSessionNotice(SESSION_TEXT.notSaved);
          else showSessionNotice(SESSION_TEXT.unknown);
        },
      };
    },

    end() {
      setButtonBusy(
        el.sendBtn,
        false,
        "Thinking…",
        "Ask agent"
      );

      el.question.focus();
    },
  };

  let pageAbort = new AbortController();

  window.addEventListener("pagehide", () => {
    pageAbort.abort();
    voice?.cancel(); // speech never outlives the page
    dictation?.cancel(); // nor does the microphone, or an upload in progress
  });
  window.addEventListener("pageshow", (event) => {
    if (event.persisted) pageAbort = new AbortController();
  });

  const streamingSupported =
    typeof window.fetch === "function" &&
    typeof window.ReadableStream === "function" &&
    typeof window.TextDecoder === "function" &&
    typeof window.AbortController === "function" &&
    typeof window.Response === "function" &&
    "body" in window.Response.prototype;

  // The session token lives only inside this object (memory, never storage).
  const sessionClient =
    window.BluePeakChat && typeof window.fetch === "function"
      ? new window.BluePeakChat.SessionClient({
          fetchImpl: (url, init) => window.fetch(url, init),
          // Abandon session creation when the page is being left.
          signal: () => pageAbort.signal,
        })
      : null;

  const chatController = window.BluePeakChat
    ? window.BluePeakChat.createChatController({
        view: chatView,
        fetchImpl: (url, init) => window.fetch(url, init),
        availability: new window.BluePeakChat.StreamAvailability({
          supported: streamingSupported,
        }),
        signal: () => pageAbort.signal,
        sessions: sessionClient,
      })
    : null;

  // -------------------------------------------------------------------
  // Voice V1: read completed answers aloud
  //
  // Browser speech synthesis with on-device voices only (static/js/voice.js);
  // no audio is recorded or uploaded. Offered only when the browser can speak
  // and GET /api/voice/status answers exactly {"tts": true} (VOICE_ENABLED,
  // off by default); the user then has to turn it on. Speech never affects
  // the chat: stopping or cancelling it never aborts a question.
  // -------------------------------------------------------------------

  const VOICE_STATUS_URL = "/api/voice/status";
  const VOICE_PROBE_TIMEOUT_MS = 5_000;
  const VOICE_CONFIRMATION = "Voice responses on.";
  const VOICE_TEXT = {
    noVoice:
      "Read aloud isn't available: this device has no on-device voice for this page's language.",
    unsupported: "Read aloud isn't supported in this browser.",
    blocked: "Your browser blocked speech. Select Read answers aloud to try again.",
    failed: "Couldn't read the answer aloud.",
  };

  function voiceIsPossible() {
    return Boolean(
      window.BluePeakVoice &&
        window.speechSynthesis &&
        typeof window.SpeechSynthesisUtterance === "function" &&
        typeof window.fetch === "function" &&
        el.voiceToggle &&
        el.voiceStop &&
        el.voiceStatus
    );
  }

  /** Only exactly {"tts": true} offers voice. */
  function isVoiceStatus(body) {
    return (
      body !== null &&
      typeof body === "object" &&
      !Array.isArray(body) &&
      Object.keys(body).length === 1 &&
      body.tts === true
    );
  }

  /** One bounded GET /api/voice/status; any failure means "not offered". */
  function probeVoiceStatus() {
    return probeFeatureStatus(VOICE_STATUS_URL, isVoiceStatus);
  }

  /**
   * One bounded, abortable GET of a feature status URL (5 seconds; aborted
   * when the page is left). Offered only when ``accept`` approves the body;
   * any failure means "not offered".
   */
  async function probeFeatureStatus(url, accept) {
    const controller = new AbortController();
    const page = pageAbort.signal;
    const abort = () => controller.abort();
    const timer = window.setTimeout(abort, VOICE_PROBE_TIMEOUT_MS);
    if (page.aborted) abort();
    else page.addEventListener("abort", abort);

    try {
      const response = await window.fetch(url, {
        method: "GET",
        headers: { Accept: "application/json" },
        cache: "no-store",
        credentials: "same-origin",
        signal: controller.signal,
      });
      if (response.status !== 200) return false;
      return accept(await response.json());
    } catch {
      return false; // network error, timeout, page left, or a body that is not JSON
    } finally {
      window.clearTimeout(timer);
      page.removeEventListener("abort", abort);
    }
  }

  function createVoiceController() {
    let enabled = false; // turned on by the user, with an on-device voice
    let attempts = 0; // each turn-on check; a newer one (or turning off) ends it
    let enabling = 0; // the attempt being checked, or 0
    let turn = 0; // the latest question
    let spokenTurn = 0; // the last question whose answer went to speech

    const speaker = window.BluePeakVoice.createSpeaker({
      synth: window.speechSynthesis,
      Utterance: window.SpeechSynthesisUtterance,
      lang: document.documentElement?.getAttribute?.("lang") || "en-US",
      setTimeout: (fn, ms) => window.setTimeout(fn, ms),
      clearTimeout: (id) => window.clearTimeout(id),
      onStateChange: showSpeakingState,
    });

    function setStatus(text) {
      el.voiceStatus.textContent = text;
    }

    function setEnabled(on) {
      enabled = on;
      el.voiceToggle.setAttribute("aria-pressed", String(on));
    }

    // "Stop speaking" exists only while something is being spoken. If it
    // had focus when speech ends, focus moves to the toggle, not the page.
    function showSpeakingState(state) {
      const speaking = state === "speaking";
      if (!speaking && document.activeElement === el.voiceStop) {
        el.voiceToggle.focus();
      }
      el.voiceStop.hidden = !speaking;
    }

    function report(result) {
      if (result.spoken || result.reason === "cancelled" || result.reason === "empty") return;

      if (result.reason === "error" && result.error === "not-allowed") {
        turnOff();
        setStatus(VOICE_TEXT.blocked);
      } else if (result.reason === "no_local_voice") {
        turnOff();
        setStatus(VOICE_TEXT.noVoice);
      } else if (result.reason === "unsupported") {
        turnOff();
        setStatus(VOICE_TEXT.unsupported);
      } else {
        setStatus(VOICE_TEXT.failed); // still on: the next answer may work
      }
    }

    function turnOff() {
      attempts += 1; // ends a turn-on check still running
      enabling = 0;
      el.voiceToggle.removeAttribute("aria-busy");
      setEnabled(false);
      speaker.cancel();
    }

    function turnOn() {
      // Synchronously, inside the click: Safari and iOS only allow speech a
      // user started. Best-effort; a block is reported by speak() below.
      speaker.prime();

      attempts += 1;
      const attempt = attempts;
      enabling = attempt;
      el.voiceToggle.setAttribute("aria-busy", "true");
      setStatus("");

      speaker.hasLocalVoice().then((ready) => {
        if (enabling !== attempt) return; // turned off (or clicked again) meanwhile
        enabling = 0;
        el.voiceToggle.removeAttribute("aria-busy");

        if (!ready) {
          setStatus(VOICE_TEXT.noVoice);
          return;
        }
        setEnabled(true);
        // Never speak into a recording (Voice V3b).
        if (!dictation?.isActive()) speaker.speak(VOICE_CONFIRMATION).then(report);
      });
    }

    function toggle() {
      if (enabled || enabling) turnOff();
      else turnOn();
    }

    return {
      start() {
        el.voiceToggle.hidden = false;
        el.voiceToggle.addEventListener("click", toggle);
        // Stops speech only: a question in progress continues.
        el.voiceStop.addEventListener("click", () => speaker.cancel());
      },

      cancel() {
        speaker.cancel();
      },

      isSpeaking() {
        return speaker.state === "speaking" || speaker.state === "loading";
      },

      questionStarted() {
        turn += 1;
        return turn;
      },

      /** Speak a completed answer: only for the latest question, and only once. */
      answerCompleted(questionTurn, answer) {
        if (!enabled || questionTurn !== turn || spokenTurn === questionTurn) return;
        if (dictation?.isActive()) return; // never speak into a recording (Voice V3b)
        spokenTurn = questionTurn;
        setStatus(""); // an earlier "couldn't read" message no longer applies
        speaker.speak(answer).then(report);
      },
    };
  }

  function startVoice() {
    if (!voiceIsPossible()) return; // nothing is requested or shown

    probeVoiceStatus().then((offered) => {
      if (!offered) return;
      voice = createVoiceController();
      voice.start();
    });
  }

  // -------------------------------------------------------------------
  // Voice V3b: microphone dictation
  //
  // Records one short clip with static/js/recorder.js (which alone owns the
  // microphone, the audio and the upload) and adds the transcript to the
  // question box for the user to review: never sent automatically. Offered
  // only in a secure context that can record, and only when
  // GET /api/voice/transcription/status answers exactly {"stt": true}
  // (VOICE_TRANSCRIPTION_ENABLED, off by default). The button and the
  // privacy note are shown together, before any recording can start.
  // -------------------------------------------------------------------

  const DICTATION_STATUS_URL = "/api/voice/transcription/status";
  const QUESTION_MAX_LENGTH = 3000;
  const DICTATION_LABEL = {
    idle: "Dictate",
    requesting: "Cancel dictation",
    recording: "Stop dictation",
    uploading: "Cancel transcription",
  };
  const DICTATION_TEXT = {
    requesting: "Allow microphone access to dictate.",
    recording: "Recording… Select Stop dictation to finish (up to 60 seconds).",
    uploading: "Transcribing…",
    added: "Transcript added to your question. Review it before you send it.",
    tooLong:
      "The transcript wasn't added: your question would be longer than 3,000 characters.",
  };

  /** Only exactly {"stt": true} offers dictation. */
  function isDictationStatus(body) {
    return (
      body !== null &&
      typeof body === "object" &&
      !Array.isArray(body) &&
      Object.keys(body).length === 1 &&
      body.stt === true
    );
  }

  function dictationIsPossible() {
    return Boolean(
      window.BluePeakRecorder &&
        window.isSecureContext === true &&
        window.navigator?.mediaDevices &&
        typeof window.MediaRecorder === "function" &&
        typeof window.fetch === "function" &&
        el.dictateBtn &&
        el.dictationNote &&
        el.dictationStatus &&
        el.question
    );
  }

  function createDictationController(recorder) {
    const { STATES, ERROR_MESSAGES } = window.BluePeakRecorder;
    const label = $(".dictate-label", el.dictateBtn);
    // Each session gets a number; an outcome whose number is no longer the
    // current one (cancelled, reset, page left) is ignored.
    let session = 0;

    function setStatus(text) {
      el.dictationStatus.textContent = text;
    }

    /** Mirrors the recorder's state on the button (the outcome sets the final message). */
    function showState(state) {
      el.dictateBtn.dataset.state = state;
      if (label) label.textContent = DICTATION_LABEL[state] || DICTATION_LABEL.idle;
      if (state === STATES.REQUESTING || state === STATES.UPLOADING) {
        el.dictateBtn.setAttribute("aria-busy", "true");
      } else {
        el.dictateBtn.removeAttribute("aria-busy");
      }
      if (DICTATION_TEXT[state]) setStatus(DICTATION_TEXT[state]);
    }

    function hide() {
      const hadFocus = document.activeElement === el.dictateBtn;
      el.dictateBtn.hidden = true;
      el.dictationNote.hidden = true;
      if (hadFocus) el.question.focus();
    }

    /** Appends to what is already there; never submits, never truncates. */
    function insertTranscript(text) {
      const current = el.question.value;
      const separator = current === "" || /\s$/.test(current) ? "" : " ";
      const combined = current + separator + text;

      if (combined.length > QUESTION_MAX_LENGTH) {
        setStatus(DICTATION_TEXT.tooLong);
        el.question.focus();
        return;
      }
      el.question.value = combined;
      updateCharacterCount();
      setStatus(DICTATION_TEXT.added);
      el.question.focus();
      if (typeof el.question.setSelectionRange === "function") {
        el.question.setSelectionRange(combined.length, combined.length);
      }
    }

    function finish(id, outcome) {
      if (id !== session) return; // cancelled, reset or page left meanwhile
      if (outcome.ok) {
        insertTranscript(outcome.text);
        return;
      }
      setStatus(ERROR_MESSAGES[outcome.code] || ERROR_MESSAGES.transcription_failed);
      // Switched off on the server since the page loaded: stop offering it.
      if (outcome.code === "transcription_disabled") hide();
    }

    function begin(event) {
      session += 1;
      const id = session;
      setStatus("");
      // Synchronously inside the click: the recorder asks for the microphone
      // before anything is awaited, and only for a trusted (real user) click.
      const outcome = recorder.start(event);
      // A session started: stop read-aloud before any audio is captured.
      if (recorder.state !== STATES.IDLE) voice?.cancel();
      outcome.then((result) => finish(id, result));
    }

    /** Ends a session at once (microphone released, upload aborted); its outcome is ignored. */
    function cancel({ announce = false } = {}) {
      if (recorder.state === STATES.IDLE) return;
      session += 1;
      recorder.cancel();
      setStatus(announce ? ERROR_MESSAGES.cancelled : "");
    }

    function onClick(event) {
      if (recorder.state === STATES.IDLE) begin(event);
      else if (recorder.state === STATES.RECORDING) recorder.stop(); // then transcribe
      else cancel({ announce: true }); // waiting for permission, or uploading
    }

    return {
      start() {
        el.dictationNote.hidden = false; // the disclosure, before any recording
        el.dictateBtn.hidden = false;
        el.dictateBtn.addEventListener("click", onClick);
      },
      showState,
      cancel,
      isActive() {
        return recorder.state !== STATES.IDLE;
      },
    };
  }

  function startDictation() {
    if (!dictationIsPossible()) return; // nothing is requested or shown

    let controller = null;
    const recorder = window.BluePeakRecorder.createRecorder({
      // Browser objects are passed in; recorder.js alone uses them.
      mediaDevices: window.navigator.mediaDevices,
      MediaRecorder: window.MediaRecorder,
      isSecureContext: window.isSecureContext === true,
      fetchImpl: (url, init) => window.fetch(url, init),
      setTimeout: (fn, ms) => window.setTimeout(fn, ms),
      clearTimeout: (id) => window.clearTimeout(id),
      onStateChange: (state) => controller?.showState(state),
    });
    if (!recorder.supported || !window.BluePeakRecorder.chooseFormat(window.MediaRecorder)) return;

    probeFeatureStatus(DICTATION_STATUS_URL, isDictationStatus).then((offered) => {
      if (!offered) return;
      controller = createDictationController(recorder);
      dictation = controller;
      controller.start();
    });
  }

  function clearSession() {
    hideSessionAlert();
    hideSessionNotice();
    if (!sessionClient) return;

    // The token is dropped at once; the DELETE is not awaited.
    sessionClient.clear().then((result) => {
      if (result.hadSession && !result.cleared && !sessionClient.active) {
        showSessionNotice(SESSION_TEXT.clearFailed);
      }
    });
  }

  function resetConversation(options = {}) {
    // New chat always silences speech and ends dictation, even while a
    // question blocks the reset.
    voice?.cancel();
    dictation?.cancel();
    if (isChatting) return;

    const keptQuestion = options.keepQuestion ? el.question.value : "";
    clearSession();

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

    el.question.value = keptQuestion;
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
      if (event.key !== "Escape") return;

      const modalOpen =
        el.uploadModal &&
        !el.uploadModal.classList.contains("hidden");

      if (modalOpen) {
        closeUploadModal(); // an open dialog comes first; speech continues
      } else if (dictation?.isActive()) {
        dictation.cancel({ announce: true }); // then dictation
      } else if (voice?.isSpeaking()) {
        voice.cancel();
      }
    });
  }

  bindEvents();
  updateCharacterCount();
  resetTrace();
  startVoice();
  startDictation();
})();
