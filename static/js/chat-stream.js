/*
 * Streaming chat client for POST /api/chat/stream, with fallback to /api/chat.
 *
 * No DOM access: app.js supplies the view. Loaded as a classic script in the
 * browser (window.BluePeakChat) and with require() in Node tests.
 *
 * Fallback policy (A1-5):
 * - Fallback to /api/chat happens at most once per question, and only before
 *   a valid event stream has started: HTTP 404 (streaming disabled; not tried
 *   again this page session), HTTP 503 (busy; not tried again until
 *   Retry-After has passed), a 200 response that is not an event stream, or
 *   a network failure before response headers arrive.
 * - A network failure before headers does not prove the backend never
 *   processed the request, so this one fallback can occasionally run the
 *   question twice on the server (two executions, two audit rows). The
 *   client never makes more than one fallback request per question.
 * - HTTP 500 and other error statuses never fall back. This is an
 *   intentional trade-off: the server has already handled the request, and
 *   retrying could duplicate backend execution. Do not change this without
 *   a separate review.
 * - Once a valid event stream has started, there is no fallback. An
 *   interrupted, malformed or inconsistent stream is reported as a failure
 *   and the user can ask again.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  } else {
    root.BluePeakChat = api;
  }
})(typeof self !== "undefined" ? self : globalThis, function () {
  "use strict";

  const STREAM_URL = "/api/chat/stream";
  const CHAT_URL = "/api/chat";
  const SESSIONS_URL = "/api/sessions";
  const SESSION_HEADER = "X-Session-ID";
  const SESSION_CREATE_RETRY_MS = 1_000; // one automatic retry after a network error
  const SESSION_CREATE_TIMEOUT_MS = 15_000; // per POST /api/sessions attempt

  const IDLE_TIMEOUT_MS = 45_000; // 3x the server's 15 s heartbeat
  const TOTAL_TIMEOUT_MS = 180_000; // above the server's 120 s deadline plus graces
  const MAX_EVENT_CHARS = 1024 * 1024;
  const DEFAULT_RETRY_AFTER_MS = 5_000;
  const MIN_RETRY_AFTER_MS = 1_000;
  const MAX_RETRY_AFTER_MS = 300_000;

  // Python's str.strip() whitespace, which the backend uses for done.answer.
  const PY_WS =
    "\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
  const PY_STRIP = new RegExp(`^[${PY_WS}]+|[${PY_WS}]+$`, "g");

  function pyStrip(text) {
    return text.replace(PY_STRIP, "");
  }

  const ERROR_MESSAGES = {
    timeout: "The request took too long. Please try again.",
    internal_error: "Unable to process the question. Please try again.",
  };
  const GENERIC_ERROR = "Unable to process the question. Please try again.";
  const INTERRUPTED = "The response was interrupted. Please try again.";

  function errorMessage(code) {
    return Object.prototype.hasOwnProperty.call(ERROR_MESSAGES, code)
      ? ERROR_MESSAGES[code]
      : GENERIC_ERROR;
  }

  // -----------------------------------------------------------------
  // Server-Sent Events parser (decoded text in, events out)
  // -----------------------------------------------------------------

  class SSEProtocolError extends Error {}

  class SSEParser {
    constructor({ maxEventChars = MAX_EVENT_CHARS } = {}) {
      this.maxEventChars = maxEventChars;
      this.buffer = "";
      this.eventName = "";
      this.dataLines = [];
      this.eventChars = 0;
    }

    /** Feed decoded text; returns the complete events, in order. */
    push(text) {
      this.buffer += text;
      const events = [];
      let start = 0;

      for (let i = 0; i < this.buffer.length; i++) {
        const ch = this.buffer[i];
        if (ch !== "\n" && ch !== "\r") continue;
        if (ch === "\r" && i === this.buffer.length - 1) break; // may be CRLF
        const line = this.buffer.slice(start, i);
        if (ch === "\r" && this.buffer[i + 1] === "\n") i++;
        start = i + 1;
        const event = this.processLine(line);
        if (event) events.push(event);
      }

      this.buffer = this.buffer.slice(start);
      if (this.buffer.length + this.eventChars > this.maxEventChars) {
        throw new SSEProtocolError("event too large");
      }
      return events;
    }

    /** End of stream: an event without its terminating blank line is dropped. */
    finish() {
      this.buffer = "";
      this.eventName = "";
      this.dataLines = [];
      this.eventChars = 0;
    }

    processLine(line) {
      if (line === "") {
        const event = this.dataLines.length
          ? { event: this.eventName || "message", data: this.dataLines.join("\n") }
          : null;
        this.eventName = "";
        this.dataLines = [];
        this.eventChars = 0;
        return event;
      }
      if (line.startsWith(":")) return null; // comment, e.g. ": keepalive"

      const colon = line.indexOf(":");
      const field = colon === -1 ? line : line.slice(0, colon);
      let value = colon === -1 ? "" : line.slice(colon + 1);
      if (value.startsWith(" ")) value = value.slice(1);

      if (field === "event") {
        this.eventName = value;
      } else if (field === "data") {
        this.dataLines.push(value);
        this.eventChars += value.length + 1;
        if (this.eventChars > this.maxEventChars) {
          throw new SSEProtocolError("event too large");
        }
      }
      return null; // id, retry and unknown fields are ignored
    }
  }

  // -----------------------------------------------------------------
  // Availability: 404 disables streaming for the page, 503 cools down
  // -----------------------------------------------------------------

  function parseRetryAfter(value, nowMs) {
    let ms = DEFAULT_RETRY_AFTER_MS;
    const text = typeof value === "string" ? value.trim() : "";
    if (/^\d+$/.test(text)) {
      ms = Number(text) * 1000;
    } else if (text) {
      const date = Date.parse(text);
      if (!Number.isNaN(date)) ms = date - nowMs;
    }
    return Math.min(MAX_RETRY_AFTER_MS, Math.max(MIN_RETRY_AFTER_MS, ms));
  }

  class StreamAvailability {
    constructor({ supported = true } = {}) {
      this.supported = supported;
      this.disabled = false;
      this.cooldownUntil = 0;
    }

    canStream(nowMs) {
      return this.supported && !this.disabled && nowMs >= this.cooldownUntil;
    }

    record(outcome, nowMs) {
      if (outcome.reason === "not_found") this.disabled = true;
      if (outcome.reason === "busy") this.cooldownUntil = nowMs + outcome.retryAfterMs;
    }
  }

  // -----------------------------------------------------------------
  // One streaming request
  // -----------------------------------------------------------------

  const realTimers = {
    setTimeout: (fn, ms) => setTimeout(fn, ms),
    clearTimeout: (id) => clearTimeout(id),
  };

  function parseJson(data) {
    try {
      return JSON.parse(data);
    } catch {
      throw new SSEProtocolError("invalid JSON");
    }
  }

  function isObject(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }

  /** turn_saved from a ``session`` event: true / false, or null if unusable. */
  function parseSessionEvent(data, requestId) {
    try {
      const payload = JSON.parse(data);
      if (isObject(payload) && payload.request_id === requestId && typeof payload.turn_saved === "boolean") {
        return payload.turn_saved;
      }
    } catch {
      // malformed: treated as unknown
    }
    return null;
  }

  /**
   * Pre-SSE session errors (session requests only):
   *   409 → session_busy (with Retry-After), 410 → session_expired,
   *   503 with code session_unavailable → session_unavailable.
   * A 503 with code stream_busy, or one whose body can't be read, is not a
   * session error: it stays the existing capacity fallback.
   */
  async function readSessionError(response, now) {
    if (response.status === 409) {
      return {
        code: "session_busy",
        retryAfterMs: parseRetryAfter(response.headers.get("Retry-After"), now()),
      };
    }
    if (response.status === 410) return { code: "session_expired" };
    if (response.status === 503) {
      let code = null;
      try {
        code = (await response.clone().json())?.detail?.code ?? null;
      } catch {
        code = null;
      }
      if (code === "session_unavailable") return { code };
    }
    return null;
  }

  /**
   * Run one streaming request. Resolves (never rejects) to one of:
   *   { type: "done", data }            done payload (same shape as /api/chat)
   *   { type: "error", code }           the server's terminal error event
   *   { type: "fallback", reason, retryAfterMs? }
   *   { type: "failed", reason, status? }
   *   { type: "session_error", code, retryAfterMs? }   session requests only (A2-7)
   * handlers: onStart(requestId), onTrace(step, index), onDelta(text),
   *           onDone(data)  (session requests only: render on "done")
   *
   * With ``sessionToken`` the request carries ``X-Session-ID``; a ``done``
   * outcome then also has ``turnSaved`` (true / false / null) from the
   * ``session`` event that follows ``done``. Without it the request and
   * every outcome are exactly as before.
   */
  async function streamChat({
    question,
    fetchImpl,
    handlers = {},
    signal,
    timers = realTimers,
    now = Date.now,
    idleMs = IDLE_TIMEOUT_MS,
    totalMs = TOTAL_TIMEOUT_MS,
    sessionToken = null,
  }) {
    const controller = new AbortController();
    let abortReason = null;
    let activeReader = null;
    const abort = (reason) => {
      if (abortReason === null) {
        abortReason = reason;
        controller.abort();
        // Also release a pending read directly, whatever the fetch
        // implementation does with the signal.
        activeReader?.cancel().catch(() => {});
      }
    };

    let idleTimer = null;
    const resetIdle = () => {
      if (idleTimer !== null) timers.clearTimeout(idleTimer);
      idleTimer = timers.setTimeout(() => abort("idle_timeout"), idleMs);
    };
    const totalTimer = timers.setTimeout(() => abort("total_timeout"), totalMs);
    const onExternalAbort = () => abort("aborted");
    if (signal) {
      if (signal.aborted) abort("aborted");
      else signal.addEventListener("abort", onExternalAbort, { once: true });
    }
    resetIdle();

    const cleanup = () => {
      if (idleTimer !== null) timers.clearTimeout(idleTimer);
      timers.clearTimeout(totalTimer);
      if (signal) signal.removeEventListener("abort", onExternalAbort);
    };

    try {
      let response;
      try {
        const headers = {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        };
        if (sessionToken) headers[SESSION_HEADER] = sessionToken;
        response = await fetchImpl(STREAM_URL, {
          method: "POST",
          headers,
          body: JSON.stringify({ question }),
          cache: "no-store",
          signal: controller.signal,
        });
      } catch {
        // Our own abort is a failure; anything else is a network failure
        // before headers, which permits the single fallback.
        return abortReason !== null
          ? { type: "failed", reason: abortReason }
          : { type: "fallback", reason: "network" };
      }

      const discardBody = () => {
        try {
          response.body?.cancel().catch(() => {});
        } catch {
          // ignore: nothing else to release
        }
      };

      if (sessionToken) {
        // Session errors come before any SSE bytes; they never fall back.
        const sessionError = await readSessionError(response, now);
        if (sessionError) {
          discardBody();
          return { type: "session_error", ...sessionError };
        }
      }
      if (response.status === 404) {
        discardBody();
        return { type: "fallback", reason: "not_found" };
      }
      if (response.status === 503) {
        discardBody();
        return {
          type: "fallback",
          reason: "busy",
          retryAfterMs: parseRetryAfter(response.headers.get("Retry-After"), now()),
        };
      }
      if (!response.ok) {
        // Intentionally no fallback (see the header comment).
        discardBody();
        return { type: "failed", reason: "http_status", status: response.status };
      }
      const contentType = response.headers.get("Content-Type") || "";
      if (!contentType.toLowerCase().startsWith("text/event-stream") || !response.body) {
        discardBody();
        return { type: "fallback", reason: "not_sse" };
      }

      if (abortReason !== null) {
        discardBody();
        return { type: "failed", reason: abortReason };
      }
      activeReader = response.body.getReader();
      return await readStream(
        activeReader, handlers, resetIdle, abort, () => abortReason, Boolean(sessionToken),
      );
    } finally {
      cleanup();
    }
  }

  async function readStream(reader, handlers, resetIdle, abort, abortReasonOf, sessionMode = false) {
    const decoder = new TextDecoder("utf-8");
    const parser = new SSEParser();
    let requestId = null;
    let text = "";
    let terminal = null;
    let sessionSeen = false;
    let turnSaved = null;

    // A session "done" outcome carries turnSaved; null unless a valid
    // ``session`` event arrived (missing, malformed or cut off → null).
    const outcome = () =>
      sessionMode && terminal && terminal.type === "done" ? { ...terminal, turnSaved } : terminal;

    const fail = (reason) => {
      abort(reason);
      reader.cancel().catch(() => {});
      return { type: "failed", reason };
    };

    const handle = ({ event, data }) => {
      if (terminal) {
        // Session streams: one non-terminal ``session`` event may follow
        // ``done``. A malformed one never fails the answer (it means null).
        if (sessionMode && terminal.type === "done" && event === "session" && !sessionSeen) {
          sessionSeen = true;
          turnSaved = parseSessionEvent(data, requestId);
        }
        return; // anything else after done/error is ignored
      }
      if (event === "start") {
        if (requestId !== null) throw new SSEProtocolError("duplicate start");
        const payload = parseJson(data);
        if (!isObject(payload) || typeof payload.request_id !== "string") {
          throw new SSEProtocolError("bad start");
        }
        requestId = payload.request_id;
        handlers.onStart?.(requestId);
        return;
      }
      if (!["trace", "delta", "done", "error"].includes(event)) return; // forward compatible
      if (requestId === null) throw new SSEProtocolError(`${event} before start`);

      const payload = parseJson(data);
      if (!isObject(payload)) throw new SSEProtocolError(`bad ${event}`);

      if (event === "trace") {
        if (typeof payload.step !== "string") throw new SSEProtocolError("bad trace");
        handlers.onTrace?.(payload.step, payload.index);
      } else if (event === "delta") {
        if (typeof payload.text !== "string") throw new SSEProtocolError("bad delta");
        text += payload.text;
        handlers.onDelta?.(payload.text);
      } else if (event === "done") {
        if (typeof payload.answer !== "string" || payload.request_id !== requestId) {
          throw new SSEProtocolError("bad done");
        }
        terminal =
          pyStrip(text) === payload.answer
            ? { type: "done", data: payload }
            : { type: "failed", reason: "mismatch" };
        if (terminal.type === "done") handlers.onDone?.(payload);
      } else {
        terminal = {
          type: "error",
          code: typeof payload.code === "string" ? payload.code : "internal_error",
        };
      }
    };

    try {
      for (;;) {
        let chunk;
        try {
          chunk = await reader.read();
        } catch {
          // After a terminal event the answer is already complete.
          if (terminal) return outcome();
          return { type: "failed", reason: abortReasonOf() ?? "interrupted" };
        }

        if (chunk.done) {
          if (terminal) return outcome();
          const reason = abortReasonOf();
          if (reason !== null) return { type: "failed", reason }; // cancelled by us
          parser.push(decoder.decode());
          parser.finish();
          return { type: "failed", reason: "interrupted" };
        }

        resetIdle(); // any bytes count as activity, including keepalive comments
        for (const event of parser.push(decoder.decode(chunk.value, { stream: true }))) {
          handle(event);
        }
        if (terminal && terminal.type === "failed") return fail(terminal.reason);
        // After done/error, keep reading to the end of the body instead of
        // closing early: closing right after "done" could reach the server
        // before it records the outcome, and turn a success into "cancelled".
      }
    } catch (error) {
      if (error instanceof SSEProtocolError) return fail("malformed");
      fail("aborted"); // e.g. a view callback threw: release the connection
      throw error;
    }
  }

  // -----------------------------------------------------------------
  // The existing /api/chat request (unchanged behaviour)
  // -----------------------------------------------------------------

  function getText(value, fallback = "") {
    return typeof value === "string" && value.trim() ? value.trim() : fallback;
  }

  /**
   * Resolves to { ok: true, data } or { ok: false, message }, or for a session
   * request { ok: false, sessionError: { code, retryAfterMs? } }. Without a
   * session token the request and results are exactly as before.
   */
  async function requestChat(question, fetchImpl, sessionToken = null, now = Date.now) {
    try {
      const headers = {
        "Content-Type": "application/json",
        Accept: "application/json",
      };
      if (sessionToken) headers[SESSION_HEADER] = sessionToken;
      const response = await fetchImpl(CHAT_URL, {
        method: "POST",
        headers,
        body: JSON.stringify({ question }),
      });

      if (sessionToken) {
        const sessionError = await readSessionError(response, now);
        if (sessionError) return { ok: false, sessionError };
      }

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        return {
          ok: false,
          message: getText(data.detail || data.message, `Request failed (HTTP ${response.status}).`),
        };
      }
      return { ok: true, data };
    } catch (error) {
      return {
        ok: false,
        message: error instanceof Error ? error.message : "Please try again.",
      };
    }
  }

  // -----------------------------------------------------------------
  // Conversation session (A2-7): the token lives only in this object
  // -----------------------------------------------------------------

  /**
   * Holds the session token in memory only (never storage, cookies or URLs).
   *
   * ensure() resolves to one of:
   *   { mode: "session", token, created }   use this token
   *   { mode: "stateless" }                 sessions are disabled (404): for the page session
   *   { mode: "blocked", reason }           do not send the question
   * Only a 404 allows stateless use; any other failure blocks. After a 410
   * (expire()) it stays blocked with reason "session_expired" until clear():
   * an expired conversation is never continued in a new session silently.
   *
   * Each create request is limited to ``timeoutMs`` (reason "timeout", not
   * retried automatically) and is abandoned when the page lifecycle
   * ``signal`` aborts (reason "aborted", never retried). An ordinary network
   * failure is retried once. Abandoning a request does not undo a session
   * the server may already have created: that session is never used and
   * expires on the server after its idle timeout.
   */
  class SessionClient {
    constructor({
      fetchImpl,
      timers = realTimers,
      retryDelayMs = SESSION_CREATE_RETRY_MS,
      timeoutMs = SESSION_CREATE_TIMEOUT_MS,
      signal = null, // an AbortSignal, or a function returning the current one
    }) {
      this.fetchImpl = fetchImpl;
      this.timers = timers;
      this.retryDelayMs = retryDelayMs;
      this.timeoutMs = timeoutMs;
      this.signal = signal;
      this.token = null;
      this.disabled = false;
      this.expired = false;
    }

    get active() {
      return this.token !== null;
    }

    async ensure() {
      if (this.disabled) return { mode: "stateless" };
      if (this.expired) return { mode: "blocked", reason: "session_expired" };
      if (this.token) return { mode: "session", token: this.token, created: false };

      const pageSignal = typeof this.signal === "function" ? this.signal() : this.signal;
      for (let attempt = 0; attempt < 2; attempt++) {
        if (pageSignal?.aborted) return { mode: "blocked", reason: "aborted" };
        const result = await this.createOnce(pageSignal);
        if (result.reason !== "network" || attempt === 1) return result;
        // One automatic retry after an ordinary network failure: a retry is
        // not a downgrade. Leaving the page cancels the wait.
        if (!(await this.wait(this.retryDelayMs, pageSignal))) {
          return { mode: "blocked", reason: "aborted" };
        }
      }
      return { mode: "blocked", reason: "network" };
    }

    /** One POST /api/sessions, bounded by the timeout and the page signal. */
    async createOnce(pageSignal) {
      const controller = new AbortController();
      let abortReason = null;
      let rejectOnAbort;
      const aborted = new Promise((_, reject) => (rejectOnAbort = reject));
      aborted.catch(() => {});
      const abort = (reason) => {
        if (abortReason !== null) return;
        abortReason = reason;
        controller.abort();
        rejectOnAbort(new Error(reason)); // even if fetchImpl ignores the signal
      };
      const timer = this.timers.setTimeout(() => abort("timeout"), this.timeoutMs);
      const onPageAbort = () => abort("aborted");
      pageSignal?.addEventListener("abort", onPageAbort, { once: true });
      const blocked = (reason) => ({ mode: "blocked", reason });

      try {
        let response;
        try {
          response = await Promise.race([
            this.fetchImpl(SESSIONS_URL, {
              method: "POST",
              headers: { "Content-Type": "application/json", Accept: "application/json" },
              body: "{}",
              cache: "no-store",
              signal: controller.signal,
            }),
            aborted,
          ]);
        } catch {
          return blocked(abortReason ?? "network");
        }

        if (response.status === 404) {
          this.disabled = true; // the feature is off: stateless for this page session
          return { mode: "stateless" };
        }

        let body = null;
        try {
          body = await Promise.race([response.json(), aborted]);
        } catch {
          body = null;
        }
        if (abortReason !== null) return blocked(abortReason);

        if (response.status === 201 && typeof body?.session_id === "string" && body.session_id) {
          this.token = body.session_id;
          return { mode: "session", token: this.token, created: true };
        }
        const code = body?.detail?.code;
        return blocked(typeof code === "string" && code ? code : `http_${response.status}`);
      } finally {
        this.timers.clearTimeout(timer);
        pageSignal?.removeEventListener("abort", onPageAbort);
      }
    }

    /** Resolves true after ``ms``, or false at once if the page signal aborts. */
    wait(ms, pageSignal) {
      return new Promise((resolve) => {
        let timer = null;
        const onAbort = () => {
          this.timers.clearTimeout(timer);
          resolve(false);
        };
        timer = this.timers.setTimeout(() => {
          pageSignal?.removeEventListener("abort", onAbort);
          resolve(true);
        }, ms);
        pageSignal?.addEventListener("abort", onAbort, { once: true });
      });
    }

    /** After a 410: forget the token; "New chat" (clear) is needed to go on. */
    expire() {
      this.token = null;
      this.expired = true;
    }

    /**
     * "New chat": forget the token immediately, then ask the server to clear
     * the memory. Resolves to { cleared: true } only on 204; a failure is not
     * retried (the token is gone) and the memory expires on its own.
     */
    async clear() {
      const token = this.token;
      this.token = null;
      this.expired = false;
      if (!token) return { cleared: true, hadSession: false };
      try {
        const response = await this.fetchImpl(SESSIONS_URL, {
          method: "DELETE",
          headers: { [SESSION_HEADER]: token },
          cache: "no-store",
        });
        return { cleared: response.status === 204, hadSession: true };
      } catch {
        return { cleared: false, hadSession: true };
      }
    }
  }

  // -----------------------------------------------------------------
  // Controller: one question at a time, one answer per question
  // -----------------------------------------------------------------

  const FAILURE_MESSAGES = {
    http_status: null, // filled with the status code
    mismatch: INTERRUPTED,
    malformed: INTERRUPTED,
    interrupted: INTERRUPTED,
    idle_timeout: INTERRUPTED,
    total_timeout: INTERRUPTED,
    aborted: INTERRUPTED,
  };

  function failureMessage(outcome) {
    if (outcome.reason === "http_status") return `Request failed (HTTP ${outcome.status}).`;
    return FAILURE_MESSAGES[outcome.reason] || GENERIC_ERROR;
  }

  /**
   * view.begin(question) returns an attempt with:
   *   trace(step, index), delta(text), complete(data), fail(message),
   *   reject({ code, retryAfterMs })        session errors: the question was not sent
   *   sessionStatus({ turnSaved, historyTurnsUsed })   after a session answer
   * The controller calls exactly one of complete / fail / reject per question.
   * view.blocked({ reason, question }) is called instead of begin() when a session is
   * needed but can't be created (503 / 403 / 415 / network): nothing is sent.
   *
   * ``sessions`` (a SessionClient) is optional: without it, or once sessions
   * turn out to be disabled (404), every question is stateless exactly as before.
   */
  function createChatController({
    view,
    fetchImpl,
    availability = new StreamAvailability(),
    now = Date.now,
    timers = realTimers,
    signal,
    idleMs = IDLE_TIMEOUT_MS,
    totalMs = TOTAL_TIMEOUT_MS,
    sessions = null,
  }) {
    let busy = false;

    async function submit(question) {
      if (busy) return false;
      busy = true;

      // A session comes first. Only "disabled" (404) allows a stateless
      // question; any other failure blocks, and nothing is sent.
      let sessionToken = null;
      if (sessions) {
        let session;
        try {
          session = await sessions.ensure();
        } catch {
          session = { mode: "blocked", reason: "error" };
        }
        if (session.mode === "blocked") {
          busy = false;
          view.blocked?.({ reason: session.reason, question });
          return false;
        }
        if (session.mode === "session") {
          sessionToken = session.token;
          if (session.created) view.sessionStarted?.();
        }
      }

      let attempt;
      try {
        attempt = view.begin(question);
      } catch (error) {
        busy = false; // a failing view must not lock later questions
        view.end?.();
        throw error;
      }
      let settled = false;
      const settle = (action) => {
        if (settled) return;
        settled = true;
        action();
      };

      let outcome = null;
      let fallbackUsed = false;
      let partialShown = false;

      const reportSession = (turnSaved, data) => {
        if (!sessionToken) return;
        const turns = data?.session?.history_turns_used;
        attempt.sessionStatus?.({
          turnSaved: typeof turnSaved === "boolean" ? turnSaved : null,
          historyTurnsUsed: Number.isInteger(turns) ? turns : null,
        });
      };

      const sessionError = ({ code, retryAfterMs }) => {
        if (code === "session_expired") sessions.expire(); // never re-sent with this token
        settle(() => attempt.reject({ code, retryAfterMs }));
      };

      const fallback = async () => {
        if (fallbackUsed) return; // never more than one /api/chat request
        fallbackUsed = true;
        // Same token as the stream: never a silent stateless retry.
        const result = await requestChat(question, fetchImpl, sessionToken, now);
        if (result.sessionError) {
          sessionError(result.sessionError);
        } else if (result.ok) {
          settle(() => attempt.complete(result.data));
          reportSession(result.data?.session?.turn_saved, result.data);
        } else {
          settle(() => attempt.fail(result.message));
        }
      };

      try {
        if (!availability.canStream(now())) {
          await fallback();
          return true;
        }

        outcome = await streamChat({
          question,
          fetchImpl,
          // A function lets the page replace its abort signal (e.g. after a
          // back/forward cache restore).
          signal: typeof signal === "function" ? signal() : signal,
          timers,
          now,
          idleMs,
          totalMs,
          sessionToken,
          handlers: {
            onTrace: (step, index) => attempt.trace(step, index),
            onDelta: (text) => {
              partialShown = true;
              attempt.delta(text);
            },
            // Session streams: show the answer as soon as "done" arrives; the
            // question stays busy until the stream ends (after "session").
            onDone: sessionToken ? (data) => settle(() => attempt.complete(data)) : undefined,
          },
        });

        if (outcome.type === "done") {
          settle(() => attempt.complete(outcome.data));
          reportSession(outcome.turnSaved, outcome.data);
        } else if (outcome.type === "session_error") {
          sessionError(outcome);
        } else if (outcome.type === "error") {
          settle(() => attempt.fail(errorMessage(outcome.code)));
        } else if (outcome.type === "fallback" && !partialShown) {
          availability.record(outcome, now());
          await fallback();
        } else {
          settle(() => attempt.fail(failureMessage(outcome)));
        }
      } catch {
        settle(() => attempt.fail(GENERIC_ERROR));
      } finally {
        settle(() => attempt.fail(GENERIC_ERROR)); // never leave a question unanswered
        busy = false;
        view.end?.();
      }
      return true;
    }

    return {
      submit,
      get busy() {
        return busy;
      },
    };
  }

  return {
    STREAM_URL,
    CHAT_URL,
    SESSIONS_URL,
    SESSION_HEADER,
    SESSION_CREATE_RETRY_MS,
    SESSION_CREATE_TIMEOUT_MS,
    SessionClient,
    IDLE_TIMEOUT_MS,
    TOTAL_TIMEOUT_MS,
    ERROR_MESSAGES,
    GENERIC_ERROR,
    INTERRUPTED,
    SSEParser,
    SSEProtocolError,
    StreamAvailability,
    createChatController,
    errorMessage,
    parseRetryAfter,
    pyStrip,
    requestChat,
    streamChat,
  };
});
