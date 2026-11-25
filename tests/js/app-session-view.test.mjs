// The page side of sessions (A2-7): static/js/app.js running against the real
// chat-stream.js controller and a minimal fake DOM. Checks the exact
// user-visible text, ARIA roles, recovery actions and announcements.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { createDocument, h } from "./fake-dom.mjs";
import { chat, DONE, fakeFetch, jsonResponse, tick } from "./helpers.mjs";

const { CHAT_URL, SESSIONS_URL, SESSION_HEADER } = chat;

const APP_SOURCE = readFileSync(new URL("../../static/js/app.js", import.meta.url), "utf-8");

const TOKEN = "tok_A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0U1v";
const TOKEN_2 = "tok_second_Zz9Yy8Xx7Ww6Vv5Uu4Tt3Ss2Rr1Qq0Pp9Oo8N";

const TEXT = {
  on: "Conversation context on: follow-up questions can use earlier answers in this chat.",
  notSaved: "This answer wasn't saved to the conversation. Follow-up questions won't include it.",
  unknown: "Couldn't confirm this answer was saved to the conversation.",
  clearFailed:
    "Earlier conversation memory couldn't be cleared right now. It will be cleared automatically after a period of inactivity.",
  expired: "This conversation has expired. Start a new chat to continue.",
  busy: "Another question in this conversation is still being answered. Your question wasn't sent. Wait a moment, then select Retry.",
  unavailable:
    "Conversation memory is temporarily unavailable. Your question wasn't sent. Select Retry to try again.",
  network:
    "Couldn't reach the server to start a conversation. Your question wasn't sent. Select Retry to try again.",
  timeout:
    "The server took too long to start a conversation. Your question wasn't sent. Select Retry to try again.",
  origin:
    "Conversations can't be started from this address. Your question wasn't sent. Open BluePeak AI Desk from its usual address to continue.",
  mediaType:
    "This page is out of date, so a conversation couldn't be started. Your question wasn't sent. Reload the page to continue.",
  clientError:
    "A conversation couldn't be started from this page. Your question wasn't sent. Reload the page to continue.",
  generic: "Couldn't start a conversation. Your question wasn't sent. Select Retry to try again.",
};

const created = (token = TOKEN) => jsonResponse({ session_id: token }, { status: 201 });
const sessionError = (status, code, headers = {}) =>
  jsonResponse({ detail: { code, message: "x" } }, { status, headers });
const answer = (session) => jsonResponse(session ? { ...DONE, session } : DONE);

/** /api/sessions: POST answers from ``creates`` in order, DELETE from ``deletes``. */
function sessionsRoute({ creates = [created()], deletes = [] } = {}) {
  return (init) => {
    if (init.method === "DELETE") {
      const next = deletes.shift() ?? new Response(null, { status: 204 });
      return typeof next === "function" ? next(init) : next;
    }
    const next = creates.shift();
    if (!next) throw new Error("unexpected session create");
    return typeof next === "function" ? next(init) : next;
  };
}

/** Never answers; rejects like fetch when the request's signal aborts. */
const hanging = (init) =>
  new Promise((_, reject) => {
    init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
  });

function loadApp(routes, { timeoutMs = 15_000 } = {}) {
  const document = createDocument();
  const form = h(document, "form#chatForm.composer", [
    h(document, "textarea#question"),
    h(document, "button#sendBtn", [h(document, "span", ["Ask agent"])]),
  ]);
  document.body.append(
    h(document, "button#clearChat"),
    h(document, "section#chat"),
    h(document, "div.composer-area", [form, h(document, "p.disclaimer")]),
    h(document, "span#characterCount"),
    h(document, "div#trace"),
    h(document, "span#traceCount"),
    h(document, "span#runStatus"),
    h(document, "div#runProgressBar"),
    h(document, "span#runStepCount"),
    h(document, "span#sourceUsed"),
    h(document, "p#sourceDescription"),
  );

  const windowListeners = new Map();
  const fetchImpl = fakeFetch(routes);
  const reloads = [];
  const window = {
    document,
    fetch: fetchImpl,
    // No ReadableStream: the page uses /api/chat (streaming is covered elsewhere).
    TextDecoder,
    AbortController,
    Response,
    setTimeout,
    clearTimeout,
    location: { origin: "http://localhost", reload: () => reloads.push(true) },
    addEventListener(type, listener) {
      if (!windowListeners.has(type)) windowListeners.set(type, []);
      windowListeners.get(type).push(listener);
    },
    BluePeakChat: {
      ...chat,
      SessionClient: class extends chat.SessionClient {
        constructor(options) {
          super({ ...options, timeoutMs, retryDelayMs: 0 });
        }
      },
    },
  };

  new Function("window", "document", APP_SOURCE)(window, document);

  const $ = (selector) => document.querySelector(selector);
  const page = {
    document,
    fetchImpl,
    reloads,
    question: $("#question"),
    chat: $("#chat"),
    async settle() {
      // Real timers run here (e.g. the 0 ms retry delay), so wait briefly too.
      for (let round = 0; round < 3; round++) {
        for (let i = 0; i < 20; i++) await tick();
        await new Promise((resolve) => setTimeout(resolve, 5));
      }
    },
    async ask(question) {
      page.question.value = question;
      $("#chatForm").dispatch("submit");
      await page.settle();
    },
    fire(type, event = {}) {
      for (const listener of windowListeners.get(type) || []) listener(event);
    },
    alerts: () => document.querySelectorAll(".session-alert"),
    alert: () => page.alerts()[0] || null,
    notice: () => document.querySelector(".session-notice"),
    noticeText: () => {
      const notice = page.notice();
      return notice && !notice.hidden ? notice.textContent : "";
    },
    buttons: () => (page.alert()?.querySelectorAll("button") || []).map((b) => b.textContent),
    clickAction: () => page.alert().querySelector("button").click(),
    newChat: () => $("#clearChat").click(),
    /** Click a welcome suggestion card (shown on an empty chat, e.g. after New chat). */
    async clickSuggestion(index = 0) {
      const card = document.querySelectorAll(".suggestion-card")[index];
      const prompt = card.querySelector(".suggestion-content span").textContent;
      card.focus();
      card.click();
      await page.settle();
      return { card, prompt };
    },
    sentQuestions: () =>
      page.fetchImpl.calls.filter((c) => c.url === CHAT_URL).map((c) => JSON.parse(c.init.body).question),
    userMessages: () => document.querySelectorAll(".message.user"),
    /** Visible live regions (role alert/status or aria-live) whose text includes ``text``. */
    announcing: (text) =>
      [...document.documentElement.descendants()].filter(
        (element) =>
          (element.getAttribute("role") === "alert" ||
            element.getAttribute("role") === "status" ||
            element.hasAttribute("aria-live")) &&
          !element.hidden &&
          element.textContent.includes(text),
      ),
  };
  return page;
}

const sessionHeaders = (page) =>
  page.fetchImpl.calls.filter((c) => c.url === CHAT_URL).map((c) => c.init.headers[SESSION_HEADER]);

// -----------------------------------------------------------------
// Status line
// -----------------------------------------------------------------

test("the status line is a polite status region after the form", async () => {
  const page = loadApp({
    [SESSIONS_URL]: sessionsRoute(),
    [CHAT_URL]: answer({ history_turns_used: 0, turn_saved: true }),
  });

  await page.ask("How do I connect to the VPN?");

  const notice = page.notice();
  assert.equal(notice.getAttribute("role"), "status");
  assert.equal(notice.getAttribute("aria-live"), "polite");
  assert.equal(notice.parentElement.className, "composer-area");
  const siblings = notice.parentElement.children;
  assert.equal(siblings[siblings.indexOf(notice) - 1].id, "chatForm");
  assert.equal(page.noticeText(), TEXT.on);
  assert.deepEqual(sessionHeaders(page), [TOKEN]);
});

for (const [turnSaved, text] of [
  [false, TEXT.notSaved],
  [null, TEXT.unknown],
]) {
  test(`turn_saved ${turnSaved} shows its status text`, async () => {
    const page = loadApp({
      [SESSIONS_URL]: sessionsRoute(),
      [CHAT_URL]: answer({ history_turns_used: 0, turn_saved: turnSaved }),
    });

    await page.ask("q?");

    assert.equal(page.noticeText(), text);
    assert.equal(page.alerts().length, 0);
  });
}

// -----------------------------------------------------------------
// Expired: alert only, announced once
// -----------------------------------------------------------------

test("410 shows the expired message once, in the alert only, and clears the stale status line", async () => {
  let n = 0;
  const page = loadApp({
    [SESSIONS_URL]: sessionsRoute(),
    [CHAT_URL]: () => (++n === 1 ? answer({ history_turns_used: 0, turn_saved: true }) : sessionError(410, "session_expired")),
  });

  await page.ask("first?");
  assert.equal(page.noticeText(), TEXT.on);

  await page.ask("second?");

  const alert = page.alert();
  assert.equal(alert.getAttribute("role"), "alert");
  assert.equal(alert.querySelector("p").textContent, TEXT.expired);
  assert.deepEqual(page.buttons(), ["Start new chat"]);
  assert.equal(page.announcing(TEXT.expired).length, 1, "announced by one live region only");
  assert.equal(page.announcing(TEXT.expired)[0], alert);
  assert.equal(page.notice().hidden, true, "stale status line cleared");
  assert.equal(page.notice().textContent, "");

  // The unsent question is given back and its message removed.
  assert.equal(page.question.value, "second?");
  assert.deepEqual(page.userMessages().map((m) => m.textContent.includes("second?")), [false]);
});

test("asking again after expiry repeats the alert without sending, still announced once", async () => {
  const page = loadApp({
    [SESSIONS_URL]: sessionsRoute(),
    [CHAT_URL]: sessionError(410, "session_expired"),
  });

  await page.ask("q?");
  await page.ask("q?");

  assert.equal(page.alerts().length, 1);
  assert.equal(page.announcing(TEXT.expired).length, 1);
  assert.equal(page.fetchImpl.count(CHAT_URL), 1, "the expired conversation is not continued");
  assert.equal(page.fetchImpl.count(SESSIONS_URL), 1, "no silent new session");
});

test("'Start new chat' keeps the question and the next question uses a new session", async () => {
  let n = 0;
  const page = loadApp({
    [SESSIONS_URL]: sessionsRoute({ creates: [created(), created(TOKEN_2)] }),
    [CHAT_URL]: () => (++n === 1 ? sessionError(410, "session_expired") : answer({ history_turns_used: 0, turn_saved: true })),
  });

  await page.ask("q?");
  page.clickAction();
  await page.settle();

  assert.equal(page.alerts().length, 0);
  assert.equal(page.question.value, "q?");
  assert.equal(page.noticeText(), "");
  assert.equal(page.fetchImpl.calls.filter((c) => c.init.method === "DELETE").length, 0, "no token to delete");

  await page.ask(page.question.value);
  assert.deepEqual(sessionHeaders(page), [TOKEN, TOKEN_2]);
});

// -----------------------------------------------------------------
// Alerts: exact text and recovery action
// -----------------------------------------------------------------

test("409 shows the busy message with Retry; Retry resends with the same token", async () => {
  let n = 0;
  const page = loadApp({
    [SESSIONS_URL]: sessionsRoute(),
    [CHAT_URL]: () => (++n === 1 ? sessionError(409, "session_busy", { "Retry-After": "2" }) : answer({ history_turns_used: 0, turn_saved: true })),
  });

  await page.ask("q?");

  assert.equal(page.alert().querySelector("p").textContent, TEXT.busy);
  assert.deepEqual(page.buttons(), ["Retry"]);
  assert.equal(page.fetchImpl.count(CHAT_URL), 1, "no automatic retry");

  page.clickAction();
  await page.settle();

  assert.deepEqual(sessionHeaders(page), [TOKEN, TOKEN]);
  assert.equal(page.alerts().length, 0);
});

test("503 session_unavailable on a question shows Retry", async () => {
  const page = loadApp({
    [SESSIONS_URL]: sessionsRoute(),
    [CHAT_URL]: sessionError(503, "session_unavailable"),
  });

  await page.ask("q?");

  assert.equal(page.alert().querySelector("p").textContent, TEXT.unavailable);
  assert.deepEqual(page.buttons(), ["Retry"]);
});

for (const [label, creates, text, buttons] of [
  ["503 session_unavailable", [sessionError(503, "session_unavailable")], TEXT.unavailable, ["Retry"]],
  ["two network errors", [new TypeError("x"), new TypeError("x")], TEXT.network, ["Retry"]],
  ["403 origin_not_allowed", [sessionError(403, "origin_not_allowed")], TEXT.origin, []],
  ["415 unsupported_media_type", [sessionError(415, "unsupported_media_type")], TEXT.mediaType, ["Reload page"]],
  ["another 4xx", [new Response("no", { status: 400 })], TEXT.clientError, ["Reload page"]],
  ["a 500", [new Response("no", { status: 500 })], TEXT.generic, ["Retry"]],
]) {
  test(`session creation ${label}: exact message, actions ${JSON.stringify(buttons)}, nothing sent`, async () => {
    const page = loadApp({ [SESSIONS_URL]: sessionsRoute({ creates }), [CHAT_URL]: answer() });

    await page.ask("q?");

    const alert = page.alert();
    assert.equal(alert.getAttribute("role"), "alert");
    assert.equal(alert.querySelector("p").textContent, text);
    assert.deepEqual(page.buttons(), buttons);
    assert.equal(page.fetchImpl.count(CHAT_URL), 0, "never a stateless question");
    assert.equal(page.question.value, "q?", "the question stays in the box");
    assert.equal(page.userMessages().length, 0);
    assert.equal(page.announcing(text).length, 1);
  });
}

test("'Reload page' reloads", async () => {
  const page = loadApp({ [SESSIONS_URL]: sessionsRoute({ creates: [sessionError(415, "unsupported_media_type")] }) });

  await page.ask("q?");
  page.clickAction();

  assert.equal(page.reloads.length, 1);
});

test("a create timeout shows the timeout message with Retry", async () => {
  const page = loadApp(
    { [SESSIONS_URL]: sessionsRoute({ creates: [hanging, created()] }), [CHAT_URL]: answer({ history_turns_used: 0, turn_saved: true }) },
    { timeoutMs: 5 },
  );

  page.question.value = "q?";
  page.document.querySelector("#chatForm").dispatch("submit");
  await new Promise((resolve) => setTimeout(resolve, 30));
  await page.settle();

  assert.equal(page.alert().querySelector("p").textContent, TEXT.timeout);
  assert.deepEqual(page.buttons(), ["Retry"]);
  assert.equal(page.fetchImpl.count(SESSIONS_URL), 1, "a timeout is not retried automatically");
  assert.equal(page.fetchImpl.count(CHAT_URL), 0);

  page.clickAction();
  await page.settle();
  assert.deepEqual(sessionHeaders(page), [TOKEN]);
});

test("leaving the page during creation abandons it: no retry, no alert, nothing sent", async () => {
  const page = loadApp({ [SESSIONS_URL]: sessionsRoute({ creates: [hanging] }), [CHAT_URL]: answer() });

  page.question.value = "q?";
  page.document.querySelector("#chatForm").dispatch("submit");
  await page.settle();
  page.fire("pagehide");
  await page.settle();

  const [create] = page.fetchImpl.calls;
  assert.equal(create.init.signal.aborted, true);
  assert.equal(page.fetchImpl.count(SESSIONS_URL), 1, "no retry after a page abort");
  assert.equal(page.fetchImpl.count(CHAT_URL), 0);
  assert.equal(page.alerts().length, 0);
  assert.equal(page.userMessages().length, 0);
});

test("a new alert replaces the previous one", async () => {
  const page = loadApp({
    [SESSIONS_URL]: sessionsRoute({ creates: [sessionError(503, "session_unavailable"), new TypeError("x"), new TypeError("x")] }),
  });

  await page.ask("q?");
  await page.ask("q?");

  assert.equal(page.alerts().length, 1);
  assert.equal(page.alert().querySelector("p").textContent, TEXT.network);
});

// -----------------------------------------------------------------
// D1: a blocked question stays in the box (typed or suggestion card)
// -----------------------------------------------------------------

test("a blocked typed question stays as typed and Retry resubmits it", async () => {
  const page = loadApp({
    [SESSIONS_URL]: sessionsRoute({ creates: [sessionError(503, "session_unavailable"), created()] }),
    [CHAT_URL]: answer({ history_turns_used: 0, turn_saved: true }),
  });

  await page.ask("  How do I reset MFA?  ");

  assert.equal(page.question.value, "  How do I reset MFA?  ", "the typed text is not rewritten");
  page.clickAction();
  await page.settle();

  assert.deepEqual(page.sentQuestions(), ["How do I reset MFA?"]);
  assert.equal(page.alerts().length, 0);
});

test("a blocked suggestion-card question is kept in the box and Retry resubmits it", async () => {
  const page = loadApp({
    [SESSIONS_URL]: sessionsRoute({ creates: [sessionError(503, "session_unavailable"), created()] }),
    [CHAT_URL]: answer({ history_turns_used: 0, turn_saved: true }),
  });
  page.newChat(); // shows the welcome suggestions (no session yet: nothing to delete)
  await page.settle();

  const { prompt } = await page.clickSuggestion(1);

  assert.equal(page.alert().querySelector("p").textContent, TEXT.unavailable);
  assert.equal(page.question.value, prompt, "the question stays visible in the box");
  assert.equal(page.document.querySelector("#characterCount").textContent, `${prompt.length} / 3000`);
  assert.deepEqual(page.sentQuestions(), [], "nothing was sent");

  page.clickAction();
  await page.settle();

  assert.deepEqual(page.sentQuestions(), [prompt], "Retry actually resubmits the card's question");
  assert.equal(page.alerts().length, 0);
});

test("a blocked suggestion-card question never overwrites text the user typed", async () => {
  const page = loadApp({
    [SESSIONS_URL]: sessionsRoute({ creates: [sessionError(503, "session_unavailable")] }),
  });
  page.newChat();
  await page.settle();
  page.question.value = "my draft";

  await page.clickSuggestion(0);

  assert.equal(page.question.value, "my draft");
  assert.equal(page.alerts().length, 1);
});

test("whitespace-only text in the box is replaced by the blocked card's question", async () => {
  const page = loadApp({
    [SESSIONS_URL]: sessionsRoute({ creates: [sessionError(503, "session_unavailable")] }),
  });
  page.newChat();
  await page.settle();
  page.question.value = "   ";

  const { prompt } = await page.clickSuggestion(0);

  assert.equal(page.question.value, prompt);
});

test("a blocked card question with a permanent error is kept, and Reload is offered", async () => {
  const page = loadApp({
    [SESSIONS_URL]: sessionsRoute({ creates: [sessionError(415, "unsupported_media_type")] }),
  });
  page.newChat();
  await page.settle();

  const { prompt } = await page.clickSuggestion(2);

  assert.equal(page.question.value, prompt);
  assert.deepEqual(page.buttons(), ["Reload page"]);
});

test("Start new chat keeps a question that was blocked as expired", async () => {
  let n = 0;
  const page = loadApp({
    [SESSIONS_URL]: sessionsRoute({ creates: [created(), created(TOKEN_2)] }),
    [CHAT_URL]: () => (++n === 1 ? sessionError(410, "session_expired") : answer({ history_turns_used: 0, turn_saved: true })),
  });

  await page.ask("first?"); // rejected as expired: the question comes back
  page.question.value = "follow-up?";
  await page.ask("follow-up?"); // blocked as expired before anything is sent

  assert.equal(page.alert().querySelector("p").textContent, TEXT.expired);
  assert.equal(page.question.value, "follow-up?");

  page.clickAction(); // Start new chat
  await page.settle();
  assert.equal(page.question.value, "follow-up?");
  assert.equal(page.document.activeElement, page.question);

  await page.ask(page.question.value);
  assert.deepEqual(page.sentQuestions(), ["first?", "follow-up?"]);
});

test("a page abort during creation leaves the box untouched", async () => {
  const page = loadApp({ [SESSIONS_URL]: sessionsRoute({ creates: [hanging] }) });
  page.newChat();
  await page.settle();

  const card = page.document.querySelectorAll(".suggestion-card")[0];
  card.click();
  await page.settle();
  page.fire("pagehide");
  await page.settle();

  assert.equal(page.question.value, "");
  assert.equal(page.alerts().length, 0);
});

// -----------------------------------------------------------------
// D2: focus after a repeated block
// -----------------------------------------------------------------

test("Retry blocked again moves focus to the question box, not the page body", async () => {
  const page = loadApp({
    [SESSIONS_URL]: sessionsRoute({
      creates: [sessionError(503, "session_unavailable"), sessionError(503, "session_unavailable")],
    }),
  });

  await page.ask("q?");
  const retry = page.alert().querySelector("button");
  retry.focus();
  retry.click();
  await page.settle();

  assert.equal(page.alerts().length, 1, "blocked again");
  assert.equal(retry.isConnected, false, "the activated Retry button was removed");
  assert.notEqual(page.document.activeElement, page.document.body);
  assert.equal(page.document.activeElement, page.question);
  assert.equal(page.question.value, "q?");
});

test("Retry blocked again after a suggestion card also focuses the box holding its question", async () => {
  const page = loadApp({
    [SESSIONS_URL]: sessionsRoute({ creates: [new TypeError("x"), new TypeError("x"), sessionError(503, "session_unavailable")] }),
  });
  page.newChat();
  await page.settle();

  const { prompt } = await page.clickSuggestion(0);
  const retry = page.alert().querySelector("button");
  retry.focus();
  retry.click();
  await page.settle();

  assert.equal(page.document.activeElement, page.question);
  assert.equal(page.question.value, prompt);
  assert.equal(page.alert().querySelector("p").textContent, TEXT.unavailable);
});

test("a first block keeps focus where the user was", async () => {
  const page = loadApp({ [SESSIONS_URL]: sessionsRoute({ creates: [sessionError(503, "session_unavailable")] }) });
  page.newChat();
  await page.settle();

  const { card } = await page.clickSuggestion(0);

  assert.equal(page.document.activeElement, card, "the card is still on the page");
});

test("a successful Retry focuses the question box as before", async () => {
  const page = loadApp({
    [SESSIONS_URL]: sessionsRoute({ creates: [sessionError(503, "session_unavailable"), created()] }),
    [CHAT_URL]: answer({ history_turns_used: 0, turn_saved: true }),
  });

  await page.ask("q?");
  const retry = page.alert().querySelector("button");
  retry.focus();
  retry.click();
  await page.settle();

  assert.deepEqual(page.sentQuestions(), ["q?"]);
  assert.equal(page.document.activeElement, page.question);
});

test("an expired question rejected after Retry focuses the box (existing end() behaviour)", async () => {
  let n = 0;
  const page = loadApp({
    [SESSIONS_URL]: sessionsRoute(),
    [CHAT_URL]: () => (++n === 1 ? sessionError(409, "session_busy") : sessionError(410, "session_expired")),
  });

  await page.ask("q?");
  const retry = page.alert().querySelector("button");
  retry.focus();
  retry.click();
  await page.settle();

  assert.equal(page.alert().querySelector("p").textContent, TEXT.expired);
  assert.equal(page.document.activeElement, page.question);
});

// -----------------------------------------------------------------
// New chat
// -----------------------------------------------------------------

test("New chat with a failed DELETE shows the clear-failed status text", async () => {
  const page = loadApp({
    [SESSIONS_URL]: sessionsRoute({ deletes: [sessionError(503, "session_unavailable")] }),
    [CHAT_URL]: answer({ history_turns_used: 0, turn_saved: true }),
  });

  await page.ask("q?");
  page.newChat();
  await page.settle();

  assert.equal(page.noticeText(), TEXT.clearFailed);
  assert.ok(!page.noticeText().includes("30"), "no hardcoded duration");
  const del = page.fetchImpl.calls.find((c) => c.init.method === "DELETE");
  assert.equal(del.init.headers[SESSION_HEADER], TOKEN);
});

test("New chat with a successful DELETE clears the status line", async () => {
  const page = loadApp({
    [SESSIONS_URL]: sessionsRoute(),
    [CHAT_URL]: answer({ history_turns_used: 0, turn_saved: true }),
  });

  await page.ask("q?");
  page.newChat();
  await page.settle();

  assert.equal(page.noticeText(), "");
  assert.equal(page.alerts().length, 0);
});

// -----------------------------------------------------------------
// Stateless
// -----------------------------------------------------------------

test("sessions disabled (404): no header, no status line, no alert, no DELETE", async () => {
  const page = loadApp({
    [SESSIONS_URL]: sessionsRoute({ creates: [jsonResponse({ detail: "Not Found" }, { status: 404 })] }),
    [CHAT_URL]: () => answer(),
  });

  await page.ask("one?");
  await page.ask("two?");
  page.newChat();
  await page.settle();

  assert.deepEqual(sessionHeaders(page), [undefined, undefined]);
  assert.equal(page.notice(), null, "the status line is never created");
  assert.equal(page.alerts().length, 0);
  assert.equal(page.fetchImpl.count(SESSIONS_URL), 1);
});
