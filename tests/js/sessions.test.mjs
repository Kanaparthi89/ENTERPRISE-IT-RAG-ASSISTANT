// Frontend session integration (A2-7): SessionClient and the controller's
// session flow. Stateless behaviour is covered by the existing tests; the
// "stateless compatibility" section here checks it stays identical.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import {
  chat,
  DONE,
  TRANSCRIPT,
  fakeFetch,
  fakeTimers,
  frame,
  jsonResponse,
  manualResponse,
  recordingView,
  sseResponse,
  tick,
} from "./helpers.mjs";

const {
  CHAT_URL,
  STREAM_URL,
  SESSIONS_URL,
  SESSION_HEADER,
  IDLE_TIMEOUT_MS,
  SESSION_CREATE_RETRY_MS,
  SESSION_CREATE_TIMEOUT_MS,
  SessionClient,
  StreamAvailability,
  createChatController,
} = chat;

// Real bytes from the backend's format_sse(); kept in sync by
// tests/test_frontend_assets.py.
const SESSION_EVENT = readFileSync(new URL("./fixtures/session-event.sse", import.meta.url), "utf-8");

const TOKEN = "tok_A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0U1v";
const TOKEN_2 = "tok_second_Zz9Yy8Xx7Ww6Vv5Uu4Tt3Ss2Rr1Qq0Pp9Oo8N";
const SESSION_DONE = { ...DONE, session: { history_turns_used: 1 } };
const SESSION_TRANSCRIPT = TRANSCRIPT.replace(
  frame("done", DONE),
  frame("done", SESSION_DONE),
);

const sessionFrame = (data) => frame("session", { request_id: "req-1", history_turns_used: 1, ...data });

// The retry delay fires at once; the create timeout never fires.
const instantTimers = {
  setTimeout(fn, ms) {
    if (ms === SESSION_CREATE_RETRY_MS) fn();
    return 0;
  },
  clearTimeout() {},
};

/**
 * Await ``promise`` but fail (instead of hanging) if it never settles: a
 * regression in the timeout / abort handling must fail the suite, not stall it.
 */
async function within(promise, ms = 2_000) {
  let timer;
  const limit = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`did not settle within ${ms} ms`)), ms);
  });
  try {
    return await Promise.race([promise, limit]);
  } finally {
    clearTimeout(timer);
  }
}

/** A fetch that never answers until its signal aborts (then rejects like fetch). */
function hangingFetch(calls = []) {
  return (url, init) => {
    calls.push({ url, init });
    return new Promise((_, reject) => {
      init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
    });
  };
}

function created(token = TOKEN) {
  return jsonResponse({ session_id: token }, { status: 201, headers: { "Cache-Control": "no-store" } });
}

function sessionErrorResponse(status, code, headers = {}) {
  return jsonResponse({ detail: { code, message: "x" } }, { status, headers });
}

/** /api/sessions route: POST answers come from ``creates`` in order; DELETE from ``deletes``. */
function sessionsRoute({ creates = [created()], deletes = [] } = {}) {
  return (init) => {
    if (init.method === "DELETE") {
      const next = deletes.shift() ?? new Response(null, { status: 204 });
      return typeof next === "function" ? next() : next;
    }
    const next = creates.shift();
    if (!next) throw new Error("unexpected session create");
    return typeof next === "function" ? next() : next;
  };
}

function sessionView() {
  const view = recordingView();
  const begin = view.begin;
  view.begin = (question) => {
    const attempt = begin(question);
    attempt.reject = (info) => view.log.push(["reject", info]);
    attempt.sessionStatus = (status) => view.log.push(["sessionStatus", status]);
    return attempt;
  };
  view.blocked = (info) => view.log.push(["blocked", info]);
  view.sessionStarted = () => view.log.push(["sessionStarted"]);
  view.kinds = () => view.log.map(([kind]) => kind);
  view.of = (kind) => view.log.filter(([k]) => k === kind).map(([, value]) => value);
  return view;
}

function setup(routes, { availability, now = () => 0, timers = fakeTimers(), sessions } = {}) {
  const view = sessionView();
  const fetchImpl = fakeFetch(routes);
  const client = sessions === null ? null : new SessionClient({ fetchImpl, timers: instantTimers });
  const controller = createChatController({ view, fetchImpl, availability, now, timers, sessions: client });
  return { view, fetchImpl, controller, timers, client };
}

const headersOf = (fetchImpl, url) =>
  fetchImpl.calls.filter((call) => call.url === url).map((call) => call.init.headers || {});

function assertNoTokenInUrls(fetchImpl) {
  for (const { url } of fetchImpl.calls) {
    assert.ok(!url.includes(TOKEN) && !url.includes(TOKEN_2), `token in URL ${url}`);
  }
}

// -----------------------------------------------------------------
// SessionClient: creation
// -----------------------------------------------------------------

test("ensure creates a session with an empty JSON body and reuses the token", async () => {
  const fetchImpl = fakeFetch({ [SESSIONS_URL]: sessionsRoute() });
  const client = new SessionClient({ fetchImpl, timers: instantTimers });

  assert.deepEqual(await client.ensure(), { mode: "session", token: TOKEN, created: true });
  assert.deepEqual(await client.ensure(), { mode: "session", token: TOKEN, created: false });

  assert.equal(fetchImpl.count(SESSIONS_URL), 1);
  const { init } = fetchImpl.calls[0];
  assert.equal(init.method, "POST");
  assert.equal(init.body, "{}");
  assert.equal(init.headers["Content-Type"], "application/json");
  assert.equal(init.cache, "no-store");
  assert.ok(!(SESSION_HEADER in init.headers), "create sends no session header");
  assert.equal(client.active, true);
});

test("404 means sessions are disabled: stateless for the page session", async () => {
  const fetchImpl = fakeFetch({
    [SESSIONS_URL]: sessionsRoute({ creates: [jsonResponse({ detail: "Not Found" }, { status: 404 })] }),
  });
  const client = new SessionClient({ fetchImpl, timers: instantTimers });

  assert.deepEqual(await client.ensure(), { mode: "stateless" });
  assert.deepEqual(await client.ensure(), { mode: "stateless" });
  assert.equal(fetchImpl.count(SESSIONS_URL), 1, "404 remembered");
  assert.equal(client.active, false);
});

for (const [status, code] of [
  [503, "session_unavailable"],
  [403, "origin_not_allowed"],
  [415, "unsupported_media_type"],
]) {
  test(`${status} ${code} blocks, is not retried automatically, and is not remembered`, async () => {
    const fetchImpl = fakeFetch({
      [SESSIONS_URL]: sessionsRoute({ creates: [sessionErrorResponse(status, code), created()] }),
    });
    const client = new SessionClient({ fetchImpl, timers: instantTimers });

    assert.deepEqual(await client.ensure(), { mode: "blocked", reason: code });
    assert.equal(fetchImpl.count(SESSIONS_URL), 1, "no automatic retry");

    // The user's Retry tries again.
    assert.deepEqual(await client.ensure(), { mode: "session", token: TOKEN, created: true });
  });
}

test("an unexpected status without a code blocks with the status", async () => {
  const fetchImpl = fakeFetch({
    [SESSIONS_URL]: sessionsRoute({ creates: [new Response("oops", { status: 500 })] }),
  });
  const client = new SessionClient({ fetchImpl, timers: instantTimers });

  assert.deepEqual(await client.ensure(), { mode: "blocked", reason: "http_500" });
});

test("a 201 without a usable session_id blocks", async () => {
  const fetchImpl = fakeFetch({
    [SESSIONS_URL]: sessionsRoute({ creates: [jsonResponse({}, { status: 201 })] }),
  });
  const client = new SessionClient({ fetchImpl, timers: instantTimers });

  assert.equal((await client.ensure()).mode, "blocked");
  assert.equal(client.active, false);
});

test("a network error is retried once after the delay, then succeeds", async () => {
  const delays = [];
  const timers = {
    setTimeout(fn, ms) {
      if (ms === SESSION_CREATE_TIMEOUT_MS) return 0; // the create timeout
      delays.push(ms);
      fn();
      return 0;
    },
    clearTimeout() {},
  };
  const fetchImpl = fakeFetch({
    [SESSIONS_URL]: sessionsRoute({ creates: [new TypeError("offline"), created()] }),
  });
  const client = new SessionClient({ fetchImpl, timers, retryDelayMs: 1234 });

  assert.deepEqual(await client.ensure(), { mode: "session", token: TOKEN, created: true });
  assert.equal(fetchImpl.count(SESSIONS_URL), 2);
  assert.deepEqual(delays, [1234]);
});

test("two network errors block with reason network (exactly two attempts)", async () => {
  const fetchImpl = fakeFetch({
    [SESSIONS_URL]: sessionsRoute({ creates: [new TypeError("offline"), new TypeError("offline")] }),
  });
  const client = new SessionClient({ fetchImpl, timers: instantTimers });

  assert.deepEqual(await client.ensure(), { mode: "blocked", reason: "network" });
  assert.equal(fetchImpl.count(SESSIONS_URL), 2);
});

// -----------------------------------------------------------------
// SessionClient: timeout and page abandonment
// -----------------------------------------------------------------

test("each create request carries an abort signal", async () => {
  const fetchImpl = fakeFetch({ [SESSIONS_URL]: sessionsRoute() });
  const client = new SessionClient({ fetchImpl, timers: instantTimers });

  await client.ensure();

  assert.ok(fetchImpl.calls[0].init.signal instanceof AbortSignal);
  assert.equal(fetchImpl.calls[0].init.signal.aborted, false);
});

test("a create request that doesn't answer in time blocks with timeout and is not retried", async () => {
  const timers = fakeTimers();
  const calls = [];
  const client = new SessionClient({ fetchImpl: hangingFetch(calls), timers });

  const pending = client.ensure();
  await tick();
  timers.fire(SESSION_CREATE_TIMEOUT_MS);

  assert.deepEqual(await within(pending), { mode: "blocked", reason: "timeout" });
  assert.equal(calls.length, 1, "a timeout is not retried automatically");
  assert.equal(calls[0].init.signal.aborted, true, "the browser request is abandoned");
  assert.equal(timers.count(), 0, "no timers left behind");
  assert.equal(client.active, false);
});

test("the timeout also bounds reading the response body", async () => {
  const timers = fakeTimers();
  const slowBody = new Response(new ReadableStream({ start() {} }), {
    status: 201,
    headers: { "Content-Type": "application/json" },
  });
  const client = new SessionClient({ fetchImpl: async () => slowBody, timers });

  const pending = client.ensure();
  await tick();
  timers.fire(SESSION_CREATE_TIMEOUT_MS);

  assert.deepEqual(await within(pending), { mode: "blocked", reason: "timeout" });
  assert.equal(client.active, false);
});

test("the timeout works even if fetchImpl ignores the signal", async () => {
  const timers = fakeTimers();
  const client = new SessionClient({ fetchImpl: () => new Promise(() => {}), timers });

  const pending = client.ensure();
  await tick();
  timers.fire(SESSION_CREATE_TIMEOUT_MS);

  assert.deepEqual(await within(pending), { mode: "blocked", reason: "timeout" });
});

test("the timeout applies to each attempt separately", async () => {
  const timers = fakeTimers();
  let n = 0;
  const fetchImpl = async () => {
    n += 1;
    if (n === 1) throw new TypeError("offline");
    return created();
  };
  const client = new SessionClient({ fetchImpl, timers });

  const pending = client.ensure();
  await tick();
  timers.fire(SESSION_CREATE_RETRY_MS);

  assert.deepEqual(await within(pending), { mode: "session", token: TOKEN, created: true });
  assert.equal(timers.count(), 0, "both attempts' timeouts cleared");
});

test("leaving the page abandons the create request and does not retry", async () => {
  const page = new AbortController();
  const calls = [];
  const timers = fakeTimers();
  const client = new SessionClient({ fetchImpl: hangingFetch(calls), timers, signal: () => page.signal });

  const pending = client.ensure();
  await tick();
  page.abort();

  assert.deepEqual(await within(pending), { mode: "blocked", reason: "aborted" });
  assert.equal(calls.length, 1, "no automatic retry after a page abort");
  assert.equal(calls[0].init.signal.aborted, true);
  assert.equal(timers.count(), 0, "no retry delay was scheduled");
});

test("a page that is already being left sends no create request", async () => {
  const page = new AbortController();
  page.abort();
  const fetchImpl = fakeFetch({ [SESSIONS_URL]: sessionsRoute() });
  const client = new SessionClient({ fetchImpl, timers: instantTimers, signal: page.signal });

  assert.deepEqual(await client.ensure(), { mode: "blocked", reason: "aborted" });
  assert.equal(fetchImpl.calls.length, 0);
});

test("leaving the page during the retry delay cancels the retry", async () => {
  const page = new AbortController();
  const timers = fakeTimers();
  const fetchImpl = fakeFetch({ [SESSIONS_URL]: sessionsRoute({ creates: [new TypeError("offline"), created()] }) });
  const client = new SessionClient({ fetchImpl, timers, signal: () => page.signal });

  const pending = client.ensure();
  for (let i = 0; i < 3; i++) await tick();
  page.abort();

  assert.deepEqual(await within(pending), { mode: "blocked", reason: "aborted" });
  assert.equal(fetchImpl.count(SESSIONS_URL), 1, "the retry was never sent");
  assert.equal(timers.count(), 0);
});

test("the page signal is read per ensure, so a restored page can create again", async () => {
  let page = new AbortController();
  page.abort();
  const fetchImpl = fakeFetch({ [SESSIONS_URL]: sessionsRoute() });
  const client = new SessionClient({ fetchImpl, timers: instantTimers, signal: () => page.signal });

  assert.equal((await client.ensure()).reason, "aborted");
  page = new AbortController(); // pageshow after a back/forward cache restore
  assert.equal((await client.ensure()).mode, "session");
});

test("a page abort during creation sends no question and shows nothing", async () => {
  const page = new AbortController();
  const calls = [];
  const view = sessionView();
  const fetchImpl = (url, init) => (url === SESSIONS_URL ? hangingFetch(calls)(url, init) : Promise.reject(new Error(url)));
  const client = new SessionClient({ fetchImpl, timers: fakeTimers(), signal: () => page.signal });
  const controller = createChatController({ view, fetchImpl, timers: fakeTimers(), sessions: client });

  const pending = controller.submit("q?");
  await tick();
  page.abort();

  assert.equal(await within(pending), false);
  assert.deepEqual(view.log, [["blocked", { reason: "aborted", question: "q?" }]]);
  assert.equal(calls.length, 1);
});

// -----------------------------------------------------------------
// SessionClient: New chat (clear) and expiry
// -----------------------------------------------------------------

test("clear drops the token before the DELETE completes and sends it in the header", async () => {
  let release;
  const fetchImpl = fakeFetch({
    [SESSIONS_URL]: sessionsRoute({
      deletes: [() => new Promise((resolve) => (release = resolve))],
    }),
  });
  const client = new SessionClient({ fetchImpl, timers: instantTimers });
  await client.ensure();

  const pending = client.clear();
  assert.equal(client.active, false, "token gone immediately");
  await tick();

  const del = fetchImpl.calls.at(-1);
  assert.equal(del.url, SESSIONS_URL);
  assert.equal(del.init.method, "DELETE");
  assert.equal(del.init.headers[SESSION_HEADER], TOKEN);
  assert.equal(del.init.body, undefined);

  release(new Response(null, { status: 204 }));
  assert.deepEqual(await within(pending), { cleared: true, hadSession: true });
  assertNoTokenInUrls(fetchImpl);
});

for (const [label, answer] of [
  ["503", () => sessionErrorResponse(503, "session_unavailable")],
  ["a network error", () => Promise.reject(new TypeError("offline"))],
]) {
  test(`clear reports failure on ${label} and is not retried`, async () => {
    const fetchImpl = fakeFetch({ [SESSIONS_URL]: sessionsRoute({ deletes: [answer] }) });
    const client = new SessionClient({ fetchImpl, timers: instantTimers });
    await client.ensure();

    assert.deepEqual(await client.clear(), { cleared: false, hadSession: true });
    assert.equal(fetchImpl.calls.filter((c) => c.init.method === "DELETE").length, 1);
    assert.equal(client.active, false);
  });
}

test("clear without a session sends nothing", async () => {
  const fetchImpl = fakeFetch({});
  const client = new SessionClient({ fetchImpl, timers: instantTimers });

  assert.deepEqual(await client.clear(), { cleared: true, hadSession: false });
  assert.equal(fetchImpl.calls.length, 0);
});

test("after expire, ensure blocks without a request until clear", async () => {
  const fetchImpl = fakeFetch({ [SESSIONS_URL]: sessionsRoute({ creates: [created(), created(TOKEN_2)] }) });
  const client = new SessionClient({ fetchImpl, timers: instantTimers });
  await client.ensure();

  client.expire();
  assert.deepEqual(await client.ensure(), { mode: "blocked", reason: "session_expired" });
  assert.equal(fetchImpl.count(SESSIONS_URL), 1);

  await client.clear(); // no token: nothing sent
  assert.equal(fetchImpl.count(SESSIONS_URL), 1);
  assert.deepEqual(await client.ensure(), { mode: "session", token: TOKEN_2, created: true });
});

// -----------------------------------------------------------------
// Controller: headers and creation order
// -----------------------------------------------------------------

test("the session is created before the question is shown, and the stream carries the header", async () => {
  const { view, fetchImpl, controller } = setup({
    [SESSIONS_URL]: sessionsRoute(),
    [STREAM_URL]: sseResponse([SESSION_TRANSCRIPT, SESSION_EVENT]),
  });

  assert.equal(await controller.submit("q?"), true);

  assert.deepEqual(fetchImpl.calls.map((c) => c.url), [SESSIONS_URL, STREAM_URL]);
  assert.equal(headersOf(fetchImpl, STREAM_URL)[0][SESSION_HEADER], TOKEN);
  assert.deepEqual(view.kinds().slice(0, 2), ["sessionStarted", "begin"]);
  assertNoTokenInUrls(fetchImpl);
});

test("later questions reuse the token without creating another session", async () => {
  const { view, fetchImpl, controller } = setup({
    [SESSIONS_URL]: sessionsRoute(),
    [STREAM_URL]: () => sseResponse([SESSION_TRANSCRIPT, SESSION_EVENT]),
  });

  await controller.submit("one?");
  await controller.submit("two?");

  assert.equal(fetchImpl.count(SESSIONS_URL), 1);
  assert.deepEqual(headersOf(fetchImpl, STREAM_URL).map((h) => h[SESSION_HEADER]), [TOKEN, TOKEN]);
  assert.equal(view.of("sessionStarted").length, 1);
});

test("/api/chat carries the header when streaming is unavailable", async () => {
  const { view, fetchImpl, controller } = setup(
    {
      [SESSIONS_URL]: sessionsRoute(),
      [CHAT_URL]: jsonResponse({ ...DONE, session: { history_turns_used: 2, turn_saved: true } }),
    },
    { availability: new StreamAvailability({ supported: false }) },
  );

  await controller.submit("q?");

  assert.equal(headersOf(fetchImpl, CHAT_URL)[0][SESSION_HEADER], TOKEN);
  assert.deepEqual(view.of("sessionStatus"), [{ turnSaved: true, historyTurnsUsed: 2 }]);
});

// -----------------------------------------------------------------
// Controller: SSE event ordering and turn_saved
// -----------------------------------------------------------------

test("the answer renders on done; the question stays busy until the session event", async () => {
  const body = manualResponse();
  const { view, controller } = setup({
    [SESSIONS_URL]: sessionsRoute(),
    [STREAM_URL]: body.response,
  });

  const done = controller.submit("q?");
  body.send(SESSION_TRANSCRIPT);
  for (let i = 0; i < 5; i++) await tick();

  assert.deepEqual(view.finals(), [["complete", SESSION_DONE]], "rendered on done");
  assert.equal(controller.busy, true, "still busy before the session event");
  assert.ok(!view.kinds().includes("end"));
  assert.ok(!view.kinds().includes("sessionStatus"));

  body.send(SESSION_EVENT);
  body.close();
  await done;

  assert.deepEqual(view.kinds().slice(-3), ["complete", "sessionStatus", "end"]);
  assert.deepEqual(view.of("sessionStatus"), [{ turnSaved: true, historyTurnsUsed: 1 }]);
  assert.equal(view.finals().length, 1, "complete is not repeated at the end");
  assert.equal(controller.busy, false);
});

for (const [label, tail, expected] of [
  ["turn_saved true", [SESSION_EVENT], true],
  ["turn_saved false", [sessionFrame({ turn_saved: false })], false],
  ["turn_saved null (not confirmed in time)", [sessionFrame({ turn_saved: null })], null],
  ["no session event", [], null],
  ["a malformed session event", ["event: session\ndata: {not json\n\n"], null],
  ["a session event for another request", [sessionFrame({ request_id: "other", turn_saved: true })], null],
  ["a non-boolean turn_saved", [sessionFrame({ turn_saved: "yes" })], null],
]) {
  test(`${label} → turnSaved ${expected}`, async () => {
    const { view, controller } = setup({
      [SESSIONS_URL]: sessionsRoute(),
      [STREAM_URL]: sseResponse([SESSION_TRANSCRIPT, ...tail]),
    });

    await controller.submit("q?");

    assert.deepEqual(view.finals(), [["complete", SESSION_DONE]], "the answer is never failed");
    assert.deepEqual(view.of("sessionStatus"), [{ turnSaved: expected, historyTurnsUsed: 1 }]);
  });
}

test("only the first session event counts", async () => {
  const { view, controller } = setup({
    [SESSIONS_URL]: sessionsRoute(),
    [STREAM_URL]: sseResponse([SESSION_TRANSCRIPT, sessionFrame({ turn_saved: false }), SESSION_EVENT]),
  });

  await controller.submit("q?");

  assert.deepEqual(view.of("sessionStatus"), [{ turnSaved: false, historyTurnsUsed: 1 }]);
});

test("a session event before done is ignored (forward compatible) and gives null", async () => {
  const early = TRANSCRIPT.replace(frame("done", DONE), SESSION_EVENT + frame("done", SESSION_DONE));
  const { view, controller } = setup({
    [SESSIONS_URL]: sessionsRoute(),
    [STREAM_URL]: sseResponse([early]),
  });

  await controller.submit("q?");

  assert.deepEqual(view.finals(), [["complete", SESSION_DONE]]);
  assert.deepEqual(view.of("sessionStatus"), [{ turnSaved: null, historyTurnsUsed: 1 }]);
});

test("an idle timeout after done keeps the answer and reports null", async () => {
  const body = manualResponse();
  const { view, controller, timers } = setup({
    [SESSIONS_URL]: sessionsRoute(),
    [STREAM_URL]: body.response,
  });

  const done = controller.submit("q?");
  body.send(SESSION_TRANSCRIPT);
  for (let i = 0; i < 5; i++) await tick();
  timers.fire(IDLE_TIMEOUT_MS);
  await done;

  assert.deepEqual(view.finals(), [["complete", SESSION_DONE]]);
  assert.deepEqual(view.of("sessionStatus"), [{ turnSaved: null, historyTurnsUsed: 1 }]);
});

test("a stream error event fails the answer and reports no save status", async () => {
  const errored = TRANSCRIPT.replace(frame("done", DONE), frame("error", { code: "generation_failed" }));
  const { view, fetchImpl, controller } = setup({
    [SESSIONS_URL]: sessionsRoute(),
    [STREAM_URL]: sseResponse([errored]),
  });

  await controller.submit("q?");

  assert.equal(view.finals()[0][0], "fail");
  assert.deepEqual(view.of("sessionStatus"), []);
  assert.equal(fetchImpl.count(CHAT_URL), 0);
});

// -----------------------------------------------------------------
// Controller: 409 / 410 / 503
// -----------------------------------------------------------------

test("409 rejects with Retry-After, keeps the token and is not retried", async () => {
  const { view, fetchImpl, controller, client } = setup(
    {
      [SESSIONS_URL]: sessionsRoute(),
      [STREAM_URL]: sessionErrorResponse(409, "session_busy", { "Retry-After": "3" }),
    },
    { now: () => 1_000 },
  );

  await controller.submit("q?");

  assert.deepEqual(view.log.filter(([k]) => k === "reject"), [
    ["reject", { code: "session_busy", retryAfterMs: 3_000 }],
  ]);
  assert.equal(view.finals().length, 0, "no answer and no failure message");
  assert.equal(fetchImpl.count(STREAM_URL), 1, "no automatic retry");
  assert.equal(fetchImpl.count(CHAT_URL), 0, "no fallback");
  assert.equal(client.active, true, "token kept");
  assert.deepEqual(view.kinds().slice(-1), ["end"]);
});

test("410 rejects as expired, drops the token, and blocks until New chat", async () => {
  const { view, fetchImpl, controller, client } = setup({
    [SESSIONS_URL]: sessionsRoute({ creates: [created(), created(TOKEN_2)] }),
    [STREAM_URL]: (_init, n) =>
      n <= 2 ? sessionErrorResponse(410, "session_expired") : sseResponse([SESSION_TRANSCRIPT, SESSION_EVENT]),
  });

  await controller.submit("q?");
  assert.deepEqual(view.of("reject"), [{ code: "session_expired", retryAfterMs: undefined }]);
  assert.equal(client.active, false);
  assert.equal(fetchImpl.count(CHAT_URL), 0, "never re-sent stateless");

  assert.equal(await controller.submit("again?"), false);
  assert.deepEqual(view.of("blocked"), [{ reason: "session_expired", question: "again?" }]);
  assert.equal(fetchImpl.count(STREAM_URL), 1, "the expired conversation is not continued");
  assert.equal(fetchImpl.count(SESSIONS_URL), 1);

  await client.clear(); // New chat
  await controller.submit("fresh?");
  assert.equal(fetchImpl.count(SESSIONS_URL), 2);
  assert.equal(headersOf(fetchImpl, STREAM_URL).at(-1)[SESSION_HEADER], TOKEN_2);
});

test("503 session_unavailable rejects without fallback and keeps the token", async () => {
  const { view, fetchImpl, controller, client } = setup({
    [SESSIONS_URL]: sessionsRoute(),
    [STREAM_URL]: sessionErrorResponse(503, "session_unavailable"),
  });

  await controller.submit("q?");

  assert.deepEqual(view.of("reject"), [{ code: "session_unavailable", retryAfterMs: undefined }]);
  assert.equal(fetchImpl.count(CHAT_URL), 0);
  assert.equal(client.active, true);
});

test("503 stream_busy falls back to /api/chat once with the same token", async () => {
  const { view, fetchImpl, controller } = setup({
    [SESSIONS_URL]: sessionsRoute(),
    [STREAM_URL]: jsonResponse(
      { detail: { code: "stream_busy", message: "busy" } },
      { status: 503, headers: { "Retry-After": "5" } },
    ),
    [CHAT_URL]: jsonResponse({ ...DONE, session: { history_turns_used: 1, turn_saved: false } }),
  });

  await controller.submit("q?");

  assert.equal(fetchImpl.count(CHAT_URL), 1);
  assert.equal(headersOf(fetchImpl, CHAT_URL)[0][SESSION_HEADER], TOKEN);
  assert.equal(view.finals()[0][0], "complete");
  assert.deepEqual(view.of("sessionStatus"), [{ turnSaved: false, historyTurnsUsed: 1 }]);
});

for (const [label, streamAnswer] of [
  ["404", () => jsonResponse({ detail: "Not Found" }, { status: 404 })],
  ["a network error before headers", () => new TypeError("offline")],
]) {
  test(`stream ${label} falls back with the same token, never stateless`, async () => {
    const { fetchImpl, controller } = setup({
      [SESSIONS_URL]: sessionsRoute(),
      [STREAM_URL]: streamAnswer,
      [CHAT_URL]: jsonResponse({ ...DONE, session: { history_turns_used: 0, turn_saved: true } }),
    });

    await controller.submit("q?");

    assert.deepEqual(headersOf(fetchImpl, CHAT_URL).map((h) => h[SESSION_HEADER]), [TOKEN]);
  });
}

for (const [status, code, retryAfterMs] of [
  [409, "session_busy", 2_000],
  [410, "session_expired", undefined],
  [503, "session_unavailable", undefined],
]) {
  test(`a ${status} from the /api/chat fallback rejects (${code})`, async () => {
    const { view, fetchImpl, controller } = setup(
      {
        [SESSIONS_URL]: sessionsRoute(),
        [CHAT_URL]: sessionErrorResponse(status, code, { "Retry-After": "2" }),
      },
      { availability: new StreamAvailability({ supported: false }) },
    );

    await controller.submit("q?");

    assert.deepEqual(view.of("reject"), [{ code, retryAfterMs }]);
    assert.equal(view.finals().length, 0);
    assert.equal(fetchImpl.count(CHAT_URL), 1, "no retry");
  });
}

test("/api/chat without a session block in the body reports null", async () => {
  const { view, controller } = setup(
    { [SESSIONS_URL]: sessionsRoute(), [CHAT_URL]: jsonResponse(DONE) },
    { availability: new StreamAvailability({ supported: false }) },
  );

  await controller.submit("q?");

  assert.deepEqual(view.of("sessionStatus"), [{ turnSaved: null, historyTurnsUsed: null }]);
});

test("a stream 500 in a session fails without fallback", async () => {
  const { view, fetchImpl, controller } = setup({
    [SESSIONS_URL]: sessionsRoute(),
    [STREAM_URL]: new Response("boom", { status: 500 }),
  });

  await controller.submit("q?");

  assert.equal(view.finals()[0][0], "fail");
  assert.equal(fetchImpl.count(CHAT_URL), 0);
});

// -----------------------------------------------------------------
// Controller: blocked creation
// -----------------------------------------------------------------

for (const [label, creates, reason] of [
  ["503", [sessionErrorResponse(503, "session_unavailable")], "session_unavailable"],
  ["403", [sessionErrorResponse(403, "origin_not_allowed")], "origin_not_allowed"],
  ["415", [sessionErrorResponse(415, "unsupported_media_type")], "unsupported_media_type"],
  ["two network errors", [new TypeError("x"), new TypeError("x")], "network"],
]) {
  test(`session creation ${label} blocks: nothing is sent or shown`, async () => {
    const { view, fetchImpl, controller } = setup({
      [SESSIONS_URL]: sessionsRoute({ creates }),
      [STREAM_URL]: sseResponse([TRANSCRIPT]),
      [CHAT_URL]: jsonResponse(DONE),
    });

    assert.equal(await controller.submit("q?"), false);

    assert.deepEqual(view.log, [["blocked", { reason, question: "q?" }]]);
    assert.equal(fetchImpl.count(STREAM_URL), 0);
    assert.equal(fetchImpl.count(CHAT_URL), 0, "never a silent stateless question");
    assert.equal(controller.busy, false);
  });
}

test("Retry after a blocked creation sends the question in a new session", async () => {
  const { view, fetchImpl, controller } = setup({
    [SESSIONS_URL]: sessionsRoute({ creates: [sessionErrorResponse(503, "session_unavailable"), created()] }),
    [STREAM_URL]: sseResponse([SESSION_TRANSCRIPT, SESSION_EVENT]),
  });

  await controller.submit("q?");
  await controller.submit("q?");

  assert.equal(view.finals()[0][0], "complete");
  assert.equal(headersOf(fetchImpl, STREAM_URL)[0][SESSION_HEADER], TOKEN);
});

test("a second submit while the session is being created is ignored", async () => {
  let release;
  const { fetchImpl, controller } = setup({
    [SESSIONS_URL]: sessionsRoute({ creates: [() => new Promise((resolve) => (release = resolve))] }),
    [STREAM_URL]: sseResponse([SESSION_TRANSCRIPT, SESSION_EVENT]),
  });

  const first = controller.submit("one?");
  await tick();
  assert.equal(await controller.submit("two?"), false);
  release(created());
  await first;

  assert.equal(fetchImpl.count(SESSIONS_URL), 1);
  assert.equal(fetchImpl.count(STREAM_URL), 1);
});

// -----------------------------------------------------------------
// New chat
// -----------------------------------------------------------------

test("New chat: the next question starts a new session with a new token", async () => {
  const { fetchImpl, controller, client } = setup({
    [SESSIONS_URL]: sessionsRoute({ creates: [created(), created(TOKEN_2)] }),
    [STREAM_URL]: () => sseResponse([SESSION_TRANSCRIPT, SESSION_EVENT]),
  });

  await controller.submit("one?");
  const cleared = client.clear();
  assert.equal(client.active, false);
  await controller.submit("two?");

  assert.deepEqual(await cleared, { cleared: true, hadSession: true });
  const del = fetchImpl.calls.find((c) => c.init.method === "DELETE");
  assert.equal(del.init.headers[SESSION_HEADER], TOKEN);
  assert.deepEqual(headersOf(fetchImpl, STREAM_URL).map((h) => h[SESSION_HEADER]), [TOKEN, TOKEN_2]);
});

test("a failed DELETE does not affect the next session", async () => {
  const { fetchImpl, controller, client } = setup({
    [SESSIONS_URL]: sessionsRoute({
      creates: [created(), created(TOKEN_2)],
      deletes: [sessionErrorResponse(503, "session_unavailable")],
    }),
    [STREAM_URL]: () => sseResponse([SESSION_TRANSCRIPT, SESSION_EVENT]),
  });

  await controller.submit("one?");
  assert.deepEqual(await client.clear(), { cleared: false, hadSession: true });
  await controller.submit("two?");

  assert.equal(headersOf(fetchImpl, STREAM_URL).at(-1)[SESSION_HEADER], TOKEN_2);
});

// -----------------------------------------------------------------
// Stateless compatibility
// -----------------------------------------------------------------

function statelessRun(sessions, routes, options = {}) {
  const view = sessionView();
  const fetchImpl = fakeFetch(routes);
  const controller = createChatController({
    view,
    fetchImpl,
    timers: fakeTimers(),
    now: () => 0,
    sessions,
    ...options,
  });
  return { view, fetchImpl, controller };
}

for (const [label, streamRoutes] of [
  ["a stream", () => ({ [STREAM_URL]: sseResponse([TRANSCRIPT]) })],
  ["a 404 fallback", () => ({
    [STREAM_URL]: jsonResponse({ detail: "Not Found" }, { status: 404 }),
    [CHAT_URL]: jsonResponse(DONE),
  })],
  ["a 503 fallback", () => ({
    [STREAM_URL]: jsonResponse({ detail: { message: "busy" } }, { status: 503, headers: { "Retry-After": "5" } }),
    [CHAT_URL]: jsonResponse(DONE),
  })],
]) {
  test(`sessions disabled (404): ${label} is identical to the stateless client`, async () => {
    const plain = statelessRun(null, streamRoutes());
    const disabledFetch = fakeFetch({
      [SESSIONS_URL]: sessionsRoute({ creates: [jsonResponse({ detail: "Not Found" }, { status: 404 })] }),
    });
    const client = new SessionClient({ fetchImpl: disabledFetch, timers: instantTimers });
    const disabled = statelessRun(client, streamRoutes());

    await plain.controller.submit("q?");
    await disabled.controller.submit("q?");

    assert.deepEqual(disabled.view.log, plain.view.log);
    assert.deepEqual(
      disabled.fetchImpl.calls.map((c) => [c.url, c.init]),
      plain.fetchImpl.calls.map((c) => [c.url, c.init]),
      "same requests, same headers and bodies",
    );
    for (const call of disabled.fetchImpl.calls) {
      assert.ok(!(SESSION_HEADER in (call.init.headers || {})));
    }
    assert.ok(!disabled.view.kinds().includes("sessionStatus"));
  });
}

test("without a SessionClient, no session request is ever made", async () => {
  const { fetchImpl, view, controller } = statelessRun(null, {
    [STREAM_URL]: () => sseResponse([TRANSCRIPT]),
  });

  await controller.submit("one?");
  await controller.submit("two?");

  assert.equal(fetchImpl.count(SESSIONS_URL), 0);
  assert.ok(!view.kinds().includes("sessionStatus"));
});

test("stateless streams ignore a stray session event", async () => {
  const { view, controller } = statelessRun(null, {
    [STREAM_URL]: sseResponse([TRANSCRIPT, SESSION_EVENT]),
  });

  await controller.submit("q?");

  assert.deepEqual(view.finals(), [["complete", DONE]]);
  assert.ok(!view.kinds().includes("sessionStatus"));
});

test("stateless 409 / 410 responses keep the existing HTTP failure (no session handling)", async () => {
  for (const status of [409, 410]) {
    const { view, fetchImpl, controller } = statelessRun(null, {
      [STREAM_URL]: sessionErrorResponse(status, "x"),
    });

    await controller.submit("q?");

    assert.deepEqual(view.finals(), [["fail", `Request failed (HTTP ${status}).`]]);
    assert.equal(fetchImpl.count(CHAT_URL), 0);
  }
});
