import assert from "node:assert/strict";
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

const { CHAT_URL, STREAM_URL, StreamAvailability, createChatController, INTERRUPTED } = chat;

const CHAT_DATA = { ...DONE, request_id: "chat-1" };

function setup(routes, { availability, now = () => 0, timers = fakeTimers() } = {}) {
  const view = recordingView();
  const fetchImpl = fakeFetch(routes);
  const controller = createChatController({ view, fetchImpl, availability, now, timers });
  return { view, fetchImpl, controller, timers };
}

function assertOneAnswer(view) {
  assert.equal(view.finals().length, 1, "exactly one complete/fail per question");
  assert.equal(view.log.filter(([k]) => k === "end").length, 1);
}

// -----------------------------------------------------------------
// Normal completion
// -----------------------------------------------------------------

test("a successful stream renders deltas, then the final answer once", async () => {
  const { view, fetchImpl, controller } = setup({ [STREAM_URL]: sseResponse([TRANSCRIPT]) });

  await controller.submit("How do I connect to the VPN?");

  assertOneAnswer(view);
  assert.deepEqual(view.finals()[0], ["complete", DONE]);
  assert.equal(view.log.filter(([k]) => k === "delta").length, 3);
  assert.equal(fetchImpl.count(CHAT_URL), 0, "no fallback after success");
});

test("the done payload and the /api/chat payload reach the same renderer", async () => {
  const streamed = setup({ [STREAM_URL]: sseResponse([TRANSCRIPT]) });
  const fallback = setup({
    [STREAM_URL]: jsonResponse({ detail: "Not Found" }, { status: 404 }),
    [CHAT_URL]: jsonResponse(DONE),
  });

  await streamed.controller.submit("q?");
  await fallback.controller.submit("q?");

  assert.deepEqual(streamed.view.finals(), fallback.view.finals());
});

// -----------------------------------------------------------------
// Fallback: once, only before a stream starts
// -----------------------------------------------------------------

test("404 falls back once and later questions skip streaming", async () => {
  const availability = new StreamAvailability();
  const { view, fetchImpl, controller } = setup(
    {
      [STREAM_URL]: () => jsonResponse({ detail: "Not Found" }, { status: 404 }),
      [CHAT_URL]: () => jsonResponse(CHAT_DATA),
    },
    { availability },
  );

  await controller.submit("first?");
  await controller.submit("second?");

  assert.equal(fetchImpl.count(STREAM_URL), 1, "404 remembered for the page session");
  assert.equal(fetchImpl.count(CHAT_URL), 2);
  assert.deepEqual(view.finals(), [["complete", CHAT_DATA], ["complete", CHAT_DATA]]);
});

test("503 falls back and honours Retry-After before streaming again", async () => {
  let clock = 1_000_000;
  const availability = new StreamAvailability();
  let streamCalls = 0;
  const { fetchImpl, controller } = setup(
    {
      [STREAM_URL]: () => {
        streamCalls += 1;
        return streamCalls === 1
          ? jsonResponse({ detail: { message: "busy" } }, { status: 503, headers: { "Retry-After": "7" } })
          : sseResponse([TRANSCRIPT]);
      },
      [CHAT_URL]: () => jsonResponse(CHAT_DATA),
    },
    { availability, now: () => clock },
  );

  await controller.submit("one?"); // 503 → fallback
  clock += 6_999;
  await controller.submit("two?"); // still cooling down → /api/chat directly
  clock += 1;
  await controller.submit("three?"); // cooldown over → stream again

  assert.equal(fetchImpl.count(STREAM_URL), 2);
  assert.equal(fetchImpl.count(CHAT_URL), 2);
});

test("a network failure before headers falls back exactly once", async () => {
  const { view, fetchImpl, controller } = setup({
    [STREAM_URL]: new TypeError("Failed to fetch"),
    [CHAT_URL]: () => jsonResponse(CHAT_DATA),
  });

  await controller.submit("q?");

  assertOneAnswer(view);
  assert.equal(fetchImpl.count(CHAT_URL), 1);
  assert.deepEqual(view.finals()[0], ["complete", CHAT_DATA]);
});

test("when the fallback also fails, the existing /api/chat error message is shown", async () => {
  const { view, fetchImpl, controller } = setup({
    [STREAM_URL]: new TypeError("Failed to fetch"),
    [CHAT_URL]: () => jsonResponse({ detail: { message: "x", request_id: "r" } }, { status: 500 }),
  });

  await controller.submit("q?");

  assertOneAnswer(view);
  assert.deepEqual(view.finals()[0], ["fail", "Request failed (HTTP 500)."]);
  assert.equal(fetchImpl.count(CHAT_URL), 1, "no second fallback");
});

test("unsupported browsers go straight to /api/chat", async () => {
  const { fetchImpl, controller } = setup(
    { [CHAT_URL]: () => jsonResponse(CHAT_DATA) },
    { availability: new StreamAvailability({ supported: false }) },
  );

  await controller.submit("q?");

  assert.equal(fetchImpl.count(STREAM_URL), 0);
  assert.equal(fetchImpl.count(CHAT_URL), 1);
});

for (const status of [500, 422]) {
  test(`HTTP ${status} from the stream never falls back`, async () => {
    const { view, fetchImpl, controller } = setup({
      [STREAM_URL]: new Response("Internal Server Error", { status }),
    });

    await controller.submit("q?");

    assertOneAnswer(view);
    assert.deepEqual(view.finals()[0], ["fail", `Request failed (HTTP ${status}).`]);
    assert.equal(fetchImpl.count(CHAT_URL), 0);
  });
}

// -----------------------------------------------------------------
// Started streams: failure, never fallback, never duplicated output
// -----------------------------------------------------------------

test("an incomplete stream after partial output fails without fallback", async () => {
  const { view, fetchImpl, controller } = setup({
    [STREAM_URL]: sseResponse([frame("start", { request_id: "req-1" }), frame("delta", { text: "Install " })]),
  });

  await controller.submit("q?");

  assertOneAnswer(view);
  assert.deepEqual(view.finals()[0], ["fail", INTERRUPTED]);
  assert.equal(fetchImpl.count(CHAT_URL), 0);
  const kinds = view.log.map(([k]) => k);
  assert.ok(kinds.indexOf("delta") < kinds.indexOf("fail"), "fail replaces the partial output");
});

test("a mid-stream disconnect fails without fallback", async () => {
  const { view, fetchImpl, controller } = setup({
    [STREAM_URL]: sseResponse([frame("start", { request_id: "req-1" }), frame("delta", { text: "a" })], {
      error: new TypeError("network error"),
    }),
  });

  await controller.submit("q?");

  assert.deepEqual(view.finals(), [["fail", INTERRUPTED]]);
  assert.equal(fetchImpl.count(CHAT_URL), 0);
});

test("a delta/answer mismatch shows the generic failure, not the answer", async () => {
  const { view, controller } = setup({
    [STREAM_URL]: sseResponse([
      frame("start", { request_id: "req-1" }),
      frame("delta", { text: "something else" }),
      frame("done", DONE),
    ]),
  });

  await controller.submit("q?");

  assert.deepEqual(view.finals(), [["fail", INTERRUPTED]]);
});

test("an error event shows the client-side message, never server text", async () => {
  const { view, controller } = setup({
    [STREAM_URL]: sseResponse([
      frame("start", { request_id: "req-1" }),
      frame("delta", { text: "partial" }),
      frame("error", { request_id: "req-1", code: "internal_error", message: "psycopg2 password=hunter2" }),
    ]),
  });

  await controller.submit("q?");

  assert.deepEqual(view.finals(), [["fail", "Unable to process the question. Please try again."]]);
  assert.ok(!JSON.stringify(view.log).includes("hunter2"));
});

// -----------------------------------------------------------------
// Duplicate submissions
// -----------------------------------------------------------------

test("a second submission while one is running is ignored", async () => {
  const stream = manualResponse();
  const { view, fetchImpl, controller } = setup({ [STREAM_URL]: () => stream.response });

  const first = controller.submit("first?");
  await tick();
  assert.equal(controller.busy, true);
  assert.equal(await controller.submit("second?"), false);

  stream.send(TRANSCRIPT);
  stream.close();
  await first;

  assert.equal(view.attempts, 1);
  assert.equal(fetchImpl.count(STREAM_URL), 1);
  assert.equal(controller.busy, false);
  assertOneAnswer(view);
});

test("a view error still ends the question exactly once", async () => {
  const view = recordingView();
  const begin = view.begin;
  view.begin = (question) => {
    const attempt = begin(question);
    return { ...attempt, delta: () => { throw new Error("render bug"); } };
  };
  let cancelled = false;
  const body = new ReadableStream({
    start(c) {
      c.enqueue(new TextEncoder().encode(TRANSCRIPT.split("event: delta")[0] + "event: delta\ndata: {\"text\":\"a\"}\n\n"));
    },
    cancel: () => { cancelled = true; },
  });
  const response = new Response(body, { headers: { "Content-Type": "text/event-stream" } });
  const controller = createChatController({ view, fetchImpl: fakeFetch({ [STREAM_URL]: response }), timers: fakeTimers() });

  await controller.submit("q?");

  assert.deepEqual(view.finals(), [["fail", chat.GENERIC_ERROR]]);
  assert.equal(controller.busy, false);
  assert.ok(cancelled, "the connection is released");
});

test("the abort signal can be supplied per question (page restored from cache)", async () => {
  let pageAbort = new AbortController();
  pageAbort.abort(); // the page was hidden earlier
  pageAbort = new AbortController(); // and restored: a fresh signal
  const view = recordingView();
  const controller = createChatController({
    view,
    fetchImpl: fakeFetch({ [STREAM_URL]: sseResponse([TRANSCRIPT]) }),
    timers: fakeTimers(),
    signal: () => pageAbort.signal,
  });

  await controller.submit("q?");

  assert.deepEqual(view.finals(), [["complete", DONE]]);
});

test("a view.begin() failure does not lock later submissions", async () => {
  const view = recordingView();
  const begin = view.begin;
  let failNext = true;
  view.begin = (question) => {
    if (failNext) {
      failNext = false;
      throw new Error("render bug");
    }
    return begin(question);
  };
  const fetchImpl = fakeFetch({ [STREAM_URL]: () => sseResponse([TRANSCRIPT]) });
  const controller = createChatController({ view, fetchImpl, timers: fakeTimers() });

  await assert.rejects(controller.submit("first?"), /render bug/);
  assert.equal(controller.busy, false);
  assert.equal(fetchImpl.count(STREAM_URL), 0, "no request for the failed question");
  assert.deepEqual(view.log, [["end"]], "the view is reset");

  assert.equal(await controller.submit("second?"), true);
  assert.deepEqual(view.finals(), [["complete", DONE]]);
  assert.equal(controller.busy, false);
});

test("page hide before response headers fails without /api/chat fallback", async () => {
  const pageAbort = new AbortController();
  let streamRequested;
  const requested = new Promise((resolve) => { streamRequested = resolve; });
  const fetchImpl = fakeFetch({
    // Like browser fetch: never answers, rejects with AbortError on abort.
    [STREAM_URL]: (init) =>
      new Promise((_resolve, reject) => {
        streamRequested();
        init.signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
      }),
    [CHAT_URL]: () => jsonResponse(CHAT_DATA),
  });
  const view = recordingView();
  const controller = createChatController({
    view,
    fetchImpl,
    timers: fakeTimers(),
    signal: () => pageAbort.signal,
  });

  const submission = controller.submit("q?");
  await requested;
  pageAbort.abort(); // pagehide

  await submission;
  assert.deepEqual(view.finals(), [["fail", INTERRUPTED]]);
  assert.equal(fetchImpl.count(CHAT_URL), 0, "no fallback");
  assert.equal(controller.busy, false);
});
