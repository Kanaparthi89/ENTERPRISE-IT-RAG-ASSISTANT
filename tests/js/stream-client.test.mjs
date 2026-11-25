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
  sseResponse,
  tick,
  transcript,
} from "./helpers.mjs";

const { STREAM_URL, IDLE_TIMEOUT_MS, TOTAL_TIMEOUT_MS, parseRetryAfter, streamChat } = chat;

function run(response, options = {}) {
  const seen = { deltas: [], traces: [], starts: [] };
  const fetchImpl = fakeFetch({ [STREAM_URL]: response });
  const promise = streamChat({
    question: "How do I connect to the VPN?",
    fetchImpl,
    timers: options.timers ?? fakeTimers(),
    now: options.now ?? (() => 0),
    signal: options.signal,
    handlers: {
      onStart: (id) => seen.starts.push(id),
      onDelta: (text) => seen.deltas.push(text),
      onTrace: (step, index) => seen.traces.push([step, index]),
    },
  });
  return { promise, seen, fetchImpl };
}

// -----------------------------------------------------------------
// Normal completion and terminal events
// -----------------------------------------------------------------

test("streams the backend transcript to completion", async () => {
  const { promise, seen, fetchImpl } = run(sseResponse([TRANSCRIPT]));

  const outcome = await promise;

  assert.equal(outcome.type, "done");
  assert.deepEqual(outcome.data, DONE);
  assert.deepEqual(seen.starts, ["req-1"]);
  assert.deepEqual(seen.deltas, ["\n Install GlobalProtect — ", "then sign in.\nGrüße ✓", "  \n"]);
  assert.deepEqual(seen.traces, [["Router → KB", 0], ["Answer generation → PRIVATE KB", 1]]);
  const [call] = fetchImpl.calls;
  assert.equal(call.init.method, "POST");
  assert.equal(call.init.headers.Accept, "text/event-stream");
  assert.deepEqual(JSON.parse(call.init.body), { question: "How do I connect to the VPN?" });
});

test("a delta/answer mismatch is treated as interrupted", async () => {
  const { promise } = run(sseResponse(transcript(["Install ", "something else"])));

  assert.deepEqual(await promise, { type: "failed", reason: "mismatch" });
});

test("an error event returns its code and never server text", async () => {
  const chunks = [
    frame("start", { request_id: "req-1", version: 1 }),
    frame("delta", { text: "partial" }),
    frame("error", { request_id: "req-1", code: "timeout", message: "Traceback: secret" }),
  ];

  const outcome = await run(sseResponse(chunks)).promise;

  assert.deepEqual(outcome, { type: "error", code: "timeout" });
  assert.equal(chat.errorMessage("timeout"), "The request took too long. Please try again.");
  assert.equal(chat.errorMessage("weird"), chat.GENERIC_ERROR);
  assert.equal(chat.errorMessage("__proto__"), chat.GENERIC_ERROR);
});

test("events after the terminal event are ignored", async () => {
  const chunks = [...transcript(["Install GlobalProtect — then sign in.\nGrüße ✓"]), frame("delta", { text: "late" }), frame("error", { code: "x" })];
  const { promise, seen } = run(sseResponse(chunks));

  assert.equal((await promise).type, "done");
  assert.ok(!seen.deltas.includes("late"));
});

test("unknown event names are ignored", async () => {
  const chunks = [frame("start", { request_id: "req-1" }), frame("progress", { pct: 5 }), ...transcript(["Install GlobalProtect — then sign in.\nGrüße ✓"]).slice(1)];

  assert.equal((await run(sseResponse(chunks)).promise).type, "done");
});

// -----------------------------------------------------------------
// Incomplete, interrupted and malformed streams: failed, never fallback
// -----------------------------------------------------------------

test("a stream that ends without done or error has failed", async () => {
  const chunks = [frame("start", { request_id: "req-1" }), frame("delta", { text: "partial" })];

  assert.deepEqual(await run(sseResponse(chunks)).promise, { type: "failed", reason: "interrupted" });
});

test("a stream that ends before start has failed (no fallback)", async () => {
  assert.deepEqual(await run(sseResponse([])).promise, { type: "failed", reason: "interrupted" });
});

test("a network error mid-stream has failed (no fallback)", async () => {
  const chunks = [frame("start", { request_id: "req-1" }), frame("delta", { text: "partial" })];
  const response = sseResponse(chunks, { error: new TypeError("network error") });

  assert.deepEqual(await run(response).promise, { type: "failed", reason: "interrupted" });
});

test("a network error after done still completes", async () => {
  const response = sseResponse(transcript(["Install GlobalProtect — then sign in.\nGrüße ✓"]), { error: new TypeError("reset") });

  assert.equal((await run(response).promise).type, "done");
});

for (const [name, chunks] of [
  ["invalid JSON", [frame("start", { request_id: "req-1" }), "event: delta\ndata: {not json\n\n"]],
  ["delta before start", [frame("delta", { text: "a" })]],
  ["non-string delta", [frame("start", { request_id: "req-1" }), frame("delta", { text: 5 })]],
  ["start without request_id", [frame("start", { version: 1 })]],
  ["duplicate start", [frame("start", { request_id: "req-1" }), frame("start", { request_id: "req-1" })]],
  ["done with another request_id", transcript(["Install GlobalProtect — then sign in.\nGrüße ✓"], { ...DONE, request_id: "other" })],
  ["done without answer", [frame("start", { request_id: "req-1" }), frame("done", { request_id: "req-1" })]],
  ["array payload", [frame("start", { request_id: "req-1" }), frame("trace", [1, 2])]],
]) {
  test(`malformed stream (${name}) has failed and is cancelled`, async () => {
    const outcome = await run(sseResponse(chunks)).promise;

    assert.deepEqual(outcome, { type: "failed", reason: "malformed" });
  });
}

// -----------------------------------------------------------------
// Watchdogs and cancellation
// -----------------------------------------------------------------

test("the idle watchdog resets on any bytes, including keepalive comments", async () => {
  const timers = fakeTimers();
  const stream = manualResponse();
  const { promise } = run(stream.response, { timers });
  await tick();

  stream.send(frame("start", { request_id: "req-1" }));
  for (let i = 0; i < 5; i++) {
    await tick();
    stream.send(": keepalive\n\n"); // no tokens, still healthy
  }
  await tick();
  assert.equal(timers.count(), 2, "one idle and one total timer pending");

  for (const text of ["Install GlobalProtect — then sign in.\nGrüße ✓"]) stream.send(frame("delta", { text }));
  stream.send(frame("done", DONE));
  stream.close();

  assert.equal((await promise).type, "done");
  assert.equal(timers.count(), 0, "timers cleared");
});

test("no bytes for the idle period fails the stream", async () => {
  const timers = fakeTimers();
  const stream = manualResponse();
  const { promise } = run(stream.response, { timers });
  await tick();
  stream.send(frame("start", { request_id: "req-1" }));
  await tick();

  timers.fire(IDLE_TIMEOUT_MS);

  assert.deepEqual(await promise, { type: "failed", reason: "idle_timeout" });
});

test("the overall watchdog fails a stream that never finishes", async () => {
  const timers = fakeTimers();
  const stream = manualResponse();
  const { promise } = run(stream.response, { timers });
  await tick();
  stream.send(frame("start", { request_id: "req-1" }));
  await tick();

  timers.fire(TOTAL_TIMEOUT_MS);

  assert.deepEqual(await promise, { type: "failed", reason: "total_timeout" });
});

test("a watchdog firing before headers is a failure, not a fallback", async () => {
  const timers = fakeTimers();
  const fetchImpl = (_url, init) =>
    new Promise((_resolve, reject) =>
      init.signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError"))),
    );
  const promise = streamChat({ question: "q?", fetchImpl, timers });
  await tick();

  timers.fire(IDLE_TIMEOUT_MS);

  assert.deepEqual(await promise, { type: "failed", reason: "idle_timeout" });
});

test("an external abort (page hide) fails the stream", async () => {
  const pageAbort = new AbortController();
  const stream = manualResponse();
  const { promise } = run(stream.response, { signal: pageAbort.signal });
  await tick();
  stream.send(frame("start", { request_id: "req-1" }));
  await tick();

  pageAbort.abort();

  assert.deepEqual(await promise, { type: "failed", reason: "aborted" });
});

// -----------------------------------------------------------------
// Responses that permit the single fallback, and those that do not
// -----------------------------------------------------------------

test("404 falls back (streaming disabled)", async () => {
  const outcome = await run(jsonResponse({ detail: "Not Found" }, { status: 404 })).promise;

  assert.deepEqual(outcome, { type: "fallback", reason: "not_found" });
});

test("503 falls back with the Retry-After delay", async () => {
  const response = jsonResponse({ detail: { message: "busy" } }, { status: 503, headers: { "Retry-After": "7" } });

  assert.deepEqual(await run(response).promise, { type: "fallback", reason: "busy", retryAfterMs: 7000 });
});

test("a network failure before headers falls back", async () => {
  const outcome = await run(new TypeError("Failed to fetch")).promise;

  assert.deepEqual(outcome, { type: "fallback", reason: "network" });
});

test("a 200 response that is not an event stream falls back and is cancelled", async () => {
  let cancelled = false;
  const body = new ReadableStream({ cancel: () => { cancelled = true; } });
  const response = new Response(body, { status: 200, headers: { "Content-Type": "text/html" } });

  assert.deepEqual(await run(response).promise, { type: "fallback", reason: "not_sse" });
  await tick();
  assert.ok(cancelled);
});

for (const status of [500, 422, 401, 502]) {
  test(`HTTP ${status} fails without fallback (intentional: avoids duplicate execution)`, async () => {
    const response = new Response("Internal Server Error", { status, headers: { "Content-Type": "text/plain" } });

    assert.deepEqual(await run(response).promise, { type: "failed", reason: "http_status", status });
  });
}

test("Retry-After accepts seconds or an HTTP date and is clamped", () => {
  const now = Date.parse("2026-10-30T12:00:00Z");

  assert.equal(parseRetryAfter("7", now), 7000);
  assert.equal(parseRetryAfter("Fri, 30 Oct 2026 12:00:20 GMT", now), 20000);
  assert.equal(parseRetryAfter(null, now), 5000);
  assert.equal(parseRetryAfter("soon", now), 5000);
  assert.equal(parseRetryAfter("0", now), 1000);
  assert.equal(parseRetryAfter("99999", now), 300000);
  assert.equal(parseRetryAfter("Fri, 30 Oct 2026 11:00:00 GMT", now), 1000);
});
