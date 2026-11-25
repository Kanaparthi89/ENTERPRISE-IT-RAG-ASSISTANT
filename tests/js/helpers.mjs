// Shared helpers for the chat-stream tests (node:test, no dependencies).
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
export const chat = require("../../static/js/chat-stream.js");

// Real SSE bytes produced by the backend's format_sse(); kept in sync by
// tests/test_frontend_assets.py.
export const TRANSCRIPT = readFileSync(
  new URL("./fixtures/stream-transcript.sse", import.meta.url),
  "utf-8",
);

export const DONE = {
  request_id: "req-1",
  answer: "Install GlobalProtect — then sign in.\nGrüße ✓",
  source_used: "private_kb",
  trace: ["Router → KB", "Answer generation → PRIVATE KB"],
  citations: [{ title: "handbook.md", url: "", type: "private_kb" }],
  rewritten_query: "How do I connect to the VPN?",
};

export function frame(event, data) {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

export function transcript(deltas, done = DONE, { requestId = "req-1" } = {}) {
  return [
    frame("start", { request_id: requestId, version: 1 }),
    frame("trace", { index: 0, step: "Router → KB" }),
    ...deltas.map((text) => frame("delta", { text })),
    frame("done", done),
  ];
}

const encoder = new TextEncoder();

/** A Response whose body yields the given chunks (strings or bytes). */
export function sseResponse(chunks, { status = 200, headers = {}, error } = {}) {
  let index = 0;
  const body = new ReadableStream({
    pull(controller) {
      if (index < chunks.length) {
        const chunk = chunks[index++];
        controller.enqueue(typeof chunk === "string" ? encoder.encode(chunk) : chunk);
      } else if (error) {
        controller.error(error);
      } else {
        controller.close();
      }
    },
  });
  return new Response(body, {
    status,
    headers: { "Content-Type": "text/event-stream; charset=utf-8", ...headers },
  });
}

/** A Response whose body the test feeds by hand. */
export function manualResponse() {
  let controller;
  const body = new ReadableStream({
    start(c) {
      controller = c;
    },
  });
  const cancelled = { value: false };
  const response = new Response(body, {
    status: 200,
    headers: { "Content-Type": "text/event-stream; charset=utf-8" },
  });
  return {
    response,
    cancelled,
    send: (text) => controller.enqueue(encoder.encode(text)),
    close: () => controller.close(),
    fail: (error) => controller.error(error),
  };
}

export function jsonResponse(data, { status = 200, headers = {} } = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  });
}

/** fetch double: routes by URL; records every call. */
export function fakeFetch(routes) {
  const calls = [];
  const fetchImpl = async (url, init = {}) => {
    calls.push({ url, init });
    const route = routes[url];
    if (!route) throw new Error(`unexpected fetch ${url}`);
    const result = typeof route === "function" ? await route(init, calls.length) : route;
    if (result instanceof Error) throw result;
    return result;
  };
  fetchImpl.calls = calls;
  fetchImpl.count = (url) => calls.filter((call) => call.url === url).length;
  return fetchImpl;
}

/** Timers that only fire when the test says so. */
export function fakeTimers() {
  let nextId = 0;
  const pending = new Map();
  return {
    setTimeout(fn, ms) {
      pending.set(++nextId, { fn, ms });
      return nextId;
    },
    clearTimeout(id) {
      pending.delete(id);
    },
    fire(ms) {
      for (const [id, timer] of [...pending]) {
        if (timer.ms === ms) {
          pending.delete(id);
          timer.fn();
        }
      }
    },
    count: () => pending.size,
  };
}

export const tick = () => new Promise((resolve) => setImmediate(resolve));

/** Records everything the controller asks the view to do. */
export function recordingView() {
  const log = [];
  const view = {
    log,
    attempts: 0,
    begin(question) {
      view.attempts += 1;
      log.push(["begin", question]);
      return {
        trace: (step, index) => log.push(["trace", step, index]),
        delta: (text) => log.push(["delta", text]),
        complete: (data) => log.push(["complete", data]),
        fail: (message) => log.push(["fail", message]),
      };
    },
    end() {
      log.push(["end"]);
    },
    finals: () => log.filter(([kind]) => kind === "complete" || kind === "fail"),
  };
  return view;
}
