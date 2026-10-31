import assert from "node:assert/strict";
import { test } from "node:test";

import { chat, TRANSCRIPT } from "./helpers.mjs";

const { SSEParser, SSEProtocolError, pyStrip } = chat;

function parseAll(chunks) {
  const parser = new SSEParser();
  const events = [];
  for (const chunk of chunks) events.push(...parser.push(chunk));
  parser.finish();
  return events;
}

function decodeChunks(byteChunks) {
  const decoder = new TextDecoder("utf-8");
  return [...byteChunks.map((bytes) => decoder.decode(bytes, { stream: true })), decoder.decode()];
}

test("parses the backend transcript into events", () => {
  const events = parseAll([TRANSCRIPT]);

  assert.deepEqual(
    events.map((e) => e.event),
    ["start", "trace", "delta", "delta", "delta", "trace", "done"],
  );
  assert.equal(JSON.parse(events[3].data).text, "then sign in.\nGrüße ✓");
});

test("every byte split point gives the same events, including inside UTF-8", () => {
  const bytes = new TextEncoder().encode(TRANSCRIPT);
  const expected = parseAll([TRANSCRIPT]);

  for (let split = 1; split < bytes.length; split++) {
    const chunks = decodeChunks([bytes.slice(0, split), bytes.slice(split)]);
    assert.deepEqual(parseAll(chunks), expected, `split at byte ${split}`);
  }
});

test("one byte at a time gives the same events", () => {
  const bytes = new TextEncoder().encode(TRANSCRIPT);
  const chunks = decodeChunks(Array.from(bytes, (b) => Uint8Array.of(b)));

  assert.deepEqual(parseAll(chunks), parseAll([TRANSCRIPT]));
});

test("CRLF, CR and LF line endings are equivalent, even split between chunks", () => {
  const lf = "event: delta\ndata: {\"text\":\"a\"}\n\n";
  const crlf = lf.replaceAll("\n", "\r\n");
  const cr = lf.replaceAll("\n", "\r");
  const expected = parseAll([lf]);

  assert.deepEqual(parseAll([crlf]), expected);
  assert.deepEqual(parseAll([cr + "\r"]), expected);
  for (let split = 1; split < crlf.length; split++) {
    assert.deepEqual(parseAll([crlf.slice(0, split), crlf.slice(split)]), expected);
  }
});

test("comments are skipped; multiple data lines are joined; leading space stripped once", () => {
  const events = parseAll([": keepalive\n\n", "event: x\ndata:  two\ndata:one\nid: 7\nretry: 5\n\n"]);

  assert.deepEqual(events, [{ event: "x", data: " two\none" }]);
});

test("an event without its terminating blank line is dropped at end of stream", () => {
  const events = parseAll([
    'event: delta\ndata: {"text":"a"}\n\n',
    'event: done\ndata: {"answer":"a"}\n',
  ]);

  assert.deepEqual(events.map((e) => e.event), ["delta"]);
});

test("an oversized event is a protocol error", () => {
  const parser = new SSEParser({ maxEventChars: 100 });

  assert.throws(() => parser.push(`data: ${"x".repeat(200)}\n`), SSEProtocolError);
  assert.throws(() => new SSEParser({ maxEventChars: 100 }).push("x".repeat(200)), SSEProtocolError);
  assert.throws(() => {
    const p = new SSEParser({ maxEventChars: 100 });
    for (let i = 0; i < 10; i++) p.push(`data: ${"x".repeat(20)}\n`);
  }, SSEProtocolError);
});

test("pyStrip matches Python str.strip(), not String.prototype.trim()", () => {
  assert.equal(pyStrip("\x1c\x85 a b \n　"), "a b");
  assert.equal(pyStrip("﻿a"), "﻿a"); // JS trim() would remove U+FEFF; Python does not
  assert.equal(pyStrip("  "), "");
});
