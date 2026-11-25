// Security S1-5 Phase C: static/js/app.js document upload messages.
// The real app.js against a minimal fake DOM and a fake fetch: structured
// errors ({ detail: { code, message } }) show their message; plain string
// details are shown exactly as before; anything else falls back.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { createDocument, h } from "./fake-dom.mjs";
import { chat, fakeFetch, jsonResponse, tick } from "./helpers.mjs";

const APP_SOURCE = readFileSync(new URL("../../static/js/app.js", import.meta.url), "utf-8");
const INGEST_URL = "/api/ingest";
const NOT_FOUND = () => jsonResponse({ detail: "Not Found" }, { status: 404 });

function loadApp(ingestRoute) {
  const document = createDocument();
  document.documentElement.setAttribute("lang", "en");
  const adminKey = h(document, "input#adminKey");
  const fileInput = h(document, "input#fileInput");
  const uploadStatus = h(document, "p#uploadStatus.upload-status");
  const uploadBtn = h(document, "button#uploadBtn", ["Index document"]);
  const fileName = h(document, "span#fileName", ["Choose a file to upload"]);
  const modal = h(document, "div#uploadModal.hidden", [adminKey, fileInput, fileName, uploadBtn, uploadStatus]);
  const form = h(document, "form#chatForm.composer", [
    h(document, "textarea#question"),
    h(document, "button#sendBtn", [h(document, "span", ["Ask agent"])]),
  ]);
  document.body.append(
    h(document, "section#chat"),
    h(document, "div.composer-area", [form]),
    h(document, "span#characterCount"),
    h(document, "div#trace"),
    h(document, "span#traceCount"),
    h(document, "span#runStatus"),
    h(document, "div#runProgressBar"),
    h(document, "span#runStepCount"),
    h(document, "span#sourceUsed"),
    h(document, "p#sourceDescription"),
    h(document, "button#openUpload"),
    modal,
  );

  const fetchImpl = fakeFetch({
    [chat.SESSIONS_URL]: NOT_FOUND,
    "/api/voice/status": NOT_FOUND,
    "/api/voice/transcription/status": NOT_FOUND,
    [INGEST_URL]: ingestRoute,
  });
  const window = {
    document,
    fetch: fetchImpl,
    TextDecoder,
    AbortController,
    Response,
    setTimeout,
    clearTimeout,
    location: { origin: "http://localhost", reload() {} },
    addEventListener() {},
    BluePeakChat: chat,
  };
  new Function("window", "document", APP_SOURCE)(window, document);

  return {
    fetchImpl,
    uploadStatus,
    adminKey,
    async upload({ key = "an-admin-key-0123456789abcdefghijkl", file = new File(["# VPN\n"], "vpn.md") } = {}) {
      adminKey.value = key;
      fileInput.files = file ? [file] : [];
      uploadBtn.click();
      for (let round = 0; round < 3; round++) {
        for (let i = 0; i < 20; i++) await tick();
        await new Promise((resolve) => setTimeout(resolve, 5));
      }
    },
    status: () => uploadStatus.textContent,
    kind: () => ["success", "error"].filter((name) => uploadStatus.classList.contains(name)),
  };
}

const structured = (status, code, message, headers = {}) => () =>
  jsonResponse({ detail: { code, message } }, { status, headers });

test("a structured 401 shows its message", async () => {
  const page = loadApp(structured(401, "admin_auth_required", "A valid admin key is required."));
  await page.upload();
  assert.equal(page.status(), "A valid admin key is required.");
  assert.deepEqual(page.kind(), ["error"]);
  assert.equal(page.fetchImpl.count(INGEST_URL), 1);
});

test("structured throttling, busy, size and timeout errors show their messages", async () => {
  const cases = [
    [429, "admin_auth_throttled", "Too many failed admin key attempts. Please try again later."],
    [503, "ingest_busy", "Another document is being indexed. Please try again shortly."],
    [503, "ingest_unavailable", "Document ingestion is not available."],
    [413, "upload_too_large", "File exceeds the 10 MB upload limit."],
    [408, "upload_timeout", "The upload took too long."],
    [415, "unsupported_media_type", "Send the document as multipart/form-data."],
  ];
  for (const [status, code, message] of cases) {
    const page = loadApp(structured(status, code, message));
    await page.upload();
    assert.equal(page.status(), message, code);
    assert.deepEqual(page.kind(), ["error"], code);
  }
});

test("a plain string detail is still shown exactly as before", async () => {
  const page = loadApp(() => jsonResponse({ detail: "The uploaded file is empty." }, { status: 400 }));
  await page.upload();
  assert.equal(page.status(), "The uploaded file is empty.");
});

test("a top-level message is still used when there is no detail", async () => {
  const page = loadApp(() => jsonResponse({ message: "Something specific." }, { status: 500 }));
  await page.upload();
  assert.equal(page.status(), "Something specific.");
});

test("unusable error bodies fall back to the HTTP status", async () => {
  for (const body of [{ detail: { code: "x" } }, { detail: { message: "   " } }, { detail: 42 }, {}, { detail: ["a"] }]) {
    const page = loadApp(() => jsonResponse(body, { status: 502 }));
    await page.upload();
    assert.equal(page.status(), "Upload failed (HTTP 502).", JSON.stringify(body));
  }
  const page = loadApp(() => new Response("not json", { status: 503 }));
  await page.upload();
  assert.equal(page.status(), "Upload failed (HTTP 503).");
});

test("a successful upload is unchanged", async () => {
  const page = loadApp(() => jsonResponse({ message: "Document indexed successfully.", file: "vpn.md", chunks: 3, ids_created: 3 }));
  await page.upload();
  assert.equal(page.status(), "Successfully indexed vpn.md. 3 chunks added to the knowledge base.");
  assert.deepEqual(page.kind(), ["success"]);
  assert.equal(page.adminKey.value, "");
  const [call] = page.fetchImpl.calls.filter((entry) => entry.url === INGEST_URL);
  assert.equal(call.init.method, "POST");
  assert.equal(call.init.headers["X-Admin-Key"], "an-admin-key-0123456789abcdefghijkl");
  assert.ok(call.init.body instanceof FormData);
});

test("the message is shown as text, never as markup", async () => {
  const page = loadApp(structured(401, "admin_auth_required", "<img src=x onerror=alert(1)>"));
  await page.upload();
  assert.equal(page.status(), "<img src=x onerror=alert(1)>");
  assert.equal(page.uploadStatus.children.length, 1); // one text node, no element
});
