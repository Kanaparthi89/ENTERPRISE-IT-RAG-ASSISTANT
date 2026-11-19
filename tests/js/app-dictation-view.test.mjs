// Voice V3b: static/js/app.js dictation, against the real recorder.js,
// chat-stream.js and voice.js, with a fake microphone, MediaRecorder, fetch,
// timers and speech engine, and a minimal fake DOM. No browser, network or
// real microphone is used.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { test } from "node:test";

import { createDocument, h } from "./fake-dom.mjs";
import { chat, DONE, fakeFetch, fakeTimers, jsonResponse, tick } from "./helpers.mjs";

const require = createRequire(import.meta.url);
const recorderModule = require("../../static/js/recorder.js");
const voiceModule = require("../../static/js/voice.js");
const APP_SOURCE = readFileSync(new URL("../../static/js/app.js", import.meta.url), "utf-8");

const { CHAT_URL, SESSIONS_URL, SESSION_HEADER } = chat;
const { ERROR_MESSAGES, MAX_DURATION_MS, TRANSCRIBE_URL, UPLOAD_TIMEOUT_MS } = recorderModule;
const { VOICE_LOAD_TIMEOUT_MS } = voiceModule;

// Every test is bounded: a regression that leaves a session pending fails fast.
const T = Object.freeze({ timeout: 10_000 });

const STT_URL = "/api/voice/transcription/status";
const TTS_URL = "/api/voice/status";
const PROBE_TIMEOUT_MS = 5_000;
const TRANSCRIPT = "How do I reset my VPN token?";
const WEBM = "audio/webm;codecs=opus";
const TEXT = {
  requesting: "Allow microphone access to dictate.",
  recording: "Recording… Select Stop dictation to finish (up to 60 seconds).",
  uploading: "Transcribing…",
  added: "Transcript added to your question. Review it before you send it.",
  tooLong: "The transcript wasn't added: your question would be longer than 3,000 characters.",
};
const LABEL = {
  idle: "Dictate",
  requesting: "Cancel dictation",
  recording: "Stop dictation",
  uploading: "Cancel transcription",
};

const STT_ON = () => jsonResponse({ stt: true });
const TTS_ON = () => jsonResponse({ tts: true });
const NOT_FOUND = () => jsonResponse({ detail: "Not Found" }, { status: 404 });
const transcribed = (text = TRANSCRIPT) => () => jsonResponse({ text });
const answer = (text = DONE.answer) => jsonResponse({ ...DONE, answer: text });
const LOCAL_EN = { name: "Samantha", lang: "en-US", localService: true, default: true };

// ---------------------------------------------------------------------------
// Fakes: microphone, MediaRecorder, speech
// ---------------------------------------------------------------------------

class FakeTrack {
  constructor() {
    this.stopCount = 0;
    this.onended = null;
  }

  stop() {
    this.stopCount += 1;
  }
}

class FakeStream {
  constructor() {
    this.tracks = [new FakeTrack()];
  }

  getTracks() {
    return [...this.tracks];
  }

  allStopped() {
    return this.tracks.every((track) => track.stopCount > 0);
  }
}

/** navigator.mediaDevices double; ``manual`` holds the permission prompt open. */
function fakeDevices({ manual = false, error = null } = {}) {
  const devices = {
    calls: [],
    streams: [],
    pending: null,
    getUserMedia(constraints) {
      devices.calls.push(constraints);
      if (error) return Promise.reject(Object.assign(new Error(error), { name: error }));
      if (manual) {
        return new Promise((resolve, reject) => {
          devices.pending = { resolve, reject };
        });
      }
      const stream = new FakeStream();
      devices.streams.push(stream);
      return Promise.resolve(stream);
    },
    grant() {
      const stream = new FakeStream();
      devices.streams.push(stream);
      devices.pending.resolve(stream);
      return stream;
    },
  };
  return devices;
}

function fakeMediaRecorder(types = [WEBM]) {
  return class FakeMediaRecorder {
    static instances = [];

    static isTypeSupported(type) {
      return types.includes(type);
    }

    constructor(stream, options) {
      this.stream = stream;
      this.mimeType = options.mimeType;
      this.state = "inactive";
      this.ondataavailable = null;
      this.onstop = null;
      this.onerror = null;
      FakeMediaRecorder.instances.push(this);
    }

    start() {
      this.state = "recording";
    }

    stop() {
      if (this.state === "inactive") return;
      this.state = "inactive";
      queueMicrotask(() => this.onstop?.({}));
    }

    emit(bytes) {
      this.ondataavailable?.({ data: new Blob([new Uint8Array(bytes)]) });
    }
  };
}

class FakeUtterance {
  constructor(text) {
    this.text = text;
    this.voice = null;
    this.lang = "";
    this.volume = 1;
    this.onend = null;
    this.onerror = null;
  }
}

class FakeSynth {
  constructor(voices) {
    this.voices = voices;
    this.queue = [];
    this.spoken = [];
    this.cancelCount = 0;
  }

  getVoices() {
    return this.voices;
  }

  speak(utterance) {
    this.queue.push(utterance);
    this.spoken.push(utterance);
  }

  cancel() {
    this.cancelCount += 1;
    const dropped = this.queue;
    this.queue = [];
    for (const utterance of dropped) utterance.onerror?.({ error: "interrupted" });
  }

  addEventListener() {}

  removeEventListener() {}

  texts() {
    return this.spoken.filter((u) => u.text !== "").map((u) => u.text);
  }

  finishAll() {
    while (this.queue.length) this.queue.shift().onend?.({});
  }
}

/** A transcription the test answers by hand; aborting it rejects like a browser. */
function heldUpload() {
  const held = { signals: [], resolve: null };
  held.route = (init) =>
    new Promise((resolve, reject) => {
      held.signals.push(init.signal);
      held.resolve = resolve;
      init.signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
    });
  return held;
}

// ---------------------------------------------------------------------------
// The page
// ---------------------------------------------------------------------------

function loadApp(
  routes = {},
  {
    secure = true,
    devices = fakeDevices(),
    types = [WEBM],
    mediaRecorder = true,
    recorderLoaded = true,
    mediaDevicesPresent = true,
    speech = false,
  } = {},
) {
  const document = createDocument();
  document.documentElement.setAttribute("lang", "en");

  const toggle = h(document, "button#voiceToggle.text-button.voice-toggle", ["Read answers aloud"]);
  toggle.setAttribute("aria-pressed", "false");
  toggle.hidden = true;
  const stop = h(document, "button#voiceStop.text-button.voice-stop", ["Stop speaking"]);
  stop.hidden = true;
  const voiceStatus = h(document, "p#voiceStatus.voice-status");

  const button = h(document, "button#dictateBtn.dictate-btn", [h(document, "span.dictate-label", ["Dictate"])]);
  button.type = "button";
  button.dataset.state = "idle";
  button.setAttribute("aria-describedby", "dictationNote");
  button.hidden = true;
  const note = h(document, "p#dictationNote.dictation-note", ["Dictation sends your recording to OpenAI."]);
  note.hidden = true;
  const status = h(document, "p#dictationStatus.dictation-status");
  status.setAttribute("role", "status");

  const modal = h(document, "div#uploadModal.hidden", [h(document, "input#adminKey")]);
  const form = h(document, "form#chatForm.composer", [
    h(document, "textarea#question"),
    h(document, "div.composer-actions", [
      button,
      h(document, "span#characterCount"),
      h(document, "button#sendBtn", [h(document, "span", ["Ask agent"])]),
    ]),
  ]);
  document.body.append(
    h(document, "div.conversation-actions", [toggle, stop, h(document, "button#clearChat")]),
    voiceStatus,
    h(document, "section#chat"),
    h(document, "div.composer-area", [form, note, status, h(document, "p.disclaimer")]),
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

  const fetchImpl = fakeFetch({ [SESSIONS_URL]: NOT_FOUND, [TTS_URL]: NOT_FOUND, ...routes });
  const timers = fakeTimers();
  // Fake time for the probe timeout, the recording and upload limits and the
  // voice wait; the page's other timers (frames, retries) run for real.
  const faked = new Set([PROBE_TIMEOUT_MS, MAX_DURATION_MS, UPLOAD_TIMEOUT_MS, VOICE_LOAD_TIMEOUT_MS]);
  const windowListeners = new Map();
  const MediaRecorder = fakeMediaRecorder(types);
  const synth = new FakeSynth([LOCAL_EN]);

  const window = {
    document,
    fetch: fetchImpl,
    TextDecoder,
    AbortController,
    Response,
    isSecureContext: secure,
    navigator: mediaDevicesPresent ? { mediaDevices: devices } : {},
    setTimeout: (fn, ms) => (faked.has(ms) ? timers.setTimeout(fn, ms) : setTimeout(fn, ms)),
    clearTimeout: (id) => (typeof id === "number" ? timers.clearTimeout(id) : clearTimeout(id)),
    location: { origin: "http://localhost", reload() {} },
    addEventListener(type, listener) {
      if (!windowListeners.has(type)) windowListeners.set(type, []);
      windowListeners.get(type).push(listener);
    },
    BluePeakChat: chat,
    BluePeakVoice: voiceModule,
  };
  if (mediaRecorder) window.MediaRecorder = MediaRecorder;
  if (recorderLoaded) window.BluePeakRecorder = recorderModule;
  if (speech) {
    window.speechSynthesis = synth;
    window.SpeechSynthesisUtterance = FakeUtterance;
  }

  new Function("window", "document", APP_SOURCE)(window, document);

  const page = {
    document,
    fetchImpl,
    timers,
    devices,
    MediaRecorder,
    synth,
    button,
    note,
    status,
    modal,
    toggle,
    question: document.querySelector("#question"),
    count: document.querySelector("#characterCount"),
    async settle() {
      for (let round = 0; round < 3; round++) {
        for (let i = 0; i < 20; i++) await tick();
        await new Promise((resolve) => setTimeout(resolve, 5));
      }
    },
    /** A real user's click (trusted). */
    click() {
      button.dispatch("click", { isTrusted: true });
    },
    keydown(key) {
      for (const listener of document.listeners.get("keydown") || []) listener({ key, preventDefault() {} });
    },
    fire(type, event = {}) {
      for (const listener of windowListeners.get(type) || []) listener(event);
    },
    async ask(question) {
      page.question.value = question;
      document.querySelector("#chatForm").dispatch("submit");
      await page.settle();
    },
    label: () => button.querySelector(".dictate-label").textContent,
    media: () => MediaRecorder.instances.at(-1),
    stream: () => devices.streams.at(-1),
    probes: () => fetchImpl.count(STT_URL),
    uploads: () => fetchImpl.count(TRANSCRIBE_URL),
    chats: () => fetchImpl.count(CHAT_URL),
    shown: () => !button.hidden && !note.hidden,
    hiddenBoth: () => button.hidden && note.hidden,
  };
  return page;
}

/** A page where dictation is offered. */
async function offered(routes = {}, options = {}) {
  const page = loadApp({ [STT_URL]: STT_ON, [TRANSCRIBE_URL]: transcribed(), ...routes }, options);
  await page.settle();
  assert.ok(page.shown(), "dictation should be offered");
  return page;
}

/** Click, let permission resolve, and record ``bytes``. */
async function recording(page, bytes = 4096) {
  page.click();
  await page.settle();
  assert.equal(page.button.dataset.state, "recording");
  page.media().emit(bytes);
}

// ---------------------------------------------------------------------------
// Gating: shown only when the browser can record securely and STT is on
// ---------------------------------------------------------------------------

test("recorder.js not loaded: nothing is requested or shown", T, async () => {
  const page = loadApp({ [STT_URL]: STT_ON }, { recorderLoaded: false });
  await page.settle();
  assert.equal(page.probes(), 0);
  assert.ok(page.hiddenBoth());
});

test("an insecure context: nothing is requested or shown", T, async () => {
  const page = loadApp({ [STT_URL]: STT_ON }, { secure: false });
  await page.settle();
  assert.equal(page.probes(), 0);
  assert.ok(page.hiddenBoth());
});

test("no microphone API, no MediaRecorder or no recordable format: nothing is requested or shown", T, async () => {
  for (const options of [{ mediaDevicesPresent: false }, { mediaRecorder: false }, { types: ["audio/x-unknown"] }]) {
    const page = loadApp({ [STT_URL]: STT_ON }, options);
    await page.settle();
    assert.equal(page.probes(), 0, JSON.stringify(options));
    assert.ok(page.hiddenBoth(), JSON.stringify(options));
  }
});

test("the probe is one bounded, same-origin, uncached GET", T, async () => {
  const page = await offered();
  assert.equal(page.probes(), 1);
  const { init } = page.fetchImpl.calls.find((call) => call.url === STT_URL);
  assert.equal(init.method, "GET");
  assert.equal(init.cache, "no-store");
  assert.equal(init.credentials, "same-origin");
  assert.ok(init.signal instanceof AbortSignal);
});

test("STT disabled on the server (404) or unreachable: the controls stay hidden", T, async () => {
  for (const route of [NOT_FOUND, () => new Error("offline")]) {
    const page = loadApp({ [STT_URL]: route });
    await page.settle();
    assert.equal(page.probes(), 1);
    assert.ok(page.hiddenBoth());
  }
});

test("only exactly {\"stt\": true} offers dictation", T, async () => {
  const bodies = [{ stt: "true" }, { stt: true, model: "whisper-1" }, { tts: true }, [], null, { stt: false }];
  for (const body of bodies) {
    const page = loadApp({ [STT_URL]: () => jsonResponse(body) });
    await page.settle();
    assert.ok(page.hiddenBoth(), JSON.stringify(body));
  }
});

test("a probe that never answers is abandoned after 5 seconds", T, async () => {
  const seen = [];
  const page = loadApp({
    [STT_URL]: (init) =>
      new Promise((_, reject) => {
        seen.push(init.signal);
        init.signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
      }),
  });
  await page.settle();
  page.timers.fire(PROBE_TIMEOUT_MS);
  await page.settle();
  assert.equal(seen[0].aborted, true);
  assert.ok(page.hiddenBoth());
});

test("leaving the page aborts the probe", T, async () => {
  const seen = [];
  const page = loadApp({
    [STT_URL]: (init) =>
      new Promise((_, reject) => {
        seen.push(init.signal);
        init.signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
      }),
  });
  await page.settle();
  page.fire("pagehide");
  await page.settle();
  assert.equal(seen[0].aborted, true);
  assert.ok(page.hiddenBoth());
});

test("{\"stt\": true}: the button and the privacy note appear together, idle", T, async () => {
  const page = await offered();
  assert.equal(page.button.hidden, false);
  assert.equal(page.note.hidden, false); // the disclosure is visible before any recording
  assert.equal(page.label(), LABEL.idle);
  assert.equal(page.button.dataset.state, "idle");
  assert.equal(page.button.getAttribute("aria-busy"), null);
  assert.equal(page.status.textContent, "");
  assert.equal(page.devices.calls.length, 0); // no microphone until the user asks
});

test("dictation and read-aloud are offered independently", T, async () => {
  const sttOnly = loadApp({ [STT_URL]: STT_ON }, { speech: true });
  await sttOnly.settle();
  assert.ok(sttOnly.shown());
  assert.equal(sttOnly.toggle.hidden, true);

  const both = loadApp({ [STT_URL]: STT_ON, [TTS_URL]: TTS_ON }, { speech: true });
  await both.settle();
  assert.ok(both.shown());
  assert.equal(both.toggle.hidden, false);

  const ttsOnly = loadApp({ [STT_URL]: NOT_FOUND, [TTS_URL]: TTS_ON }, { speech: true });
  await ttsOnly.settle();
  assert.ok(ttsOnly.hiddenBoth());
  assert.equal(ttsOnly.toggle.hidden, false);
});

// ---------------------------------------------------------------------------
// User action and permission
// ---------------------------------------------------------------------------

test("a synthetic (untrusted) click never touches the microphone", T, async () => {
  const page = await offered();
  page.button.click(); // fake-dom's click() is not a trusted user action
  await page.settle();
  assert.equal(page.devices.calls.length, 0);
  assert.equal(page.status.textContent, ERROR_MESSAGES.not_user_action);
  assert.equal(page.button.dataset.state, "idle");
});

test("a real click asks for the microphone synchronously and shows that it is waiting", T, async () => {
  const page = await offered({}, { devices: fakeDevices({ manual: true }) });
  page.click();
  assert.equal(page.devices.calls.length, 1); // inside the click, before anything is awaited
  assert.deepEqual(page.devices.calls[0], { audio: true, video: false });
  assert.equal(page.button.dataset.state, "requesting");
  assert.equal(page.label(), LABEL.requesting);
  assert.equal(page.button.getAttribute("aria-busy"), "true");
  assert.equal(page.status.textContent, TEXT.requesting);
});

test("microphone errors show their message and leave nothing running", T, async () => {
  for (const [name, code] of [
    ["NotAllowedError", "permission_denied"],
    ["NotFoundError", "no_microphone"],
    ["NotReadableError", "microphone_busy"],
  ]) {
    const page = await offered({}, { devices: fakeDevices({ error: name }) });
    page.question.value = "kept";
    page.click();
    await page.settle();
    assert.equal(page.status.textContent, ERROR_MESSAGES[code], name);
    assert.equal(page.button.dataset.state, "idle");
    assert.equal(page.label(), LABEL.idle);
    assert.equal(page.uploads(), 0);
    assert.equal(page.question.value, "kept");
  }
});

test("selecting the button while permission is pending cancels and releases a late grant", T, async () => {
  const page = await offered({}, { devices: fakeDevices({ manual: true }) });
  page.click();
  page.click(); // "Cancel dictation"
  await page.settle();
  assert.equal(page.status.textContent, ERROR_MESSAGES.cancelled);
  assert.equal(page.button.dataset.state, "idle");

  const late = page.devices.grant();
  await page.settle();
  assert.ok(late.allStopped()); // the microphone indicator turns off at once
  assert.equal(page.MediaRecorder.instances.length, 0);
  assert.equal(page.uploads(), 0);
});

// ---------------------------------------------------------------------------
// Recording and transcription
// ---------------------------------------------------------------------------

test("a full dictation: record, stop, transcribe, and add the text without sending it", T, async () => {
  const held = heldUpload();
  const page = await offered({ [TRANSCRIBE_URL]: held.route, [CHAT_URL]: () => answer() });
  await recording(page);
  assert.equal(page.label(), LABEL.recording);
  assert.equal(page.status.textContent, TEXT.recording);
  assert.equal(page.button.getAttribute("aria-busy"), null);

  page.click(); // "Stop dictation"
  await page.settle();
  assert.equal(page.button.dataset.state, "uploading");
  assert.equal(page.label(), LABEL.uploading);
  assert.equal(page.button.getAttribute("aria-busy"), "true");
  assert.equal(page.status.textContent, TEXT.uploading);
  assert.ok(page.stream().allStopped()); // the microphone is released before the upload ends

  held.resolve(jsonResponse({ text: TRANSCRIPT }));
  await page.settle();
  assert.equal(page.question.value, TRANSCRIPT);
  assert.equal(page.count.textContent, `${TRANSCRIPT.length} / 3000`);
  assert.equal(page.document.activeElement, page.question);
  assert.equal(page.status.textContent, TEXT.added);
  assert.equal(page.button.dataset.state, "idle");
  assert.equal(page.label(), LABEL.idle);
  // Never sent: no chat request and no message in the conversation.
  assert.equal(page.chats(), 0);
  assert.equal(page.document.querySelectorAll(".message").length, 0);
});

test("the upload goes to the transcription endpoint only, with no session header", T, async () => {
  const page = await offered();
  await recording(page);
  page.click();
  await page.settle();
  const { init } = page.fetchImpl.calls.find((call) => call.url === TRANSCRIBE_URL);
  assert.equal(init.method, "POST");
  assert.equal(init.credentials, "same-origin");
  assert.ok(init.body instanceof FormData);
  assert.equal(init.headers[SESSION_HEADER], undefined);
  assert.deepEqual(
    page.fetchImpl.calls.map((call) => call.url).filter((url) => url !== SESSIONS_URL && url !== TTS_URL),
    [STT_URL, TRANSCRIBE_URL],
  );
});

test("a transcript is appended to existing text, keeping it as typed", T, async () => {
  for (const [before, after] of [
    ["My laptop", `My laptop ${TRANSCRIPT}`],
    ["My laptop ", `My laptop ${TRANSCRIPT}`],
    ["Line one\n", `Line one\n${TRANSCRIPT}`],
  ]) {
    const page = await offered();
    page.question.value = before;
    await recording(page);
    page.click();
    await page.settle();
    assert.equal(page.question.value, after, JSON.stringify(before));
  }
});

test("a transcript that would pass 3,000 characters is refused, never truncated", T, async () => {
  const page = await offered();
  const typed = "x".repeat(3000 - TRANSCRIPT.length - 1); // one space short of fitting
  page.question.value = typed + "y";
  await recording(page);
  page.click();
  await page.settle();
  assert.equal(page.question.value, typed + "y"); // unchanged
  assert.equal(page.status.textContent, TEXT.tooLong);

  const fits = await offered();
  fits.question.value = typed;
  await recording(fits);
  fits.click();
  await fits.settle();
  assert.equal(fits.question.value.length, 3000); // exactly at the limit is allowed
  assert.equal(fits.status.textContent, TEXT.added);
});

test("server and network failures show the recorder's messages and add nothing", T, async () => {
  const cases = [
    [() => jsonResponse({ detail: { code: "transcription_unavailable" } }, { status: 503 }), "transcription_unavailable"],
    [() => jsonResponse({ detail: { code: "audio_too_large" } }, { status: 413 }), "too_large"],
    [() => jsonResponse({ detail: { code: "audio_not_transcribable" } }, { status: 422 }), "audio_not_transcribable"],
    [() => jsonResponse({ detail: { code: "transcription_failed" } }, { status: 500 }), "transcription_failed"],
    [() => new Error("offline"), "network_error"],
    [transcribed("   "), "no_speech"],
  ];
  for (const [route, code] of cases) {
    const page = await offered({ [TRANSCRIBE_URL]: route });
    page.question.value = "typed";
    await recording(page);
    page.click();
    await page.settle();
    assert.equal(page.status.textContent, ERROR_MESSAGES[code], code);
    assert.equal(page.question.value, "typed", code);
    assert.equal(page.button.dataset.state, "idle", code);
    assert.ok(page.shown(), code); // still offered: the next attempt may work
  }
});

test("dictation switched off on the server (404): the controls are hidden and focus moves to the question", T, async () => {
  const page = await offered({ [TRANSCRIBE_URL]: NOT_FOUND });
  await recording(page);
  page.button.focus();
  page.click();
  await page.settle();
  assert.equal(page.status.textContent, ERROR_MESSAGES.transcription_disabled);
  assert.ok(page.hiddenBoth());
  assert.equal(page.document.activeElement, page.question);
});

test("recording stops by itself at 60 seconds and is transcribed", T, async () => {
  const page = await offered();
  await recording(page);
  page.timers.fire(MAX_DURATION_MS);
  await page.settle();
  assert.equal(page.question.value, TRANSCRIPT);
});

test("an upload that takes too long is abandoned after 45 seconds", T, async () => {
  const held = heldUpload();
  const page = await offered({ [TRANSCRIBE_URL]: held.route });
  await recording(page);
  page.click();
  await page.settle();
  page.timers.fire(UPLOAD_TIMEOUT_MS);
  await page.settle();
  assert.equal(held.signals[0].aborted, true);
  assert.equal(page.status.textContent, ERROR_MESSAGES.upload_timeout);
  assert.equal(page.question.value, "");
});

// ---------------------------------------------------------------------------
// Cancellation and late results
// ---------------------------------------------------------------------------

test("selecting the button during the upload aborts it and ignores its result", T, async () => {
  const held = heldUpload();
  const page = await offered({ [TRANSCRIBE_URL]: held.route });
  await recording(page);
  page.click(); // stop
  await page.settle();
  page.click(); // "Cancel transcription"
  await page.settle();
  assert.equal(held.signals[0].aborted, true);
  assert.equal(page.status.textContent, ERROR_MESSAGES.cancelled);
  assert.equal(page.button.dataset.state, "idle");
  assert.equal(page.question.value, "");
});

test("Escape cancels dictation, after an open dialog and before speech", T, async () => {
  const page = await offered();
  await recording(page);

  page.modal.classList.remove("hidden"); // the upload dialog is open
  page.keydown("Escape");
  await page.settle();
  assert.ok(page.modal.classList.contains("hidden")); // the dialog closed first
  assert.equal(page.button.dataset.state, "recording"); // dictation continues

  page.keydown("Escape");
  await page.settle();
  assert.equal(page.button.dataset.state, "idle");
  assert.ok(page.stream().allStopped());
  assert.equal(page.status.textContent, ERROR_MESSAGES.cancelled);
  assert.equal(page.uploads(), 0);
});

test("submitting a question ends dictation; the question is sent, the transcript never added", T, async () => {
  const held = heldUpload();
  const page = await offered({ [TRANSCRIBE_URL]: held.route, [CHAT_URL]: () => answer() });
  await recording(page);
  page.click(); // stop: uploading
  await page.settle();

  await page.ask("My printer is offline");
  assert.equal(held.signals[0].aborted, true);
  assert.equal(page.chats(), 1);
  assert.equal(page.button.dataset.state, "idle");
  assert.equal(page.status.textContent, ERROR_MESSAGES.cancelled);

  held.resolve?.(jsonResponse({ text: TRANSCRIPT })); // a late answer changes nothing
  await page.settle();
  assert.equal(page.question.value, "");
});

test("submitting while recording releases the microphone and uploads nothing", T, async () => {
  const page = await offered({ [CHAT_URL]: () => answer() });
  await recording(page);
  await page.ask("VPN is down");
  assert.ok(page.stream().allStopped());
  assert.equal(page.uploads(), 0);
  assert.equal(page.chats(), 1);
});

test("New chat ends dictation and a late transcript is ignored", T, async () => {
  const held = heldUpload();
  const page = await offered({ [TRANSCRIBE_URL]: held.route });
  await recording(page);
  page.click();
  await page.settle();

  page.document.querySelector("#clearChat").click();
  await page.settle();
  assert.equal(held.signals[0].aborted, true);
  assert.equal(page.button.dataset.state, "idle");
  assert.equal(page.status.textContent, ""); // a reset, not an error
  assert.equal(page.question.value, "");
});

test("leaving the page releases the microphone and aborts an upload", T, async () => {
  const page = await offered();
  await recording(page);
  page.fire("pagehide");
  await page.settle();
  assert.ok(page.stream().allStopped());
  assert.equal(page.uploads(), 0);

  const held = heldUpload();
  const uploading = await offered({ [TRANSCRIBE_URL]: held.route });
  await recording(uploading);
  uploading.click();
  await uploading.settle();
  uploading.fire("pagehide");
  await uploading.settle();
  assert.equal(held.signals[0].aborted, true);
});

test("a cancelled session's late result never lands in the next session", T, async () => {
  let call = 0;
  const first = heldUpload();
  const page = await offered({
    [TRANSCRIBE_URL]: (init) => (++call === 1 ? first.route(init) : jsonResponse({ text: "second" })),
  });
  await recording(page);
  page.click();
  await page.settle();
  page.click(); // cancel the first upload
  await page.settle();

  await recording(page); // a new session
  page.click();
  await page.settle();
  assert.equal(page.question.value, "second");

  first.resolve?.(jsonResponse({ text: "first" })); // too late
  await page.settle();
  assert.equal(page.question.value, "second");
});

test("one session at a time: clicks follow the state, never starting a second recording", T, async () => {
  const page = await offered();
  await recording(page); // one click: recording (with some audio)
  page.click(); // stop, not a second start
  await page.settle();
  assert.equal(page.devices.calls.length, 1);
  assert.equal(page.uploads(), 1);
});

// ---------------------------------------------------------------------------
// Read-aloud (Voice V1) while dictating
// ---------------------------------------------------------------------------

async function bothOn(routes = {}) {
  const page = await offered({ [TTS_URL]: TTS_ON, ...routes }, { speech: true });
  page.toggle.click(); // turn read-aloud on
  await page.settle();
  page.synth.finishAll(); // the confirmation
  await page.settle();
  return page;
}

test("starting dictation stops speech first", T, async () => {
  const page = await bothOn({ [CHAT_URL]: () => answer("The VPN uses GlobalProtect.") });
  await page.ask("VPN?");
  assert.ok(page.synth.queue.length > 0); // the answer is being spoken
  const cancels = page.synth.cancelCount;

  page.click();
  assert.ok(page.synth.cancelCount > cancels); // silenced before any audio is captured
  assert.equal(page.synth.queue.length, 0);
});

test("an untrusted click does not stop speech", T, async () => {
  const page = await bothOn({ [CHAT_URL]: () => answer("The VPN uses GlobalProtect.") });
  await page.ask("VPN?");
  const cancels = page.synth.cancelCount;
  page.button.click();
  await page.settle();
  assert.equal(page.synth.cancelCount, cancels);
});

test("an answer completing while dictating is not spoken; later answers are", T, async () => {
  let reply = "First answer.";
  const page = await bothOn({ [CHAT_URL]: () => answer(reply) });
  const spokenBefore = page.synth.texts().length;

  // Dictation starts while a question is in progress; the answer arrives meanwhile.
  page.question.value = "First?";
  page.document.querySelector("#chatForm").dispatch("submit");
  page.click(); // dictation is now active (permission pending → recording)
  await page.settle();
  assert.ok(page.button.dataset.state !== "idle");
  assert.equal(page.synth.texts().length, spokenBefore); // nothing spoken into the recording

  page.keydown("Escape"); // end dictation
  await page.settle();
  reply = "Second answer.";
  await page.ask("Second?");
  assert.ok(page.synth.texts().some((text) => text.includes("Second answer")));
});

test("turning read-aloud on while dictating does not speak the confirmation", T, async () => {
  const page = await offered({ [TTS_URL]: TTS_ON }, { speech: true });
  await recording(page);
  page.toggle.click();
  await page.settle();
  assert.equal(page.synth.texts().length, 0);
  assert.equal(page.toggle.getAttribute("aria-pressed"), "true"); // still turned on
});
