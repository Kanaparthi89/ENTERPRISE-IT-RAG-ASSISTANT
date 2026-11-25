// Voice V1c: static/js/app.js reading answers aloud, against the real
// chat-stream.js controller, the real voice.js module, a fake speech engine
// and a minimal fake DOM. No browser, network or real voice is used.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { test } from "node:test";

import { createDocument, h } from "./fake-dom.mjs";
import { chat, DONE, fakeFetch, fakeTimers, jsonResponse, sseResponse, tick, transcript } from "./helpers.mjs";

const require = createRequire(import.meta.url);
const voiceModule = require("../../static/js/voice.js");
const APP_SOURCE = readFileSync(new URL("../../static/js/app.js", import.meta.url), "utf-8");

const { CHAT_URL, STREAM_URL, SESSIONS_URL, SESSION_HEADER } = chat;
const { toSpeechText, splitIntoChunks, VOICE_LOAD_TIMEOUT_MS } = voiceModule;

const VOICE_URL = "/api/voice/status";
const PROBE_TIMEOUT_MS = 5_000;
const CONFIRMATION = "Voice responses on.";
const TEXT = {
  noVoice: "Read aloud isn't available: this device has no on-device voice for this page's language.",
  blocked: "Your browser blocked speech. Select Read answers aloud to try again.",
  failed: "Couldn't read the answer aloud.",
};

const LOCAL_EN = { name: "Samantha", lang: "en-US", localService: true, default: true };
const NETWORK_EN = { name: "Google US English", lang: "en-US", localService: false, default: true };

const ON = () => jsonResponse({ tts: true });
const NOT_FOUND = () => jsonResponse({ detail: "Not Found" }, { status: 404 });
const answer = (text = DONE.answer) => jsonResponse({ ...DONE, answer: text });

// ---------------------------------------------------------------------------
// Fakes
// ---------------------------------------------------------------------------

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

/** speechSynthesis double: records utterances; tests finish or fail them. */
class FakeSynth {
  constructor(voices) {
    this.voices = voices;
    this.queue = [];
    this.spoken = [];
    this.cancelCount = 0;
    this.listeners = new Map();
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

  addEventListener(type, listener) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type).add(listener);
  }

  removeEventListener(type, listener) {
    this.listeners.get(type)?.delete(listener);
  }

  loadVoices(voices) {
    this.voices = voices;
    for (const listener of [...(this.listeners.get("voiceschanged") ?? [])]) listener();
  }

  /** Texts actually spoken (the silent, empty primer excluded). */
  texts() {
    return this.spoken.filter((u) => u.text !== "").map((u) => u.text);
  }

  finishAll() {
    while (this.queue.length) this.queue.shift().onend?.({});
  }

  failNext(error) {
    let utterance = this.queue.shift();
    while (utterance && utterance.text === "") utterance = this.queue.shift(); // skip the primer
    utterance?.onerror?.({ error });
  }
}

// ---------------------------------------------------------------------------
// The page
// ---------------------------------------------------------------------------

function loadApp(routes = {}, { speech = true, voices = [LOCAL_EN], voiceModuleLoaded = true, streaming = false } = {}) {
  const document = createDocument();
  document.documentElement.setAttribute("lang", "en");

  const toggle = h(document, "button#voiceToggle.text-button.voice-toggle", ["Read answers aloud"]);
  toggle.setAttribute("aria-pressed", "false");
  toggle.setAttribute("aria-describedby", "voiceHint");
  toggle.hidden = true;
  const stop = h(document, "button#voiceStop.text-button.voice-stop", ["Stop speaking"]);
  stop.hidden = true;
  const status = h(document, "p#voiceStatus.voice-status");
  status.setAttribute("role", "status");

  const modal = h(document, "div#uploadModal.hidden", [h(document, "input#adminKey")]);
  const form = h(document, "form#chatForm.composer", [
    h(document, "textarea#question"),
    h(document, "button#sendBtn", [h(document, "span", ["Ask agent"])]),
  ]);
  document.body.append(
    h(document, "div.conversation-actions", [toggle, stop, h(document, "button#clearChat")]),
    status,
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
    h(document, "button#openUpload"),
    modal,
  );

  const allRoutes = { [SESSIONS_URL]: NOT_FOUND, ...routes };
  const fetchImpl = fakeFetch(allRoutes);
  const synth = new FakeSynth(voices);
  // Fake time only for the 5 s probe timeout and the 3 s voice wait; the
  // page's other timers (frames, retries) run for real.
  const timers = fakeTimers();
  const faked = new Set([PROBE_TIMEOUT_MS, VOICE_LOAD_TIMEOUT_MS]);
  const windowListeners = new Map();

  const window = {
    document,
    fetch: fetchImpl,
    TextDecoder,
    AbortController,
    Response,
    setTimeout: (fn, ms) => (faked.has(ms) ? timers.setTimeout(fn, ms) : setTimeout(fn, ms)),
    clearTimeout: (id) => (typeof id === "number" ? timers.clearTimeout(id) : clearTimeout(id)),
    location: { origin: "http://localhost", reload() {} },
    addEventListener(type, listener) {
      if (!windowListeners.has(type)) windowListeners.set(type, []);
      windowListeners.get(type).push(listener);
    },
    BluePeakChat: chat,
  };
  if (streaming) window.ReadableStream = ReadableStream;
  if (speech) {
    window.speechSynthesis = synth;
    window.SpeechSynthesisUtterance = FakeUtterance;
  }
  if (voiceModuleLoaded) window.BluePeakVoice = voiceModule;

  new Function("window", "document", APP_SOURCE)(window, document);

  const page = {
    document,
    fetchImpl,
    synth,
    timers,
    toggle,
    stop,
    status,
    modal,
    question: document.querySelector("#question"),
    chat: document.querySelector("#chat"),
    async settle() {
      for (let round = 0; round < 3; round++) {
        for (let i = 0; i < 20; i++) await tick();
        await new Promise((resolve) => setTimeout(resolve, 5));
      }
    },
    async ask(question) {
      page.question.value = question;
      document.querySelector("#chatForm").dispatch("submit");
      await page.settle();
    },
    async turnOn() {
      toggle.click();
      await page.settle();
    },
    fire(type, event = {}) {
      for (const listener of windowListeners.get(type) || []) listener(event);
    },
    keydown(key) {
      for (const listener of document.listeners.get("keydown") || []) listener({ key, preventDefault() {} });
    },
    pressed: () => toggle.getAttribute("aria-pressed"),
    probes: () => fetchImpl.count(VOICE_URL),
    answers: () => document.querySelectorAll(".message.assistant"),
  };
  return page;
}

/** The chunks voice.js speaks for an answer. */
const spokenChunks = (markdown) => splitIntoChunks(toSpeechText(markdown));

/** A page where voice is offered and already turned on (confirmation finished). */
async function voiceOn(routes = {}, options = {}) {
  const page = loadApp({ [VOICE_URL]: ON, ...routes }, options);
  await page.settle();
  await page.turnOn();
  page.synth.finishAll(); // the confirmation
  await page.settle();
  return page;
}

// ---------------------------------------------------------------------------
// The status probe
// ---------------------------------------------------------------------------

test("no speech synthesis: nothing is requested or shown", async () => {
  const page = loadApp({}, { speech: false });
  await page.settle();
  assert.equal(page.probes(), 0);
  assert.equal(page.toggle.hidden, true);
  assert.equal(page.stop.hidden, true);
});

test("voice.js not loaded: nothing is requested or shown", async () => {
  const page = loadApp({}, { voiceModuleLoaded: false });
  await page.settle();
  assert.equal(page.probes(), 0);
  assert.equal(page.toggle.hidden, true);
});

test("the probe is one bounded, same-origin GET", async () => {
  const page = loadApp({ [VOICE_URL]: ON });
  await page.settle();

  assert.equal(page.probes(), 1);
  const [call] = page.fetchImpl.calls.filter((c) => c.url === VOICE_URL);
  assert.equal(call.init.method, "GET");
  assert.equal(call.init.headers.Accept, "application/json");
  assert.equal(call.init.credentials, "same-origin");
  assert.equal(call.init.cache, "no-store");
  assert.ok(call.init.signal instanceof AbortSignal);
  assert.equal(page.timers.count(), 0, "the probe timeout was cleared");
});

test("voice disabled on the server (404): the toggle stays hidden", async () => {
  const page = loadApp({ [VOICE_URL]: NOT_FOUND });
  await page.settle();
  assert.equal(page.probes(), 1);
  assert.equal(page.toggle.hidden, true);
  assert.equal(page.status.textContent, ""); // nothing is announced
});

test("a network error hides the toggle silently", async () => {
  const page = loadApp({ [VOICE_URL]: () => new TypeError("Failed to fetch") });
  await page.settle();
  assert.equal(page.toggle.hidden, true);
  assert.equal(page.status.textContent, "");
});

test("a probe that never answers is abandoned after 5 seconds", async () => {
  let signal;
  const page = loadApp({
    [VOICE_URL]: (init) =>
      new Promise((_, reject) => {
        signal = init.signal;
        signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
      }),
  });
  await page.settle();
  assert.equal(page.toggle.hidden, true);

  page.timers.fire(PROBE_TIMEOUT_MS);
  await page.settle();
  assert.equal(signal.aborted, true);
  assert.equal(page.toggle.hidden, true);
});

test("leaving the page aborts the probe", async () => {
  let signal;
  const page = loadApp({
    [VOICE_URL]: (init) =>
      new Promise((_, reject) => {
        signal = init.signal;
        signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
      }),
  });
  await page.settle();
  page.fire("pagehide");
  await page.settle();
  assert.equal(signal.aborted, true);
  assert.equal(page.toggle.hidden, true);
});

test("only exactly {\"tts\": true} offers voice", async () => {
  const bodies = [
    () => jsonResponse({ tts: "true" }),
    () => jsonResponse({ tts: 1 }),
    () => jsonResponse({}),
    () => jsonResponse({ tts: true, stt: true }),
    () => jsonResponse([true]),
    () => jsonResponse(null),
    () => new Response("not json", { status: 200 }),
    () => jsonResponse({ tts: true }, { status: 201 }),
    () => jsonResponse({ tts: true }, { status: 500 }),
  ];
  for (const body of bodies) {
    const page = loadApp({ [VOICE_URL]: body });
    await page.settle();
    assert.equal(page.toggle.hidden, true, String(body));
  }
});

test("{\"tts\": true} shows the toggle, off, with Stop hidden", async () => {
  const page = loadApp({ [VOICE_URL]: ON });
  await page.settle();

  assert.equal(page.toggle.hidden, false);
  assert.equal(page.pressed(), "false");
  assert.equal(page.stop.hidden, true);
  assert.equal(page.synth.spoken.length, 0); // nothing spoken until the user opts in
});

// ---------------------------------------------------------------------------
// Turning voice on
// ---------------------------------------------------------------------------

test("turning on primes synchronously inside the click, then confirms once", async () => {
  const page = loadApp({ [VOICE_URL]: ON });
  await page.settle();

  page.toggle.click();
  // Before anything is awaited: the silent primer is already queued.
  assert.equal(page.synth.spoken.length, 1);
  assert.equal(page.synth.spoken[0].text, "");
  assert.equal(page.synth.spoken[0].volume, 0);
  assert.equal(page.toggle.getAttribute("aria-busy"), "true");
  assert.equal(page.pressed(), "false"); // not on until a voice is confirmed

  await page.settle();
  assert.equal(page.pressed(), "true");
  assert.equal(page.toggle.hasAttribute("aria-busy"), false);
  assert.deepEqual(page.synth.texts(), [CONFIRMATION]);
  assert.ok(page.synth.spoken.slice(1).every((u) => u.voice === LOCAL_EN));
  assert.equal(page.stop.hidden, false); // something is being spoken

  page.synth.finishAll();
  await page.settle();
  assert.equal(page.stop.hidden, true);
  assert.deepEqual(page.synth.texts(), [CONFIRMATION]); // confirmed once
});

test("no on-device voice: voice stays off and the reason is shown", async () => {
  const page = loadApp({ [VOICE_URL]: ON }, { voices: [NETWORK_EN] });
  await page.settle();
  await page.turnOn();

  assert.equal(page.pressed(), "false");
  assert.equal(page.status.textContent, TEXT.noVoice);
  assert.deepEqual(page.synth.texts(), []); // a network voice is never used

  await page.ask("VPN?");
  page.question.value = "";
  assert.deepEqual(page.synth.texts(), []);
});

test("voices that load late are waited for; voices that never load mean unavailable", async () => {
  const late = loadApp({ [VOICE_URL]: ON }, { voices: [] });
  await late.settle();
  await late.turnOn();
  assert.equal(late.pressed(), "false");
  late.synth.loadVoices([LOCAL_EN]);
  await late.settle();
  assert.equal(late.pressed(), "true");
  assert.deepEqual(late.synth.texts(), [CONFIRMATION]);

  const never = loadApp({ [VOICE_URL]: ON }, { voices: [] });
  await never.settle();
  await never.turnOn();
  never.timers.fire(VOICE_LOAD_TIMEOUT_MS);
  await never.settle();
  assert.equal(never.pressed(), "false");
  assert.equal(never.status.textContent, TEXT.noVoice);
});

test("clicking again while the voice check runs turns it off for good", async () => {
  const page = loadApp({ [VOICE_URL]: ON }, { voices: [] });
  await page.settle();
  page.toggle.click();
  page.toggle.click(); // second click before the check finished
  await page.settle();

  page.synth.loadVoices([LOCAL_EN]); // the first check completes late
  await page.settle();
  assert.equal(page.pressed(), "false");
  assert.equal(page.toggle.hasAttribute("aria-busy"), false);
  assert.deepEqual(page.synth.texts(), []);
});

test("a browser that blocks speech is reported and voice turns off", async () => {
  const page = loadApp({ [VOICE_URL]: ON });
  await page.settle();
  await page.turnOn();

  page.synth.failNext("not-allowed"); // the confirmation is refused
  await page.settle();
  assert.equal(page.pressed(), "false");
  assert.equal(page.status.textContent, TEXT.blocked);

  // The next click is a new user gesture: it tries again and clears the message.
  await page.turnOn();
  assert.equal(page.status.textContent, "");
  assert.equal(page.pressed(), "true");
});

// ---------------------------------------------------------------------------
// Speaking answers
// ---------------------------------------------------------------------------

test("voice off: completed answers are never spoken", async () => {
  const page = loadApp({ [VOICE_URL]: ON, [CHAT_URL]: () => answer() });
  await page.settle();
  await page.ask("VPN?");
  assert.equal(page.answers().length, 1);
  assert.equal(page.synth.spoken.length, 0);
});

test("a completed /api/chat answer is spoken exactly once", async () => {
  const page = await voiceOn({ [CHAT_URL]: () => answer() });
  await page.ask("VPN?");

  assert.equal(page.answers().length, 1); // rendering is unchanged
  assert.deepEqual(page.synth.texts(), [CONFIRMATION, ...spokenChunks(DONE.answer)]);
  page.synth.finishAll();
  await page.settle();
  assert.deepEqual(page.synth.texts(), [CONFIRMATION, ...spokenChunks(DONE.answer)]); // not repeated
});

test("a streamed answer is spoken once, after done, never delta by delta", async () => {
  const deltas = ["Install GlobalProtect", " — then sign in.", "\nGrüße ✓"];
  const page = await voiceOn({ [STREAM_URL]: () => sseResponse(transcript(deltas)) }, { streaming: true });
  await page.ask("VPN?");

  assert.equal(page.fetchImpl.count(STREAM_URL), 1);
  assert.equal(page.fetchImpl.count(CHAT_URL), 0);
  const spoken = page.synth.texts().slice(1);
  assert.deepEqual(spoken, spokenChunks(DONE.answer));
  for (const delta of deltas) assert.ok(!spoken.includes(delta.trim()), `delta spoken: ${delta}`);
});

test("failed questions are never spoken", async () => {
  const page = await voiceOn({ [CHAT_URL]: () => jsonResponse({ detail: { message: "x" } }, { status: 500 }) });
  await page.ask("VPN?");
  assert.deepEqual(page.synth.texts(), [CONFIRMATION]);
  assert.equal(page.pressed(), "true"); // voice stays on
});

test("Markdown, links and citations are spoken as plain text", async () => {
  const markdown = "## Steps\n- Open [the guide](https://it.example.com/vpn) [1]\n```\nsecret\n```";
  const page = await voiceOn({ [CHAT_URL]: () => answer(markdown) });
  await page.ask("VPN?");
  const spoken = page.synth.texts().slice(1).join(" ");
  assert.equal(spoken, toSpeechText(markdown));
  assert.ok(!spoken.includes("https") && !spoken.includes("secret") && !spoken.includes("[1]"));
});

test("an engine error is reported and voice stays on", async () => {
  const page = await voiceOn({ [CHAT_URL]: () => answer() });
  await page.ask("VPN?");
  page.synth.failNext("audio-busy");
  await page.settle();
  assert.equal(page.status.textContent, TEXT.failed);
  assert.equal(page.pressed(), "true");
});

test("the error message clears when the next answer is spoken", async () => {
  // Regression: "Couldn't read the answer aloud." stayed after later answers spoke.
  const page = await voiceOn({ [CHAT_URL]: () => answer() });
  await page.ask("first?");
  page.synth.failNext("audio-busy");
  await page.settle();
  assert.equal(page.status.textContent, TEXT.failed);

  await page.ask("second?");
  assert.equal(page.status.textContent, "");
  page.synth.finishAll();
  await page.settle();
  assert.equal(page.status.textContent, "");
});

// ---------------------------------------------------------------------------
// Cancellation
// ---------------------------------------------------------------------------

/** A page that is in the middle of speaking an answer. */
async function speakingPage(routes = {}) {
  const page = await voiceOn({ [CHAT_URL]: () => answer("First answer. ".repeat(40)), ...routes });
  await page.ask("first?");
  assert.equal(page.stop.hidden, false);
  assert.ok(page.synth.queue.length > 0);
  return page;
}

test("a new question silences the previous answer before it is sent", async () => {
  const page = await speakingPage();
  const cancels = page.synth.cancelCount;

  page.question.value = "second?";
  page.document.querySelector("#chatForm").dispatch("submit");
  assert.ok(page.synth.cancelCount > cancels, "cancelled synchronously on submit");
  await page.settle();
  assert.equal(page.fetchImpl.count(CHAT_URL), 2);
});

test("New chat silences speech", async () => {
  const page = await speakingPage();
  page.document.querySelector("#clearChat").click();
  await page.settle();
  assert.equal(page.synth.queue.length, 0);
  assert.equal(page.stop.hidden, true);
});

test("turning voice off silences speech", async () => {
  const page = await speakingPage();
  page.toggle.click();
  await page.settle();
  assert.equal(page.pressed(), "false");
  assert.equal(page.synth.queue.length, 0);
  assert.equal(page.stop.hidden, true);
});

test("Stop speaking silences speech, keeps voice on and returns focus to the toggle", async () => {
  const page = await speakingPage();
  page.stop.focus();
  page.stop.click();
  await page.settle();

  assert.equal(page.synth.queue.length, 0);
  assert.equal(page.stop.hidden, true);
  assert.equal(page.pressed(), "true");
  assert.equal(page.document.activeElement, page.toggle);
});

test("leaving the page silences speech", async () => {
  const page = await speakingPage();
  page.fire("pagehide");
  await page.settle();
  assert.equal(page.synth.queue.length, 0);
});

test("Escape stops speech, but an open dialog takes precedence", async () => {
  const page = await speakingPage();

  page.modal.classList.remove("hidden"); // the upload dialog is open
  page.keydown("Escape");
  assert.equal(page.modal.classList.contains("hidden"), true); // the dialog closed
  assert.ok(page.synth.queue.length > 0, "speech continues");

  page.keydown("Escape"); // no dialog now
  await page.settle();
  assert.equal(page.synth.queue.length, 0);

  page.keydown("Enter"); // other keys do nothing
});

test("Stop speaking never aborts a question in progress", async () => {
  let release;
  let chatSignal;
  const page = loadApp({
    [VOICE_URL]: ON,
    [CHAT_URL]: (init) => {
      chatSignal = init.signal;
      return new Promise((resolve) => {
        release = () => resolve(answer());
      });
    },
  });
  await page.settle();

  page.question.value = "VPN?";
  page.document.querySelector("#chatForm").dispatch("submit");
  await page.settle();
  await page.turnOn(); // the confirmation plays while the question runs
  assert.equal(page.stop.hidden, false);

  page.stop.click();
  await page.settle();
  assert.equal(chatSignal?.aborted ?? false, false);

  release();
  await page.settle();
  assert.equal(page.answers().length, 1);
  assert.deepEqual(page.synth.texts(), [CONFIRMATION, ...spokenChunks(DONE.answer)]);
});

// ---------------------------------------------------------------------------
// Sessions and accessibility
// ---------------------------------------------------------------------------

test("sessions are unchanged with voice on, and each answer is spoken once", async () => {
  const token = "tok_A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0U1v";
  const page = await voiceOn({
    [SESSIONS_URL]: () => jsonResponse({ session_id: token }, { status: 201 }),
    [CHAT_URL]: () => jsonResponse({ ...DONE, session: { turn_saved: true, history_turns_used: 0 } }),
  });

  await page.ask("one?");
  page.synth.finishAll();
  await page.ask("two?");

  const headers = page.fetchImpl.calls.filter((c) => c.url === CHAT_URL).map((c) => c.init.headers[SESSION_HEADER]);
  assert.deepEqual(headers, [token, token]);
  assert.equal(page.fetchImpl.count(SESSIONS_URL), 1);
  const chunks = spokenChunks(DONE.answer);
  assert.deepEqual(page.synth.texts(), [CONFIRMATION, ...chunks, ...chunks]);
  assert.equal(page.probes(), 1); // the status is probed once per page
});

test("the toggle keeps aria-pressed in sync and its description", async () => {
  const page = loadApp({ [VOICE_URL]: ON });
  await page.settle();
  assert.equal(page.toggle.getAttribute("aria-describedby"), "voiceHint");
  assert.equal(page.pressed(), "false");

  await page.turnOn();
  assert.equal(page.pressed(), "true");
  page.toggle.click();
  assert.equal(page.pressed(), "false");
  assert.equal(page.status.getAttribute("role"), "status");
});
