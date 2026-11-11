// Voice V3a: static/js/recorder.js (microphone dictation, no UI).
// Deterministic: the microphone, MediaRecorder, fetch and timers are fakes;
// audio chunks are real Blobs so sizes are real byte counts.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { test } from "node:test";

import { fakeFetch, fakeTimers, jsonResponse, tick } from "./helpers.mjs";

const require = createRequire(import.meta.url);
const rec = require("../../static/js/recorder.js");
const SOURCE = readFileSync(new URL("../../static/js/recorder.js", import.meta.url), "utf-8");

const {
  AUDIO_BITS_PER_SECOND,
  EARLY_STOP_BYTES,
  ERROR_MESSAGES,
  FORMATS,
  MAX_AUDIO_BYTES,
  MAX_DURATION_MS,
  STATES,
  TIMESLICE_MS,
  TRANSCRIBE_URL,
  UPLOAD_TIMEOUT_MS,
  chooseFormat,
  createRecorder,
} = rec;

// Every test is bounded: a regression that leaves a session pending fails
// quickly instead of stalling the suite.
const T = Object.freeze({ timeout: 10_000 });
const CLICK = Object.freeze({ isTrusted: true });
const TRANSCRIPT = "How do I reset my VPN token?";
const WEBM = "audio/webm;codecs=opus";
const OGG = "audio/ogg;codecs=opus";
const MP4 = "audio/mp4";
const KiB = 1024;

// ---------------------------------------------------------------------------
// Fakes
// ---------------------------------------------------------------------------

class FakeTrack {
  constructor() {
    this.stopCount = 0;
    this.onended = null;
  }

  stop() {
    this.stopCount += 1;
  }

  /** The device went away (unplugged, revoked). */
  end() {
    this.onended?.({});
  }
}

class FakeStream {
  constructor(trackCount = 2) {
    this.tracks = Array.from({ length: trackCount }, () => new FakeTrack());
  }

  getTracks() {
    return [...this.tracks];
  }

  allStopped() {
    return this.tracks.every((track) => track.stopCount > 0);
  }
}

/** navigator.mediaDevices double; ``manual`` holds the permission prompt open. */
function fakeDevices({ manual = false, error = null, throwSync = null } = {}) {
  const devices = {
    calls: [],
    streams: [],
    pending: null,
    getUserMedia(constraints) {
      devices.calls.push(constraints);
      if (throwSync) throw throwSync;
      if (error) return Promise.reject(error);
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
    deny(name) {
      devices.pending.reject(Object.assign(new Error(name), { name }));
    },
  };
  return devices;
}

/** A MediaRecorder class that supports ``types``. */
function fakeMediaRecorder(types = [WEBM], { throwOnConstruct = false, throwOnStart = false, reportedType } = {}) {
  class FakeMediaRecorder {
    static instances = [];

    static isTypeSupported(type) {
      return types.includes(type);
    }

    constructor(stream, options) {
      if (throwOnConstruct) throw new Error("NotSupportedError");
      this.stream = stream;
      this.options = options;
      this.mimeType = reportedType ?? options.mimeType;
      this.state = "inactive";
      this.stopCount = 0;
      this.finalChunk = 0; // bytes delivered with the last dataavailable on stop()
      this.ondataavailable = null;
      this.onstop = null;
      this.onerror = null;
      FakeMediaRecorder.instances.push(this);
    }

    start(timeslice) {
      if (throwOnStart) throw new Error("InvalidStateError");
      this.timeslice = timeslice;
      this.state = "recording";
    }

    stop() {
      if (this.state === "inactive") throw new Error("InvalidStateError");
      this.state = "inactive";
      this.stopCount += 1;
      const final = this.finalChunk;
      queueMicrotask(() => {
        if (final) this.ondataavailable?.({ data: new Blob([new Uint8Array(final)]) });
        this.onstop?.({});
      });
    }

    emit(bytes) {
      this.ondataavailable?.({ data: new Blob([new Uint8Array(bytes)]) });
    }

    fail() {
      this.onerror?.({ error: { name: "UnknownError" } });
    }
  }
  return FakeMediaRecorder;
}

const ok = (text = TRANSCRIPT) => jsonResponse({ text });

/** A fetch route that only settles when aborted (rejecting like a browser). */
function hangingRoute(seen) {
  return (init) =>
    new Promise((_, reject) => {
      seen.push(init.signal);
      init.signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
    });
}

function setup({ types = [WEBM], devices = fakeDevices(), route = () => ok(), secure = true, recorderOptions = {}, ...options } = {}) {
  const MediaRecorder = fakeMediaRecorder(types, recorderOptions);
  const timers = fakeTimers();
  const fetchImpl = fakeFetch({ [TRANSCRIBE_URL]: route });
  const states = [];
  const recorder = createRecorder({
    mediaDevices: devices,
    MediaRecorder,
    isSecureContext: secure,
    fetchImpl,
    setTimeout: timers.setTimeout,
    clearTimeout: timers.clearTimeout,
    now: () => 1_000,
    onStateChange: (state, detail) => states.push(detail.code ? `${state}:${detail.code}` : state),
    ...options,
  });
  return {
    recorder,
    devices,
    MediaRecorder,
    timers,
    fetchImpl,
    states,
    media: () => MediaRecorder.instances.at(-1),
    stream: () => devices.streams.at(-1),
  };
}

/**
 * Start and record ``chunks`` (bytes each). Returns ``{ outcome }``: the
 * pending promise is wrapped, because an async function returning a promise
 * would adopt it and wait for the whole session.
 */
async function recording(ctx, chunks = [4 * KiB]) {
  const outcome = ctx.recorder.start(CLICK);
  await tick();
  for (const bytes of chunks) ctx.media().emit(bytes);
  return { outcome };
}

async function settle() {
  for (let i = 0; i < 10; i++) await tick();
}

// ---------------------------------------------------------------------------
// Preconditions: user action, secure context, APIs, formats
// ---------------------------------------------------------------------------

test("the requested limits are the approved ones", T, () => {
  assert.equal(MAX_DURATION_MS, 60_000);
  assert.equal(MAX_AUDIO_BYTES, 2 * 1024 * 1024);
  assert.equal(EARLY_STOP_BYTES, Math.floor(1.9 * 1024 * 1024));
  assert.ok(EARLY_STOP_BYTES < MAX_AUDIO_BYTES);
  assert.equal(AUDIO_BITS_PER_SECOND, 32_000);
  assert.equal(TIMESLICE_MS, 1_000);
  assert.equal(UPLOAD_TIMEOUT_MS, 45_000);
  assert.equal(TRANSCRIBE_URL, "/api/voice/transcribe");
});

test("only a real user action starts a session", T, async () => {
  for (const event of [undefined, null, {}, { isTrusted: false }, { isTrusted: "true" }]) {
    const ctx = setup();
    assert.deepEqual(await ctx.recorder.start(event), { ok: false, code: "not_user_action" });
    assert.equal(ctx.devices.calls.length, 0, "the microphone is never requested");
    assert.equal(ctx.recorder.state, STATES.IDLE);
  }
});

test("an insecure context never touches the microphone", T, async () => {
  const ctx = setup({ secure: false });
  assert.deepEqual(await ctx.recorder.start(CLICK), { ok: false, code: "insecure_context" });
  assert.equal(ctx.devices.calls.length, 0);
});

test("missing browser APIs are reported as unsupported", T, async () => {
  const MediaRecorder = fakeMediaRecorder();
  const base = {
    mediaDevices: fakeDevices(),
    MediaRecorder,
    isSecureContext: true,
    fetchImpl: () => ok(),
  };
  const variants = [
    { mediaDevices: undefined },
    { mediaDevices: {} },
    { MediaRecorder: undefined },
    { MediaRecorder: function NoTypeCheck() {} },
    { fetchImpl: undefined },
    { FormData: null },
    { Blob: null },
    { AbortController: null },
  ];
  for (const variant of variants) {
    const recorder = createRecorder({ ...base, ...variant });
    assert.equal(recorder.supported, false, JSON.stringify(Object.keys(variant)));
    assert.deepEqual(await recorder.start(CLICK), { ok: false, code: "unsupported" });
  }
  assert.equal(base.mediaDevices.calls.length, 0);
});

test("formats are chosen in order: WebM/Opus, Ogg/Opus, MP4", T, () => {
  assert.equal(chooseFormat(fakeMediaRecorder([WEBM, OGG, MP4])).extension, ".webm");
  assert.equal(chooseFormat(fakeMediaRecorder([OGG, MP4])).mimeType, OGG);
  assert.equal(chooseFormat(fakeMediaRecorder([MP4])).mimeType, MP4);
  assert.equal(chooseFormat(fakeMediaRecorder(["audio/webm"])).extension, ".webm");
  assert.equal(chooseFormat(fakeMediaRecorder(["audio/wav"])), null);
  assert.equal(chooseFormat(null), null);

  const throwing = fakeMediaRecorder();
  throwing.isTypeSupported = () => {
    throw new Error("boom");
  };
  assert.equal(chooseFormat(throwing), null);
  assert.ok(Object.isFrozen(FORMATS) && FORMATS.every(Object.isFrozen));
});

test("no recordable format: unsupported, and the microphone is not requested", T, async () => {
  const ctx = setup({ types: ["audio/wav"] });
  assert.deepEqual(await ctx.recorder.start(CLICK), { ok: false, code: "unsupported" });
  assert.equal(ctx.devices.calls.length, 0);
});

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

test("the microphone is requested synchronously inside the user action", T, () => {
  const ctx = setup();
  ctx.recorder.start(CLICK); // not awaited
  assert.deepEqual(ctx.devices.calls, [{ audio: true, video: false }]);
  assert.equal(ctx.recorder.state, STATES.REQUESTING);
  ctx.recorder.cancel();
});

test("recording uses the chosen format, 32 kbps and one-second slices", T, async () => {
  const ctx = setup({ types: [OGG] });
  const outcome = ctx.recorder.start(CLICK);
  await tick();

  assert.equal(ctx.recorder.state, STATES.RECORDING);
  assert.deepEqual(ctx.media().options, { mimeType: OGG, audioBitsPerSecond: 32_000 });
  assert.equal(ctx.media().timeslice, 1_000);
  assert.deepEqual(ctx.states, ["requesting", "recording"]);
  ctx.recorder.cancel();
  assert.deepEqual(await outcome, { ok: false, code: "cancelled" });
});

test("a full session: record, stop, upload, transcript (and nothing else)", T, async () => {
  const ctx = setup();
  const { outcome } = await recording(ctx, [3 * KiB, 2 * KiB]);
  const stream = ctx.stream();

  ctx.recorder.stop();
  assert.equal(ctx.media().stopCount, 1);
  assert.ok(stream.allStopped(), "the microphone is released as soon as recording stops");
  await settle();

  assert.deepEqual(await outcome, { ok: true, text: TRANSCRIPT, stoppedBy: "user" });
  assert.deepEqual(ctx.states, ["requesting", "recording", "uploading", "idle"]);
  assert.equal(ctx.recorder.state, STATES.IDLE);

  // Exactly one request, to the transcription endpoint only (never the chat).
  assert.deepEqual(ctx.fetchImpl.calls.map((c) => c.url), [TRANSCRIBE_URL]);
  const [{ init }] = ctx.fetchImpl.calls;
  assert.equal(init.method, "POST");
  assert.equal(init.credentials, "same-origin");
  assert.equal(init.cache, "no-store");
  assert.equal(init.headers.Accept, "application/json");
  assert.ok(init.signal instanceof AbortSignal);
  const audio = init.body.get("audio");
  assert.equal(audio.name, "recording.webm");
  assert.equal(audio.type, WEBM);
  assert.equal(audio.size, 5 * KiB);
  assert.deepEqual([...init.body.keys()], ["audio"]);
  assert.equal(ctx.timers.count(), 0, "no timer is left behind");
});

test("the uploaded file follows the format the recorder actually used", T, async () => {
  for (const [types, reportedType, name] of [
    [[MP4], "audio/mp4", "recording.m4a"], // Safari
    [[OGG], "audio/ogg; codecs=opus", "recording.ogg"],
    [[WEBM], "video/webm;codecs=opus", "recording.webm"],
    [[WEBM], "", "recording.webm"], // nothing reported: the requested format
  ]) {
    const ctx = setup({ types, recorderOptions: { reportedType } });
    const { outcome } = await recording(ctx);
    ctx.recorder.stop();
    await settle();
    assert.equal((await outcome).ok, true);
    assert.equal(ctx.fetchImpl.calls[0].init.body.get("audio").name, name, reportedType);
  }
});

test("recording stops by itself at 60 seconds and is transcribed", T, async () => {
  const ctx = setup();
  const { outcome } = await recording(ctx);
  assert.equal(ctx.timers.count(), 1);

  ctx.timers.fire(MAX_DURATION_MS);
  assert.ok(ctx.stream().allStopped());
  await settle();

  assert.deepEqual(await outcome, { ok: true, text: TRANSCRIPT, stoppedBy: "time_limit" });
});

test("recording stops early at 1.9 MiB and is still uploaded", T, async () => {
  const ctx = setup();
  // The threshold is 1.9 MiB = 1945.6 KiB.
  const { outcome } = await recording(ctx, [1024 * KiB, 800 * KiB, 100 * KiB]); // 1924 KiB
  assert.equal(ctx.recorder.state, STATES.RECORDING, "still under the threshold");
  assert.equal(ctx.media().stopCount, 0);

  ctx.media().emit(30 * KiB); // 1954 KiB: past 1.9 MiB, still under 2 MiB
  assert.equal(ctx.media().stopCount, 1, "stopped at the early threshold");
  await settle();

  const result = await outcome;
  assert.deepEqual(result, { ok: true, text: TRANSCRIPT, stoppedBy: "size_limit" });
  assert.ok(ctx.fetchImpl.calls[0].init.body.get("audio").size <= MAX_AUDIO_BYTES);
});

test("audio over 2 MiB is never uploaded", T, async () => {
  const ctx = setup();
  // 1900 KiB keeps recording; the next one-second slice crosses both the
  // 1.9 MiB early-stop threshold and the 2 MiB limit at once.
  const { outcome } = await recording(ctx, [1900 * KiB]);
  assert.equal(ctx.recorder.state, STATES.RECORDING);
  ctx.media().emit(200 * KiB);
  assert.equal(ctx.media().stopCount, 1);
  await settle();

  assert.deepEqual(await outcome, { ok: false, code: "too_large" });
  assert.equal(ctx.fetchImpl.calls.length, 0);
  assert.ok(ctx.stream().allStopped());
});

test("a final chunk on stop that crosses 2 MiB is refused too", T, async () => {
  const ctx = setup();
  const { outcome } = await recording(ctx, [1500 * KiB]);
  ctx.media().finalChunk = 600 * KiB; // delivered by stop(): 2.05 MiB in all
  ctx.recorder.stop();
  await settle();

  assert.deepEqual(await outcome, { ok: false, code: "too_large" });
  assert.equal(ctx.fetchImpl.calls.length, 0);
});

test("a recording with no audio is not uploaded", T, async () => {
  const ctx = setup();
  const { outcome } = await recording(ctx, [0]); // empty slices are ignored
  ctx.recorder.stop();
  await settle();
  assert.deepEqual(await outcome, { ok: false, code: "empty_recording" });
  assert.equal(ctx.fetchImpl.calls.length, 0);
});

test("a microphone that goes away ends the recording, which is still transcribed", T, async () => {
  const ctx = setup();
  const { outcome } = await recording(ctx);
  ctx.stream().tracks[0].end();
  await settle();
  assert.deepEqual(await outcome, { ok: true, text: TRANSCRIPT, stoppedBy: "microphone_ended" });
  assert.ok(ctx.stream().allStopped());
});

test("only one session at a time", T, async () => {
  const ctx = setup();
  const { outcome: first } = await recording(ctx);
  assert.deepEqual(await ctx.recorder.start(CLICK), { ok: false, code: "busy" });
  assert.equal(ctx.devices.calls.length, 1);
  ctx.recorder.cancel();
  assert.equal((await first).code, "cancelled");
});

// ---------------------------------------------------------------------------
// Microphone errors
// ---------------------------------------------------------------------------

test("microphone errors are mapped and leave nothing running", T, async () => {
  const cases = [
    ["NotAllowedError", "permission_denied"],
    ["SecurityError", "permission_denied"],
    ["NotFoundError", "no_microphone"],
    ["OverconstrainedError", "no_microphone"],
    ["NotReadableError", "microphone_busy"],
    ["AbortError", "microphone_busy"],
    ["TypeError", "recording_failed"],
  ];
  for (const [name, code] of cases) {
    const ctx = setup({ devices: fakeDevices({ error: Object.assign(new Error(name), { name }) }) });
    assert.deepEqual(await ctx.recorder.start(CLICK), { ok: false, code }, name);
    assert.equal(ctx.recorder.state, STATES.IDLE);
    assert.equal(ctx.MediaRecorder.instances.length, 0);
    assert.equal(ctx.timers.count(), 0);
  }
});

test("getUserMedia throwing synchronously is handled", T, async () => {
  const error = Object.assign(new Error("x"), { name: "NotAllowedError" });
  const ctx = setup({ devices: fakeDevices({ throwSync: error }) });
  assert.deepEqual(await ctx.recorder.start(CLICK), { ok: false, code: "permission_denied" });
  assert.equal(ctx.recorder.state, STATES.IDLE);
});

test("a recorder that cannot be created or started releases the microphone", T, async () => {
  for (const recorderOptions of [{ throwOnConstruct: true }, { throwOnStart: true }]) {
    const ctx = setup({ recorderOptions });
    assert.deepEqual(await ctx.recorder.start(CLICK), { ok: false, code: "recording_failed" });
    assert.ok(ctx.stream().allStopped(), JSON.stringify(recorderOptions));
    assert.equal(ctx.timers.count(), 0);
  }
});

test("a recorder error stops everything and uploads nothing", T, async () => {
  const ctx = setup();
  const { outcome } = await recording(ctx);
  ctx.media().fail();
  assert.deepEqual(await outcome, { ok: false, code: "recording_failed" });
  assert.ok(ctx.stream().allStopped());
  assert.equal(ctx.timers.count(), 0);
  await settle();
  assert.equal(ctx.fetchImpl.calls.length, 0);
});

// ---------------------------------------------------------------------------
// Cancellation and stale callbacks
// ---------------------------------------------------------------------------

test("cancel while permission is pending releases a late grant at once", T, async () => {
  const devices = fakeDevices({ manual: true });
  const ctx = setup({ devices });
  const outcome = ctx.recorder.start(CLICK);

  ctx.recorder.cancel();
  assert.deepEqual(await outcome, { ok: false, code: "cancelled" });

  const stream = devices.grant(); // the user allows the microphone afterwards
  await settle();
  assert.ok(stream.allStopped(), "a late grant is released immediately");
  assert.equal(ctx.MediaRecorder.instances.length, 0, "nothing records");
  assert.equal(ctx.recorder.state, STATES.IDLE);
});

test("stop while permission is pending cancels", T, async () => {
  const devices = fakeDevices({ manual: true });
  const ctx = setup({ devices });
  const outcome = ctx.recorder.start(CLICK);
  ctx.recorder.stop();
  assert.deepEqual(await outcome, { ok: false, code: "cancelled" });
  devices.deny("NotAllowedError"); // a late denial changes nothing
  await settle();
  assert.equal(ctx.recorder.state, STATES.IDLE);
});

test("cancel while recording discards the audio and uploads nothing", T, async () => {
  const ctx = setup();
  const { outcome } = await recording(ctx, [8 * KiB]);
  const media = ctx.media();

  ctx.recorder.cancel();
  assert.deepEqual(await outcome, { ok: false, code: "cancelled" });
  assert.ok(ctx.stream().allStopped());
  assert.equal(media.stopCount, 1);
  await settle(); // the recorder's late "stop" event is ignored
  assert.equal(ctx.fetchImpl.calls.length, 0);
  assert.equal(ctx.timers.count(), 0);
});

test("cancel during the upload aborts the request and ignores its result", T, async () => {
  const signals = [];
  const ctx = setup({ route: hangingRoute(signals) });
  const { outcome } = await recording(ctx);
  ctx.recorder.stop();
  await settle();
  assert.equal(ctx.recorder.state, STATES.UPLOADING);

  ctx.recorder.cancel();
  assert.deepEqual(await outcome, { ok: false, code: "cancelled" });
  assert.equal(signals[0].aborted, true);
  await settle();
  assert.equal(ctx.recorder.state, STATES.IDLE);
  assert.equal(ctx.timers.count(), 0);
});

test("callbacks from an earlier session never affect the next one", T, async () => {
  const ctx = setup();
  const { outcome: first } = await recording(ctx);
  const oldMedia = ctx.media();
  const oldStream = ctx.stream();
  ctx.recorder.cancel();
  await first;

  const { outcome: second } = await recording(ctx, [2 * KiB]);
  // Late events from the first session:
  oldMedia.emit(4 * KiB);
  oldMedia.onstop?.({});
  oldMedia.onerror?.({});
  oldStream.tracks[0].onended?.({});
  await settle();
  assert.equal(ctx.recorder.state, STATES.RECORDING);
  assert.equal(ctx.fetchImpl.calls.length, 0);

  ctx.recorder.stop();
  await settle();
  const result = await second;
  assert.equal(result.ok, true);
  assert.equal(ctx.fetchImpl.calls[0].init.body.get("audio").size, 2 * KiB); // only its own audio
});

test("a late upload result after a new session started is ignored", T, async () => {
  let release;
  const ctx = setup({
    route: () =>
      new Promise((resolve) => {
        release = () => resolve(ok("STALE"));
      }),
  });
  const { outcome: first } = await recording(ctx);
  ctx.recorder.stop();
  await settle();
  ctx.recorder.cancel(); // while uploading
  await first;

  const { outcome: second } = await recording(ctx);
  release(); // the first upload answers late
  await settle();
  assert.equal(ctx.recorder.state, STATES.RECORDING);
  ctx.recorder.cancel();
  assert.equal((await second).code, "cancelled");
});

test("cancel and stop are safe at any time", T, () => {
  const ctx = setup();
  ctx.recorder.cancel();
  ctx.recorder.stop();
  assert.equal(ctx.recorder.state, STATES.IDLE);
  assert.deepEqual(ctx.states, []);
});

// ---------------------------------------------------------------------------
// Upload: timeout, network and server responses
// ---------------------------------------------------------------------------

test("an upload that takes too long is aborted after 45 seconds", T, async () => {
  const signals = [];
  const ctx = setup({ route: hangingRoute(signals) });
  const { outcome } = await recording(ctx);
  ctx.recorder.stop();
  await settle();

  ctx.timers.fire(UPLOAD_TIMEOUT_MS);
  assert.deepEqual(await outcome, { ok: false, code: "upload_timeout" });
  assert.equal(signals[0].aborted, true);
  assert.equal(ctx.recorder.state, STATES.IDLE);
});

test("a network failure is reported", T, async () => {
  const ctx = setup({ route: () => new TypeError("Failed to fetch") });
  const { outcome } = await recording(ctx);
  ctx.recorder.stop();
  await settle();
  assert.deepEqual(await outcome, { ok: false, code: "network_error" });
});

test("server errors map to safe codes", T, async () => {
  const detail = (status, code) => () => jsonResponse({ detail: { code, message: "server text" } }, { status });
  const cases = [
    [detail(403, "origin_not_allowed"), "origin_not_allowed"],
    [detail(415, "unsupported_media_type"), "invalid_upload"],
    [detail(413, "audio_too_large"), "too_large"],
    [detail(408, "upload_timeout"), "upload_timeout"],
    [detail(400, "invalid_upload"), "invalid_upload"],
    [detail(415, "unsupported_audio"), "unsupported_audio"],
    [detail(422, "audio_not_transcribable"), "audio_not_transcribable"],
    [detail(504, "transcription_timeout"), "transcription_timeout"],
    [detail(503, "transcription_unavailable"), "transcription_unavailable"],
    [detail(500, "transcription_failed"), "transcription_failed"],
    [() => jsonResponse({ detail: "Not Found" }, { status: 404 }), "transcription_disabled"],
    [() => jsonResponse({ detail: { code: "something_new" } }, { status: 503 }), "transcription_unavailable"],
    [() => jsonResponse({}, { status: 429 }), "transcription_unavailable"],
    [() => new Response("<html>bad gateway</html>", { status: 502 }), "transcription_failed"],
  ];
  for (const [route, code] of cases) {
    const ctx = setup({ route });
    const { outcome } = await recording(ctx);
    ctx.recorder.stop();
    await settle();
    assert.deepEqual(await outcome, { ok: false, code });
    assert.ok(ERROR_MESSAGES[code], `a message exists for ${code}`);
  }
});

test("a 200 without a text string is a failure; empty text means no speech", T, async () => {
  for (const [route, expected] of [
    [() => jsonResponse({ text: 42 }), { ok: false, code: "transcription_failed" }],
    [() => jsonResponse(null), { ok: false, code: "transcription_failed" }],
    [() => new Response("not json", { status: 200 }), { ok: false, code: "transcription_failed" }],
    [() => ok("   \n "), { ok: false, code: "no_speech" }],
    [() => ok(`  ${TRANSCRIPT}\n`), { ok: true, text: TRANSCRIPT, stoppedBy: "user" }],
  ]) {
    const ctx = setup({ route });
    const { outcome } = await recording(ctx);
    ctx.recorder.stop();
    await settle();
    assert.deepEqual(await outcome, expected);
  }
});

test("a throwing state listener does not break the session", T, async () => {
  const ctx = setup({
    onStateChange: () => {
      throw new Error("view broke");
    },
  });
  const { outcome } = await recording(ctx);
  ctx.recorder.stop();
  await settle();
  assert.equal((await outcome).ok, true);
  assert.ok(ctx.stream().allStopped());
});

test("every outcome code has a user-facing message", T, () => {
  const produced = [
    "unsupported", "insecure_context", "not_user_action", "busy", "permission_denied",
    "no_microphone", "microphone_busy", "recording_failed", "empty_recording", "too_large",
    "upload_timeout", "network_error", "no_speech", "origin_not_allowed", "invalid_upload",
    "unsupported_audio", "audio_not_transcribable", "transcription_timeout",
    "transcription_unavailable", "transcription_disabled", "transcription_failed", "cancelled",
  ];
  for (const code of produced) assert.equal(typeof ERROR_MESSAGES[code], "string", code);
  assert.ok(Object.isFrozen(ERROR_MESSAGES));
});

// ---------------------------------------------------------------------------
// Module boundaries
// ---------------------------------------------------------------------------

test("no DOM, storage, chat or ambient browser access", T, () => {
  const code = SOURCE.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "");
  for (const forbidden of [
    "document",
    "window",
    "navigator",
    "localStorage",
    "sessionStorage",
    "indexedDB",
    "cookie",
    "/api/chat",
    "innerHTML",
    "speechSynthesis",
    "XMLHttpRequest",
    "sendBeacon",
  ]) {
    assert.ok(!code.includes(forbidden), `recorder.js must not use ${forbidden}`);
  }
  assert.ok(!/\(\?<[=!]/.test(SOURCE), "no regex lookbehind (Safari before 16.4)");
});

test("the browser global is BluePeakRecorder", T, () => {
  assert.match(SOURCE, /root\.BluePeakRecorder = api;/);
  const recorder = createRecorder({});
  for (const name of ["start", "stop", "cancel"]) assert.equal(typeof recorder[name], "function");
  assert.equal(recorder.supported, false);
});
