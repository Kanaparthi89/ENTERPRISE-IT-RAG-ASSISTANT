// Voice V3a; Security S1-4c-2: static/js/recorder.js (microphone dictation, no UI).
// Deterministic: the microphone, AudioContext, AudioWorkletNode (and its
// port), fetch and timers are fakes. The worklet itself is tested in
// pcm-worklet.test.mjs; here its messages are simulated.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { test } from "node:test";

import { fakeFetch, fakeTimers, jsonResponse, tick } from "./helpers.mjs";

const require = createRequire(import.meta.url);
const rec = require("../../static/js/recorder.js");
const SOURCE = readFileSync(new URL("../../static/js/recorder.js", import.meta.url), "utf-8");
const HEADER_HEX = readFileSync(new URL("./fixtures/canonical-wav-header.hex", import.meta.url), "utf-8").trim();

const {
  CHUNK_SAMPLES,
  DURATION_BACKSTOP_MS,
  ERROR_MESSAGES,
  FLUSH_TIMEOUT_MS,
  LOCAL_COOLDOWN_MAX_SECONDS,
  MAX_AUDIO_BYTES,
  MAX_DURATION_MS,
  MAX_SAMPLES,
  MAX_WAV_BYTES,
  PROCESSOR_NAME,
  SAMPLE_RATE,
  SETUP_TIMEOUT_MS,
  STATES,
  TRANSCRIBE_URL,
  UPLOAD_TIMEOUT_MS,
  WORKLET_URL,
  browserDependencies,
  canRecord,
  createRecorder,
  encodeWav,
  messageFor,
  wavHeader,
} = rec;

const T = Object.freeze({ timeout: 10_000 });
const CLICK = Object.freeze({ isTrusted: true });
const TRANSCRIPT = "How do I reset my VPN token?";

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

function fakeDevices({ manual = false, error = null, throwSync = null } = {}) {
  const devices = {
    calls: [],
    streams: [],
    pending: null,
    getUserMedia(constraints) {
      devices.calls.push(constraints);
      if (throwSync) throw Object.assign(new Error(throwSync), { name: throwSync });
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

/**
 * AudioContext and AudioWorkletNode doubles. ``autoFlush``: the node answers
 * a flush with "done" (as the worklet does); off to test the flush timeout.
 */
function fakeAudio({
  autoFlush = true,
  constructThrows = false,
  resumeFails = false,
  addModuleFails = false,
  nodeThrows = false,
  sourceThrows = false,
  stateAfterResume = "running",
  worklet = true,
  addModuleHangs = false,
  resumePlan = null, // per call: "running" | "interrupted" | "hang" | "reject"
} = {}) {
  const audio = { contexts: [], nodes: [], autoFlush, log: [], resumeCalls: 0, pending: [] };

  class FakeAudioWorkletNode {
    constructor(context, name, options) {
      if (nodeThrows) throw new Error("cannot create node");
      this.context = context;
      this.name = name;
      this.options = options;
      this.connected = [];
      this.disconnected = 0;
      this.onprocessorerror = null;
      const node = this;
      this.port = {
        posted: [],
        onmessage: null,
        onmessageerror: null,
        closed: 0,
        postMessage(message) {
          node.port.posted.push(message);
          if (message && message.type === "flush" && audio.autoFlush) queueMicrotask(() => node.send({ type: "done" }));
        },
        close() {
          node.port.closed += 1;
        },
      };
      audio.nodes.push(this);
    }

    connect(target) {
      this.connected.push(target);
    }

    disconnect() {
      this.disconnected += 1;
    }

    send(data) {
      this.port.onmessage?.({ data });
    }

    /** Post ``count`` samples as the worklet would. */
    emit(count, value = 1000) {
      this.send({ type: "chunk", samples: new Int16Array(count).fill(value) });
    }
  }

  class FakeAudioContext {
    constructor() {
      if (constructThrows) throw new Error("no audio");
      this.state = "suspended";
      this.sampleRate = 48000;
      this.closed = 0;
      this.onstatechange = null;
      this.destination = { kind: "destination" };
      this.source = null;
      this._worklet = {
        modules: [],
        addModule: (url) => {
          this._worklet.modules.push(url);
          if (addModuleHangs) {
            // Held open; the test can settle it late with audio.settleLate().
            return new Promise((resolve) => audio.pending.push(resolve));
          }
          return addModuleFails ? Promise.reject(new Error("404")) : Promise.resolve();
        },
      };
      audio.contexts.push(this);
      audio.log.push("context");
    }

    resume() {
      audio.log.push("resume");
      audio.resumeCalls += 1;
      if (resumePlan) {
        const step = resumePlan[Math.min(audio.resumeCalls, resumePlan.length) - 1];
        if (step === "hang") return new Promise((resolve) => audio.pending.push(resolve));
        if (step === "reject") return Promise.reject(new Error("not allowed"));
        this.setState(step);
        return Promise.resolve();
      }
      if (resumeFails) return Promise.reject(new Error("not allowed"));
      this.setState(stateAfterResume);
      return Promise.resolve();
    }

    close() {
      this.closed += 1;
      this.setState("closed");
      return Promise.resolve();
    }

    createMediaStreamSource(stream) {
      if (sourceThrows) throw new Error("no source");
      const source = {
        stream,
        connected: [],
        disconnected: 0,
        connect(target) {
          source.connected.push(target);
        },
        disconnect() {
          source.disconnected += 1;
        },
      };
      this.source = source;
      return source;
    }

    setState(next) {
      this.state = next;
      this.onstatechange?.();
    }
  }
  if (worklet) {
    Object.defineProperty(FakeAudioContext.prototype, "audioWorklet", {
      get() {
        return this._worklet;
      },
    });
  }

  audio.AudioContext = FakeAudioContext;
  audio.AudioWorkletNode = FakeAudioWorkletNode;
  audio.context = () => audio.contexts.at(-1);
  audio.node = () => audio.nodes.at(-1);
  // Closed by the recorder, or already closed by the browser (not closed twice).
  audio.settleLate = () => {
    for (const resolve of audio.pending.splice(0)) resolve();
  };
  audio.allClosed = () => audio.contexts.every((context) => context.closed >= 1 || context.state === "closed");
  return audio;
}

const ok = (text = TRANSCRIPT) => jsonResponse({ text });

function hangingRoute(seen) {
  return (init) =>
    new Promise((_, reject) => {
      seen.push(init.signal);
      init.signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
    });
}

function setup({ devices = fakeDevices(), audio = fakeAudio(), route = () => ok(), secure = true, clock = { now: 1_000 }, ...options } = {}) {
  const timers = fakeTimers();
  const fetchImpl = fakeFetch({ [TRANSCRIBE_URL]: route });
  const states = [];
  const recorder = createRecorder({
    mediaDevices: devices,
    AudioContext: audio.AudioContext,
    AudioWorkletNode: audio.AudioWorkletNode,
    isSecureContext: secure,
    fetchImpl,
    setTimeout: timers.setTimeout,
    clearTimeout: timers.clearTimeout,
    now: () => clock.now,
    onStateChange: (state, detail) => states.push(detail.code ? `${state}:${detail.code}` : state),
    ...options,
  });
  return {
    recorder,
    devices,
    audio,
    timers,
    fetchImpl,
    states,
    clock,
    node: () => audio.node(),
    context: () => audio.context(),
    stream: () => devices.streams.at(-1),
    /** Everything released: tracks stopped, nodes disconnected, contexts closed. */
    released() {
      const streams = devices.streams.every((stream) => stream.allStopped());
      const nodes = audio.nodes.every((node) => node.disconnected >= 1 && node.port.onmessage === null);
      return streams && nodes && audio.allClosed();
    },
  };
}

/** Start, let permission and audio resolve, and post ``chunks`` (sample counts). */
async function recording(ctx, chunks = [CHUNK_SAMPLES]) {
  const outcome = ctx.recorder.start(CLICK);
  await settle();
  assert.equal(ctx.recorder.state, STATES.RECORDING);
  for (const count of chunks) ctx.node().emit(count);
  return { outcome };
}

async function settle() {
  for (let i = 0; i < 12; i++) await tick();
}

async function uploadedBytes(ctx) {
  const { init } = ctx.fetchImpl.calls.find((call) => call.url === TRANSCRIBE_URL);
  const file = init.body.get("audio");
  return { file, bytes: new Uint8Array(await file.arrayBuffer()) };
}

// ---------------------------------------------------------------------------
// Limits, WAV construction and helpers
// ---------------------------------------------------------------------------

test("the limits are the approved ones", T, () => {
  assert.equal(SAMPLE_RATE, 16000);
  assert.equal(MAX_DURATION_MS, 60_000);
  assert.equal(MAX_SAMPLES, 960_000);
  assert.equal(MAX_WAV_BYTES, 1_920_044);
  assert.ok(MAX_WAV_BYTES < MAX_AUDIO_BYTES);
  assert.equal(MAX_AUDIO_BYTES, 2 * 1024 * 1024);
  assert.equal(UPLOAD_TIMEOUT_MS, 75_000);
  assert.equal(DURATION_BACKSTOP_MS, 61_000);
  assert.equal(SETUP_TIMEOUT_MS, 10_000);
  assert.equal(LOCAL_COOLDOWN_MAX_SECONDS, 60);
  assert.equal(WORKLET_URL, "/static/js/pcm-recorder-worklet.js");
  assert.equal(PROCESSOR_NAME, "bluepeak-pcm-recorder");
});

test("the WAV header is the canonical 44 bytes (shared reference with the server tests)", T, () => {
  const header = wavHeader(16000);
  assert.equal(header.length, 44);
  assert.equal(Buffer.from(header).toString("hex"), HEADER_HEX);
  const view = new DataView(header.buffer);
  assert.equal(view.getUint32(4, true), 36 + 32000);
  assert.equal(view.getUint32(40, true), 32000);
  assert.equal(Buffer.from(wavHeader(MAX_SAMPLES).slice(40)).readUInt32LE(0), 1_920_000);
  for (const bad of [-1, 1.5, MAX_SAMPLES + 1, Number.NaN, "10"]) assert.throws(() => wavHeader(bad), RangeError);
});

test("encodeWav builds header + little-endian samples, exactly", T, async () => {
  const blob = encodeWav([Int16Array.from([1, -2, 32767]), Int16Array.from([-32768])], 4);
  assert.equal(blob.type, "audio/wav");
  const bytes = new Uint8Array(await blob.arrayBuffer());
  assert.equal(bytes.length, 44 + 8);
  assert.deepEqual([...bytes.slice(44)], [1, 0, 0xfe, 0xff, 0xff, 0x7f, 0x00, 0x80]);
  assert.equal(new DataView(bytes.buffer).getUint32(40, true), 8);
  assert.throws(() => encodeWav([Int16Array.from([1])], 2), RangeError); // count mismatch
  assert.throws(() => encodeWav([], 0), RangeError); // empty
  assert.throws(() => encodeWav([new Float32Array(2)], 2), TypeError);
  assert.throws(() => encodeWav([new Int16Array(MAX_SAMPLES + 1)], MAX_SAMPLES + 1), RangeError);
});

test("browserDependencies reads only what the recorder needs; canRecord checks it", T, () => {
  const audio = fakeAudio();
  const scope = {
    navigator: { mediaDevices: fakeDevices() },
    AudioContext: audio.AudioContext,
    AudioWorkletNode: audio.AudioWorkletNode,
    isSecureContext: true,
    localStorage: "never read",
  };
  const deps = browserDependencies(scope);
  assert.deepEqual(Object.keys(deps).sort(), ["AudioContext", "AudioWorkletNode", "isSecureContext", "mediaDevices"]);
  assert.equal(canRecord(scope), true);
  for (const change of [
    { isSecureContext: false },
    { isSecureContext: "true" },
    { navigator: {} },
    { navigator: { mediaDevices: {} } },
    { AudioContext: undefined },
    { AudioWorkletNode: undefined },
    { AudioContext: fakeAudio({ worklet: false }).AudioContext }, // no audioWorklet: no ScriptProcessor fallback
  ]) {
    assert.equal(canRecord({ ...scope, ...change }), false, JSON.stringify(Object.keys(change)));
  }
  assert.equal(canRecord(undefined), false);
  assert.equal(browserDependencies(undefined).isSecureContext, false);
});

test("messageFor adds the wait when there is one", T, () => {
  assert.equal(messageFor({ code: "voice_busy" }), ERROR_MESSAGES.voice_busy);
  assert.equal(messageFor({ code: "voice_busy", retryAfter: 5 }), `${ERROR_MESSAGES.voice_busy} You can try again in 5 seconds.`);
  assert.equal(messageFor({ code: "quota_busy", retryAfter: 1 }), `${ERROR_MESSAGES.quota_busy} You can try again in 1 second.`);
  assert.match(messageFor({ code: "quota_minutes_limit", retryAfter: 125 }), /about 3 minutes\.$/);
  assert.match(messageFor({ code: "quota_daily_limit", retryAfter: 7_200 }), /about 2 hours\.$/);
  assert.equal(messageFor({ code: "something_unknown" }), ERROR_MESSAGES.transcription_failed);
  assert.equal(messageFor(undefined), ERROR_MESSAGES.transcription_failed);
});

// ---------------------------------------------------------------------------
// Preconditions
// ---------------------------------------------------------------------------

test("only a real user action starts a session", T, async () => {
  const ctx = setup();
  for (const event of [undefined, {}, { isTrusted: false }]) {
    assert.deepEqual(await ctx.recorder.start(event), { ok: false, code: "not_user_action" });
  }
  assert.equal(ctx.devices.calls.length, 0);
  assert.equal(ctx.audio.contexts.length, 0);
});

test("an insecure context never touches the microphone or audio", T, async () => {
  const ctx = setup({ secure: false });
  assert.deepEqual(await ctx.recorder.start(CLICK), { ok: false, code: "insecure_context" });
  assert.equal(ctx.devices.calls.length, 0);
  assert.equal(ctx.audio.contexts.length, 0);
});

test("missing APIs (including AudioWorklet) are reported as unsupported", T, async () => {
  for (const options of [
    { mediaDevices: undefined },
    { AudioContext: undefined },
    { AudioWorkletNode: undefined },
    { AudioContext: fakeAudio({ worklet: false }).AudioContext },
    { fetchImpl: undefined },
  ]) {
    const ctx = setup(options);
    assert.equal(ctx.recorder.supported, false);
    assert.deepEqual(await ctx.recorder.start(CLICK), { ok: false, code: "unsupported" });
    assert.equal(ctx.devices.calls.length, 0);
  }
});

test("the context, resume, worklet module and microphone are all requested synchronously in the gesture", T, () => {
  const ctx = setup();
  ctx.recorder.start(CLICK);
  assert.deepEqual(ctx.audio.log, ["context", "resume"]);
  assert.equal(ctx.devices.calls.length, 1);
  assert.deepEqual(ctx.devices.calls[0], { audio: true, video: false });
  assert.deepEqual(ctx.context()._worklet.modules, [WORKLET_URL]);
  assert.equal(ctx.recorder.state, STATES.REQUESTING);
  ctx.recorder.cancel();
});

// ---------------------------------------------------------------------------
// A full session
// ---------------------------------------------------------------------------

test("a full session: record, stop, flush, upload a canonical WAV, transcript", T, async () => {
  const ctx = setup();
  const { outcome } = await recording(ctx, [CHUNK_SAMPLES, CHUNK_SAMPLES, 1000]);
  const node = ctx.node();
  assert.equal(node.name, PROCESSOR_NAME);
  assert.deepEqual(node.options.processorOptions, { maxSamples: MAX_SAMPLES, chunkSamples: CHUNK_SAMPLES });
  assert.deepEqual(ctx.context().source.connected, [node]);
  assert.deepEqual(node.connected, [ctx.context().destination]);

  ctx.recorder.stop();
  assert.ok(ctx.stream().allStopped(), "the microphone is released at once");
  await settle();
  assert.deepEqual(node.port.posted, [{ type: "flush" }]);
  assert.deepEqual(await outcome, { ok: true, text: TRANSCRIPT, stoppedBy: "user" });

  const { file, bytes } = await uploadedBytes(ctx);
  const samples = 2 * CHUNK_SAMPLES + 1000;
  assert.equal(file.name, "recording.wav");
  assert.equal(file.type, "audio/wav");
  assert.equal(bytes.length, 44 + 2 * samples);
  assert.equal(Buffer.from(bytes.slice(0, 4)).toString(), "RIFF");
  assert.equal(new DataView(bytes.buffer).getUint32(40, true), 2 * samples);
  const { init } = ctx.fetchImpl.calls[0];
  assert.equal(init.method, "POST");
  assert.equal(init.credentials, "same-origin");
  assert.equal(init.cache, "no-store");
  assert.ok(ctx.released());
  assert.deepEqual(ctx.states, ["requesting", "recording", "uploading", "idle"]);
});

test("the sample cap (the worklet's limit message) stops and uploads at exactly 960,000", T, async () => {
  const ctx = setup();
  const counts = [];
  for (let left = MAX_SAMPLES; left > 0; left -= CHUNK_SAMPLES) counts.push(Math.min(CHUNK_SAMPLES, left));
  const { outcome } = await recording(ctx, counts);
  ctx.node().send({ type: "limit", total: MAX_SAMPLES });
  await settle();
  assert.deepEqual(await outcome, { ok: true, text: TRANSCRIPT, stoppedBy: "time_limit" });
  const { bytes } = await uploadedBytes(ctx);
  assert.equal(bytes.length, MAX_WAV_BYTES);
});

test("the backstop timer stops the recording if the cap message never comes", T, async () => {
  const ctx = setup();
  const { outcome } = await recording(ctx);
  ctx.timers.fire(DURATION_BACKSTOP_MS);
  await settle();
  assert.deepEqual(await outcome, { ok: true, text: TRANSCRIPT, stoppedBy: "time_limit" });
});

test("samples beyond the cap are refused and nothing is uploaded", T, async () => {
  const ctx = setup({ maxSamples: 5000 });
  const { outcome } = await recording(ctx, [4096, 4096]);
  assert.deepEqual(await outcome, { ok: false, code: "too_large" });
  assert.equal(ctx.fetchImpl.calls.length, 0);
  assert.ok(ctx.released());
});

test("a recording with no samples is not uploaded", T, async () => {
  const ctx = setup();
  const { outcome } = await recording(ctx, []);
  ctx.recorder.stop();
  await settle();
  assert.deepEqual(await outcome, { ok: false, code: "empty_recording" });
  assert.equal(ctx.fetchImpl.calls.length, 0);
  assert.ok(ctx.released());
});

test("a chunk that isn't Int16 PCM fails the session", T, async () => {
  const ctx = setup();
  const { outcome } = await recording(ctx, []);
  ctx.node().send({ type: "chunk", samples: new Float32Array(10) });
  assert.deepEqual(await outcome, { ok: false, code: "recording_failed" });
  assert.equal(ctx.fetchImpl.calls.length, 0);
});

test("unknown or malformed worklet messages are ignored", T, async () => {
  const ctx = setup();
  const { outcome } = await recording(ctx);
  for (const data of [null, "chunk", { type: "other" }, { type: "done" }]) ctx.node().send(data);
  ctx.recorder.stop();
  await settle();
  assert.equal((await outcome).ok, true);
});

// ---------------------------------------------------------------------------
// Never upload partial or interrupted audio
// ---------------------------------------------------------------------------

test("a flush that never completes uploads nothing", T, async () => {
  const audio = fakeAudio({ autoFlush: false });
  const ctx = setup({ audio });
  const { outcome } = await recording(ctx);
  ctx.recorder.stop();
  await settle();
  ctx.timers.fire(FLUSH_TIMEOUT_MS);
  assert.deepEqual(await outcome, { ok: false, code: "recording_failed" });
  assert.equal(ctx.fetchImpl.calls.length, 0);
  assert.ok(ctx.released());
  ctx.node().send({ type: "done" }); // a late answer changes nothing
  await settle();
  assert.equal(ctx.fetchImpl.calls.length, 0);
});

test("a flush that can't be sent uploads nothing", T, async () => {
  const ctx = setup();
  const { outcome } = await recording(ctx);
  ctx.node().port.postMessage = () => {
    throw new Error("port closed");
  };
  ctx.recorder.stop();
  assert.deepEqual(await outcome, { ok: false, code: "recording_failed" });
  assert.equal(ctx.fetchImpl.calls.length, 0);
});

test("a microphone that goes away discards the recording", T, async () => {
  const ctx = setup();
  const { outcome } = await recording(ctx);
  ctx.stream().tracks[0].onended();
  assert.deepEqual(await outcome, { ok: false, code: "recording_interrupted" });
  assert.equal(ctx.fetchImpl.calls.length, 0);
  assert.ok(ctx.released());
});

for (const contextState of ["suspended", "interrupted", "closed"]) {
  test(`an AudioContext that becomes ${contextState} discards the recording`, T, async () => {
    const ctx = setup();
    const { outcome } = await recording(ctx);
    ctx.context().setState(contextState);
    assert.deepEqual(await outcome, { ok: false, code: "recording_interrupted" });
    assert.equal(ctx.fetchImpl.calls.length, 0);
    assert.ok(ctx.released());
  });
}

test("an interruption while flushing also discards the recording", T, async () => {
  const audio = fakeAudio({ autoFlush: false });
  const ctx = setup({ audio });
  const { outcome } = await recording(ctx);
  ctx.recorder.stop();
  ctx.context().setState("interrupted");
  assert.deepEqual(await outcome, { ok: false, code: "recording_interrupted" });
  ctx.node().send({ type: "done" });
  await settle();
  assert.equal(ctx.fetchImpl.calls.length, 0);
});

test("a processor error or an undeliverable message fails the session", T, async () => {
  for (const trigger of [(node) => node.onprocessorerror(), (node) => node.port.onmessageerror()]) {
    const ctx = setup();
    const { outcome } = await recording(ctx);
    trigger(ctx.node());
    assert.deepEqual(await outcome, { ok: false, code: "recording_failed" });
    assert.equal(ctx.fetchImpl.calls.length, 0);
    assert.ok(ctx.released());
  }
});

// ---------------------------------------------------------------------------
// Failures while starting: everything is released
// ---------------------------------------------------------------------------

test("microphone errors are mapped and release the audio", T, async () => {
  for (const [name, code] of [
    ["NotAllowedError", "permission_denied"],
    ["SecurityError", "permission_denied"],
    ["NotFoundError", "no_microphone"],
    ["OverconstrainedError", "no_microphone"],
    ["NotReadableError", "microphone_busy"],
    ["AbortError", "microphone_busy"],
    ["TypeError", "recording_failed"],
  ]) {
    const ctx = setup({ devices: fakeDevices({ error: name }) });
    assert.deepEqual(await ctx.recorder.start(CLICK), { ok: false, code }, name);
    assert.equal(ctx.audio.nodes.length, 0);
    assert.ok(ctx.audio.allClosed(), name);
  }
});

test("getUserMedia throwing synchronously is handled", T, async () => {
  const ctx = setup({ devices: fakeDevices({ throwSync: "NotAllowedError" }) });
  assert.deepEqual(await ctx.recorder.start(CLICK), { ok: false, code: "permission_denied" });
  assert.ok(ctx.audio.allClosed());
});

for (const [label, options, code] of [
  ["the AudioContext can't be created", { constructThrows: true }, "unsupported"],
  ["resume() is refused", { resumeFails: true }, "recording_failed"],
  ["the worklet module doesn't load", { addModuleFails: true }, "recording_failed"],
  ["the context isn't running after resume", { stateAfterResume: "suspended" }, "recording_failed"],
  ["the worklet node can't be created", { nodeThrows: true }, "recording_failed"],
  ["the microphone source can't be created", { sourceThrows: true }, "recording_failed"],
]) {
  test(`${label}: ${code}, microphone and audio released`, T, async () => {
    const ctx = setup({ audio: fakeAudio(options) });
    assert.deepEqual(await ctx.recorder.start(CLICK), { ok: false, code });
    await settle();
    assert.ok(ctx.devices.streams.every((stream) => stream.allStopped()));
    assert.ok(ctx.audio.allClosed());
    assert.equal(ctx.recorder.state, STATES.IDLE);
    assert.equal(ctx.fetchImpl.calls.length, 0);
  });
}

// ---------------------------------------------------------------------------
// Cancellation, late callbacks and repeated sessions
// ---------------------------------------------------------------------------

test("cancel while permission is pending releases a late grant at once", T, async () => {
  const ctx = setup({ devices: fakeDevices({ manual: true }) });
  const outcome = ctx.recorder.start(CLICK);
  ctx.recorder.cancel();
  assert.deepEqual(await outcome, { ok: false, code: "cancelled" });
  assert.ok(ctx.audio.allClosed());
  const late = ctx.devices.grant();
  await settle();
  assert.ok(late.allStopped());
  assert.equal(ctx.audio.nodes.length, 0, "nothing records");
});

test("stop while permission is pending cancels", T, async () => {
  const ctx = setup({ devices: fakeDevices({ manual: true }) });
  const outcome = ctx.recorder.start(CLICK);
  ctx.recorder.stop();
  assert.deepEqual(await outcome, { ok: false, code: "cancelled" });
});

test("cancel while recording or flushing discards the audio", T, async () => {
  for (const flushing of [false, true]) {
    const ctx = setup({ audio: fakeAudio({ autoFlush: false }) });
    const { outcome } = await recording(ctx);
    if (flushing) ctx.recorder.stop();
    ctx.recorder.cancel();
    assert.deepEqual(await outcome, { ok: false, code: "cancelled" });
    ctx.node().send({ type: "done" });
    ctx.node().emit(100);
    await settle();
    assert.equal(ctx.fetchImpl.calls.length, 0);
    assert.ok(ctx.released());
  }
});

test("cancel during the upload aborts it and ignores its result", T, async () => {
  const signals = [];
  const ctx = setup({ route: hangingRoute(signals) });
  const { outcome } = await recording(ctx);
  ctx.recorder.stop();
  await settle();
  assert.equal(ctx.recorder.state, STATES.UPLOADING);
  ctx.recorder.cancel();
  assert.deepEqual(await outcome, { ok: false, code: "cancelled" });
  assert.equal(signals[0].aborted, true);
});

test("callbacks from an earlier session never affect the next one", T, async () => {
  const ctx = setup({ audio: fakeAudio({ autoFlush: false }) });
  const first = await recording(ctx);
  const oldNode = ctx.node();
  const oldContext = ctx.context();
  const oldTrack = ctx.stream().tracks[0];
  ctx.recorder.cancel();
  await first.outcome;

  const second = await recording(ctx);
  oldNode.send({ type: "done" });
  oldNode.emit(500);
  oldNode.onprocessorerror?.();
  oldContext.setState("interrupted");
  oldTrack.onended?.();
  await settle();
  assert.equal(ctx.recorder.state, STATES.RECORDING);
  ctx.node().send({ type: "done" }); // not flushing yet: ignored
  ctx.audio.autoFlush = true;
  ctx.recorder.stop();
  await settle();
  const result = await second.outcome;
  assert.equal(result.ok, true);
  const { bytes } = await uploadedBytes(ctx);
  assert.equal(bytes.length, 44 + 2 * CHUNK_SAMPLES); // only the second session's audio
});

test("a worklet message delivered to an old handler after the session ended changes nothing", T, async () => {
  // Even if a browser delivered an already-queued message to the handler that
  // was installed before cleanup, the finished session can't be revived.
  const ctx = setup({ audio: fakeAudio({ autoFlush: false }) });
  const { outcome } = await recording(ctx);
  const handler = ctx.node().port.onmessage;
  ctx.recorder.stop(); // flushing
  ctx.recorder.cancel();
  assert.deepEqual(await outcome, { ok: false, code: "cancelled" });
  handler({ data: { type: "chunk", samples: new Int16Array(100) } });
  handler({ data: { type: "done" } });
  handler({ data: { type: "limit" } });
  await settle();
  assert.equal(ctx.fetchImpl.calls.length, 0);
  assert.equal(ctx.recorder.state, STATES.IDLE);
  assert.ok(ctx.released());
});

test("a session ends exactly once, whatever arrives late", T, async () => {
  const ctx = setup();
  const resolved = [];
  const outcome = ctx.recorder.start(CLICK).then((value) => resolved.push(value));
  await settle();
  ctx.node().emit(100);
  ctx.recorder.stop();
  await settle();
  ctx.recorder.cancel();
  ctx.node().send({ type: "done" });
  ctx.context().setState("closed");
  await outcome;
  await settle();
  assert.equal(resolved.length, 1);
  assert.equal(ctx.states.filter((state) => state.startsWith("idle")).length, 1);
});

test("twenty start/stop and start/cancel cycles leave nothing behind", T, async () => {
  const ctx = setup();
  for (let i = 0; i < 20; i++) {
    const { outcome } = await recording(ctx, [256]);
    if (i % 2) ctx.recorder.cancel();
    else ctx.recorder.stop();
    await settle();
    await outcome;
  }
  assert.equal(ctx.audio.contexts.length, 20);
  assert.ok(ctx.released());
  assert.ok(ctx.audio.contexts.every((context) => context.closed === 1 && context.onstatechange === null));
  assert.equal(ctx.timers.count(), 0);
  assert.equal(ctx.recorder.state, STATES.IDLE);
});

test("only one session at a time", T, async () => {
  const ctx = setup();
  const { outcome } = await recording(ctx);
  assert.deepEqual(await ctx.recorder.start(CLICK), { ok: false, code: "busy" });
  assert.equal(ctx.audio.contexts.length, 1);
  ctx.recorder.cancel();
  await outcome;
});

test("cancel and stop are safe at any time", T, () => {
  const ctx = setup();
  ctx.recorder.cancel();
  ctx.recorder.stop();
  assert.equal(ctx.recorder.state, STATES.IDLE);
});

// ---------------------------------------------------------------------------
// The upload: timeout, errors, Retry-After
// ---------------------------------------------------------------------------

test("an upload that takes too long is aborted after 75 seconds", T, async () => {
  const signals = [];
  const ctx = setup({ route: hangingRoute(signals) });
  const { outcome } = await recording(ctx);
  ctx.recorder.stop();
  await settle();
  ctx.timers.fire(UPLOAD_TIMEOUT_MS);
  assert.deepEqual(await outcome, { ok: false, code: "upload_timeout" });
  assert.equal(signals[0].aborted, true);
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
    [detail(413, "audio_too_long"), "audio_too_long"],
    [detail(408, "upload_timeout"), "upload_timeout"],
    [detail(400, "invalid_upload"), "invalid_upload"],
    [detail(400, "invalid_audio"), "invalid_audio"],
    [detail(415, "unsupported_audio"), "unsupported_audio"],
    [detail(422, "audio_not_transcribable"), "audio_not_transcribable"],
    [detail(504, "transcription_timeout"), "transcription_timeout"],
    [detail(503, "transcription_unavailable"), "transcription_unavailable"],
    [detail(500, "transcription_failed"), "transcription_failed"],
    [detail(503, "voice_busy"), "voice_busy"],
    [detail(429, "quota_busy"), "quota_busy"],
    [detail(429, "quota_capacity"), "quota_capacity"],
    [detail(429, "quota_daily_limit"), "quota_daily_limit"],
    [detail(429, "quota_minutes_limit"), "quota_minutes_limit"],
    [detail(503, "quota_unavailable"), "quota_unavailable"],
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

test("Retry-After on a busy or quota refusal is reported and respected locally", T, async () => {
  const clock = { now: 1_000 };
  const ctx = setup({
    clock,
    route: () =>
      jsonResponse({ detail: { code: "voice_busy", message: "x" } }, { status: 503, headers: { "Retry-After": "5" } }),
  });
  const { outcome } = await recording(ctx);
  ctx.recorder.stop();
  await settle();
  assert.deepEqual(await outcome, { ok: false, code: "voice_busy", retryAfter: 5 });

  clock.now += 2_000; // too soon: refused without the microphone, audio or server
  assert.deepEqual(await ctx.recorder.start(CLICK), { ok: false, code: "voice_busy", retryAfter: 3 });
  assert.equal(ctx.devices.calls.length, 1);
  assert.equal(ctx.audio.contexts.length, 1);
  assert.equal(ctx.fetchImpl.calls.length, 1);

  clock.now += 3_000; // after the wait: a new session starts
  const next = ctx.recorder.start(CLICK);
  await settle();
  assert.equal(ctx.recorder.state, STATES.RECORDING);
  ctx.recorder.cancel();
  await next;
});

test("Retry-After is ignored when malformed, and only for retryable codes", T, async () => {
  for (const [code, value, expected] of [
    ["quota_busy", "abc", undefined],
    ["quota_busy", "0", undefined],
    ["quota_busy", "Wed, 21 Oct 2026 07:28:00 GMT", undefined],
    ["quota_daily_limit", "999999999", 86_400],
    ["transcription_unavailable", "30", undefined], // not a busy or quota refusal
  ]) {
    const ctx = setup({
      route: () => jsonResponse({ detail: { code } }, { status: 429, headers: { "Retry-After": value } }),
    });
    const { outcome } = await recording(ctx);
    ctx.recorder.stop();
    await settle();
    assert.equal((await outcome).retryAfter, expected, `${code} ${value}`);
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
  assert.ok(ctx.released());
});

test("every outcome code has a user-facing message", T, () => {
  const produced = [
    "unsupported", "insecure_context", "not_user_action", "busy", "permission_denied",
    "no_microphone", "microphone_busy", "recording_failed", "recording_interrupted", "empty_recording",
    "too_large", "audio_too_long", "upload_timeout", "network_error", "no_speech", "origin_not_allowed",
    "invalid_upload", "invalid_audio", "unsupported_audio", "audio_not_transcribable",
    "transcription_timeout", "transcription_unavailable", "transcription_disabled", "transcription_failed",
    "voice_busy", "quota_busy", "quota_capacity", "quota_daily_limit", "quota_minutes_limit",
    "quota_unavailable", "cancelled",
  ];
  for (const code of produced) assert.equal(typeof ERROR_MESSAGES[code], "string", code);
  assert.ok(Object.isFrozen(ERROR_MESSAGES));
});

// ---------------------------------------------------------------------------
// Module boundaries
// ---------------------------------------------------------------------------

test("no DOM, storage, chat or ambient browser access; MediaRecorder is gone", T, () => {
  const code = SOURCE.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "");
  for (const forbidden of [
    "document",
    "window",
    "localStorage",
    "sessionStorage",
    "indexedDB",
    "cookie",
    "/api/chat",
    "innerHTML",
    "speechSynthesis",
    "XMLHttpRequest",
    "sendBeacon",
    "MediaRecorder",
    "ScriptProcessor",
    "createScriptProcessor",
  ]) {
    assert.ok(!code.includes(forbidden), `recorder.js must not use ${forbidden}`);
  }
  // The browser objects come only from the scope passed to browserDependencies().
  assert.deepEqual(code.match(/\bnavigator\b/g), ["navigator", "navigator"]);
  assert.match(code, /source\.navigator \? source\.navigator\.mediaDevices : undefined/);
  assert.ok(!/\(\?<[=!]/.test(SOURCE), "no regex lookbehind (Safari before 16.4)");
});

test("the browser global is BluePeakRecorder", T, () => {
  assert.match(SOURCE, /root\.BluePeakRecorder = api;/);
  const recorder = createRecorder({});
  for (const name of ["start", "stop", "cancel"]) assert.equal(typeof recorder[name], "function");
  assert.equal(recorder.supported, false);
});


// ---------------------------------------------------------------------------
// S1-4c-2 remediation: bounded setup after the microphone grant (MEDIUM-1)
// ---------------------------------------------------------------------------

const micLive = (ctx) => ctx.devices.streams.some((stream) => !stream.allStopped());

for (const [label, audioOptions] of [
  ["addModule never settles", { addModuleHangs: true }],
  ["resume never settles", { resumePlan: ["hang"] }],
]) {
  test(`${label}: the session fails after 10 s and releases the microphone and audio`, T, async () => {
    const ctx = setup({ audio: fakeAudio(audioOptions) });
    const outcome = ctx.recorder.start(CLICK);
    await settle();
    assert.equal(ctx.recorder.state, STATES.REQUESTING);
    assert.ok(micLive(ctx), "granted, so live while setting up");
    ctx.timers.fire(SETUP_TIMEOUT_MS);
    assert.deepEqual(await outcome, { ok: false, code: "recording_failed" });
    assert.equal(micLive(ctx), false);
    assert.ok(ctx.audio.allClosed());
    assert.equal(ctx.timers.count(), 0);
    assert.equal(ctx.context().onstatechange, null);
    assert.equal(ctx.recorder.state, STATES.IDLE);

    ctx.audio.settleLate(); // the old promises finish after the timeout
    await settle();
    assert.equal(ctx.audio.nodes.length, 0, "nothing is revived");
    assert.equal(ctx.recorder.state, STATES.IDLE);
    assert.equal(ctx.fetchImpl.calls.length, 0);
  });
}

test("the setup timer never runs while the user is deciding on permission", T, async () => {
  const ctx = setup({ devices: fakeDevices({ manual: true }), audio: fakeAudio({ addModuleHangs: true }) });
  const outcome = ctx.recorder.start(CLICK);
  await settle();
  assert.equal(ctx.timers.count(), 0); // the prompt may take as long as the user needs
  ctx.timers.fire(SETUP_TIMEOUT_MS);
  assert.equal(ctx.recorder.state, STATES.REQUESTING);
  ctx.devices.grant();
  await settle();
  assert.equal(ctx.timers.count(), 1); // from the grant on, bounded
  ctx.timers.fire(SETUP_TIMEOUT_MS);
  assert.deepEqual(await outcome, { ok: false, code: "recording_failed" });
});

test("cancelling before the timeout clears it and releases everything", T, async () => {
  const ctx = setup({ audio: fakeAudio({ addModuleHangs: true }) });
  const outcome = ctx.recorder.start(CLICK);
  await settle();
  ctx.recorder.cancel();
  assert.deepEqual(await outcome, { ok: false, code: "cancelled" });
  assert.equal(ctx.timers.count(), 0);
  assert.equal(micLive(ctx), false);
  ctx.audio.settleLate();
  await settle();
  assert.equal(ctx.audio.nodes.length, 0);
});

test("a setup that finishes in time clears the timer and records normally", T, async () => {
  const ctx = setup({ audio: fakeAudio({ addModuleHangs: true }) });
  const outcome = ctx.recorder.start(CLICK);
  await settle();
  ctx.audio.settleLate(); // within the 10 s
  await settle();
  assert.equal(ctx.recorder.state, STATES.RECORDING);
  ctx.timers.fire(SETUP_TIMEOUT_MS); // a stale timeout does nothing
  assert.equal(ctx.recorder.state, STATES.RECORDING);
  ctx.node().emit(1000);
  ctx.recorder.stop();
  await settle();
  assert.equal((await outcome).ok, true);
  assert.ok(ctx.released());
});

test("ten sessions that time out in setup leak no microphone, context or timer", T, async () => {
  const ctx = setup({ audio: fakeAudio({ addModuleHangs: true }) });
  for (let i = 0; i < 10; i++) {
    const outcome = ctx.recorder.start(CLICK);
    await settle();
    ctx.timers.fire(SETUP_TIMEOUT_MS);
    assert.deepEqual(await outcome, { ok: false, code: "recording_failed" });
  }
  ctx.audio.settleLate();
  await settle();
  assert.equal(ctx.audio.contexts.length, 10);
  assert.ok(ctx.audio.allClosed());
  assert.equal(micLive(ctx), false);
  assert.equal(ctx.timers.count(), 0);
  assert.equal(ctx.audio.nodes.length, 0);
});

// ---------------------------------------------------------------------------
// One bounded extra resume() when the context isn't running (LOW-2)
// ---------------------------------------------------------------------------

test("a context interrupted during setup is resumed once more and records", T, async () => {
  const ctx = setup({ audio: fakeAudio({ resumePlan: ["interrupted", "running"] }) });
  const { outcome } = await recording(ctx);
  assert.equal(ctx.audio.resumeCalls, 2);
  ctx.recorder.stop();
  await settle();
  assert.equal((await outcome).ok, true);
});

test("a context still not running after the one retry fails, without a third try", T, async () => {
  for (const plan of [["interrupted", "interrupted"], ["suspended", "reject"]]) {
    const ctx = setup({ audio: fakeAudio({ resumePlan: plan }) });
    assert.deepEqual(await ctx.recorder.start(CLICK), { ok: false, code: "recording_failed" });
    assert.equal(ctx.audio.resumeCalls, 2, plan.join());
    assert.equal(micLive(ctx), false);
    assert.ok(ctx.audio.allClosed());
  }
});

test("a retry that never settles is bounded by the same setup timer", T, async () => {
  const ctx = setup({ audio: fakeAudio({ resumePlan: ["interrupted", "hang"] }) });
  const outcome = ctx.recorder.start(CLICK);
  await settle();
  assert.equal(ctx.audio.resumeCalls, 2);
  ctx.timers.fire(SETUP_TIMEOUT_MS);
  assert.deepEqual(await outcome, { ok: false, code: "recording_failed" });
  assert.equal(micLive(ctx), false);
});

test("an interruption while recording is never retried: the recording is discarded", T, async () => {
  const ctx = setup({ audio: fakeAudio({ resumePlan: ["running"] }) });
  const { outcome } = await recording(ctx);
  ctx.context().setState("interrupted");
  assert.deepEqual(await outcome, { ok: false, code: "recording_interrupted" });
  assert.equal(ctx.audio.resumeCalls, 1);
  assert.equal(ctx.fetchImpl.calls.length, 0);
});

// ---------------------------------------------------------------------------
// Local cooldown: short-term and capped; never for daily limits (LOW-3)
// ---------------------------------------------------------------------------

function refusing(code, retryAfter, status = 429) {
  return () => jsonResponse({ detail: { code, message: "x" } }, { status, headers: { "Retry-After": String(retryAfter) } });
}

test("a daily limit is reported with its wait but never blocks the next session locally", T, async () => {
  for (const code of ["quota_daily_limit", "quota_minutes_limit"]) {
    const clock = { now: 1_000 };
    const ctx = setup({ clock, route: refusing(code, 7_200) });
    const { outcome } = await recording(ctx);
    ctx.recorder.stop();
    await settle();
    assert.deepEqual(await outcome, { ok: false, code, retryAfter: 7_200 }); // the message shows the wait
    // A different identity (for example a newly redeemed access code) may be
    // allowed: the next attempt goes to the server, which decides.
    const next = ctx.recorder.start(CLICK);
    await settle();
    assert.equal(ctx.recorder.state, STATES.RECORDING, code);
    ctx.recorder.cancel();
    await next;
  }
});

test("a short-term refusal blocks locally for at most 60 seconds", T, async () => {
  const clock = { now: 1_000 };
  const ctx = setup({ clock, route: refusing("quota_busy", 3_600) });
  const { outcome } = await recording(ctx);
  ctx.recorder.stop();
  await settle();
  assert.equal((await outcome).retryAfter, 3_600);
  clock.now += 59_000;
  assert.deepEqual(await ctx.recorder.start(CLICK), { ok: false, code: "quota_busy", retryAfter: 1 });
  clock.now += 1_000;
  const next = ctx.recorder.start(CLICK);
  await settle();
  assert.equal(ctx.recorder.state, STATES.RECORDING);
  ctx.recorder.cancel();
  await next;
});
