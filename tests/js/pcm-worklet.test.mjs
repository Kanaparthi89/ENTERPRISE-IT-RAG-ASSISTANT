// Security S1-4c-2: static/js/pcm-recorder-worklet.js, the shipped file,
// loaded in a VM with the AudioWorklet globals stubbed. Deterministic PCM
// fixtures only (sines, DC, square waves); no audio device.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

const SOURCE = readFileSync(new URL("../../static/js/pcm-recorder-worklet.js", import.meta.url), "utf-8");
const T = Object.freeze({ timeout: 60_000 });
const OUT = 16000;

/** Load the worklet at ``rate`` and return a driver for one processor instance. */
function load(rate, processorOptions = {}) {
  const posted = [];
  const registered = {};
  class AudioWorkletProcessor {
    constructor() {
      this.port = {
        onmessage: null,
        postMessage(message, transfer) {
          posted.push({ message, transfer });
        },
      };
    }
  }
  const context = vm.createContext({
    sampleRate: rate,
    AudioWorkletProcessor,
    registerProcessor(name, processor) {
      registered[name] = processor;
    },
  });
  vm.runInContext(SOURCE, context);
  const Processor = registered["bluepeak-pcm-recorder"];
  assert.equal(typeof Processor, "function", "registers bluepeak-pcm-recorder");
  const processor = new Processor({ processorOptions });
  return {
    processor,
    posted,
    context,
    /** Feed mono (or multi-channel) samples in render quanta of ``block`` frames. */
    feed(channels, block = 128) {
      const frames = channels[0].length;
      for (let start = 0; start < frames; start += block) {
        const end = Math.min(frames, start + block);
        assert.equal(processor.process([channels.map((channel) => Float32Array.from(channel.subarray(start, end)))]), true);
      }
    },
    flush() {
      processor.port.onmessage({ data: { type: "flush" } });
    },
    samples() {
      const chunks = posted.filter((entry) => entry.message.type === "chunk").map((entry) => entry.message.samples);
      const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
      const out = new Int16Array(total);
      let offset = 0;
      for (const chunk of chunks) {
        out.set(chunk, offset);
        offset += chunk.length;
      }
      return out;
    },
    events: () => posted.map((entry) => entry.message.type).filter((type) => type !== "chunk"),
  };
}

function sine(rate, frequency, seconds, amplitude = 0.5) {
  const frames = Math.round(rate * seconds);
  const out = new Float32Array(frames);
  for (let i = 0; i < frames; i++) out[i] = amplitude * Math.sin((2 * Math.PI * frequency * i) / rate);
  return out;
}

/** RMS of the steady-state middle of ``samples`` (as a fraction of full scale). */
function rms(samples, skip = 800) {
  let sum = 0;
  let count = 0;
  for (let i = skip; i < samples.length - skip; i++) {
    const value = samples[i] / 32767;
    sum += value * value;
    count += 1;
  }
  return Math.sqrt(sum / count);
}

/** The amplitude at ``frequency`` (Goertzel), as a fraction of full scale. */
function toneAmplitude(samples, frequency, rate = OUT, skip = 800) {
  const n = samples.length - 2 * skip;
  const k = (2 * Math.PI * frequency) / rate;
  let s1 = 0;
  let s2 = 0;
  for (let i = skip; i < samples.length - skip; i++) {
    const s0 = samples[i] / 32767 + 2 * Math.cos(k) * s1 - s2;
    s2 = s1;
    s1 = s0;
  }
  const power = s1 * s1 + s2 * s2 - 2 * Math.cos(k) * s1 * s2;
  return (2 * Math.sqrt(power)) / n;
}

const dB = (ratio) => 20 * Math.log10(ratio);

// ---------------------------------------------------------------------------
// Exact sample counts and the cap
// ---------------------------------------------------------------------------

for (const [rate, seconds] of [[48000, 1], [44100, 1], [48000, 2.5], [44100, 3.3], [22050, 1], [96000, 1], [16000, 1], [8000, 1]]) {
  test(`${rate} Hz for ${seconds} s gives exactly ceil(N x 16000 / rate) samples`, T, () => {
    const worklet = load(rate);
    const input = sine(rate, 440, seconds);
    worklet.feed([input]);
    worklet.flush();
    assert.equal(worklet.samples().length, Math.ceil((input.length * OUT) / rate));
    assert.deepEqual(worklet.events(), ["done"]);
    assert.equal(worklet.posted.at(-1).message.total, worklet.samples().length);
  });
}

for (const rate of [44100, 48000]) {
  test(`60 s at ${rate} Hz is exactly 960,000 samples; 61 s is capped there`, T, () => {
    const exact = load(rate);
    exact.feed([new Float32Array(rate * 60)]);
    exact.flush();
    assert.equal(exact.samples().length, 960000);

    const over = load(rate);
    over.feed([new Float32Array(rate * 61)]);
    assert.equal(over.samples().length, 960000);
    assert.deepEqual(over.events(), ["limit"]);
    over.flush();
    assert.equal(over.samples().length, 960000); // nothing after the cap
    assert.deepEqual(over.events(), ["limit", "done"]);
  });
}

test("a smaller configured cap is honoured exactly", T, () => {
  const worklet = load(48000, { maxSamples: 1000, chunkSamples: 300 });
  worklet.feed([new Float32Array(48000)]);
  assert.equal(worklet.samples().length, 1000);
  assert.deepEqual(worklet.posted.filter((e) => e.message.type === "chunk").map((e) => e.message.samples.length), [300, 300, 300, 100]);
});

test("chunks are Int16Array, posted with their buffers transferred", T, () => {
  const worklet = load(48000);
  worklet.feed([sine(48000, 440, 1)]);
  worklet.flush();
  const chunks = worklet.posted.filter((entry) => entry.message.type === "chunk");
  assert.ok(chunks.length >= 3);
  for (const { message, transfer } of chunks) {
    // Created in the worklet's realm (the VM here): checked by type tag, not instanceof.
    assert.equal(Object.prototype.toString.call(message.samples), "[object Int16Array]");
    assert.equal(transfer.length, 1);
    assert.equal(transfer[0], message.samples.buffer); // transferred, not copied
  }
  assert.ok(chunks.slice(0, -1).every(({ message }) => message.samples.length === 4096));
});

test("the result does not depend on the render quantum size", T, () => {
  const input = sine(44100, 1234, 0.7);
  const small = load(44100);
  small.feed([input], 128);
  small.flush();
  const large = load(44100);
  large.feed([input], 44100);
  large.flush();
  const odd = load(44100);
  odd.feed([input], 97);
  odd.flush();
  assert.deepEqual(small.samples(), large.samples());
  assert.deepEqual(small.samples(), odd.samples());
});

// ---------------------------------------------------------------------------
// Accuracy: passband, anti-aliasing, DC, clipping, mono mix
// ---------------------------------------------------------------------------

for (const rate of [44100, 48000]) {
  for (const frequency of [300, 1000, 3000, 5000]) {
    test(`${frequency} Hz at ${rate} Hz keeps its amplitude (within 0.1 dB) and frequency`, T, () => {
      const worklet = load(rate);
      worklet.feed([sine(rate, frequency, 1, 0.5)]);
      worklet.flush();
      const out = worklet.samples();
      const amplitude = toneAmplitude(out, frequency);
      assert.ok(Math.abs(dB(amplitude / 0.5)) < 0.1, `${frequency} Hz: ${dB(amplitude / 0.5).toFixed(3)} dB`);
      assert.ok(Math.abs(rms(out) - 0.5 / Math.SQRT2) < 0.002); // nothing else added
    });
  }

  for (const frequency of [9500, 10000, 12000, 15000, rate / 2 - 500]) {
    test(`${frequency} Hz at ${rate} Hz is attenuated by at least 50 dB (no aliasing)`, T, () => {
      const worklet = load(rate);
      worklet.feed([sine(rate, frequency, 1, 0.9)]);
      worklet.flush();
      const out = worklet.samples();
      const attenuation = dB(rms(out) / (0.9 / Math.SQRT2));
      assert.ok(attenuation <= -50, `${frequency} Hz: ${attenuation.toFixed(1)} dB`);
      // In particular, nothing appears at the alias frequency 16000 - f.
      assert.ok(toneAmplitude(out, OUT - frequency > 0 ? Math.abs(OUT - frequency) : 100) < 0.9 * 10 ** (-50 / 20));
    });
  }
}

test("DC is preserved exactly (unit gain)", T, () => {
  for (const rate of [44100, 48000]) {
    const worklet = load(rate);
    worklet.feed([new Float32Array(rate).fill(0.25)]);
    worklet.flush();
    const out = worklet.samples();
    for (let i = 100; i < out.length - 100; i++) assert.ok(Math.abs(out[i] - Math.round(0.25 * 32767)) <= 1);
  }
});

test("near full scale, DC comes out exactly (each kernel phase has unit gain)", T, () => {
  for (const rate of [44100, 48000]) {
    const worklet = load(rate);
    worklet.feed([new Float32Array(rate).fill(0.99)]);
    worklet.flush();
    const out = worklet.samples();
    const expected = Math.round(0.99 * 32767);
    for (let i = 100; i < out.length - 100; i++) assert.equal(out[i], expected, `${rate} Hz, sample ${i}`);
  }
});

test("over-range input is clipped to the 16-bit range, never wrapped", T, () => {
  const worklet = load(48000);
  worklet.feed([new Float32Array(48000).fill(2.0)]);
  worklet.flush();
  const high = worklet.samples();
  assert.equal(high[8000], 32767);
  const negative = load(48000);
  negative.feed([new Float32Array(48000).fill(-2.0)]);
  negative.flush();
  assert.equal(negative.samples()[8000], -32767);
  assert.ok([...high, ...negative.samples()].every((value) => value >= -32767 && value <= 32767));
});

test("channels are averaged to mono", T, () => {
  const left = sine(48000, 1000, 1, 0.5);
  const right = new Float32Array(left.length);
  const worklet = load(48000);
  worklet.feed([left, right]);
  worklet.flush();
  const amplitude = toneAmplitude(worklet.samples(), 1000);
  assert.ok(Math.abs(dB(amplitude / 0.25)) < 0.1);
});

test("silence stays silent and an impulse produces a short, symmetric response", T, () => {
  const worklet = load(48000);
  const input = new Float32Array(4800);
  input[2400] = 1;
  worklet.feed([input]);
  worklet.flush();
  const out = worklet.samples();
  const peak = out.indexOf(Math.max(...out));
  assert.equal(peak, 800); // 2400 / 3: aligned, no group delay left
  for (let i = 0; i < out.length; i++) {
    if (Math.abs(i - peak) > 12) assert.ok(Math.abs(out[i]) <= 1, `ringing at ${i}`);
  }
  assert.ok(Math.abs(out[peak - 3] - out[peak + 3]) <= 1);
});

// ---------------------------------------------------------------------------
// Lifecycle and module boundaries
// ---------------------------------------------------------------------------

test("flush posts what is left, then done; afterwards the processor ends", T, () => {
  const worklet = load(48000);
  worklet.feed([sine(48000, 440, 0.1)]);
  worklet.flush();
  assert.deepEqual(worklet.events(), ["done"]);
  assert.equal(worklet.samples().length, 1600);
  assert.equal(worklet.processor.process([[new Float32Array(128)]]), false);
  worklet.flush(); // a second flush is ignored
  assert.deepEqual(worklet.events(), ["done"]);
});

test("missing input (no channels) produces nothing and keeps the processor alive", T, () => {
  const worklet = load(48000);
  assert.equal(worklet.processor.process([[]]), true);
  assert.equal(worklet.processor.process([]), true);
  worklet.flush();
  assert.equal(worklet.samples().length, 0);
});

test("unsupported input rates are refused", T, () => {
  assert.throws(() => load(1000));
  assert.throws(() => load(Number.NaN));
});

test("the worklet has no network, storage, DOM, eval or imports", T, () => {
  const code = SOURCE.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "");
  for (const forbidden of [
    "fetch", "XMLHttpRequest", "WebSocket", "importScripts", "import ", "require(", "eval", "Function(",
    "localStorage", "sessionStorage", "indexedDB", "document", "window", "navigator", "cookie",
  ]) {
    assert.ok(!code.includes(forbidden), `worklet must not use ${forbidden}`);
  }
});

// ---------------------------------------------------------------------------
// S1-4c-2 remediation: the output is always silent
// ---------------------------------------------------------------------------

function garbageOutputs(channels = 1) {
  // Outputs that start full of non-zero data: only an explicit zero-fill silences them.
  return [Array.from({ length: channels }, () => new Float32Array(128).fill(0.7))];
}

const silent = (outputs) => outputs.every((output) => output.every((channel) => channel.every((value) => value === 0)));

test("every output buffer is zeroed while recording, with microphone audio at the input", T, () => {
  const worklet = load(48000);
  const input = sine(48000, 1000, 0.1, 0.9);
  for (let start = 0; start < input.length; start += 128) {
    const outputs = garbageOutputs(2);
    assert.equal(worklet.processor.process([[input.slice(start, start + 128)]], outputs), true);
    assert.ok(silent(outputs), `render quantum at ${start}`);
  }
  worklet.flush();
  assert.ok(worklet.samples().length > 0); // the input was still recorded
});

test("the output is silent on every other path too", T, () => {
  const capped = load(48000, { maxSamples: 100 });
  capped.feed([new Float32Array(48000).fill(0.5)]);
  const paths = [
    ["no input channels", load(48000).processor, [[]]],
    ["no inputs at all", load(48000).processor, []],
    ["after the cap", capped.processor, [[new Float32Array(128).fill(0.5)]]],
  ];
  for (const [label, processor, inputs] of paths) {
    const outputs = garbageOutputs();
    processor.process(inputs, outputs);
    assert.ok(silent(outputs), label);
  }
  const finished = load(48000);
  finished.flush();
  const outputs = garbageOutputs();
  assert.equal(finished.processor.process([[new Float32Array(128).fill(0.5)]], outputs), false);
  assert.ok(silent(outputs), "after the flush");
});

test("the output is silenced even if processing throws", T, () => {
  const worklet = load(48000);
  worklet.processor.resampler.push = () => {
    throw new Error("boom");
  };
  const outputs = garbageOutputs();
  assert.throws(() => worklet.processor.process([[new Float32Array(128).fill(0.5)]], outputs), /boom/);
  assert.ok(silent(outputs));
});

test("even an engine sharing the input and output buffer records the input and plays silence", T, () => {
  // Not allowed by the spec; checked anyway: the input is read before the zero-fill.
  const worklet = load(16000);
  const shared = new Float32Array(128).fill(0.5);
  worklet.processor.process([[shared]], [[shared]]);
  assert.ok(shared.every((value) => value === 0));
  worklet.flush();
  const out = worklet.samples();
  // Away from the edges of this 128-sample step (the 64-tap filter rings there), it is the input.
  assert.ok(out.slice(40, 88).every((value) => Math.abs(value - Math.round(0.5 * 32767)) <= 1));
});
