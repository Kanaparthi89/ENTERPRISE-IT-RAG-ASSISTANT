/*
 * Dictation capture (Security S1-4c-2): an AudioWorklet processor that turns
 * the microphone into 16 kHz mono 16-bit PCM for the canonical WAV the
 * server accepts (app/services/audio_duration.py).
 *
 * Runs on the audio rendering thread, loaded by recorder.js with
 * audioWorklet.addModule(). Self-contained (no imports); no network, storage
 * or DOM access. Its output is always silence: every output buffer is
 * zero-filled on every render quantum (the node is connected to the
 * destination only so that every browser keeps running it), so the
 * microphone never reaches the speakers. Messages to the main thread:
 *   { type: "chunk", samples: Int16Array }  (buffer transferred, not copied)
 *   { type: "limit", total }  the sample cap was reached; nothing more is produced
 *   { type: "done", total }   answer to { type: "flush" }: everything was posted
 *
 * Resampling: channels are averaged to mono, then a 64-tap windowed-sinc
 * low-pass (Kaiser, beta 8, cutoff 7 kHz) is evaluated at each 16 kHz output
 * position, with 256 precomputed fractional phases (interpolated linearly).
 * Positions come from integer counters (n * inputRate / 16000), so there is
 * no drift: N input samples give exactly ceil(N * 16000 / inputRate) outputs.
 */
"use strict";

const PCM_OUTPUT_RATE = 16000;
const PCM_TAPS = 64;
const PCM_HALF = PCM_TAPS / 2;
const PCM_PHASES = 256;
const PCM_KAISER_BETA = 8;
const PCM_CUTOFF_HZ = 7000;
const PCM_DEFAULT_MAX_SAMPLES = 960000; // 60 seconds
const PCM_DEFAULT_CHUNK_SAMPLES = 4096; // 256 ms per message

function pcmBesselI0(x) {
  let sum = 1;
  let term = 1;
  for (let k = 1; k < 64; k++) {
    term *= (x / (2 * k)) ** 2;
    sum += term;
    if (term < sum * 1e-15) break;
  }
  return sum;
}

/** Kernel rows for PCM_PHASES + 1 fractional offsets; each row sums to 1 (unit DC gain). */
function pcmKernelTable(inputRate) {
  const fc = PCM_CUTOFF_HZ / inputRate; // cycles per input sample
  const denominator = pcmBesselI0(PCM_KAISER_BETA);
  const table = new Float64Array((PCM_PHASES + 1) * PCM_TAPS);
  for (let phase = 0; phase <= PCM_PHASES; phase++) {
    const frac = phase / PCM_PHASES;
    let sum = 0;
    for (let j = 0; j < PCM_TAPS; j++) {
      const d = j - (PCM_HALF - 1) - frac; // input tap minus the output position
      const ratio = d / PCM_HALF;
      let value = 0;
      if (Math.abs(ratio) < 1) {
        const x = 2 * fc * d;
        const sinc = x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
        const taper = pcmBesselI0(PCM_KAISER_BETA * Math.sqrt(1 - ratio * ratio)) / denominator;
        value = 2 * fc * sinc * taper;
      }
      table[phase * PCM_TAPS + j] = value;
      sum += value;
    }
    for (let j = 0; j < PCM_TAPS; j++) table[phase * PCM_TAPS + j] /= sum;
  }
  return table;
}

/** Streaming mono resampler to 16 kHz Int16, with a hard sample cap. */
class PcmResampler {
  constructor(inputRate, { maxSamples = PCM_DEFAULT_MAX_SAMPLES, chunkSamples = PCM_DEFAULT_CHUNK_SAMPLES, emit } = {}) {
    if (!Number.isFinite(inputRate) || inputRate < 8000 || inputRate > 384000) {
      throw new RangeError("unsupported input sample rate");
    }
    this.inputRate = Math.round(inputRate);
    this.maxSamples = maxSamples;
    this.chunkSamples = chunkSamples;
    this.emit = emit;
    this.table = pcmKernelTable(this.inputRate);
    this.input = new Float64Array(16384);
    this.inputOffset = 0; // global index of input[0]
    this.inputLength = 0; // valid samples in input
    this.totalInput = 0;
    this.produced = 0; // output samples so far
    this.chunk = new Int16Array(chunkSamples);
    this.chunkLength = 0;
    this.limitReached = false;
  }

  /** The input position of output ``n``: [base, frac) from exact integer arithmetic. */
  position(n) {
    const numerator = n * this.inputRate;
    const base = Math.floor(numerator / PCM_OUTPUT_RATE);
    return [base, (numerator - base * PCM_OUTPUT_RATE) / PCM_OUTPUT_RATE];
  }

  sampleAt(index) {
    if (index < 0 || index >= this.totalInput) return 0; // before the start, or padding at the end
    return this.input[index - this.inputOffset];
  }

  push(samples) {
    if (this.limitReached) return;
    this.compact();
    if (this.inputLength + samples.length > this.input.length) {
      const grown = new Float64Array(Math.max(this.input.length * 2, this.inputLength + samples.length));
      grown.set(this.input.subarray(0, this.inputLength));
      this.input = grown;
    }
    this.input.set(samples, this.inputLength);
    this.inputLength += samples.length;
    this.totalInput += samples.length;
    this.produce(false);
  }

  /** Drop input no future output needs (its first tap is at base - HALF + 1). */
  compact() {
    const [base] = this.position(this.produced);
    const keepFrom = Math.max(0, base - PCM_HALF + 1);
    const drop = keepFrom - this.inputOffset;
    if (drop > 0 && drop <= this.inputLength) {
      this.input.copyWithin(0, drop, this.inputLength);
      this.inputLength -= drop;
      this.inputOffset += drop;
    }
  }

  produce(final) {
    while (!this.limitReached) {
      const [base, frac] = this.position(this.produced);
      if (final ? base >= this.totalInput : base + PCM_HALF > this.totalInput - 1) break;
      const scaled = frac * PCM_PHASES;
      const phase = Math.min(PCM_PHASES - 1, Math.floor(scaled));
      const t = scaled - phase;
      const row0 = phase * PCM_TAPS;
      const row1 = row0 + PCM_TAPS;
      const first = base - PCM_HALF + 1;
      let acc = 0;
      for (let j = 0; j < PCM_TAPS; j++) {
        const weight = this.table[row0 + j] + t * (this.table[row1 + j] - this.table[row0 + j]);
        acc += this.sampleAt(first + j) * weight;
      }
      const clamped = acc > 1 ? 1 : acc < -1 ? -1 : acc;
      this.chunk[this.chunkLength++] = Math.round(clamped * 32767);
      this.produced += 1;
      if (this.chunkLength === this.chunkSamples) this.post();
      if (this.produced >= this.maxSamples) {
        this.limitReached = true;
        this.post();
        this.emit({ type: "limit", total: this.produced });
      }
    }
  }

  post() {
    if (this.chunkLength === 0) return;
    const samples = this.chunk.slice(0, this.chunkLength);
    this.chunk = new Int16Array(this.chunkSamples);
    this.chunkLength = 0;
    this.emit({ type: "chunk", samples }, [samples.buffer]);
  }

  /** Produce what is left (zero padding after the last input), then report the total. */
  flush() {
    this.produce(true);
    this.post();
    this.emit({ type: "done", total: this.produced });
  }
}

/** Zero every output channel: nothing captured is ever played. */
function pcmSilence(outputs) {
  if (!outputs) return;
  for (const output of outputs) {
    if (!output) continue;
    for (const channel of output) {
      if (channel && typeof channel.fill === "function") channel.fill(0);
    }
  }
}

if (typeof AudioWorkletProcessor === "function" && typeof registerProcessor === "function") {
  class BluePeakPcmRecorder extends AudioWorkletProcessor {
    constructor(options) {
      super();
      const settings = (options && options.processorOptions) || {};
      this.finished = false;
      this.scratch = new Float64Array(128);
      this.resampler = new PcmResampler(sampleRate, {
        maxSamples: settings.maxSamples || PCM_DEFAULT_MAX_SAMPLES,
        chunkSamples: settings.chunkSamples || PCM_DEFAULT_CHUNK_SAMPLES,
        emit: (message, transfer) => this.port.postMessage(message, transfer || []),
      });
      this.port.onmessage = (event) => {
        if (this.finished || !event || !event.data || event.data.type !== "flush") return;
        this.finished = true;
        this.resampler.flush();
      };
    }

    process(inputs, outputs) {
      try {
        if (this.finished) return false;
        const channels = inputs && inputs[0];
        if (!channels || channels.length === 0 || this.resampler.limitReached) return true;
        const frames = channels[0].length;
        if (this.scratch.length !== frames) this.scratch = new Float64Array(frames);
        const mono = this.scratch;
        mono.fill(0);
        for (const channel of channels) {
          for (let i = 0; i < frames; i++) mono[i] += channel[i];
        }
        if (channels.length > 1) {
          for (let i = 0; i < frames; i++) mono[i] /= channels.length;
        }
        this.resampler.push(mono);
        return true;
      } finally {
        // On every path, exceptions included, and after the input was read
        // (even an engine that shared buffers could not echo it): silence.
        pcmSilence(outputs);
      }
    }
  }

  registerProcessor("bluepeak-pcm-recorder", BluePeakPcmRecorder);
}
