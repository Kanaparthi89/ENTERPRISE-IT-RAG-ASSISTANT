/*
 * Microphone dictation (Voice V3a; Security S1-4c-2): record one short clip
 * as canonical 16 kHz mono 16-bit PCM WAV and transcribe it with
 * POST /api/voice/transcribe.
 *
 * No DOM access: the browser objects (mediaDevices, AudioContext,
 * AudioWorkletNode, fetch, isSecureContext, timers) are injected; the page
 * gets them with ``browserDependencies(scope)``. Loaded as a classic script in
 * the browser (window.BluePeakRecorder) and with require() in Node tests.
 *
 * Capture: an AudioContext at the device's native rate feeds an AudioWorklet
 * (pcm-recorder-worklet.js) that mixes to mono, low-pass filters, resamples
 * to 16 kHz and converts to Int16 on the audio thread, posting chunks here.
 * The server accepts only the canonical WAV built from them (44-byte header,
 * one data chunk), at most 960,000 samples (60 s, 1,920,044 bytes).
 *
 * Contract:
 * - A session starts only from a real user action (``start(event)`` with a
 *   trusted event) in a secure context with the needed APIs. The
 *   AudioContext is created and resumed, and getUserMedia called,
 *   synchronously inside that call, before anything is awaited (Safari only
 *   lets audio start during the user's gesture).
 * - ``start()`` resolves once per session, never rejects:
 *   ``{ ok: true, text, stoppedBy }`` or ``{ ok: false, code[, retryAfter] }``
 *   (codes in ERROR_MESSAGES; "cancelled" when the caller cancelled).
 * - Recording stops on ``stop()``, at 960,000 samples (with a timer as a
 *   backstop), and is then flushed and uploaded. If the microphone goes away
 *   or the audio is interrupted (a call, the screen locking, a suspended or
 *   closed context), the recording is discarded, never uploaded; so is one
 *   whose final flush fails or doesn't arrive in time.
 * - Every exit stops every microphone track, disconnects the audio nodes and
 *   closes the AudioContext, so the "microphone in use" indicator turns off.
 *   Once the microphone is granted, the audio must be ready within
 *   SETUP_TIMEOUT_MS (one extra resume() is tried if the context isn't
 *   running, for example after an iOS audio-session interruption); otherwise
 *   the session fails and the microphone is released. The user's own time to
 *   answer the permission prompt is not limited.
 * - The upload is abortable (``cancel()``) and bounded (UPLOAD_TIMEOUT_MS).
 *   After a short-term refusal with Retry-After (busy, capacity, quota
 *   unavailable), a new session is refused locally until that time has passed,
 *   for at most LOCAL_COOLDOWN_MAX_SECONDS. Daily limits are only reported:
 *   they belong to an identity (an access code, which this module can't see),
 *   so they are never applied locally, and the server stays authoritative.
 * - Every callback checks that its session is still the current, unfinished
 *   one, and a session ends exactly once.
 * - Nothing is stored: samples are dropped as soon as the upload starts or
 *   the session ends. The transcript is only returned.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  } else {
    root.BluePeakRecorder = api;
  }
})(typeof self !== "undefined" ? self : globalThis, function () {
  "use strict";

  const TRANSCRIBE_URL = "/api/voice/transcribe";
  const WORKLET_URL = "/static/js/pcm-recorder-worklet.js";
  const PROCESSOR_NAME = "bluepeak-pcm-recorder";
  const SAMPLE_RATE = 16_000;
  const MAX_DURATION_MS = 60_000;
  const MAX_SAMPLES = 960_000; // 60 s at 16 kHz
  const WAV_HEADER_BYTES = 44;
  const MAX_WAV_BYTES = WAV_HEADER_BYTES + 2 * MAX_SAMPLES; // 1,920,044
  const MAX_AUDIO_BYTES = 2 * 1024 * 1024; // the server's upload limit
  const CHUNK_SAMPLES = 4_096; // 256 ms per message from the worklet
  const DURATION_BACKSTOP_MS = MAX_DURATION_MS + 1_000; // the sample cap normally stops first
  const FLUSH_TIMEOUT_MS = 1_000;
  const SETUP_TIMEOUT_MS = 10_000; // from the microphone grant to a running worklet
  const UPLOAD_TIMEOUT_MS = 75_000; // the server allows 30 s to upload and 35 s for the provider
  const MAX_RETRY_AFTER_SECONDS = 86_400;
  const LOCAL_COOLDOWN_MAX_SECONDS = 60;

  const STATES = Object.freeze({
    IDLE: "idle",
    REQUESTING: "requesting", // waiting for microphone permission
    RECORDING: "recording",
    UPLOADING: "uploading",
  });

  const ERROR_MESSAGES = Object.freeze({
    unsupported: "Dictation isn't supported in this browser.",
    insecure_context: "Dictation needs a secure (HTTPS) connection.",
    not_user_action: "Select the microphone button to start dictating.",
    busy: "Dictation is already in progress.",
    permission_denied:
      "Microphone access is blocked. Allow it in your browser's site settings to dictate.",
    no_microphone: "No microphone was found.",
    microphone_busy: "The microphone is in use by another application.",
    recording_failed: "Recording failed. Please try again.",
    recording_interrupted: "Recording was interrupted, so nothing was sent. Please try again.",
    empty_recording: "Nothing was recorded. Please try again.",
    too_large: "The recording is too large. Please record a shorter question.",
    audio_too_long: "Recordings can be up to 60 seconds. Please record a shorter question.",
    upload_timeout: "Transcription took too long. Please try again.",
    network_error: "Couldn't reach the server. Please try again.",
    no_speech: "No speech was recognised. Please try again.",
    origin_not_allowed: "Dictation can't be used from this address.",
    invalid_upload: "The recording couldn't be sent. Please try again.",
    invalid_audio: "The recording couldn't be processed. Please try again.",
    unsupported_audio: "This recording format isn't supported.",
    audio_not_transcribable: "The recording couldn't be transcribed. Please try again.",
    transcription_timeout: "Transcription took too long. Please try again.",
    transcription_unavailable: "Transcription is temporarily unavailable. Please try again later.",
    transcription_disabled: "Dictation isn't available right now.",
    transcription_failed: "Transcription failed. Please try again.",
    voice_busy: "Too many recordings are being sent right now. Please try again shortly.",
    quota_busy: "Another recording is being transcribed. Please try again shortly.",
    quota_capacity: "Dictation is at capacity right now. Please try again later.",
    quota_daily_limit: "You've reached today's dictation limit. Please try again tomorrow.",
    quota_minutes_limit: "You've used today's dictation time. Please try again tomorrow.",
    quota_unavailable: "Dictation is temporarily unavailable. Please try again shortly.",
    cancelled: "Dictation was cancelled.",
  });

  // Codes the server sends in {"detail": {"code"}}, passed through as-is.
  const SERVER_CODES = new Set([
    "origin_not_allowed",
    "unsupported_media_type",
    "audio_too_large",
    "audio_too_long",
    "upload_timeout",
    "invalid_upload",
    "invalid_audio",
    "unsupported_audio",
    "audio_not_transcribable",
    "transcription_timeout",
    "transcription_unavailable",
    "transcription_failed",
    "voice_busy",
    "quota_busy",
    "quota_capacity",
    "quota_daily_limit",
    "quota_minutes_limit",
    "quota_unavailable",
  ]);

  // Refusals whose Retry-After is reported to the user.
  const RETRYABLE = new Set([
    "voice_busy",
    "quota_busy",
    "quota_capacity",
    "quota_daily_limit",
    "quota_minutes_limit",
    "quota_unavailable",
  ]);

  // Short-term refusals also applied locally (capped): the server is busy for
  // everyone, not for one identity. Daily limits are not: a new access code
  // must not be blocked by an earlier identity's limit.
  const LOCAL_COOLDOWN = new Set(["voice_busy", "quota_busy", "quota_capacity", "quota_unavailable"]);

  const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

  /** getUserMedia failures, by DOMException name. */
  function microphoneErrorCode(error) {
    const name = error && error.name;
    if (name === "NotAllowedError" || name === "SecurityError") return "permission_denied";
    if (name === "NotFoundError" || name === "OverconstrainedError") return "no_microphone";
    if (name === "NotReadableError" || name === "AbortError") return "microphone_busy";
    return "recording_failed";
  }

  function serverErrorCode(status, body) {
    const code = body && body.detail && typeof body.detail === "object" ? body.detail.code : undefined;
    if (typeof code === "string" && SERVER_CODES.has(code)) {
      if (code === "audio_too_large") return "too_large";
      if (code === "unsupported_media_type") return "invalid_upload";
      return code;
    }
    if (status === 404) return "transcription_disabled";
    if (status === 413) return "too_large";
    if (status === 408) return "upload_timeout";
    if (status === 504) return "transcription_timeout";
    if (status === 429 || status === 503) return "transcription_unavailable";
    return "transcription_failed";
  }

  /** Whole seconds from a Retry-After header (delay form only), or null. */
  function retryAfterSeconds(response) {
    let value = null;
    try {
      value = response && response.headers && typeof response.headers.get === "function"
        ? response.headers.get("retry-after")
        : null;
    } catch {
      value = null;
    }
    if (typeof value !== "string" || !/^\s*\d{1,9}\s*$/.test(value)) return null;
    const seconds = Number.parseInt(value, 10);
    if (!Number.isFinite(seconds) || seconds < 1) return null;
    return Math.min(seconds, MAX_RETRY_AFTER_SECONDS);
  }

  function describeWait(seconds) {
    if (seconds < 60) return seconds === 1 ? "1 second" : `${seconds} seconds`;
    if (seconds < 3_600) {
      const minutes = Math.ceil(seconds / 60);
      return minutes === 1 ? "about a minute" : `about ${minutes} minutes`;
    }
    const hours = Math.ceil(seconds / 3_600);
    return hours === 1 ? "about an hour" : `about ${hours} hours`;
  }

  /** The user-facing message for an outcome, with the wait when there is one. */
  function messageFor(outcome) {
    const code = outcome && outcome.code;
    const message = ERROR_MESSAGES[code] || ERROR_MESSAGES.transcription_failed;
    const wait = outcome && outcome.retryAfter;
    return Number.isInteger(wait) && wait > 0 ? `${message} You can try again in ${describeWait(wait)}.` : message;
  }

  /** The 44-byte canonical header for ``sampleCount`` 16 kHz mono 16-bit samples. */
  function wavHeader(sampleCount) {
    if (!Number.isInteger(sampleCount) || sampleCount < 0 || sampleCount > MAX_SAMPLES) {
      throw new RangeError("invalid sample count");
    }
    const dataBytes = sampleCount * 2;
    const header = new ArrayBuffer(WAV_HEADER_BYTES);
    const view = new DataView(header);
    const ascii = (offset, text) => {
      for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
    };
    ascii(0, "RIFF");
    view.setUint32(4, 36 + dataBytes, true);
    ascii(8, "WAVE");
    ascii(12, "fmt ");
    view.setUint32(16, 16, true); // fmt chunk size
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // mono
    view.setUint32(24, SAMPLE_RATE, true);
    view.setUint32(28, SAMPLE_RATE * 2, true); // byte rate
    view.setUint16(32, 2, true); // block align
    view.setUint16(34, 16, true); // bits per sample
    ascii(36, "data");
    view.setUint32(40, dataBytes, true);
    return new Uint8Array(header);
  }

  /** A canonical WAV Blob from Int16 chunks holding exactly ``sampleCount`` samples. */
  function encodeWav(chunks, sampleCount, BlobImpl = globalThis.Blob) {
    let counted = 0;
    for (const chunk of chunks) {
      if (!(chunk instanceof Int16Array)) throw new TypeError("PCM chunks must be Int16Array");
      counted += chunk.length;
    }
    if (counted !== sampleCount || sampleCount < 1) throw new RangeError("sample count mismatch");
    const parts = [wavHeader(sampleCount)];
    for (const chunk of chunks) {
      if (LITTLE_ENDIAN) {
        parts.push(new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength));
      } else {
        const bytes = new Uint8Array(chunk.length * 2);
        const view = new DataView(bytes.buffer);
        for (let i = 0; i < chunk.length; i++) view.setInt16(i * 2, chunk[i], true);
        parts.push(bytes);
      }
    }
    return new BlobImpl(parts, { type: "audio/wav" });
  }

  function hasAudioWorklet(AudioContextImpl) {
    try {
      return Boolean(AudioContextImpl && AudioContextImpl.prototype && "audioWorklet" in AudioContextImpl.prototype);
    } catch {
      return false;
    }
  }

  /** The browser objects the recorder needs, read from ``scope`` (the page's window). */
  function browserDependencies(scope) {
    const source = scope || {};
    return {
      mediaDevices: source.navigator ? source.navigator.mediaDevices : undefined,
      AudioContext: source.AudioContext,
      AudioWorkletNode: source.AudioWorkletNode,
      isSecureContext: source.isSecureContext === true,
    };
  }

  /** Whether ``scope`` can record (secure context, microphone, AudioWorklet). */
  function canRecord(scope) {
    const deps = browserDependencies(scope);
    return Boolean(
      deps.isSecureContext &&
        deps.mediaDevices &&
        typeof deps.mediaDevices.getUserMedia === "function" &&
        typeof deps.AudioContext === "function" &&
        typeof deps.AudioWorkletNode === "function" &&
        hasAudioWorklet(deps.AudioContext),
    );
  }

  /**
   * A recorder over injected browser dependencies.
   *
   * Options: ``mediaDevices``, ``AudioContext``, ``AudioWorkletNode``,
   * ``isSecureContext``, ``fetchImpl``, and optionally ``FormData``, ``Blob``,
   * ``AbortController``, ``setTimeout``/``clearTimeout``, ``now``, ``url``,
   * ``workletUrl``, the limits (``maxDurationMs``, ``maxSamples``,
   * ``chunkSamples``, ``flushTimeoutMs``, ``uploadTimeoutMs``) and
   * ``onStateChange(state, detail)``.
   */
  function createRecorder(options = {}) {
    const {
      mediaDevices,
      AudioContext: AudioContextImpl,
      AudioWorkletNode: WorkletNodeImpl,
      isSecureContext = false,
      fetchImpl,
      FormData: FormDataImpl = globalThis.FormData,
      Blob: BlobImpl = globalThis.Blob,
      AbortController: AbortControllerImpl = globalThis.AbortController,
      setTimeout: setTimer = globalThis.setTimeout,
      clearTimeout: clearTimer = globalThis.clearTimeout,
      now = () => Date.now(),
      url = TRANSCRIBE_URL,
      workletUrl = WORKLET_URL,
      maxDurationMs = MAX_DURATION_MS,
      maxSamples = MAX_SAMPLES,
      chunkSamples = CHUNK_SAMPLES,
      flushTimeoutMs = FLUSH_TIMEOUT_MS,
      uploadTimeoutMs = UPLOAD_TIMEOUT_MS,
      onStateChange = null,
    } = options;

    const supported = Boolean(
      mediaDevices &&
        typeof mediaDevices.getUserMedia === "function" &&
        typeof AudioContextImpl === "function" &&
        typeof WorkletNodeImpl === "function" &&
        hasAudioWorklet(AudioContextImpl) &&
        typeof fetchImpl === "function" &&
        typeof FormDataImpl === "function" &&
        typeof BlobImpl === "function" &&
        typeof AbortControllerImpl === "function",
    );

    let state = STATES.IDLE;
    let current = null; // the active session, or null; a new one starts only once it has ended
    let retryUntil = 0; // after a refusal with Retry-After: no new session before this time
    let retryCode = null;

    function setState(next, detail = {}) {
      state = next;
      if (typeof onStateChange === "function") {
        try {
          onStateChange(next, detail);
        } catch {
          // A view error must not break recording or cleanup.
        }
      }
    }

    function stopTracks(session) {
      if (!session.stream) return;
      let tracks = [];
      try {
        tracks = session.stream.getTracks();
      } catch {
        tracks = [];
      }
      for (const track of tracks) {
        try {
          track.onended = null;
          track.stop();
        } catch {
          // Keep stopping the others.
        }
      }
    }

    /** Disconnect the nodes and close the AudioContext (once; every step guarded). */
    function releaseAudio(session) {
      if (session.audioReleased) return;
      session.audioReleased = true;
      const node = session.node;
      if (node) {
        try {
          node.onprocessorerror = null;
          if (node.port) {
            node.port.onmessage = null;
            node.port.onmessageerror = null;
            if (typeof node.port.close === "function") node.port.close();
          }
        } catch {
          // Keep releasing.
        }
        try {
          node.disconnect();
        } catch {
          // Already disconnected.
        }
      }
      if (session.source) {
        try {
          session.source.disconnect();
        } catch {
          // Already disconnected.
        }
      }
      const context = session.context;
      if (context) {
        try {
          context.onstatechange = null;
          if (context.state !== "closed") {
            const closing = context.close();
            if (closing && typeof closing.catch === "function") closing.catch(() => {});
          }
        } catch {
          // Already closed.
        }
      }
    }

    /** End a session exactly once: release everything, then resolve. */
    function finish(session, outcome) {
      if (session.finished) return;
      session.finished = true;
      if (current === session) current = null;

      for (const name of ["setupTimer", "durationTimer", "flushTimer", "uploadTimer"]) {
        if (session[name] !== null) clearTimer(session[name]);
        session[name] = null;
      }
      if (session.uploadAbort) session.uploadAbort.abort();
      releaseAudio(session);
      stopTracks(session);
      session.chunks = []; // nothing is kept
      session.samples = 0;

      if (!outcome.ok && Number.isInteger(outcome.retryAfter) && LOCAL_COOLDOWN.has(outcome.code)) {
        retryUntil = now() + Math.min(outcome.retryAfter, LOCAL_COOLDOWN_MAX_SECONDS) * 1_000;
        retryCode = outcome.code;
      }
      setState(STATES.IDLE, outcome.ok ? {} : { code: outcome.code });
      session.resolve(outcome);
    }

    function isCurrent(session) {
      return !session.finished && current === session;
    }

    function fail(session, code) {
      if (isCurrent(session)) finish(session, { ok: false, code });
    }

    /** Stop recording (once): the microphone is released at once, then the worklet is flushed. */
    function requestStop(session, reason) {
      if (!isCurrent(session) || session.phase !== "recording") return;
      session.phase = "flushing";
      session.stoppedBy = reason;
      if (session.durationTimer !== null) clearTimer(session.durationTimer);
      session.durationTimer = null;
      stopTracks(session);
      // Partial audio is never uploaded: no complete flush in time, no upload.
      session.flushTimer = setTimer(() => {
        session.flushTimer = null;
        fail(session, "recording_failed");
      }, flushTimeoutMs);
      try {
        session.node.port.postMessage({ type: "flush" });
      } catch {
        fail(session, "recording_failed");
      }
    }

    function onWorkletMessage(session, event) {
      if (!isCurrent(session)) return;
      const data = event && event.data;
      if (!data || typeof data !== "object") return;
      if (data.type === "chunk") {
        if (session.phase !== "recording" && session.phase !== "flushing") return;
        const samples = data.samples;
        if (!(samples instanceof Int16Array)) {
          fail(session, "recording_failed");
          return;
        }
        if (session.samples + samples.length > maxSamples) {
          fail(session, "too_large"); // never more than the cap (the worklet stops there itself)
          return;
        }
        session.chunks.push(samples);
        session.samples += samples.length;
      } else if (data.type === "limit") {
        requestStop(session, "time_limit");
      } else if (data.type === "done") {
        if (session.phase !== "flushing") return;
        if (session.flushTimer !== null) clearTimer(session.flushTimer);
        session.flushTimer = null;
        session.phase = "uploading";
        releaseAudio(session);
        upload(session);
      }
    }

    function onContextState(session) {
      if (!isCurrent(session)) return;
      if (session.phase !== "recording" && session.phase !== "flushing") return;
      let contextState = "closed";
      try {
        contextState = session.context.state;
      } catch {
        contextState = "closed";
      }
      // Suspended, interrupted (Safari: a call, Siri, the screen locking) or
      // closed while recording: the recording is incomplete, so it is discarded.
      if (contextState !== "running") fail(session, "recording_interrupted");
    }

    /**
     * True once the context is running. If it isn't (Safari can interrupt it
     * while the microphone starts), resume() is tried once more; the setup
     * timer still bounds the wait. Never retried more than once.
     */
    function ensureRunning(session) {
      const context = session.context;
      if (!isCurrent(session)) return Promise.resolve(false);
      if (context.state === "running") return Promise.resolve(true);
      let retry;
      try {
        retry = context.resume();
      } catch {
        return Promise.resolve(false);
      }
      return Promise.resolve(retry).then(
        () => isCurrent(session) && context.state === "running",
        () => false,
      );
    }

    function onReady(session, stream) {
      if (!isCurrent(session)) {
        stopTracks(session);
        return;
      }
      const context = session.context;
      try {
        if (context.state !== "running") {
          fail(session, "recording_failed");
          return;
        }
        session.source = context.createMediaStreamSource(stream);
        session.node = new WorkletNodeImpl(context, PROCESSOR_NAME, {
          numberOfInputs: 1,
          numberOfOutputs: 1,
          outputChannelCount: [1],
          processorOptions: { maxSamples, chunkSamples },
        });
        session.node.port.onmessage = (event) => onWorkletMessage(session, event);
        session.node.port.onmessageerror = () => fail(session, "recording_failed");
        session.node.onprocessorerror = () => fail(session, "recording_failed");
        session.source.connect(session.node);
        // A silent output keeps the node rendering in every browser.
        session.node.connect(context.destination);
        for (const track of stream.getTracks()) {
          track.onended = () => fail(session, "recording_interrupted");
        }
      } catch {
        fail(session, "recording_failed");
        return;
      }

      session.phase = "recording";
      setState(STATES.RECORDING, { startedAt: now(), maxDurationMs });
      session.durationTimer = setTimer(() => {
        session.durationTimer = null;
        requestStop(session, "time_limit");
      }, DURATION_BACKSTOP_MS);
    }

    function upload(session) {
      if (session.samples === 0) {
        finish(session, { ok: false, code: "empty_recording" });
        return;
      }
      if (session.samples > maxSamples) {
        finish(session, { ok: false, code: "too_large" });
        return;
      }

      let body;
      try {
        const audio = encodeWav(session.chunks, session.samples, BlobImpl);
        if (audio.size > MAX_WAV_BYTES || audio.size > MAX_AUDIO_BYTES) {
          finish(session, { ok: false, code: "too_large" });
          return;
        }
        body = new FormDataImpl();
        body.append("audio", audio, "recording.wav");
      } catch {
        finish(session, { ok: false, code: "recording_failed" });
        return;
      }
      session.chunks = []; // the blob holds the audio until the upload ends

      const abort = new AbortControllerImpl();
      session.uploadAbort = abort;
      session.uploadTimer = setTimer(() => {
        session.uploadTimer = null;
        session.timedOut = true;
        abort.abort();
      }, uploadTimeoutMs);
      setState(STATES.UPLOADING);

      Promise.resolve()
        .then(() =>
          fetchImpl(url, {
            method: "POST",
            body,
            headers: { Accept: "application/json" },
            credentials: "same-origin",
            cache: "no-store",
            signal: abort.signal,
          }),
        )
        .then(async (response) => {
          let data = null;
          try {
            data = await response.json();
          } catch {
            data = null;
          }
          if (!isCurrent(session)) return;
          if (response.status !== 200) {
            const code = serverErrorCode(response.status, data);
            const outcome = { ok: false, code };
            const wait = RETRYABLE.has(code) ? retryAfterSeconds(response) : null;
            if (wait !== null) outcome.retryAfter = wait;
            finish(session, outcome);
            return;
          }
          if (!data || typeof data.text !== "string") {
            finish(session, { ok: false, code: "transcription_failed" });
            return;
          }
          const text = data.text.trim();
          if (!text) {
            finish(session, { ok: false, code: "no_speech" });
            return;
          }
          finish(session, { ok: true, text, stoppedBy: session.stoppedBy });
        })
        .catch(() => {
          if (!isCurrent(session)) return;
          finish(session, { ok: false, code: session.timedOut ? "upload_timeout" : "network_error" });
        });
    }

    /**
     * Start one dictation session from a user action. Call it from the click
     * handler with the click event: the audio and the microphone are
     * requested synchronously, inside the gesture.
     */
    function start(event) {
      if (state !== STATES.IDLE) return Promise.resolve({ ok: false, code: "busy" });
      if (!event || event.isTrusted !== true) {
        return Promise.resolve({ ok: false, code: "not_user_action" });
      }
      if (isSecureContext !== true) return Promise.resolve({ ok: false, code: "insecure_context" });
      if (!supported) return Promise.resolve({ ok: false, code: "unsupported" });
      const waitMs = retryUntil - now();
      if (waitMs > 0) {
        return Promise.resolve({ ok: false, code: retryCode, retryAfter: Math.max(1, Math.ceil(waitMs / 1_000)) });
      }

      return new Promise((resolve) => {
        const session = {
          resolve,
          phase: "requesting",
          context: null,
          stream: null,
          source: null,
          node: null,
          chunks: [],
          samples: 0,
          stoppedBy: null,
          setupTimer: null,
          durationTimer: null,
          flushTimer: null,
          uploadTimer: null,
          uploadAbort: null,
          timedOut: false,
          audioReleased: false,
          finished: false,
        };
        current = session;
        setState(STATES.REQUESTING);

        try {
          session.context = new AudioContextImpl();
        } catch {
          finish(session, { ok: false, code: "unsupported" });
          return;
        }
        let request;
        try {
          request = mediaDevices.getUserMedia({ audio: true, video: false });
        } catch (error) {
          finish(session, { ok: false, code: microphoneErrorCode(error) });
          return;
        }
        let audioReady;
        try {
          session.context.onstatechange = () => onContextState(session);
          audioReady = Promise.all([session.context.resume(), session.context.audioWorklet.addModule(workletUrl)]);
        } catch (error) {
          audioReady = Promise.reject(error);
        }
        audioReady.catch(() => {}); // handled below, after the microphone answers

        Promise.resolve(request)
          .then(
            (stream) => {
              session.stream = stream;
              if (!isCurrent(session)) {
                stopTracks(session); // permission arrived after the session ended: release it now
                return null;
              }
              // From the grant on, the microphone is live: the audio must be
              // ready in time, or the session fails and releases everything.
              session.setupTimer = setTimer(() => {
                session.setupTimer = null;
                fail(session, "recording_failed");
              }, SETUP_TIMEOUT_MS);
              return audioReady
                .then(() => ensureRunning(session))
                .then(
                  (running) => {
                    if (!isCurrent(session)) return null; // timed out or cancelled meanwhile
                    if (!running) {
                      fail(session, "recording_failed");
                      return null;
                    }
                    return stream;
                  },
                  () => {
                    fail(session, "recording_failed");
                    return null;
                  },
                );
            },
            (error) => {
              fail(session, microphoneErrorCode(error));
              return null;
            },
          )
          .then((stream) => {
            if (!stream || !isCurrent(session)) return;
            if (session.setupTimer !== null) clearTimer(session.setupTimer);
            session.setupTimer = null;
            onReady(session, stream);
          });
      });
    }

    /** Stop recording and transcribe it. While permission is pending, it cancels. */
    function stop() {
      if (!current) return;
      if (current.phase === "requesting") {
        cancel();
        return;
      }
      requestStop(current, "user");
    }

    /** Discard everything: no upload, or abort one in progress. Safe at any time. */
    function cancel() {
      if (current) finish(current, { ok: false, code: "cancelled" });
    }

    return {
      start,
      stop,
      cancel,
      get state() {
        return state;
      },
      get supported() {
        return supported;
      },
    };
  }

  return {
    TRANSCRIBE_URL,
    WORKLET_URL,
    PROCESSOR_NAME,
    SAMPLE_RATE,
    MAX_DURATION_MS,
    MAX_SAMPLES,
    WAV_HEADER_BYTES,
    MAX_WAV_BYTES,
    MAX_AUDIO_BYTES,
    CHUNK_SAMPLES,
    DURATION_BACKSTOP_MS,
    FLUSH_TIMEOUT_MS,
    SETUP_TIMEOUT_MS,
    LOCAL_COOLDOWN_MAX_SECONDS,
    UPLOAD_TIMEOUT_MS,
    STATES,
    ERROR_MESSAGES,
    browserDependencies,
    canRecord,
    createRecorder,
    encodeWav,
    messageFor,
    wavHeader,
  };
});
