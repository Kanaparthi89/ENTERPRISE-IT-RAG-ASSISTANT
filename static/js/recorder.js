/*
 * Microphone dictation (Voice V3a): record one short clip and transcribe it
 * with POST /api/voice/transcribe.
 *
 * No DOM access: the browser objects (mediaDevices, MediaRecorder, fetch,
 * isSecureContext, timers) are injected. Loaded as a classic script in the
 * browser (window.BluePeakRecorder) and with require() in Node tests.
 *
 * Contract:
 * - A session starts only from a real user action (``start(event)`` with a
 *   trusted event) in a secure context with the needed APIs. getUserMedia is
 *   called synchronously inside that call, before anything is awaited.
 * - ``start()`` resolves once per session, never rejects:
 *   ``{ ok: true, text, stoppedBy }`` or ``{ ok: false, code }`` (codes in
 *   ERROR_MESSAGES; "cancelled" when the caller cancelled).
 * - Recording stops on ``stop()``, at MAX_DURATION_MS, once the recorded
 *   size reaches EARLY_STOP_BYTES, or if the microphone goes away. Audio over
 *   MAX_AUDIO_BYTES is never uploaded.
 * - Every exit stops every microphone track at once, so the browser's
 *   "microphone in use" indicator turns off; a permission granted after
 *   ``cancel()`` is released immediately.
 * - The upload is abortable (``cancel()``) and bounded (UPLOAD_TIMEOUT_MS).
 * - Every callback checks that its session is still the current, unfinished
 *   one, and a session ends exactly once: events from an earlier session
 *   (late recorder events, a late permission grant, a late fetch result) are
 *   ignored.
 * - Nothing is stored: chunks are dropped as soon as the upload starts or
 *   the session ends. The transcript is only returned; this module never
 *   sends anything to the chat.
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
  const MAX_DURATION_MS = 60_000;
  const MAX_AUDIO_BYTES = 2 * 1024 * 1024; // the server's audio limit
  const EARLY_STOP_BYTES = Math.floor(1.9 * 1024 * 1024); // stop before reaching it
  const AUDIO_BITS_PER_SECOND = 32_000;
  const TIMESLICE_MS = 1_000; // collect data every second, so size is tracked while recording
  const UPLOAD_TIMEOUT_MS = 45_000; // the server allows 15 s to upload and 30 s for the provider

  const STATES = Object.freeze({
    IDLE: "idle",
    REQUESTING: "requesting", // waiting for microphone permission
    RECORDING: "recording",
    UPLOADING: "uploading",
  });

  // Preferred first. Each is accepted by the server (extension, MIME type and
  // container signature must agree there).
  const FORMATS = Object.freeze([
    Object.freeze({ mimeType: "audio/webm;codecs=opus", extension: ".webm" }), // Chrome, Edge
    Object.freeze({ mimeType: "audio/ogg;codecs=opus", extension: ".ogg" }), // Firefox
    Object.freeze({ mimeType: "audio/mp4", extension: ".m4a" }), // Safari
    Object.freeze({ mimeType: "audio/webm", extension: ".webm" }),
    Object.freeze({ mimeType: "audio/ogg", extension: ".ogg" }),
  ]);

  const EXTENSION_BY_TYPE = Object.freeze({
    "audio/webm": ".webm",
    "video/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/mp4": ".m4a",
    "video/mp4": ".m4a",
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
    empty_recording: "Nothing was recorded. Please try again.",
    too_large: "The recording is too large. Please record a shorter question.",
    upload_timeout: "Transcription took too long. Please try again.",
    network_error: "Couldn't reach the server. Please try again.",
    no_speech: "No speech was recognised. Please try again.",
    origin_not_allowed: "Dictation can't be used from this address.",
    invalid_upload: "The recording couldn't be sent. Please try again.",
    unsupported_audio: "This browser's recording format isn't supported.",
    audio_not_transcribable: "The recording couldn't be transcribed. Please try again.",
    transcription_timeout: "Transcription took too long. Please try again.",
    transcription_unavailable: "Transcription is temporarily unavailable. Please try again later.",
    transcription_disabled: "Dictation isn't available right now.",
    transcription_failed: "Transcription failed. Please try again.",
    cancelled: "Dictation was cancelled.",
  });

  // Codes the server sends in {"detail": {"code"}} (V2), passed through as-is.
  const SERVER_CODES = new Set([
    "origin_not_allowed",
    "unsupported_media_type",
    "audio_too_large",
    "upload_timeout",
    "invalid_upload",
    "unsupported_audio",
    "audio_not_transcribable",
    "transcription_timeout",
    "transcription_unavailable",
    "transcription_failed",
  ]);

  /** The first format the browser can record, or null. */
  function chooseFormat(MediaRecorderImpl) {
    if (!MediaRecorderImpl || typeof MediaRecorderImpl.isTypeSupported !== "function") return null;
    for (const format of FORMATS) {
      try {
        if (MediaRecorderImpl.isTypeSupported(format.mimeType)) return format;
      } catch {
        // A browser that throws for a type does not support it.
      }
    }
    return null;
  }

  function baseType(mimeType) {
    return typeof mimeType === "string" ? mimeType.split(";")[0].trim().toLowerCase() : "";
  }

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

  /**
   * A recorder over injected browser dependencies.
   *
   * Options: ``mediaDevices`` (navigator.mediaDevices), ``MediaRecorder``,
   * ``isSecureContext``, ``fetchImpl``, and optionally ``FormData``,
   * ``Blob``, ``AbortController``, ``setTimeout``/``clearTimeout``, ``now``,
   * ``url``, the limits (``maxDurationMs``, ``maxAudioBytes``,
   * ``earlyStopBytes``, ``audioBitsPerSecond``, ``timesliceMs``,
   * ``uploadTimeoutMs``) and ``onStateChange(state, detail)``.
   */
  function createRecorder(options = {}) {
    const {
      mediaDevices,
      MediaRecorder: MediaRecorderImpl,
      isSecureContext = false,
      fetchImpl,
      FormData: FormDataImpl = globalThis.FormData,
      Blob: BlobImpl = globalThis.Blob,
      AbortController: AbortControllerImpl = globalThis.AbortController,
      setTimeout: setTimer = globalThis.setTimeout,
      clearTimeout: clearTimer = globalThis.clearTimeout,
      now = () => Date.now(),
      url = TRANSCRIBE_URL,
      maxDurationMs = MAX_DURATION_MS,
      maxAudioBytes = MAX_AUDIO_BYTES,
      earlyStopBytes = EARLY_STOP_BYTES,
      audioBitsPerSecond = AUDIO_BITS_PER_SECOND,
      timesliceMs = TIMESLICE_MS,
      uploadTimeoutMs = UPLOAD_TIMEOUT_MS,
      onStateChange = null,
    } = options;

    const supported = Boolean(
      mediaDevices &&
        typeof mediaDevices.getUserMedia === "function" &&
        typeof MediaRecorderImpl === "function" &&
        typeof MediaRecorderImpl.isTypeSupported === "function" &&
        typeof fetchImpl === "function" &&
        typeof FormDataImpl === "function" &&
        typeof BlobImpl === "function" &&
        typeof AbortControllerImpl === "function",
    );

    let state = STATES.IDLE;
    let current = null; // the active session, or null; a new one starts only once it has ended

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

    function stopRecorder(session) {
      const recorder = session.recorder;
      if (!recorder) return;
      try {
        if (recorder.state !== "inactive") recorder.stop();
      } catch {
        // Already stopped.
      }
    }

    /** End a session exactly once: release everything, then resolve. */
    function finish(session, outcome) {
      if (session.finished) return;
      session.finished = true;
      if (current === session) current = null;

      if (session.durationTimer !== null) clearTimer(session.durationTimer);
      if (session.uploadTimer !== null) clearTimer(session.uploadTimer);
      session.durationTimer = null;
      session.uploadTimer = null;
      if (session.uploadAbort) session.uploadAbort.abort();
      stopRecorder(session);
      stopTracks(session);
      session.chunks = []; // nothing is kept
      session.bytes = 0;

      setState(STATES.IDLE, outcome.ok ? {} : { code: outcome.code });
      session.resolve(outcome);
    }

    /** Stop recording (once) and upload what was recorded. */
    function requestStop(session, reason) {
      if (session.finished || session.stoppedBy !== null || state !== STATES.RECORDING) return;
      session.stoppedBy = reason;
      if (session.durationTimer !== null) clearTimer(session.durationTimer);
      session.durationTimer = null;
      stopRecorder(session); // the recorder delivers its last data, then "stop"
      stopTracks(session); // the microphone is released at once
    }

    function isCurrent(session) {
      return !session.finished && current === session;
    }

    function onStream(session, stream) {
      session.stream = stream;
      if (!isCurrent(session)) {
        stopTracks(session); // permission arrived after cancel(): release it now
        return;
      }

      let recorder;
      try {
        recorder = new MediaRecorderImpl(stream, {
          mimeType: session.format.mimeType,
          audioBitsPerSecond,
        });
      } catch {
        finish(session, { ok: false, code: "recording_failed" });
        return;
      }
      session.recorder = recorder;

      recorder.ondataavailable = (event) => {
        if (!isCurrent(session)) return;
        const data = event && event.data;
        if (!data || !data.size) return;
        session.chunks.push(data);
        session.bytes += data.size;
        if (session.bytes >= earlyStopBytes) requestStop(session, "size_limit");
      };
      recorder.onerror = () => {
        if (!isCurrent(session)) return;
        finish(session, { ok: false, code: "recording_failed" });
      };
      recorder.onstop = () => {
        if (!isCurrent(session)) return;
        upload(session, recorder.mimeType);
      };

      try {
        for (const track of stream.getTracks()) {
          track.onended = () => {
            if (isCurrent(session)) requestStop(session, "microphone_ended");
          };
        }
        recorder.start(timesliceMs);
      } catch {
        finish(session, { ok: false, code: "recording_failed" });
        return;
      }

      setState(STATES.RECORDING, { startedAt: now(), maxDurationMs });
      session.durationTimer = setTimer(() => {
        session.durationTimer = null;
        if (isCurrent(session)) requestStop(session, "time_limit");
      }, maxDurationMs);
    }

    function upload(session, recordedType) {
      if (session.bytes === 0) {
        finish(session, { ok: false, code: "empty_recording" });
        return;
      }
      if (session.bytes > maxAudioBytes) {
        finish(session, { ok: false, code: "too_large" });
        return;
      }

      const type = baseType(recordedType) || baseType(session.format.mimeType);
      const extension = EXTENSION_BY_TYPE[type] || session.format.extension;
      let body;
      try {
        const audio = new BlobImpl(session.chunks, { type: recordedType || session.format.mimeType });
        body = new FormDataImpl();
        body.append("audio", audio, `recording${extension}`);
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
            finish(session, { ok: false, code: serverErrorCode(response.status, data) });
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
     * Start one dictation session from a user action. Call it from the
     * click handler with the click event: getUserMedia runs synchronously.
     */
    function start(event) {
      if (state !== STATES.IDLE) return Promise.resolve({ ok: false, code: "busy" });
      if (!event || event.isTrusted !== true) {
        return Promise.resolve({ ok: false, code: "not_user_action" });
      }
      if (isSecureContext !== true) return Promise.resolve({ ok: false, code: "insecure_context" });
      if (!supported) return Promise.resolve({ ok: false, code: "unsupported" });
      const format = chooseFormat(MediaRecorderImpl);
      if (!format) return Promise.resolve({ ok: false, code: "unsupported" });

      return new Promise((resolve) => {
        const session = {
          resolve,
          format,
          stream: null,
          recorder: null,
          chunks: [],
          bytes: 0,
          stoppedBy: null,
          durationTimer: null,
          uploadTimer: null,
          uploadAbort: null,
          timedOut: false,
          finished: false,
        };
        current = session;
        setState(STATES.REQUESTING);

        let request;
        try {
          request = mediaDevices.getUserMedia({ audio: true, video: false });
        } catch (error) {
          finish(session, { ok: false, code: microphoneErrorCode(error) });
          return;
        }
        Promise.resolve(request).then(
          (stream) => onStream(session, stream),
          (error) => {
            if (isCurrent(session)) finish(session, { ok: false, code: microphoneErrorCode(error) });
          },
        );
      });
    }

    /** Stop recording and transcribe it. While permission is pending, it cancels. */
    function stop() {
      if (!current) return;
      if (state === STATES.REQUESTING) {
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
    MAX_DURATION_MS,
    MAX_AUDIO_BYTES,
    EARLY_STOP_BYTES,
    AUDIO_BITS_PER_SECOND,
    TIMESLICE_MS,
    UPLOAD_TIMEOUT_MS,
    STATES,
    FORMATS,
    ERROR_MESSAGES,
    chooseFormat,
    createRecorder,
  };
});
