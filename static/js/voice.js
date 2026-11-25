/*
 * Spoken answers (Voice V1): Markdown-to-speech text and a speaker built on
 * the browser's speech synthesis.
 *
 * No DOM access and no network: the browser's speech objects, timers and
 * listeners are injected, and nothing is uploaded or stored. Loaded as a
 * classic script in the browser (window.BluePeakVoice) and with require() in
 * Node tests.
 *
 * Privacy policy (V1):
 * - Only on-device voices are used (SpeechSynthesisVoice.localService).
 *   Network voices, such as some browsers' cloud voices, would send the
 *   answer text to a third party, so they are never selected. Without an
 *   on-device voice for the page's language, nothing is spoken.
 * - Speech is built from the answer's Markdown as plain text; HTML is never
 *   parsed or rendered here.
 *
 * Speaking:
 * - Answers are capped at SPEECH_MAX_CHARS characters, cut at a sentence
 *   boundary and followed by TRUNCATION_NOTICE (the notice counts toward the
 *   cap).
 * - Text is queued as sentence-sized utterances of at most CHUNK_MAX_CHARS
 *   characters, because some engines stop long utterances partway through.
 * - Every speak() and cancel() starts a new generation. Callbacks from an
 *   earlier generation (for example the "interrupted" error an engine reports
 *   after cancel()) are ignored, so a cancelled answer never resumes or
 *   changes the state.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  } else {
    root.BluePeakVoice = api;
  }
})(typeof self !== "undefined" ? self : globalThis, function () {
  "use strict";

  const SPEECH_MAX_CHARS = 1500;
  // Only this much of an answer is converted: far more than is ever spoken,
  // and it bounds the conversion time for unusually long or hostile input.
  const MARKDOWN_SCAN_MAX_CHARS = 20000;
  const CHUNK_MAX_CHARS = 200;
  const VOICE_LOAD_TIMEOUT_MS = 3000;

  const CODE_PLACEHOLDER = "Code example omitted, see the screen.";
  const TABLE_PLACEHOLDER = "A table is shown on screen.";
  const LINK_PLACEHOLDER = "a link";
  const TRUNCATION_NOTICE = "The full answer is on screen.";

  const STATES = Object.freeze({
    IDLE: "idle",
    LOADING: "loading",
    SPEAKING: "speaking",
    UNAVAILABLE: "unavailable",
  });

  // -------------------------------------------------------------------
  // Markdown to speech text
  // -------------------------------------------------------------------

  const ENTITIES = {
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&#39;": "'",
    "&apos;": "'",
    "&nbsp;": " ",
  };

  function ensureSentence(text) {
    const trimmed = text.trim();
    if (!trimmed) return "";
    return /[.!?:;]$/.test(trimmed) ? trimmed : `${trimmed}.`;
  }

  function isSeparatorRow(line) {
    const trimmed = line.trim();
    return trimmed.includes("|") && /^\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?$/.test(trimmed);
  }

  /**
   * Indexes of table lines. As in GitHub-flavoured Markdown, a table is a
   * header row with a pipe, then a separator row (``| --- |``), then any
   * following rows with a pipe. Pipes alone do not make a table: prose and
   * shell pipelines such as ``ps aux | grep vpn | head`` are spoken.
   */
  function tableLineIndexes(lines) {
    const rows = new Set();
    for (let i = 1; i < lines.length; i += 1) {
      if (!isSeparatorRow(lines[i]) || !lines[i - 1].includes("|") || rows.has(i - 1)) continue;
      rows.add(i - 1);
      rows.add(i);
      for (let j = i + 1; j < lines.length && lines[j].trim() && lines[j].includes("|"); j += 1) {
        rows.add(j);
      }
    }
    return rows;
  }

  /** Block-level Markdown, line by line: headings, lists, quotes, tables, rules. */
  function speakLines(text) {
    const out = [];
    const lines = text.split("\n");
    const tableRows = tableLineIndexes(lines);
    let inTable = false;

    for (const [index, rawLine] of lines.entries()) {
      if (tableRows.has(index)) {
        if (!inTable) out.push(TABLE_PLACEHOLDER);
        inTable = true;
        continue;
      }
      inTable = false;

      let line = rawLine.replace(/^\s{0,3}>+\s?/, ""); // blockquote

      if (/^\s*([-*_])(?:\s*\1){2,}\s*$/.test(line)) continue; // horizontal rule
      if (/^\s{0,3}\[[^\]\n]{1,200}\]:\s*\S/.test(line)) continue; // link reference definition

      const heading = line.match(/^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$/);
      if (heading) {
        out.push(ensureSentence(heading[1]));
        continue;
      }

      const item = line.match(/^\s*(?:[-*+]|\d{1,9}[.)])\s+(.*)$/);
      if (item) {
        out.push(ensureSentence(item[1]));
        continue;
      }

      out.push(line.trim());
    }

    return out.filter(Boolean).join(" ");
  }

  /** Inline Markdown: images, links, URLs, citations, code and emphasis. */
  function speakInline(text) {
    return text
      .replace(/!\[([^\]\n]{0,500})\]\([^)\n]{0,2000}\)/g, "$1") // image: alt text
      .replace(/\[([^\]\n]{1,500})\]\([^)\n]{0,2000}\)/g, "$1") // inline link: its text
      .replace(/\[([^\]\n]{1,500})\]\[[^\]\n]{0,200}\]/g, "$1") // reference link
      .replace(/\[\^[^\]\n]{1,50}\]/g, "") // footnote reference
      .replace(/\s*\[\d{1,4}(?:\s*[,–-]\s*\d{1,4}){0,20}\]/g, "") // citations: [1], [1, 2], [2-3]
      .replace(/\b(?:https?:\/\/|www\.)[^\s<>()]*[^\s<>().,;:!?'"]/gi, LINK_PLACEHOLDER) // bare URL, not its trailing punctuation
      .replace(/`([^`\n]{1,500})`/g, "$1") // inline code: its text
      .replace(/(\*\*|__|~~)(?=\S)([^\n]{0,1000}?\S)\1/g, "$2") // strong, strikethrough
      .replace(/(^|[^\w*])\*(?=\S)([^*\n]{0,1000}?\S)\*(?!\w)/g, "$1$2") // *emphasis*
      .replace(/(^|[^\w])_(?=\S)([^_\n]{0,1000}?\S)_(?![\w])/g, "$1$2") // _emphasis_, not snake_case
      .replace(/[`*]+/g, "") // leftover markers
      .replace(/\p{Extended_Pictographic}|️/gu, ""); // emoji
  }

  function capSpeech(text, maxChars, alreadyCut = false) {
    if (alreadyCut) {
      // The answer was longer than MARKDOWN_SCAN_MAX_CHARS: always say so.
      const withNotice = text ? `${text} ${TRUNCATION_NOTICE}` : TRUNCATION_NOTICE;
      if (withNotice.length <= maxChars) return withNotice;
    } else if (text.length <= maxChars) {
      return text;
    }

    const room = maxChars - TRUNCATION_NOTICE.length - 1;
    if (room <= 0) return TRUNCATION_NOTICE.slice(0, maxChars);

    const head = text.slice(0, room + 1); // +1: a sentence may end exactly at the limit
    let cut = -1;
    const boundary = /[.!?](?=\s)/g;
    let match;
    while ((match = boundary.exec(head)) !== null) {
      if (match.index + 1 <= room) cut = match.index + 1;
    }
    if (cut < room / 2) {
      // No sentence boundary in the second half: cut at the last word instead.
      const space = text.lastIndexOf(" ", room);
      cut = space > 0 ? space : room;
    }
    return `${text.slice(0, cut).trim()} ${TRUNCATION_NOTICE}`;
  }

  /**
   * Plain text to speak for an answer written in Markdown. Never throws;
   * returns "" for anything that is not a non-empty string.
   */
  function toSpeechText(markdown, { maxChars = SPEECH_MAX_CHARS } = {}) {
    if (typeof markdown !== "string" || !markdown.trim()) return "";

    const alreadyCut = markdown.length > MARKDOWN_SCAN_MAX_CHARS;
    let text = markdown.slice(0, MARKDOWN_SCAN_MAX_CHARS).replace(/\r\n?/g, "\n");

    // Escaped Markdown characters, as app.js unescapes them before rendering.
    text = text.replace(/\\([*_`#[\]()])/g, "$1");

    // Fenced code blocks (also one left open at the end). The fence length is
    // bounded and the info string may not contain the fence character (as in
    // CommonMark), so a long run of backticks fails fast instead of
    // backtracking.
    text = text
      .replace(/^[ \t]*(`{3,10}|~{3,10})[^`~\n]*\n[\s\S]*?^[ \t]*\1[ \t]*$/gm, `\n${CODE_PLACEHOLDER}\n`)
      .replace(/^[ \t]*(`{3,10}|~{3,10})[^`~\n]*(?:\n[\s\S]*)?$/m, `\n${CODE_PLACEHOLDER}\n`);

    // HTML: autolinks first, then whole script/style elements, comments, tags.
    text = text
      .replace(/<(?:https?:\/\/|mailto:)[^>\s]{1,2000}>/gi, LINK_PLACEHOLDER)
      .replace(/<(script|style)\b[^>]*>[\s\S]*?<\/\1\s*>/gi, " ")
      .replace(/<!--[\s\S]*?-->/g, " ")
      .replace(/<\/?[a-zA-Z][^<>]{0,2000}>/g, " ")
      .replace(/&(?:amp|lt|gt|quot|#39|apos|nbsp);/g, (entity) => ENTITIES[entity]);

    text = speakInline(speakLines(text));

    text = text
      .replace(/\s+/g, " ")
      .replace(/\s+([.,;:!?])/g, "$1")
      .replace(/([.!?])\.(?!\.)/g, "$1")
      .trim();

    return capSpeech(text, maxChars, alreadyCut);
  }

  // -------------------------------------------------------------------
  // Sentence-sized chunks
  // -------------------------------------------------------------------

  function splitLong(piece, maxChars) {
    const parts = [];
    let current = "";
    for (const word of piece.split(" ")) {
      if (word.length > maxChars) {
        if (current) parts.push(current);
        current = "";
        for (let i = 0; i < word.length; i += maxChars) parts.push(word.slice(i, i + maxChars));
        continue;
      }
      const next = current ? `${current} ${word}` : word;
      if (next.length > maxChars) {
        parts.push(current);
        current = word;
      } else {
        current = next;
      }
    }
    if (current) parts.push(current);
    return parts;
  }

  /** Split speech text into chunks of at most ``maxChars``, packing whole sentences. */
  function splitIntoChunks(text, maxChars = CHUNK_MAX_CHARS) {
    if (typeof text !== "string") return [];
    const normalized = text.replace(/\s+/g, " ").trim();
    if (!normalized) return [];

    // Sentences end at . ! or ? followed by whitespace. Found with exec()
    // rather than a lookbehind split: lookbehind is a syntax error in older
    // Safari (before 16.4) and would stop this whole script from loading.
    const sentences = [];
    const boundary = /[.!?]\s+/g;
    let start = 0;
    let match;
    while ((match = boundary.exec(normalized)) !== null) {
      sentences.push(normalized.slice(start, match.index + 1));
      start = match.index + match[0].length;
    }
    if (start < normalized.length) sentences.push(normalized.slice(start));

    const pieces = [];
    for (const sentence of sentences) {
      if (sentence.length <= maxChars) pieces.push(sentence);
      else pieces.push(...splitLong(sentence, maxChars));
    }

    const chunks = [];
    let current = "";
    for (const piece of pieces) {
      const next = current ? `${current} ${piece}` : piece;
      if (next.length > maxChars) {
        chunks.push(current);
        current = piece;
      } else {
        current = next;
      }
    }
    if (current) chunks.push(current);
    return chunks;
  }

  // -------------------------------------------------------------------
  // Voice selection: on-device voices only
  // -------------------------------------------------------------------

  function normalizeLang(lang) {
    return typeof lang === "string" ? lang.trim().replace(/_/g, "-").toLowerCase() : "";
  }

  /**
   * The on-device voice to use for ``lang``, or null. Network voices
   * (``localService !== true``) are never returned. Prefers an exact language
   * match, then the same primary language; within each, the default voice.
   */
  function pickLocalVoice(voices, lang) {
    if (!Array.isArray(voices) && !(voices && typeof voices.length === "number")) return null;
    const wanted = normalizeLang(lang) || "en-us";
    const primary = wanted.split("-")[0];
    const local = Array.from(voices).filter((voice) => voice && voice.localService === true);

    const exact = local.filter((voice) => normalizeLang(voice.lang) === wanted);
    const sameLanguage = local.filter((voice) => normalizeLang(voice.lang).split("-")[0] === primary);
    for (const group of [exact, sameLanguage]) {
      if (group.length) return group.find((voice) => voice.default) || group[0];
    }
    return null;
  }

  // -------------------------------------------------------------------
  // Speaker
  // -------------------------------------------------------------------

  /**
   * A speaker over injected speech dependencies.
   *
   * Options: ``synth`` (a SpeechSynthesis), ``Utterance`` (the
   * SpeechSynthesisUtterance constructor), ``lang`` (BCP 47, default
   * "en-US"), ``setTimeout``/``clearTimeout``, ``voiceLoadTimeoutMs``,
   * ``maxChars``, ``chunkMaxChars`` and ``onStateChange(state, detail)``.
   *
   * ``speak(markdown)`` cancels anything playing, then resolves (never
   * rejects) with ``{ spoken: true }`` after the last chunk, or
   * ``{ spoken: false, reason }`` where reason is "unsupported", "empty",
   * "no_local_voice", "cancelled" or "error" (with the engine's ``error``,
   * for example "not-allowed" when the browser blocks speech).
   *
   * ``prime()`` is the synchronous user-gesture unlock; ``hasLocalVoice()``
   * checks for an on-device voice before speech is offered.
   */
  function createSpeaker(options = {}) {
    const {
      synth,
      Utterance,
      lang = "en-US",
      setTimeout: setTimer = globalThis.setTimeout,
      clearTimeout: clearTimer = globalThis.clearTimeout,
      voiceLoadTimeoutMs = VOICE_LOAD_TIMEOUT_MS,
      maxChars = SPEECH_MAX_CHARS,
      chunkMaxChars = CHUNK_MAX_CHARS,
      onStateChange = null,
    } = options;

    const supported = Boolean(
      synth &&
        typeof synth.speak === "function" &&
        typeof synth.cancel === "function" &&
        typeof synth.getVoices === "function" &&
        typeof Utterance === "function",
    );

    let state = supported ? STATES.IDLE : STATES.UNAVAILABLE;
    let generation = 0;
    let settleCurrent = null; // resolves the pending speak() promise
    let cachedVoice; // undefined until the voice list has been read once
    let voiceWait = null; // the one in-flight wait for "voiceschanged", shared by every speak()
    // The utterances of the current generation. Some browsers (Chrome) can
    // garbage-collect an utterance referenced only by the engine's queue, and
    // its onend then never fires; holding them until the answer settles keeps
    // speak() from hanging in the "speaking" state.
    let activeUtterances = [];

    function setState(next, detail = {}) {
      if (state === next && !detail.reason) return;
      state = next;
      if (typeof onStateChange === "function") {
        try {
          onStateChange(next, detail);
        } catch {
          // A view error must not break speech.
        }
      }
    }

    function readVoices() {
      try {
        const voices = synth.getVoices();
        return voices && voices.length ? Array.from(voices) : [];
      } catch {
        return [];
      }
    }

    /**
     * The voice for ``lang``; waits (bounded) for the voice list to load.
     * Concurrent callers share one wait, so one listener and one timer exist
     * at most. A cancelled answer leaves the wait running to its end: it only
     * fills the voice cache for the next answer.
     */
    function resolveVoice() {
      if (cachedVoice !== undefined) return Promise.resolve(cachedVoice);
      if (voiceWait) return voiceWait;

      const voices = readVoices();
      if (voices.length) {
        cachedVoice = pickLocalVoice(voices, lang);
        return Promise.resolve(cachedVoice);
      }

      // Some browsers fill the list asynchronously and fire "voiceschanged".
      voiceWait = new Promise((resolve) => {
        let timer = null;
        const canListen = typeof synth.addEventListener === "function";

        function finish() {
          if (timer !== null) clearTimer(timer);
          timer = null;
          if (canListen) synth.removeEventListener("voiceschanged", onChanged);
          voiceWait = null;
          const loaded = readVoices();
          if (loaded.length) cachedVoice = pickLocalVoice(loaded, lang); // keep retrying while empty
          resolve(loaded.length ? cachedVoice : null);
        }

        function onChanged() {
          if (readVoices().length) finish();
        }

        if (canListen) synth.addEventListener("voiceschanged", onChanged);
        timer = setTimer(finish, voiceLoadTimeoutMs);
      });
      return voiceWait;
    }

    function settle(result) {
      const resolve = settleCurrent;
      settleCurrent = null;
      activeUtterances = [];
      if (resolve) resolve(result);
    }

    /** Stop speaking now and drop anything queued. Safe to call at any time. */
    function cancel() {
      generation += 1;
      settle({ spoken: false, reason: "cancelled" });
      if (supported) {
        try {
          synth.cancel();
        } catch {
          // Nothing more can be done; the generation change ignores late callbacks.
        }
      }
      if (state === STATES.SPEAKING || state === STATES.LOADING) setState(STATES.IDLE);
    }

    /**
     * Best-effort unlock for browsers that only allow speech a user started
     * (Safari, iOS). Call it synchronously inside the click handler, before
     * anything is awaited. It cancels anything playing (a new generation, so
     * every stale-callback guarantee holds), then speaks an empty, silent
     * utterance with no handlers: no text reaches any voice and the speaker's
     * state is unchanged. Returns false if unsupported or the engine throws.
     * True does not prove later speech is allowed; speak() reports that.
     */
    function prime() {
      if (!supported) return false;
      cancel();
      try {
        const utterance = new Utterance("");
        utterance.volume = 0;
        synth.speak(utterance);
        return true;
      } catch {
        return false;
      }
    }

    /**
     * Whether an on-device voice exists for ``lang``. Waits (bounded, shared
     * with speak()) for the voice list; resolves, never rejects.
     */
    async function hasLocalVoice() {
      if (!supported) return false;
      return Boolean(await resolveVoice());
    }

    async function speak(markdown) {
      if (!supported) {
        setState(STATES.UNAVAILABLE, { reason: "unsupported" });
        return { spoken: false, reason: "unsupported" };
      }

      cancel();
      const mine = generation;

      const text = toSpeechText(markdown, { maxChars });
      const chunks = splitIntoChunks(text, chunkMaxChars);
      if (!chunks.length) return { spoken: false, reason: "empty" };

      return new Promise((resolve) => {
        settleCurrent = resolve;
        setState(STATES.LOADING);

        resolveVoice().then((voice) => {
          if (mine !== generation) return; // cancelled while loading (already settled)

          if (!voice) {
            setState(STATES.UNAVAILABLE, { reason: "no_local_voice" });
            settle({ spoken: false, reason: "no_local_voice" });
            return;
          }

          setState(STATES.SPEAKING);
          const last = chunks.length - 1;

          function fail(error) {
            if (mine !== generation) return; // a stale callback
            generation += 1;
            try {
              synth.cancel();
            } catch {
              // Ignore; the queue belongs to a finished generation now.
            }
            setState(STATES.IDLE);
            settle({ spoken: false, reason: "error", error });
          }

          try {
            chunks.forEach((chunk, index) => {
              const utterance = new Utterance(chunk);
              utterance.voice = voice;
              utterance.lang = voice.lang || lang;
              utterance.onend = () => {
                if (mine !== generation || index !== last) return;
                setState(STATES.IDLE);
                settle({ spoken: true });
              };
              utterance.onerror = (event) => fail((event && event.error) || "unknown");
              activeUtterances.push(utterance);
              synth.speak(utterance);
            });
          } catch (error) {
            fail((error && error.message) || "speak_failed");
          }
        });
      });
    }

    return {
      speak,
      cancel,
      prime,
      hasLocalVoice,
      get state() {
        return state;
      },
      get supported() {
        return supported;
      },
    };
  }

  return {
    SPEECH_MAX_CHARS,
    MARKDOWN_SCAN_MAX_CHARS,
    CHUNK_MAX_CHARS,
    VOICE_LOAD_TIMEOUT_MS,
    CODE_PLACEHOLDER,
    TABLE_PLACEHOLDER,
    LINK_PLACEHOLDER,
    TRUNCATION_NOTICE,
    STATES,
    toSpeechText,
    splitIntoChunks,
    pickLocalVoice,
    createSpeaker,
  };
});
