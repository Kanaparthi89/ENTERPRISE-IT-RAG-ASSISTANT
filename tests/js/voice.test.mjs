// Voice V1a: static/js/voice.js (Markdown-to-speech text and the speaker).
// Deterministic: speech synthesis, utterances and timers are fakes; nothing
// touches a browser, the network or a real voice.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { test } from "node:test";

import { fakeTimers, tick } from "./helpers.mjs";

const require = createRequire(import.meta.url);
const voice = require("../../static/js/voice.js");
const VOICE_SOURCE = readFileSync(new URL("../../static/js/voice.js", import.meta.url), "utf-8");

const {
  CHUNK_MAX_CHARS,
  CODE_PLACEHOLDER,
  LINK_PLACEHOLDER,
  SPEECH_MAX_CHARS,
  STATES,
  TABLE_PLACEHOLDER,
  TRUNCATION_NOTICE,
  VOICE_LOAD_TIMEOUT_MS,
  createSpeaker,
  pickLocalVoice,
  splitIntoChunks,
  toSpeechText,
} = voice;

// ---------------------------------------------------------------------------
// Fakes
// ---------------------------------------------------------------------------

const LOCAL_EN = { name: "Samantha", lang: "en-US", localService: true, default: false };
const LOCAL_EN_DEFAULT = { name: "Alex", lang: "en-US", localService: true, default: true };
const LOCAL_EN_GB = { name: "Daniel", lang: "en-GB", localService: true, default: false };
const NETWORK_EN = { name: "Google US English", lang: "en-US", localService: false, default: true };
const LOCAL_FR = { name: "Thomas", lang: "fr-FR", localService: true, default: false };

class FakeUtterance {
  constructor(text) {
    this.text = text;
    this.voice = null;
    this.lang = "";
    this.onend = null;
    this.onerror = null;
  }
}

/** A SpeechSynthesis double that records utterances and lets tests drive callbacks. */
class FakeSynth {
  constructor(voices = [LOCAL_EN]) {
    this.voices = voices;
    this.queue = [];
    this.spoken = [];
    this.cancelCount = 0;
    this.listeners = new Map();
    this.speakError = null;
  }

  getVoices() {
    return this.voices;
  }

  speak(utterance) {
    if (this.speakError) throw this.speakError;
    this.queue.push(utterance);
    this.spoken.push(utterance);
  }

  /** Like browsers, report "interrupted" to every utterance dropped by cancel(). */
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

  listenerCount(type) {
    return this.listeners.get(type)?.size ?? 0;
  }

  loadVoices(voices) {
    this.voices = voices;
    for (const listener of [...(this.listeners.get("voiceschanged") ?? [])]) listener();
  }

  /** Finish the first queued utterance. */
  finishNext() {
    const utterance = this.queue.shift();
    utterance?.onend?.({});
    return utterance;
  }

  finishAll() {
    while (this.queue.length) this.finishNext();
  }

  /** Fail the first queued utterance with ``error`` (not a cancel). */
  failNext(error = "synthesis-failed") {
    const utterance = this.queue.shift();
    utterance?.onerror?.({ error });
  }
}

function speakerWith({ synth = new FakeSynth(), timers = fakeTimers(), ...options } = {}) {
  const states = [];
  const speaker = createSpeaker({
    synth,
    Utterance: FakeUtterance,
    setTimeout: timers.setTimeout,
    clearTimeout: timers.clearTimeout,
    onStateChange: (state, detail) => states.push(detail.reason ? `${state}:${detail.reason}` : state),
    ...options,
  });
  return { speaker, synth, timers, states };
}

// ---------------------------------------------------------------------------
// toSpeechText: Markdown to plain speech text
// ---------------------------------------------------------------------------

test("headings become sentences without markers", () => {
  assert.equal(toSpeechText("# Connect to the VPN\nOpen the app"), "Connect to the VPN. Open the app");
  assert.equal(toSpeechText("### Steps ###"), "Steps.");
  assert.equal(toSpeechText("## Done!"), "Done!");
});

test("bulleted and numbered list items become sentences", () => {
  const markdown = "Steps:\n- Open GlobalProtect\n* Sign in\n+ Connect\n1. Restart\n2) Retry";
  assert.equal(toSpeechText(markdown), "Steps: Open GlobalProtect. Sign in. Connect. Restart. Retry.");
});

test("fenced code blocks are replaced, including one left open", () => {
  const closed = "Run this:\n```bash\nrm -rf /tmp/cache\n```\nThen sign in.";
  assert.equal(toSpeechText(closed), `Run this: ${CODE_PLACEHOLDER} Then sign in.`);

  const tildes = "Before\n~~~\nsecret = 1\n~~~\nAfter";
  assert.equal(toSpeechText(tildes), `Before ${CODE_PLACEHOLDER} After`);

  const open = "Use:\n```\nkubectl get pods\nkubectl logs";
  assert.equal(toSpeechText(open), `Use: ${CODE_PLACEHOLDER}`);
  assert.ok(!toSpeechText(open).includes("kubectl"));
});

test("inline code keeps its text without backticks", () => {
  assert.equal(toSpeechText("Run `ipconfig /flushdns` now."), "Run ipconfig /flushdns now.");
});

test("links keep their text; URLs are never read out", () => {
  assert.equal(toSpeechText("See [the VPN guide](https://it.example.com/vpn)."), "See the VPN guide.");
  assert.equal(toSpeechText("Docs: https://it.example.com/vpn?x=1&y=2 today"), `Docs: ${LINK_PLACEHOLDER} today`);
  assert.equal(toSpeechText("Visit www.example.com."), `Visit ${LINK_PLACEHOLDER}.`);
  assert.equal(toSpeechText("Mail <https://example.com/a>."), `Mail ${LINK_PLACEHOLDER}.`);
  assert.equal(toSpeechText("![VPN icon](https://x.example/i.png) Connected"), "VPN icon Connected");
  assert.equal(toSpeechText("Read [the guide][vpn].\n\n[vpn]: https://it.example.com/vpn"), "Read the guide.");
});

test("citation markers and footnotes are removed", () => {
  assert.equal(
    toSpeechText("MFA is required [1]. Use the app [2, 3] or a key [4-5].[^note]"),
    "MFA is required. Use the app or a key.",
  );
});

test("a table is announced once instead of read cell by cell", () => {
  const markdown = "Options:\n\n| OS | Client |\n| --- | :---: |\n| macOS | GlobalProtect |\n| Windows | GlobalProtect |\n\nPick one.";
  assert.equal(toSpeechText(markdown), `Options: ${TABLE_PLACEHOLDER} Pick one.`);
});

test("pipes outside a table are spoken, not announced as a table", () => {
  // Regression: any line with two pipes used to become the table placeholder.
  assert.equal(
    toSpeechText("Run `ps aux | grep vpn | head` to check the client."),
    "Run ps aux | grep vpn | head to check the client.",
  );
  assert.equal(toSpeechText("Choose A | B | C, then save."), "Choose A | B | C, then save.");
  // A separator row without a header row above it is not a table either.
  assert.equal(toSpeechText("| --- |\nText"), "| --- | Text");
});

test("tables in every GFM form are announced once", () => {
  assert.equal(toSpeechText("OS | Client\n--- | ---\nmacOS | GP\nWindows | GP\nAfter."), `${TABLE_PLACEHOLDER} After.`);
  assert.equal(toSpeechText("| Step |\n| :--- |\n| Restart |"), TABLE_PLACEHOLDER);
  assert.equal(
    toSpeechText("| A | B |\n| - | - |\n| 1 | 2 |"), // separators shorter than three dashes are not a table
    "| A | B | | - | - | | 1 | 2 |",
  );
  assert.equal(
    toSpeechText("| A | B |\n|---|---|\n| 1 | 2 |\n\nText\n\n| C | D |\n|---|---|\n| 3 | 4 |"),
    `${TABLE_PLACEHOLDER} Text ${TABLE_PLACEHOLDER}`,
  );
});

test("HTML is stripped as text, and script and style content is dropped", () => {
  const html = '<p>Hello <b>there</b></p><script>alert("x")</script><style>p{}</style><!-- note -->';
  assert.equal(toSpeechText(html), "Hello there");
  assert.equal(toSpeechText("Use &lt;Ctrl&gt; &amp; &quot;Alt&quot;&nbsp;now"), 'Use <Ctrl> & "Alt" now');
  assert.ok(!toSpeechText('<img src=x onerror="alert(1)">Hi').includes("alert"));
});

test("emphasis markers are removed but snake_case and arithmetic survive", () => {
  assert.equal(toSpeechText("**Important:** use *two* factors and ~~SMS~~ __keys__."), "Important: use two factors and SMS keys.");
  assert.equal(toSpeechText("Set max_retry_count to 3."), "Set max_retry_count to 3.");
  assert.equal(toSpeechText("Escaped \\*not emphasis\\* and \\[1\\]"), "Escaped not emphasis and");
});

test("blockquotes and horizontal rules are flattened", () => {
  assert.equal(toSpeechText("> Note: restart first\n\n---\n\nThen connect"), "Note: restart first Then connect");
});

test("whitespace is collapsed and emoji are dropped", () => {
  assert.equal(toSpeechText("  Hello\n\n\n   world\t !  🚀✅  "), "Hello world!");
});

test("non-text and empty answers give empty speech", () => {
  for (const value of [undefined, null, 42, {}, [], "", "   \n\t"]) {
    assert.equal(toSpeechText(value), "");
  }
  assert.equal(toSpeechText("```\ncode only\n```"), CODE_PLACEHOLDER);
});

test("long answers are capped at 1,500 characters at a sentence boundary", () => {
  const sentence = "This sentence explains one step of the procedure in detail.";
  const markdown = Array.from({ length: 60 }, () => sentence).join(" ");
  const speech = toSpeechText(markdown);

  assert.equal(SPEECH_MAX_CHARS, 1500);
  assert.ok(speech.length <= SPEECH_MAX_CHARS, `length ${speech.length}`);
  assert.ok(speech.endsWith(` ${TRUNCATION_NOTICE}`));
  const spokenPart = speech.slice(0, -TRUNCATION_NOTICE.length - 1);
  assert.ok(spokenPart.endsWith("detail."), "cut after a whole sentence");
});

test("a long answer without sentence breaks is cut at a word", () => {
  const speech = toSpeechText(Array.from({ length: 400 }, () => "word").join(" "));
  assert.ok(speech.length <= SPEECH_MAX_CHARS);
  assert.ok(speech.endsWith(` ${TRUNCATION_NOTICE}`));
  assert.match(speech.slice(0, -TRUNCATION_NOTICE.length - 1), /(?:^| )word$/);
});

test("short answers are not capped, and the cap is configurable", () => {
  assert.equal(toSpeechText("Short answer."), "Short answer.");
  const speech = toSpeechText("One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten.", { maxChars: 50 });
  assert.ok(speech.length <= 50);
  assert.equal(speech, `One. Two. Three. ${TRUNCATION_NOTICE}`);
});

test("only a bounded prefix is converted; a cut answer always ends with the notice", () => {
  const { MARKDOWN_SCAN_MAX_CHARS } = voice;
  assert.equal(MARKDOWN_SCAN_MAX_CHARS, 20000);

  // Short visible prefix, then a huge code block that runs past the scan limit.
  const markdown = `Short intro.\n\`\`\`\n${"x = 1\n".repeat(5000)}\`\`\`\nHidden ending.`;
  assert.ok(markdown.length > MARKDOWN_SCAN_MAX_CHARS);
  const speech = toSpeechText(markdown);
  assert.equal(speech, `Short intro. ${CODE_PLACEHOLDER} ${TRUNCATION_NOTICE}`);
  assert.ok(!speech.includes("Hidden ending"));

  // Converted text that fits but leaves no room for the notice is still cut.
  const tight = toSpeechText(`${"Fits. ".repeat(8)}\n\`\`\`\n${"y\n".repeat(11000)}`, { maxChars: 50 });
  assert.ok(tight.length <= 50, tight);
  assert.ok(tight.endsWith(TRUNCATION_NOTICE), tight);
});

test("pathological input is processed quickly (no catastrophic backtracking)", () => {
  const inputs = [
    "[".repeat(50_000),
    "*".repeat(50_000),
    "_a".repeat(25_000),
    "`".repeat(50_000),
    "|".repeat(50_000),
    "<".repeat(50_000),
    "**a ".repeat(12_000),
    "```\n".repeat(10_000),
    `[x](${"(".repeat(20_000)}`,
    "1. ".repeat(20_000),
  ];
  for (const input of inputs) {
    const started = performance.now();
    const speech = toSpeechText(input);
    const elapsed = performance.now() - started;
    assert.ok(elapsed < 500, `${JSON.stringify(input.slice(0, 12))}… took ${elapsed.toFixed(0)} ms`);
    assert.ok(speech.length <= SPEECH_MAX_CHARS);
  }
});

// ---------------------------------------------------------------------------
// splitIntoChunks: sentence-sized pieces
// ---------------------------------------------------------------------------

test("chunks pack whole sentences up to the limit", () => {
  const text = "First step. Second step! Third step? Fourth.";
  assert.deepEqual(splitIntoChunks(text, 30), ["First step. Second step!", "Third step? Fourth."]);
  assert.deepEqual(splitIntoChunks(text), [text]);
  assert.equal(CHUNK_MAX_CHARS, 200);
});

test("a long sentence is split at words and a giant word is hard-split", () => {
  const sentence = Array.from({ length: 80 }, (_, i) => `w${i}`).join(" ") + ".";
  const chunks = splitIntoChunks(sentence, 40);
  assert.ok(chunks.every((chunk) => chunk.length <= 40));
  assert.equal(chunks.join(" "), sentence);

  const giant = "x".repeat(95);
  assert.deepEqual(splitIntoChunks(`Go ${giant} now.`, 40), ["Go", "x".repeat(40), "x".repeat(40), "x".repeat(15) + " now."]);
});

test("every chunk of a capped answer respects the limit and nothing is lost", () => {
  const speech = toSpeechText(Array.from({ length: 50 }, (_, i) => `Step ${i} is done, then continue.`).join(" "));
  const chunks = splitIntoChunks(speech);
  assert.ok(chunks.length > 1);
  assert.ok(chunks.every((chunk) => chunk.length > 0 && chunk.length <= CHUNK_MAX_CHARS));
  assert.equal(chunks.join(" "), speech);
});

test("empty or non-text input gives no chunks", () => {
  assert.deepEqual(splitIntoChunks(""), []);
  assert.deepEqual(splitIntoChunks("   "), []);
  assert.deepEqual(splitIntoChunks(null), []);
});

// ---------------------------------------------------------------------------
// pickLocalVoice: on-device voices only
// ---------------------------------------------------------------------------

test("network voices are never chosen", () => {
  assert.equal(pickLocalVoice([NETWORK_EN], "en-US"), null);
  assert.equal(pickLocalVoice([NETWORK_EN, LOCAL_EN], "en-US"), LOCAL_EN);
  assert.equal(pickLocalVoice([{ ...LOCAL_EN, localService: undefined }], "en-US"), null);
});

test("an exact language match wins, preferring the default voice", () => {
  assert.equal(pickLocalVoice([LOCAL_EN_GB, LOCAL_EN, LOCAL_EN_DEFAULT], "en-US"), LOCAL_EN_DEFAULT);
  assert.equal(pickLocalVoice([LOCAL_EN_GB, LOCAL_EN], "en_us"), LOCAL_EN);
});

test("the same primary language is used when no exact match exists", () => {
  assert.equal(pickLocalVoice([LOCAL_FR, LOCAL_EN_GB], "en-US"), LOCAL_EN_GB);
  assert.equal(pickLocalVoice([LOCAL_FR], "en-US"), null); // never a voice for another language
  assert.equal(pickLocalVoice([LOCAL_EN], ""), LOCAL_EN); // defaults to en-US
});

test("empty or invalid voice lists give no voice", () => {
  assert.equal(pickLocalVoice([], "en-US"), null);
  assert.equal(pickLocalVoice(null, "en-US"), null);
  assert.equal(pickLocalVoice(undefined, "en-US"), null);
});

// ---------------------------------------------------------------------------
// createSpeaker
// ---------------------------------------------------------------------------

test("an unsupported browser never throws and reports unavailable", async () => {
  for (const options of [{}, { synth: new FakeSynth() }, { Utterance: FakeUtterance }, { synth: {}, Utterance: FakeUtterance }]) {
    const speaker = createSpeaker(options);
    assert.equal(speaker.supported, false);
    assert.equal(speaker.state, STATES.UNAVAILABLE);
    assert.deepEqual(await speaker.speak("Hello."), { spoken: false, reason: "unsupported" });
    speaker.cancel(); // safe
  }
});

test("an answer is spoken in sentence chunks with the on-device voice", async () => {
  const { speaker, synth, states } = speakerWith({ synth: new FakeSynth([NETWORK_EN, LOCAL_EN]) });
  const answer = Array.from({ length: 12 }, (_, i) => `Step ${i}: open the settings panel and check it.`).join(" ");

  const result = speaker.speak(`## VPN\n${answer}`);
  await tick();

  assert.equal(speaker.state, STATES.SPEAKING);
  assert.ok(synth.spoken.length > 1);
  assert.ok(synth.spoken.every((u) => u.voice === LOCAL_EN && u.lang === "en-US"));
  assert.ok(synth.spoken.every((u) => u.text.length <= CHUNK_MAX_CHARS));
  assert.equal(synth.spoken.map((u) => u.text).join(" "), toSpeechText(`## VPN\n${answer}`));

  synth.finishAll();
  assert.deepEqual(await result, { spoken: true });
  assert.equal(speaker.state, STATES.IDLE);
  assert.deepEqual(states, ["loading", "speaking", "idle"]);
});

test("only network voices: nothing is spoken", async () => {
  const { speaker, synth, states } = speakerWith({ synth: new FakeSynth([NETWORK_EN]) });

  assert.deepEqual(await speaker.speak("Hello."), { spoken: false, reason: "no_local_voice" });
  assert.equal(synth.spoken.length, 0);
  assert.equal(speaker.state, STATES.UNAVAILABLE);
  assert.deepEqual(states, ["loading", "unavailable:no_local_voice"]);
});

test("voices that load later are waited for (voiceschanged)", async () => {
  const synth = new FakeSynth([]);
  const { speaker, timers } = speakerWith({ synth });

  const result = speaker.speak("Hello there.");
  await tick();
  assert.equal(synth.spoken.length, 0);
  assert.equal(synth.listenerCount("voiceschanged"), 1);
  assert.equal(timers.count(), 1);

  synth.loadVoices([LOCAL_EN]);
  await tick();
  assert.equal(synth.listenerCount("voiceschanged"), 0); // listener removed
  assert.equal(timers.count(), 0); // timer cleared
  assert.equal(synth.spoken.length, 1);

  synth.finishAll();
  assert.deepEqual(await result, { spoken: true });
});

test("an empty voiceschanged is ignored until real voices arrive", async () => {
  const synth = new FakeSynth([]);
  const { speaker } = speakerWith({ synth });

  const result = speaker.speak("Hello.");
  await tick();
  synth.loadVoices([]); // still empty: keep waiting
  await tick();
  assert.equal(synth.listenerCount("voiceschanged"), 1);

  synth.loadVoices([LOCAL_EN]);
  await tick();
  synth.finishAll();
  assert.deepEqual(await result, { spoken: true });
});

test("voices that never load: the wait is bounded, then nothing is spoken", async () => {
  const synth = new FakeSynth([]);
  const { speaker, timers } = speakerWith({ synth });

  const result = speaker.speak("Hello.");
  await tick();
  timers.fire(VOICE_LOAD_TIMEOUT_MS);

  assert.deepEqual(await result, { spoken: false, reason: "no_local_voice" });
  assert.equal(synth.listenerCount("voiceschanged"), 0);
  assert.equal(synth.spoken.length, 0);

  // Not cached as "no voice": a later answer tries again once voices exist.
  synth.voices = [LOCAL_EN];
  const second = speaker.speak("Again.");
  await tick();
  synth.finishAll();
  assert.deepEqual(await second, { spoken: true });
});

test("rapid answers while voices load share one listener and one timer", async () => {
  // Regression: each speak() used to start its own wait, piling up listeners and timers.
  const synth = new FakeSynth([]);
  const { speaker, timers } = speakerWith({ synth });

  const first = speaker.speak("First.");
  await tick();
  const second = speaker.speak("Second.");
  const third = speaker.speak("Third.");
  await tick();

  assert.equal(synth.listenerCount("voiceschanged"), 1);
  assert.equal(timers.count(), 1);
  assert.deepEqual(await first, { spoken: false, reason: "cancelled" });
  assert.deepEqual(await second, { spoken: false, reason: "cancelled" });

  synth.loadVoices([LOCAL_EN]);
  await tick();
  assert.equal(synth.listenerCount("voiceschanged"), 0);
  assert.equal(timers.count(), 0);
  assert.deepEqual(synth.spoken.map((u) => u.text), ["Third."]); // only the latest answer
  synth.finishAll();
  assert.deepEqual(await third, { spoken: true });
});

test("a browser without voiceschanged still gets a bounded wait", async () => {
  const synth = new FakeSynth([]);
  synth.addEventListener = undefined;
  synth.removeEventListener = undefined;
  const { speaker, timers } = speakerWith({ synth });

  const result = speaker.speak("Hello.");
  await tick();
  synth.voices = [LOCAL_EN];
  timers.fire(VOICE_LOAD_TIMEOUT_MS);
  await tick();
  synth.finishAll();
  assert.deepEqual(await result, { spoken: true });
});

test("cancel while waiting for voices: nothing is spoken later", async () => {
  const synth = new FakeSynth([]);
  const { speaker, timers } = speakerWith({ synth });

  const result = speaker.speak("Hello.");
  await tick();
  speaker.cancel();
  assert.deepEqual(await result, { spoken: false, reason: "cancelled" });
  assert.equal(speaker.state, STATES.IDLE);

  synth.loadVoices([LOCAL_EN]); // the load completes after the cancel
  await tick();
  timers.fire(VOICE_LOAD_TIMEOUT_MS);
  await tick();
  assert.equal(synth.spoken.length, 0);
});

test("cancel while speaking stops the queue and ignores stale callbacks", async () => {
  const { speaker, synth, states } = speakerWith();
  const result = speaker.speak("One. Two. Three.".repeat(40));
  await tick();
  const first = synth.spoken[0];
  const queued = synth.queue.length;
  assert.ok(queued > 1);

  speaker.cancel(); // the fake reports "interrupted" to every dropped utterance
  assert.deepEqual(await result, { spoken: false, reason: "cancelled" });
  assert.equal(synth.cancelCount, 2); // one from speak() starting, one from cancel()
  assert.equal(synth.queue.length, 0);

  first.onend({}); // a late end event from the engine
  first.onerror({ error: "interrupted" });
  assert.equal(speaker.state, STATES.IDLE);
  assert.deepEqual(states, ["loading", "speaking", "idle"]); // no extra transitions
});

test("a new answer replaces the one being spoken", async () => {
  const { speaker, synth } = speakerWith();
  const first = speaker.speak("First answer. It is long enough.");
  await tick();
  const firstUtterances = [...synth.spoken];

  const second = speaker.speak("Second answer.");
  assert.deepEqual(await first, { spoken: false, reason: "cancelled" });
  await tick();

  for (const utterance of firstUtterances) utterance.onend({}); // stale
  assert.equal(speaker.state, STATES.SPEAKING);

  assert.equal(synth.queue.at(-1).text, "Second answer.");
  synth.finishAll();
  assert.deepEqual(await second, { spoken: true });
  assert.equal(speaker.state, STATES.IDLE);
});

test("an engine error ends the answer and drops the rest of the queue", async () => {
  const { speaker, synth } = speakerWith();
  const result = speaker.speak("One. Two. Three.".repeat(40));
  await tick();
  assert.ok(synth.queue.length > 2);

  synth.failNext("audio-busy");
  assert.deepEqual(await result, { spoken: false, reason: "error", error: "audio-busy" });
  assert.equal(synth.queue.length, 0);
  assert.equal(speaker.state, STATES.IDLE);
});

test("a throwing speak() is reported, never thrown", async () => {
  const synth = new FakeSynth();
  synth.speakError = new Error("not allowed");
  const { speaker } = speakerWith({ synth });

  assert.deepEqual(await speaker.speak("Hello."), { spoken: false, reason: "error", error: "not allowed" });
  assert.equal(speaker.state, STATES.IDLE);
});

test("empty answers speak nothing and leave the state unchanged", async () => {
  const { speaker, synth, states } = speakerWith();
  assert.deepEqual(await speaker.speak("   "), { spoken: false, reason: "empty" });
  assert.deepEqual(await speaker.speak("<script>x()</script>"), { spoken: false, reason: "empty" });
  assert.equal(synth.spoken.length, 0);
  assert.deepEqual(states, []);
});

test("the spoken text honours the cap", async () => {
  const { speaker, synth } = speakerWith();
  const long = Array.from({ length: 80 }, () => "This sentence is part of a very long answer.").join(" ");
  const result = speaker.speak(long);
  await tick();
  const spoken = synth.spoken.map((u) => u.text).join(" ");
  assert.ok(spoken.length <= SPEECH_MAX_CHARS);
  assert.ok(spoken.endsWith(TRUNCATION_NOTICE));
  synth.finishAll();
  await result;
});

test("a throwing state listener does not break speech", async () => {
  const synth = new FakeSynth();
  const timers = fakeTimers();
  const speaker = createSpeaker({
    synth,
    Utterance: FakeUtterance,
    setTimeout: timers.setTimeout,
    clearTimeout: timers.clearTimeout,
    onStateChange: () => {
      throw new Error("view broke");
    },
  });
  const result = speaker.speak("Hello.");
  await tick();
  synth.finishAll();
  assert.deepEqual(await result, { spoken: true });
});

test("the voice is chosen once and reused", async () => {
  const synth = new FakeSynth([LOCAL_EN]);
  let reads = 0;
  const getVoices = synth.getVoices.bind(synth);
  synth.getVoices = () => {
    reads += 1;
    return getVoices();
  };
  const { speaker } = speakerWith({ synth });

  for (const text of ["One.", "Two."]) {
    const result = speaker.speak(text);
    await tick();
    synth.finishAll();
    await result;
  }
  assert.equal(reads, 1);
});

// ---------------------------------------------------------------------------
// prime(): the synchronous user-gesture unlock (V1c)
// ---------------------------------------------------------------------------

test("prime speaks one empty, silent utterance synchronously", () => {
  const { speaker, synth, states } = speakerWith();

  assert.equal(speaker.prime(), true); // no await: it must run inside the click
  assert.equal(synth.spoken.length, 1);
  const [primer] = synth.spoken;
  assert.equal(primer.text, ""); // no answer text reaches any voice
  assert.equal(primer.volume, 0);
  assert.equal(primer.voice, null);
  assert.equal(primer.onend, null); // no handlers: it cannot change the speaker
  assert.equal(primer.onerror, null);
  assert.equal(speaker.state, STATES.IDLE);
  assert.deepEqual(states, []);
});

test("prime cancels speech in progress and keeps stale callbacks ignored", async () => {
  const { speaker, synth, states } = speakerWith();
  const result = speaker.speak("One. Two. Three.".repeat(30));
  await tick();
  const first = synth.spoken[0];

  speaker.prime();
  assert.deepEqual(await result, { spoken: false, reason: "cancelled" });
  first.onend({}); // late events from the cancelled answer
  first.onerror({ error: "interrupted" });
  assert.equal(speaker.state, STATES.IDLE);
  assert.deepEqual(states, ["loading", "speaking", "idle"]);
});

test("speech after prime works, and the primer is dropped", async () => {
  const { speaker, synth } = speakerWith();
  speaker.prime();

  const result = speaker.speak("Hello there.");
  await tick();
  assert.deepEqual(synth.queue.map((u) => u.text), ["Hello there."]); // primer cancelled by speak()
  synth.finishAll();
  assert.deepEqual(await result, { spoken: true });
});

test("prime is safe when unsupported or when the engine throws", () => {
  assert.equal(createSpeaker({}).prime(), false);

  const synth = new FakeSynth();
  synth.speakError = new Error("not allowed");
  const { speaker } = speakerWith({ synth });
  assert.equal(speaker.prime(), false);
  assert.equal(speaker.state, STATES.IDLE);
});

// ---------------------------------------------------------------------------
// hasLocalVoice(): check before offering speech (V1c)
// ---------------------------------------------------------------------------

test("hasLocalVoice is true only for an on-device voice", async () => {
  assert.equal(await speakerWith({ synth: new FakeSynth([LOCAL_EN]) }).speaker.hasLocalVoice(), true);
  assert.equal(await speakerWith({ synth: new FakeSynth([NETWORK_EN]) }).speaker.hasLocalVoice(), false);
  assert.equal(await speakerWith({ synth: new FakeSynth([LOCAL_FR]) }).speaker.hasLocalVoice(), false);
  assert.equal(await createSpeaker({}).hasLocalVoice(), false);
});

test("hasLocalVoice waits for late voices and shares the wait with speak", async () => {
  const synth = new FakeSynth([]);
  const { speaker, timers } = speakerWith({ synth });

  const check = speaker.hasLocalVoice();
  const spoken = speaker.speak("Hi.");
  await tick();
  assert.equal(synth.listenerCount("voiceschanged"), 1); // one shared wait
  assert.equal(timers.count(), 1);

  synth.loadVoices([LOCAL_EN]);
  assert.equal(await check, true);
  await tick();
  synth.finishAll();
  assert.deepEqual(await spoken, { spoken: true });
});

test("hasLocalVoice resolves false when voices never load", async () => {
  const { speaker, timers, synth } = speakerWith({ synth: new FakeSynth([]) });
  const check = speaker.hasLocalVoice();
  await tick();
  timers.fire(VOICE_LOAD_TIMEOUT_MS);
  assert.equal(await check, false);
  assert.equal(synth.listenerCount("voiceschanged"), 0);
  assert.equal(speaker.state, STATES.IDLE); // checking never changes the state
});

// ---------------------------------------------------------------------------
// Module boundaries
// ---------------------------------------------------------------------------

test("the module has no DOM, network or storage access", () => {
  const code = VOICE_SOURCE.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "");
  for (const forbidden of [
    "document",
    "window",
    "fetch(",
    "XMLHttpRequest",
    "WebSocket",
    "sendBeacon",
    "localStorage",
    "sessionStorage",
    "indexedDB",
    "innerHTML",
    "getUserMedia",
    "MediaRecorder",
    "speechSynthesis",
  ]) {
    assert.ok(!code.includes(forbidden), `voice.js must not use ${forbidden}`);
  }
});

test("no regex lookbehind: it would stop the script loading in Safari before 16.4", () => {
  // Regression: splitIntoChunks used split(/(?<=[.!?])\s+/), a parse-time
  // SyntaxError there; chat-stream.js already ships without lookbehind.
  assert.ok(!/\(\?<[=!]/.test(VOICE_SOURCE), "voice.js must not use lookbehind");
});

test("the browser global is BluePeakVoice and the API is complete", () => {
  assert.match(VOICE_SOURCE, /root\.BluePeakVoice = api;/);
  for (const name of ["toSpeechText", "splitIntoChunks", "pickLocalVoice", "createSpeaker"]) {
    assert.equal(typeof voice[name], "function", name);
  }
  const speaker = createSpeaker({});
  for (const name of ["speak", "cancel", "prime", "hasLocalVoice"]) {
    assert.equal(typeof speaker[name], "function", name);
  }
  assert.ok(Object.isFrozen(STATES));
});
