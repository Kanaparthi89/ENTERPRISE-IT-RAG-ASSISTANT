// Security S2 (LOW-1): Markdown answers are sanitized with an explicit allowlist.
//
// The real static/js/app.js runs against the fake DOM. What is verified here:
// 1. the exact DOMPurify configuration app.js passes (allowlist, URI rule);
// 2. fail-closed behaviour: any missing, unsupported or failing library means
//    plain text, never unsanitized HTML;
// 3. allowlist coverage of REAL marked output: for a corpus of malicious
//    Markdown, every dangerous tag, attribute and URL marked emits is outside
//    the allowlist (so DOMPurify's allowlist removes it), and every construct of
//    benign Markdown is inside it (so formatting survives).
// DOMPurify itself needs a real DOM, so its actual removal is verified in a
// browser (see the S2 browser checklist), not here.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

import { createDocument, h } from "./fake-dom.mjs";
import { chat, fakeFetch, jsonResponse } from "./helpers.mjs";

const APP_SOURCE = readFileSync(new URL("../../static/js/app.js", import.meta.url), "utf-8");
const MARKED_SOURCE = readFileSync(new URL("../../static/vendor/marked-18.2.1/marked.umd.js", import.meta.url), "utf-8");
const NOT_FOUND = () => jsonResponse({ detail: "Not Found" }, { status: 404 });

function loadApp(extra = {}) {
  const document = createDocument();
  document.documentElement.setAttribute("lang", "en");
  const form = h(document, "form#chatForm.composer", [
    h(document, "textarea#question"),
    h(document, "button#sendBtn", [h(document, "span", ["Ask agent"])]),
  ]);
  document.body.append(
    h(document, "section#chat"),
    h(document, "div.composer-area", [form]),
    h(document, "span#characterCount"),
    h(document, "div#trace"),
    h(document, "span#traceCount"),
    h(document, "span#runStatus"),
    h(document, "div#runProgressBar"),
    h(document, "span#runStepCount"),
    h(document, "span#sourceUsed"),
    h(document, "p#sourceDescription"),
  );
  const window = {
    document,
    fetch: fakeFetch({
      [chat.SESSIONS_URL]: NOT_FOUND,
      "/api/voice/status": NOT_FOUND,
      "/api/voice/transcription/status": NOT_FOUND,
    }),
    TextDecoder, AbortController, Response, setTimeout, clearTimeout,
    location: { origin: "http://localhost", reload() {} },
    addEventListener() {},
    BluePeakChat: chat,
    ...extra,
  };
  new Function("window", "document", APP_SOURCE)(window, document);
  return { window, document };
}

function realMarked() {
  const sandbox = {};
  sandbox.window = sandbox;
  vm.runInContext(MARKED_SOURCE, vm.createContext(sandbox));
  return sandbox.window.marked;
}

const { window: APP } = loadApp();
const CONFIG = APP.BluePeakMarkdown.sanitizeConfig;
const ALLOWED_TAGS = new Set(CONFIG.ALLOWED_TAGS);
const ALLOWED_ATTR = new Set(CONFIG.ALLOWED_ATTR);

// ---------------------------------------------------------------------------
// 1. The configuration
// ---------------------------------------------------------------------------

test("the allowlist is exactly the Markdown formatting marked produces", () => {
  assert.deepEqual([...ALLOWED_TAGS].sort(), [
    "a", "blockquote", "br", "code", "del", "em", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "li", "ol",
    "p", "pre", "strong", "table", "tbody", "td", "th", "thead", "tr", "ul",
  ]);
  assert.deepEqual([...ALLOWED_ATTR].sort(), ["align", "href", "start", "title"]);
  assert.equal(CONFIG.ALLOW_DATA_ATTR, false);
  assert.equal(CONFIG.ALLOW_ARIA_ATTR, false);
  assert.equal(CONFIG.ALLOW_UNKNOWN_PROTOCOLS, false);
  assert.equal(CONFIG.SANITIZE_DOM, true);
  assert.equal(CONFIG.SANITIZE_NAMED_PROPS, true);
  assert.equal(CONFIG.RETURN_DOM, false);
  assert.equal(CONFIG.RETURN_DOM_FRAGMENT, false);
  assert.ok(Object.isFrozen(CONFIG) && Object.isFrozen(CONFIG.ALLOWED_TAGS) && Object.isFrozen(CONFIG.ALLOWED_ATTR));
});

test("interactive, styling, embedded and spoofing elements and attributes are never allowed", () => {
  for (const tag of ["form", "input", "button", "textarea", "select", "option", "label", "fieldset", "style", "link",
                     "meta", "base", "script", "noscript", "template", "svg", "math", "img", "picture", "source",
                     "video", "audio", "iframe", "frame", "object", "embed", "canvas", "dialog", "details",
                     "summary", "marquee", "div", "span", "font", "center", "abbr", "area", "map"]) {
    assert.equal(ALLOWED_TAGS.has(tag), false, tag);
  }
  for (const attr of ["style", "class", "id", "name", "src", "srcset", "action", "formaction", "method", "target",
                      "form", "xlink:href", "background", "poster", "ping", "download", "onerror", "onclick", "onload"]) {
    assert.equal(ALLOWED_ATTR.has(attr), false, attr);
  }
});

test("links must be http, https or mailto", () => {
  const ok = CONFIG.ALLOWED_URI_REGEXP;
  for (const url of ["https://example.com/a", "http://intranet.example/kb", "HTTPS://EXAMPLE.COM", "mailto:it@example.com"]) {
    assert.equal(ok.test(url), true, url);
  }
  for (const url of ["javascript:alert(1)", "JaVaScRiPt:alert(1)", "data:text/html;base64,PHNjcmlwdD4=", "vbscript:x",
                     "file:///etc/passwd", "//evil.example/x", "/api/access/logout", "#top", "blob:https://x/y", "ftp://x"]) {
    assert.equal(ok.test(url), false, url);
  }
});

// ---------------------------------------------------------------------------
// 2. Fail closed: plain text unless both libraries work
// ---------------------------------------------------------------------------

function renderWith(libs, text = "**bold** <img src=x onerror=alert(1)>") {
  const { window, document } = loadApp(libs);
  const target = h(document, "p.answer-text");
  const mode = window.BluePeakMarkdown.render(target, text);
  return { mode, target, config: window.BluePeakMarkdown.sanitizeConfig };
}

const okMarked = { parse: (text) => `<p>${text}</p>` };
const sanitizeCalls = [];
const okPurify = {
  isSupported: true,
  sanitize: (html, config) => { sanitizeCalls.push({ html, config }); return "<p>SANITIZED</p>"; },
};

test("with working libraries the sanitizer's output (and only it) is inserted, with the explicit config", () => {
  sanitizeCalls.length = 0;
  const { mode, target, config } = renderWith({ marked: okMarked, DOMPurify: okPurify }, "hello");
  assert.equal(mode, "markdown");
  assert.equal(target.textContent, "<p>SANITIZED</p>"); // the fake DOM records innerHTML as text
  assert.equal(sanitizeCalls.length, 1);
  assert.equal(sanitizeCalls[0].html, "<p>hello</p>");
  assert.equal(sanitizeCalls[0].config, config); // this app instance's own frozen allowlist config
  assert.deepEqual(config, CONFIG);
});

for (const [label, libs] of [
  ["no DOMPurify", { marked: okMarked }],
  ["no marked", { DOMPurify: okPurify }],
  ["neither library", {}],
  ["DOMPurify unsupported (no sanitize: old or DOM-less browser)", { marked: okMarked, DOMPurify: { isSupported: false } }],
  ["DOMPurify reports isSupported === false but still exposes sanitize", {
    marked: okMarked, DOMPurify: { isSupported: false, sanitize: () => "<p>UNTRUSTED SANITIZER OUTPUT</p>" } }],
  ["DOMPurify.sanitize is not a function", { marked: okMarked, DOMPurify: { isSupported: true, sanitize: "nope" } }],
  ["marked.parse is not a function", { marked: { parse: null }, DOMPurify: okPurify }],
  ["sanitize throws", { marked: okMarked, DOMPurify: { isSupported: true, sanitize() { throw new Error("boom"); } } }],
  ["marked.parse throws", { marked: { parse() { throw new Error("boom"); } }, DOMPurify: okPurify }],
  ["marked.parse returns a Promise (async mode)", { marked: { parse: async () => "<p>x</p>" }, DOMPurify: okPurify }],
  ["sanitize returns a non-string (DOM node)", { marked: okMarked, DOMPurify: { isSupported: true, sanitize: () => ({}) } }],
]) {
  test(`fails closed to plain text: ${label}`, () => {
    const text = "**bold** <img src=x onerror=alert(1)>";
    const { mode, target } = renderWith(libs, text);
    assert.equal(mode, "text");
    assert.equal(target.textContent, text); // shown literally as text, never parsed as HTML
  });
}

test("app.js renders every answer through the safe renderer", () => {
  assert.equal((APP_SOURCE.match(/renderAnswerMarkdown\(answerText, normalizedAnswer\)/g) || []).length, 1);
  assert.equal((APP_SOURCE.match(/\.innerHTML\s*=/g) || []).length, 1);
  assert.doesNotMatch(APP_SOURCE, /window\.DOMPurify\.sanitize\(/); // no default-profile call left
});

// ---------------------------------------------------------------------------
// 3. Allowlist coverage of real marked output
// ---------------------------------------------------------------------------

function inventory(html) {
  const tags = [], attrs = [], urls = [];
  for (const m of html.matchAll(/<([a-zA-Z][a-zA-Z0-9:-]*)((?:[^>"']|"[^"]*"|'[^']*')*)>/g)) {
    const tag = m[1].toLowerCase();
    tags.push(tag);
    for (const a of m[2].matchAll(/([^\s"'=<>/]+)(?:\s*=\s*("[^"]*"|'[^']*'|[^\s>]+))?/g)) {
      const name = a[1].toLowerCase();
      attrs.push({ tag, name });
      if (["href", "src", "action", "formaction", "xlink:href", "srcset", "poster", "data"].includes(name) && a[2]) {
        urls.push({ tag, name, value: a[2].replace(/^["']|["']$/g, "").replace(/&amp;/g, "&").replace(/&#x?[0-9a-f]+;/gi, "") });
      }
    }
  }
  return { tags, attrs, urls };
}

const MALICIOUS = {
  "a form posting to the app": '<form action="/api/access" method="post"><input name="code"><button>Verify</button></form>',
  "a credential prompt": '<label>Admin key <input type="password" name="k"></label><textarea>x</textarea><select><option>a</option></select>',
  "event handler on an image": "<img src=x onerror=alert(1)>",
  "event handler on allowed tag": '<p onclick="alert(1)" onmouseover="x()">hi</p><a href="https://ok.example" onfocus="x()">ok</a>',
  "javascript: link (Markdown)": "[click](javascript:alert(1))",
  "mixed-case javascript: link": "[click](JaVaScRiPt:alert(1))",
  "data: link": "[d](data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==)",
  "vbscript: link": "<a href=\"vbscript:msgbox(1)\">v</a>",
  "protocol-relative link": '<a href="//evil.example/phish">support</a>',
  "same-origin action link": '<a href="/api/access/logout">log out</a>',
  "SVG with script": "<svg><script>alert(1)</script><a xlink:href=\"javascript:alert(1)\"><text>x</text></a></svg>",
  "MathML": '<math><mi xlink:href="javascript:alert(1)">x</mi></math>',
  "style element (CSS spoofing)": "<style>body{display:none}.answer-text::after{content:'Session expired'}</style>",
  "style attribute overlay": '<p style="position:fixed;inset:0;background:#fff">Session expired, re-enter your code</p>',
  "app class reuse (UI spoofing)": '<p class="modal upload-status success">Upload complete</p>',
  "DOM clobbering": '<a id="DOMPurify" name="marked">x</a><img name="getElementById"><form id="chatForm"></form>',
  "iframe/object/embed": '<iframe src="https://evil.example"></iframe><object data="x.swf"></object><embed src="x">',
  "dialog/details overlays": "<dialog open>Enter admin key</dialog><details open><summary>x</summary>y</details>",
  "Markdown image (tracking pixel)": "![x](https://evil.example/pixel.png)",
  "meta/base/link": '<meta http-equiv="refresh" content="0;url=https://evil.example"><base href="https://evil.example/"><link rel="stylesheet" href="https://evil.example/x.css">',
  "task-list checkbox": "- [x] done\n- [ ] todo",
  "noscript/template": "<noscript><p>x</p></noscript><template><p>x</p></template>",
};

test("every dangerous construct in real marked output falls outside the allowlist", () => {
  const marked = realMarked();
  for (const [label, markdown] of Object.entries(MALICIOUS)) {
    const html = marked.parse(markdown);
    const { tags, attrs, urls } = inventory(html);
    const allowedTags = tags.filter((tag) => ALLOWED_TAGS.has(tag));
    const keptAttrs = attrs.filter(({ tag, name }) => ALLOWED_TAGS.has(tag) && ALLOWED_ATTR.has(name));
    // Nothing interactive, embedded or styling survives the allowlist.
    for (const tag of allowedTags) {
      assert.ok(!["form", "input", "button", "textarea", "select", "style", "svg", "math", "img", "iframe", "object",
                  "embed", "dialog", "details", "meta", "base", "link"].includes(tag), `${label}: ${tag}`);
    }
    for (const { name } of keptAttrs) {
      assert.ok(["href", "title", "start", "align"].includes(name), `${label}: ${name}`);
    }
    // Every dangerous URL in marked's output is rejected by the URI rule (DOMPurify then drops the attribute).
    for (const url of urls) {
      const value = url.value.replace(/[\u0000- ]/g, "");
      if (/^(javascript|data|vbscript|file|blob):|^\/\/|^\//i.test(value)) {
        assert.equal(CONFIG.ALLOWED_URI_REGEXP.test(value), false, `${label}: ${url.value} would be kept`);
      }
    }
    // And the dangerous constructs really were present in marked's output (the corpus is meaningful).
    assert.ok(tags.length > 0, `${label}: marked emitted no markup`);
  }
});

test("the malicious corpus actually contains what it claims (marked passes raw HTML through)", () => {
  const marked = realMarked();
  const html = Object.values(MALICIOUS).map((markdown) => marked.parse(markdown)).join("\n");
  const { tags, attrs, urls } = inventory(html);
  for (const tag of ["form", "input", "button", "img", "svg", "math", "style", "iframe", "dialog", "meta", "base"]) {
    assert.ok(tags.includes(tag), `corpus lacks <${tag}>`);
  }
  for (const name of ["onerror", "onclick", "style", "class", "id", "name", "action"]) {
    assert.ok(attrs.some((attr) => attr.name === name), `corpus lacks ${name}=`);
  }
  assert.ok(urls.some((url) => /^javascript:/i.test(url.value)), "corpus lacks a javascript: URL");
  assert.ok(urls.some((url) => /^data:/i.test(url.value)), "corpus lacks a data: URL");
});

test("benign Markdown formatting is entirely inside the allowlist (nothing useful is lost)", () => {
  const marked = realMarked();
  const markdown = [
    "# Reset your VPN token", "## Steps", "1. Open **Settings**.\n2. Choose *Security*.\n3. Select `Reset`.",
    "3. three\n4. four", "> Note: this signs you out.", "---", "- one\n- two", "~~old step~~",
    "```bash\nsudo vpn reset\n```", "| Step | Action |\n|:-----|-------:|\n| 1 | Open |",
    "See [the IT portal](https://it.example.com/vpn \"IT portal\") or email <mailto:it@example.com>.", "line one  \nline two",
  ].join("\n\n");
  const { tags, attrs, urls } = inventory(marked.parse(markdown));
  for (const tag of tags) assert.ok(ALLOWED_TAGS.has(tag), `benign <${tag}> would be removed`);
  const dropped = attrs.filter(({ name }) => !ALLOWED_ATTR.has(name)).map(({ tag, name }) => `${tag}[${name}]`);
  assert.deepEqual([...new Set(dropped)], ["code[class]"]); // only the language hint (no highlighter) is dropped
  for (const url of urls) assert.ok(CONFIG.ALLOWED_URI_REGEXP.test(url.value), url.value);
  for (const tag of ["h1", "h2", "ol", "ul", "li", "strong", "em", "code", "pre", "blockquote", "hr", "del", "table", "th", "td", "a", "br"]) {
    assert.ok(tags.includes(tag), `benign sample lacks <${tag}>`);
  }
});

// ---------------------------------------------------------------------------
// B1: align and start survive sanitizing, but only with safe values and placement
// ---------------------------------------------------------------------------

test("align and start are URI-safe for DOMPurify (so marked's values are not dropped)", () => {
  assert.deepEqual([...CONFIG.ADD_URI_SAFE_ATTR].sort(), ["align", "start"]);
  assert.ok(Object.isFrozen(CONFIG.ADD_URI_SAFE_ATTR));
  assert.equal(String(CONFIG.ALLOWED_URI_REGEXP), "/^(?:https?:|mailto:)/i"); // links stay strict
  assert.equal(ALLOWED_ATTR.has("align") && ALLOWED_ATTR.has("start"), true);
});

test("the attribute value rules are exactly the approved ones", () => {
  const rules = APP.BluePeakMarkdown.safeAttributeValues;
  assert.deepEqual(Object.keys(rules).sort(), ["align", "start"]);
  assert.deepEqual([...rules.align.tags], ["th", "td"]);
  assert.deepEqual([...rules.start.tags], ["ol"]);
  assert.ok(Object.isFrozen(rules) && Object.isFrozen(rules.align) && Object.isFrozen(rules.start));
});

function fragment(build) {
  const { window, document } = loadApp();
  const root = h(document, "div");
  const nodes = build(document);
  for (const node of nodes) root.appendChild(node);
  window.BluePeakMarkdown.enforceSafeAttributeValues(root);
  return nodes;
}

function withAttr(document, tag, name, value) {
  const node = document.createElement(tag); // h() cannot build tags with digits (h1)
  node.setAttribute(name, value);
  return node;
}

test("an ordered list starting at 3 keeps its start; valid alignments are kept", () => {
  const [ol, th, td, center] = fragment((d) => [
    withAttr(d, "ol", "start", "3"), withAttr(d, "th", "align", "left"),
    withAttr(d, "td", "align", "right"), withAttr(d, "td", "align", "center"),
  ]);
  assert.equal(ol.getAttribute("start"), "3");
  assert.equal(th.getAttribute("align"), "left");
  assert.equal(td.getAttribute("align"), "right");
  assert.equal(center.getAttribute("align"), "center");
});

test("malicious or malformed values are removed", () => {
  const bad = [
    ["ol", "start", "javascript:alert(1)"], ["ol", "start", "3; background:url(x)"], ["ol", "start", "-1"],
    ["ol", "start", "1e3"], ["ol", "start", " 3"], ["ol", "start", "3 "], ["ol", "start", ""], ["ol", "start", "1234567890"],
    ["th", "align", "javascript:alert(1)"], ["td", "align", "LEFT"], ["td", "align", "justify"], ["td", "align", "left\"onclick"],
    ["td", "align", ""], ["th", "align", "left right"],
  ];
  const nodes = fragment((d) => bad.map(([tag, name, value]) => withAttr(d, tag, name, value)));
  nodes.forEach((node, index) => assert.equal(node.hasAttribute(bad[index][1]), false, bad[index].join(" ")));
});

test("align and start are removed from tags they do not belong on", () => {
  const misplaced = [["p", "align", "left"], ["h1", "align", "center"], ["table", "align", "right"], ["a", "align", "left"],
                     ["ul", "start", "3"], ["li", "start", "3"], ["p", "start", "1"]];
  const nodes = fragment((d) => misplaced.map(([tag, name, value]) => withAttr(d, tag, name, value)));
  nodes.forEach((node, index) => assert.equal(node.hasAttribute(misplaced[index][1]), false, misplaced[index].join(" ")));
});

test("the value check never touches links or other attributes", () => {
  const [link, titled] = fragment((d) => {
    const a = withAttr(d, "a", "href", "https://example.com");
    a.setAttribute("title", "Portal");
    return [a, withAttr(d, "p", "title", "x")];
  });
  assert.equal(link.getAttribute("href"), "https://example.com");
  assert.equal(link.getAttribute("title"), "Portal");
  assert.equal(titled.getAttribute("title"), "x");
});

test("the renderer runs the value check right after inserting sanitized HTML", () => {
  const body = APP_SOURCE.slice(APP_SOURCE.indexOf("function renderAnswerMarkdown"));
  const insert = body.indexOf("target.innerHTML = html;");
  const enforce = body.indexOf("enforceSafeAttributeValues(target);");
  const links = body.indexOf('for (const link of target.querySelectorAll("a"))');
  assert.ok(insert > 0 && enforce > insert && links > enforce, "order: insert, enforce values, harden links");
});

test("the value check reaches nested elements at any depth", () => {
  const { window, document } = loadApp();
  const good = withAttr(document, "th", "align", "center");
  const evil = withAttr(document, "td", "align", "javascript:alert(1)");
  const deepP = withAttr(document, "p", "align", "left");
  const root = h(document, "div", [
    h(document, "table", [h(document, "thead", [h(document, "tr", [good])]), h(document, "tbody", [h(document, "tr", [evil])])]),
    h(document, "blockquote", [h(document, "blockquote", [deepP])]),
  ]);
  window.BluePeakMarkdown.enforceSafeAttributeValues(root);
  assert.equal(good.getAttribute("align"), "center");
  assert.equal(evil.hasAttribute("align"), false);
  assert.equal(deepP.hasAttribute("align"), false);
});
