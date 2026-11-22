// Security S2a: the vendored marked and DOMPurify builds work as classic
// <script> files, exposing exactly the globals static/js/app.js uses:
// window.marked.parse() and window.DOMPurify.sanitize().
//
// Each file runs in a fresh vm context shaped like a classic browser script:
// no module, exports, define or require; window and self are the global.
// DOMPurify only attaches sanitize() when a real DOM is present, so here it
// must report isSupported === false; sanitize() itself is checked in a browser.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

const VENDOR = new URL("../../static/vendor/", import.meta.url);
const manifest = JSON.parse(readFileSync(new URL("manifest.json", VENDOR), "utf-8"));
const scriptOf = (name) =>
  manifest.packages.find((pkg) => pkg.package === name).files.find((file) => file.role === "script");
const MARKED = scriptOf("marked");
const PURIFY = scriptOf("dompurify");

function classicScriptGlobal(...entries) {
  const sandbox = {};
  sandbox.window = sandbox;
  sandbox.self = sandbox;
  const context = vm.createContext(sandbox);
  const before = new Set(Object.keys(sandbox));
  for (const entry of entries) {
    const path = new URL(`../../${entry.vendored_path}`, import.meta.url);
    vm.runInContext(readFileSync(path, "utf-8"), context, { filename: entry.vendored_path });
  }
  return { sandbox, added: Object.keys(sandbox).filter((key) => !before.has(key)).sort() };
}

test("the two scripts add exactly window.marked and window.DOMPurify, nothing CommonJS or AMD", () => {
  const { sandbox, added } = classicScriptGlobal(MARKED, PURIFY);
  assert.deepEqual(added, ["DOMPurify", "marked"]);
  for (const name of ["module", "exports", "define", "require"]) {
    assert.equal(name in sandbox, false, name);
  }
  assert.equal(MARKED.browser_global, "marked");
  assert.equal(PURIFY.browser_global, "DOMPurify");
});

test("window.marked.parse renders Markdown synchronously to a string, as app.js expects", () => {
  const { sandbox } = classicScriptGlobal(MARKED);
  assert.equal(typeof sandbox.window.marked, "object");
  assert.equal(typeof sandbox.window.marked.parse, "function");
  const html = sandbox.window.marked.parse("# Title\n\n**bold** and `code`\n\n- one\n- two");
  assert.equal(typeof html, "string");
  assert.match(html, /<h1>Title<\/h1>/);
  assert.match(html, /<strong>bold<\/strong>/);
  assert.match(html, /<code>code<\/code>/);
  assert.match(html, /<li>one<\/li>/);
});

test("marked does not sanitize by itself: raw HTML passes through to DOMPurify", () => {
  // Why app.js must keep DOMPurify.sanitize() around marked.parse().
  const { sandbox } = classicScriptGlobal(MARKED);
  assert.match(sandbox.window.marked.parse('<img src=x onerror="alert(1)">'), /onerror/);
});

test("window.DOMPurify is the recorded version and attaches sanitize only with a real DOM", () => {
  const { sandbox } = classicScriptGlobal(PURIFY);
  const purify = sandbox.window.DOMPurify;
  assert.equal(typeof purify, "function");
  assert.equal(purify.version, "3.4.18");
  assert.equal(purify.isSupported, false); // no document here: sanitize() is verified in a browser
  assert.equal(typeof purify.sanitize, "undefined");
  assert.match(readFileSync(new URL(`../../${PURIFY.vendored_path}`, import.meta.url), "utf-8"), /\.sanitize=function/);
});
