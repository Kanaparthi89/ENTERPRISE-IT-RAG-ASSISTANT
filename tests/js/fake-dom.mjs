// A minimal DOM for running static/js/app.js under node:test (no dependencies).
// It implements only what app.js uses; selectors support tag, #id, .class,
// their combinations, and the descendant combinator.

class FakeText {
  constructor(data) {
    this.data = String(data);
    this.parentElement = null;
  }

  get textContent() {
    return this.data;
  }

  appendData(text) {
    this.data += text;
  }

  remove() {
    this.parentElement?._detach(this);
  }
}

function matchesSimple(element, part) {
  const match = /^([a-z]+)?((?:[#.][\w-]+)*)$/i.exec(part);
  if (!match) throw new Error(`unsupported selector ${part}`);
  const [, tag, rest] = match;
  if (tag && element.tagName !== tag.toUpperCase()) return false;
  for (const token of rest.match(/[#.][\w-]+/g) || []) {
    if (token[0] === "#" && element.id !== token.slice(1)) return false;
    if (token[0] === "." && !element.classList.contains(token.slice(1))) return false;
  }
  return true;
}

function matches(element, selector, root) {
  const parts = selector.trim().split(/\s+/);
  if (!matchesSimple(element, parts.at(-1))) return false;
  let index = parts.length - 2;
  let node = element.parentElement;
  while (index >= 0 && node && node !== root) {
    if (matchesSimple(node, parts[index])) index -= 1;
    node = node.parentElement;
  }
  return index < 0;
}

export class FakeElement {
  constructor(tagName, document) {
    this.tagName = tagName.toUpperCase();
    this.ownerDocument = document;
    this.children = [];
    this.parentElement = null;
    this.attributes = new Map();
    this.listeners = new Map();
    this.style = {};
    this.dataset = {};
    this.hidden = false;
    this.disabled = false;
    this.value = "";
    this.id = "";
    this.className = "";
    this.type = "";
    this.scrollTop = 0;
    this.scrollHeight = 0;
    this.clientHeight = 0;
    const element = this;
    this.classList = {
      contains: (name) => element.className.split(/\s+/).includes(name),
      add: (...names) => {
        const set = new Set(element.className.split(/\s+/).filter(Boolean));
        names.forEach((name) => set.add(name));
        element.className = [...set].join(" ");
      },
      remove: (...names) => {
        element.className = element.className
          .split(/\s+/)
          .filter((name) => name && !names.includes(name))
          .join(" ");
      },
    };
  }

  get textContent() {
    return this.children.map((child) => child.textContent).join("");
  }

  set textContent(value) {
    this.replaceChildren(new FakeText(value ?? ""));
  }

  set innerHTML(value) {
    this.textContent = value;
  }

  _adopt(node) {
    const child = typeof node === "string" ? new FakeText(node) : node;
    child.parentElement?._detach(child);
    child.parentElement = this;
    return child;
  }

  _detach(child) {
    const index = this.children.indexOf(child);
    if (index >= 0) this.children.splice(index, 1);
    child.parentElement = null;
    // Like a browser: removing the focused element moves focus to the body.
    const document = this.ownerDocument;
    if (child instanceof FakeElement && document.activeElement && child.contains(document.activeElement)) {
      document.activeElement = document.body;
    }
  }

  contains(node) {
    for (let current = node; current; current = current.parentElement) {
      if (current === this) return true;
    }
    return false;
  }

  append(...nodes) {
    for (const node of nodes) this.children.push(this._adopt(node));
  }

  appendChild(node) {
    this.append(node);
    return node;
  }

  insertBefore(node, reference) {
    const child = this._adopt(node);
    const index = reference ? this.children.indexOf(reference) : -1;
    if (index < 0) this.children.push(child);
    else this.children.splice(index, 0, child);
    return node;
  }

  after(...nodes) {
    const parent = this.parentElement;
    let index = parent.children.indexOf(this) + 1;
    for (const node of nodes) {
      const child = parent._adopt(node);
      parent.children.splice(index, 0, child);
      index += 1;
    }
  }

  remove() {
    this.parentElement?._detach(this);
  }

  replaceChildren(...nodes) {
    for (const child of [...this.children]) this._detach(child);
    this.append(...nodes);
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
    if (name === "id") this.id = String(value);
    if (name === "class") this.className = String(value);
  }

  getAttribute(name) {
    return this.attributes.has(name) ? this.attributes.get(name) : null;
  }

  hasAttribute(name) {
    return this.attributes.has(name);
  }

  removeAttribute(name) {
    this.attributes.delete(name);
  }

  addEventListener(type, listener) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(listener);
  }

  dispatch(type, event = {}) {
    const full = { target: this, preventDefault() {}, ...event };
    for (const listener of this.listeners.get(type) || []) listener(full);
  }

  click() {
    this.dispatch("click");
  }

  focus() {
    this.ownerDocument.activeElement = this;
  }

  *descendants() {
    for (const child of this.children) {
      if (child instanceof FakeElement) {
        yield child;
        yield* child.descendants();
      }
    }
  }

  querySelectorAll(selector) {
    return [...this.descendants()].filter((element) => matches(element, selector, this));
  }

  querySelector(selector) {
    return this.querySelectorAll(selector)[0] || null;
  }

  get isConnected() {
    return this.ownerDocument.documentElement.contains(this);
  }
}

export function createDocument() {
  const document = {
    activeElement: null,
    listeners: new Map(),
    createElement: (tag) => new FakeElement(tag, document),
    createTextNode: (text) => new FakeText(text),
    querySelector: (selector) => document.documentElement.querySelector(selector),
    querySelectorAll: (selector) => document.documentElement.querySelectorAll(selector),
    addEventListener(type, listener) {
      if (!document.listeners.has(type)) document.listeners.set(type, []);
      document.listeners.get(type).push(listener);
    },
  };
  document.documentElement = new FakeElement("html", document);
  document.body = new FakeElement("body", document);
  document.documentElement.append(document.body);
  return document;
}

/** Build an element tree: h(doc, "tag#id.class", [children]). */
export function h(document, spec, children = []) {
  const [, tag = "div", id, classes] = /^([a-z]+)?(?:#([\w-]+))?((?:\.[\w-]+)*)$/i.exec(spec);
  const element = document.createElement(tag);
  if (id) element.id = id;
  if (classes) element.className = classes.split(".").filter(Boolean).join(" ");
  element.append(...children);
  return element;
}
