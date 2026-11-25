"""Browser security headers on every HTTP response (Security S2b, finding D3-2).

``install_security_headers(app, settings)`` wraps the application's whole
middleware stack, outside Starlette's ServerErrorMiddleware, so the headers
are also on unhandled-exception 500 responses. It is pure ASGI: it edits only
the ``http.response.start`` message and passes every body message through
unchanged and at once (streaming stays incremental; nothing is buffered except
the three small development API docs pages, see below). It never touches
Cache-Control and sets a header only when the response did not set it itself.

Every response gets X-Content-Type-Options, X-Frame-Options, Referrer-Policy,
Permissions-Policy, Cross-Origin-Opener-Policy and a Content-Security-Policy:

* HTML pages: DOCUMENT_CSP. Scripts, styles, images and connections only from
  this origin (marked and DOMPurify are vendored, S2a); the Google Fonts
  stylesheet and font files are the only third-party sources. No
  'unsafe-inline', no 'unsafe-eval'.
* Everything else (JSON, static files, errors): NON_DOCUMENT_CSP, which allows
  nothing at all.
* Development API docs (/docs, /redoc, /docs/oauth2-redirect), only when
  API_DOCS_ENABLED=true: an explicit per-page policy listing the exact URLs
  FastAPI's generated pages load, with their inline scripts allowed by SHA-256
  hash. ReDoc needs style-src 'unsafe-inline' (its runtime injects styles);
  that exception is limited to the /redoc page in development.

Permissions-Policy keeps the microphone available to this origin only (voice
dictation); speech synthesis needs no permission. HSTS is deliberately absent
until the deployment's HTTPS behaviour is verified at the edge.
"""

import base64
import hashlib
import re

DOCUMENT_CSP = "; ".join((
    "default-src 'none'",
    "script-src 'self'",
    "style-src 'self' https://fonts.googleapis.com",
    "font-src https://fonts.gstatic.com",
    "img-src 'self'",
    "connect-src 'self'",
    "worker-src 'self'",
    "manifest-src 'none'",
    "media-src 'none'",
    "object-src 'none'",
    "frame-src 'none'",
    "base-uri 'none'",
    "form-action 'self'",
    "frame-ancestors 'none'",
))
NON_DOCUMENT_CSP = "default-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"

PERMISSIONS_POLICY = "microphone=(self), camera=(), geolocation=(), payment=(), usb=()"

COMMON_HEADERS = (
    ("x-content-type-options", "nosniff"),
    ("x-frame-options", "DENY"),
    ("referrer-policy", "no-referrer"),
    ("permissions-policy", PERMISSIONS_POLICY),
    ("cross-origin-opener-policy", "same-origin"),
)

# The exact resources FastAPI's generated docs pages load (checked by the tests).
SWAGGER_JS = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"
SWAGGER_CSS = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css"
REDOC_JS = "https://cdn.jsdelivr.net/npm/redoc@2/bundles/redoc.standalone.js"
REDOC_FONTS_CSS = "https://fonts.googleapis.com/css"
FASTAPI_FAVICON = "https://fastapi.tiangolo.com/img/favicon.png"

_INLINE_SCRIPT = re.compile(rb"<script>(.*?)</script>", re.DOTALL)
# How many inline scripts FastAPI's generator puts on each docs page. A page with
# any other number gets no script hashes at all (fail closed): an unexpected
# inline script is never authorized just because it appears in the served page.
EXPECTED_INLINE_SCRIPTS = {"swagger": 1, "redoc": 0, "oauth2-redirect": 1}


def _sha256_sources(body: bytes, expected: int) -> list:
    scripts = _INLINE_SCRIPT.findall(body)
    if len(scripts) != expected:
        return []
    return [f"'sha256-{base64.b64encode(hashlib.sha256(script).digest()).decode()}'" for script in scripts]


def docs_csp(kind: str, body: bytes) -> str:
    """The policy for one development docs page; its expected inline scripts are allowed only by hash."""
    if kind not in EXPECTED_INLINE_SCRIPTS:
        raise ValueError(kind)
    hashes = " ".join(_sha256_sources(body, EXPECTED_INLINE_SCRIPTS[kind]))
    tail = "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
    if kind == "swagger":
        return (f"default-src 'none'; script-src {SWAGGER_JS} {hashes}; style-src {SWAGGER_CSS}; "
                f"img-src 'self' data: {FASTAPI_FAVICON}; connect-src 'self'; form-action 'self'; {tail}")
    if kind == "redoc":
        return (f"default-src 'none'; script-src {REDOC_JS}; style-src 'unsafe-inline' {REDOC_FONTS_CSS}; "
                f"font-src https://fonts.gstatic.com; img-src 'self' data: {FASTAPI_FAVICON}; "
                f"connect-src 'self'; worker-src blob:; form-action 'none'; {tail}")
    # kind == "oauth2-redirect"; with no valid hash, script-src is 'none'.
    script_src = hashes or "'none'"
    return f"default-src 'none'; script-src {script_src}; form-action 'none'; {tail}"


def _with_headers(headers, csp: str) -> list:
    present = {name.lower() for name, _ in headers}
    added = [(name.encode(), value.encode()) for name, value in COMMON_HEADERS if name.encode() not in present]
    if b"content-security-policy" not in present:
        added.append((b"content-security-policy", csp.encode()))
    return list(headers) + added


def _is_html(headers) -> bool:
    for name, value in headers:
        if name.lower() == b"content-type":
            return value.lower().startswith(b"text/html")
    return False


class SecurityHeadersMiddleware:
    """Adds the security headers to every HTTP response; other ASGI scopes pass through untouched."""

    def __init__(self, app, *, docs_pages: dict | None = None):
        self.app = app
        self.docs_pages = dict(docs_pages or {})  # path -> "swagger" | "redoc" | "oauth2-redirect"

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        kind = self.docs_pages.get(scope.get("path", ""))
        if kind is not None:
            await self._docs_page(kind, scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = message.get("headers", [])
                csp = DOCUMENT_CSP if _is_html(headers) else NON_DOCUMENT_CSP
                message = {**message, "headers": _with_headers(headers, csp)}
            await send(message)

        await self.app(scope, receive, send_with_headers)

    async def _docs_page(self, kind, scope, receive, send):
        """A small generated docs page: buffered to hash its inline scripts, then sent at once."""
        start, chunks = None, []

        async def collect(message):
            nonlocal start
            if message["type"] == "http.response.start":
                start = message
                return
            chunks.append(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                headers = start.get("headers", [])
                body = b"".join(chunk.get("body", b"") for chunk in chunks if chunk["type"] == "http.response.body")
                csp = docs_csp(kind, body) if _is_html(headers) else NON_DOCUMENT_CSP
                await send({**start, "headers": _with_headers(headers, csp)})
                await send({"type": "http.response.body", "body": body, "more_body": False})

        await self.app(scope, receive, collect)


def docs_pages_for(settings) -> dict:
    """The docs pages present when API_DOCS_ENABLED=true; none otherwise."""
    if getattr(settings, "api_docs_enabled", False) is not True:
        return {}
    return {"/docs": "swagger", "/redoc": "redoc", "/docs/oauth2-redirect": "oauth2-redirect"}


def install_security_headers(app, settings) -> None:
    """Wrap the app's full middleware stack (outside ServerErrorMiddleware) with SecurityHeadersMiddleware.

    Must run before the app serves its first request: Starlette builds the stack
    once and caches it, so a late install would silently do nothing. It raises
    instead. Installing twice is a no-op.
    """
    if getattr(app, "middleware_stack", None) is not None:
        raise RuntimeError(
            "install_security_headers() must run before the application serves its first request."
        )
    if getattr(app.build_middleware_stack, "security_headers_installed", False):
        return
    build = app.build_middleware_stack
    pages = docs_pages_for(settings)

    def build_with_security_headers():
        return SecurityHeadersMiddleware(build(), docs_pages=pages)

    build_with_security_headers.security_headers_installed = True
    app.build_middleware_stack = build_with_security_headers
