"""Verified access for paid AI routes (Security S1-5, Phase A).

Every route that can call a paid provider (chat, streaming, sessions,
transcription) is decided here, before FastAPI reads the request body:

* ``evaluate(request, settings)`` returns an ``AccessDecision``:
  ``granted`` (a verified access cookie), ``unauthenticated`` (HTTP 401
  ``access_required``) or ``unavailable`` (HTTP 503 ``access_unavailable``).
* ``configuration_problem(settings)`` is why paid operations are refused
  outright (read-only mode), or None. Paid operations need, on every
  request: ``ACCESS_ENABLED`` with valid access settings, ``QUOTA_ENABLED``
  with valid quota settings, and a valid ``PUBLIC_ORIGINS``.
* ``origin_allowed(request, origins)`` and ``is_json_content_type(request)``
  are the cross-site (CSRF) checks for cookie-authenticated mutations.

Only a signed access cookie that verifies grants anything. The client
address, ``Host``, ``Origin``, ``Sec-Fetch-Site``, forwarded headers and
``APP_ENV`` never do: there is no development, loopback or proxy exception.
``Origin`` and ``Sec-Fetch-Site`` are only ever used to refuse a request,
and the expected origins come only from ``PUBLIC_ORIGINS``, never from the
request. Nothing here logs or returns a cookie, a code or a code id.
"""

import hashlib
import ipaddress
import re
from dataclasses import dataclass, field

from app.services import access as access_service

GRANTED = "granted"
UNAUTHENTICATED = "unauthenticated"
UNAVAILABLE = "unavailable"

# Reasons (for logs only; responses never say which one applied).
ACCESS_DISABLED = "access_disabled"
ACCESS_INVALID = "access_invalid"
QUOTA_DISABLED = "quota_disabled"
QUOTA_INVALID = "quota_invalid"
ORIGINS_MISSING = "origins_missing"
ORIGINS_INVALID = "origins_invalid"
COOKIE_MISSING = "cookie_missing"
COOKIE_AMBIGUOUS = "cookie_ambiguous"
COOKIE_INVALID = "cookie_invalid"
VERIFIED = "verified"

# Session owner identity (S1-5 Phase B): versioned domain separator.
OWNER_REF_DOMAIN = b"bluepeak-session-owner-v1\x00"
OWNER_REF_PATTERN = re.compile(r"[0-9a-f]{64}")

MAX_PUBLIC_ORIGINS = 10
MAX_ORIGIN_LENGTH = 300
MAX_HOSTNAME_LENGTH = 253
DEFAULT_PORTS = {"http": 80, "https": 443}

# The ASCII serialization of an origin (RFC 6454): scheme "://" host [":" port],
# nothing else (no path, query, fragment, user info or wildcard).
_ORIGIN_PATTERN = re.compile(r"(https?)://(\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9.-]+)(?::([0-9]{1,5}))?")
_LABEL_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
# OWS that HTTP allows around a field value.
_OWS = " \t"


class PublicOriginsError(ValueError):
    """PUBLIC_ORIGINS is malformed; the message never includes the value."""


@dataclass(frozen=True)
class AccessDecision:
    """The access outcome for one request; ``grant`` and ``owner_ref`` are set only when granted.

    ``owner_ref`` (S1-5 Phase B) identifies the verified code for session
    ownership. It is server-side only: never returned to clients or logged
    (excluded from ``repr``).
    """

    outcome: str
    reason: str
    grant: access_service.AccessGrant | None = None
    code_ref: str | None = None
    owner_ref: str | None = field(default=None, repr=False)

    @property
    def granted(self) -> bool:
        return self.outcome == GRANTED and isinstance(self.grant, access_service.AccessGrant)


# ---------------------------------------------------------------------------
# Session owner identity (Phase B)
# ---------------------------------------------------------------------------

def _length_prefixed(value: str) -> bytes:
    data = value.encode("utf-8")
    return len(data).to_bytes(4, "big") + data


def derive_owner_ref(code_id: str, fingerprint: str) -> str:
    """The session owner for one configured code: 64 lowercase hex characters.

    SHA-256 over a versioned domain separator and the length-prefixed code
    id and code fingerprint (so no two different pairs share an encoding).
    The fingerprint is derived from the configured code's digest, so
    replacing a code under the same id changes the owner; redeeming the same
    code again keeps it. The cookie-signing secret is not an input, so
    rotating it does not change any owner. Not a credential: it only names
    an owner that a verified access cookie must match.
    """
    if not isinstance(code_id, str) or not access_service.CODE_ID_PATTERN.fullmatch(code_id):
        raise ValueError("invalid code id")
    if not isinstance(fingerprint, str) or not access_service.FINGERPRINT_PATTERN.fullmatch(fingerprint):
        raise ValueError("invalid code fingerprint")
    return hashlib.sha256(
        OWNER_REF_DOMAIN + _length_prefixed(code_id) + _length_prefixed(fingerprint)
    ).hexdigest()


def is_valid_owner_ref(value) -> bool:
    return isinstance(value, str) and OWNER_REF_PATTERN.fullmatch(value) is not None


# ---------------------------------------------------------------------------
# Origins
# ---------------------------------------------------------------------------

def _valid_host(host: str) -> str | None:
    """The host, lower-cased, if it is a DNS name or an IP literal; else None."""
    if host.startswith("["):
        try:
            return f"[{ipaddress.IPv6Address(host[1:-1]).compressed}]"
        except ValueError:
            return None
    host = host.lower()
    if not host or len(host) > MAX_HOSTNAME_LENGTH:
        return None
    labels = host.split(".")
    if all(label.isdigit() for label in labels):
        try:
            return str(ipaddress.IPv4Address(host))
        except ValueError:
            return None
    if not all(_LABEL_PATTERN.fullmatch(label) for label in labels):
        return None  # empty labels (a trailing dot), underscores, wildcards
    return host


def parse_origin(value) -> tuple[str, str, int] | None:
    """(scheme, host, port) for an exact origin serialization, or None.

    The default port is made explicit, so ``https://a.example`` and
    ``https://a.example:443`` are the same origin. ``null``, paths and
    anything else are None.
    """
    if not isinstance(value, str):
        return None
    value = value.strip(_OWS)
    if len(value) > MAX_ORIGIN_LENGTH:
        return None
    match = _ORIGIN_PATTERN.fullmatch(value)
    if match is None:
        return None
    scheme, raw_host, raw_port = match.groups()
    host = _valid_host(raw_host)
    if host is None:
        return None
    if raw_port is None:
        port = DEFAULT_PORTS[scheme]
    else:
        port = int(raw_port)
        if not 1 <= port <= 65535 or raw_port != str(port):
            return None  # out of range, or leading zeros
    return scheme, host, port


def parse_public_origins(raw, *, production: bool) -> frozenset:
    """The configured origins; raises ``PublicOriginsError``. Empty means none.

    Comma-separated exact origins, at most ``MAX_PUBLIC_ORIGINS``. In
    production every origin must be https.
    """
    if raw is None:
        return frozenset()
    if not isinstance(raw, str):
        raise PublicOriginsError("PUBLIC_ORIGINS must be text.")
    items = [item.strip() for item in raw.split(",")]
    if items == [""]:
        return frozenset()
    if len(items) > MAX_PUBLIC_ORIGINS:
        raise PublicOriginsError(f"PUBLIC_ORIGINS may list at most {MAX_PUBLIC_ORIGINS} origins.")
    origins = set()
    for item in items:
        origin = parse_origin(item)
        if origin is None:
            raise PublicOriginsError(
                "PUBLIC_ORIGINS must list exact origins (scheme://host[:port]), comma-separated, "
                "with no paths or wildcards."
            )
        if production and origin[0] != "https":
            raise PublicOriginsError("In production, every PUBLIC_ORIGINS entry must use https.")
        origins.add(origin)
    return frozenset(origins)


def public_origins(settings) -> frozenset:
    """The configured origins (re-validated per request); raises ``PublicOriginsError``."""
    return parse_public_origins(settings.public_origins, production=settings.app_env == "production")


def _single_header(request, name: str) -> tuple[bool, str | None]:
    """(ok, value): ok is False if the header was sent more than once."""
    values = request.headers.getlist(name)
    if len(values) > 1:
        return False, None
    return True, (values[0] if values else None)


def origin_allowed(request, origins) -> bool:
    """True if the request is a same-origin one by the browser's own evidence.

    ``Sec-Fetch-Site``, if sent, must be ``same-origin``; ``Origin``, if
    sent, must be one of ``origins`` exactly. At least one of the two must
    be sent (every current browser sends at least one on a POST or DELETE),
    each at most once, and they must not disagree.
    """
    ok_site, fetch_site = _single_header(request, "sec-fetch-site")
    ok_origin, origin = _single_header(request, "origin")
    if not (ok_site and ok_origin):
        return False  # ambiguous
    if fetch_site is None and origin is None:
        return False  # no evidence
    if fetch_site is not None and fetch_site.strip(_OWS).lower() != "same-origin":
        return False
    if origin is not None:
        parsed = parse_origin(origin)
        if parsed is None or parsed not in origins:
            return False
    return True


def is_json_content_type(request) -> bool:
    """Exactly one ``Content-Type: application/json``, optionally ``; charset=utf-8``."""
    ok, value = _single_header(request, "content-type")
    if not ok or value is None:
        return False
    media_type, *params = [part.strip(_OWS) for part in value.split(";")]
    if media_type.lower() != "application/json":
        return False
    if not params:
        return True
    if len(params) != 1:
        return False
    name, separator, charset = params[0].partition("=")
    return (
        bool(separator)
        and name.strip(_OWS).lower() == "charset"
        and charset.strip(_OWS).strip('"').lower() == "utf-8"
    )


# ---------------------------------------------------------------------------
# Configuration and the decision
# ---------------------------------------------------------------------------

def configuration_problem(settings) -> str | None:
    """Why paid AI operations are refused (read-only mode), or None if they may run."""
    from app.services import quota as quota_service
    from app.services import quota_runtime

    if settings.access_enabled is not True:
        return ACCESS_DISABLED
    try:
        access_service.AccessConfig.from_settings(settings)
    except access_service.AccessConfigError:
        return ACCESS_INVALID
    if settings.quota_enabled is not True:
        return QUOTA_DISABLED
    try:
        quota_runtime.check_settings(settings)
    except quota_service.QuotaConfigError:
        return QUOTA_INVALID
    try:
        origins = public_origins(settings)
    except PublicOriginsError:
        return ORIGINS_INVALID
    if not origins:
        return ORIGINS_MISSING
    return None


def _cookie_values(request, name: str) -> list[str]:
    """Every value sent for cookie ``name``, across all Cookie headers."""
    values = []
    for header in request.headers.getlist("cookie"):
        for part in header.split(";"):
            key, separator, value = part.strip(_OWS).partition("=")
            if separator and key.strip(_OWS) == name:
                values.append(value.strip(_OWS))
    return values


def evaluate(request, settings) -> AccessDecision:
    """The access decision for one request (see the module docstring)."""
    problem = configuration_problem(settings)
    if problem is not None:
        return AccessDecision(UNAVAILABLE, problem)
    try:
        config = access_service.AccessConfig.from_settings(settings)
    except access_service.AccessConfigError:
        return AccessDecision(UNAVAILABLE, ACCESS_INVALID)
    values = _cookie_values(request, config.cookie_name)
    if not values:
        return AccessDecision(UNAUTHENTICATED, COOKIE_MISSING)
    if len(values) > 1:
        return AccessDecision(UNAUTHENTICATED, COOKIE_AMBIGUOUS)
    grant = access_service.verify_cookie(config, values[0], access_service.wall_clock())
    if grant is None or not access_service.CODE_ID_PATTERN.fullmatch(grant.code_id):
        return AccessDecision(UNAUTHENTICATED, COOKIE_INVALID)
    # The configured code the cookie was verified against (its fingerprint
    # matched the cookie's): the owner is derived from it, never from input.
    code = next((item for item in config.codes if item.code_id == grant.code_id), None)
    if code is None:
        return AccessDecision(UNAUTHENTICATED, COOKIE_INVALID)
    return AccessDecision(
        GRANTED, VERIFIED, grant=grant, code_ref=access_service.code_reference(config, grant.code_id),
        owner_ref=derive_owner_ref(grant.code_id, access_service.code_fingerprint(code)),
    )
