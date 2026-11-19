"""Access codes and signed access cookies (Security S1-3).

Architecture C: the public page is a read-only demo; live AI features will
require an access code (enforcement comes in a later phase; this module only
redeems codes and issues and verifies cookies).

* Codes are random 192-bit tokens (``generate_code``), given to each
  recipient. Only their SHA-256 digests are configured, as
  ``id:digest`` or ``id:digest:YYYY-MM-DD`` (valid through that UTC date).
  SHA-256 without a work factor is appropriate because the codes are high
  entropy; a candidate is compared with every configured digest in constant
  time. Revocation is removing an entry (or letting it expire).
* A redeemed code becomes a cookie ``<payload>.<signature>``: the payload is
  base64url JSON ``{"v", "cid", "cfp", "iat", "exp"}`` and the signature is
  HMAC-SHA256 over ``"v2." + payload`` with the cookie secret. ``cfp``
  fingerprints the code's digest, so a cookie is valid only while that exact
  code is configured and unexpired: removing a code, or replacing it under
  the same id, invalidates its cookies. New cookies are signed with the
  current secret; cookies signed with a listed previous secret stay valid
  until they expire (rotation). With Secure cookies the name is
  ``__Host-bp_access`` (Secure, Path=/, no Domain: sibling subdomains can't
  set it); ``bp_access`` only for local development over plain HTTP.
* ``FailureThrottle`` limits failed redemptions per client and in total.
  The client is the socket peer or one configured platform header (sent
  exactly once); IPv6 addresses are grouped by /64. In production one of
  these must be configured explicitly, or redemption refuses to start: a
  hosting edge can make every visitor share one address.

This module imports nothing from the application, so the settings can use
it to validate themselves. It never logs codes, digests, secrets or cookies.

Operators create codes and secrets with:
``python -m app.services.access new-code <id> [--expires YYYY-MM-DD]`` and
``python -m app.services.access new-secret``.

Client identity on DigitalOcean App Platform (not yet verified; nothing is
configured by default). Before setting ``ACCESS_CLIENT_IP_HEADER`` on a
test deployment, confirm all of the following:

1. The edge sets the header (for example ``do-connecting-ip``) on every
   request, holding the visitor's real address (compare with an outside
   "what is my IP" service).
2. A client that sends that header itself is overwritten, not passed
   through or appended to: send a forged value and check what the app sees.
3. It arrives exactly once (a repeated header is refused here anyway) and
   holds one address, for both IPv4 and IPv6 visitors.
4. Without the header, the socket peer is the edge's address, shared by all
   visitors: ``ACCESS_TRUST_SOCKET_PEER=true`` is wrong on App Platform.

Until one is configured, ``ACCESS_ENABLED=true`` with ``APP_ENV=production``
refuses to start (fails closed); with access disabled nothing changes.
"""

import argparse
import base64
import hashlib
import hmac
import ipaddress
import json
import re
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

COOKIE_NAME = "bp_access"  # local development over plain HTTP only
SECURE_COOKIE_NAME = "__Host-bp_access"
COOKIE_VERSION = 2
FINGERPRINT_PATTERN = re.compile(r"[A-Za-z0-9_-]{22}")
CODE_BYTES = 24  # 192 bits: token_urlsafe(24) is 32 characters
CODE_PATTERN = re.compile(r"[A-Za-z0-9_-]{32,128}")
CODE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
HEADER_NAME_PATTERN = re.compile(r"[a-z0-9-]{1,64}")
COOKIE_PART_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
# Headers a client can append to or forge through most proxies: never trusted.
SPOOFABLE_HEADERS = frozenset({"x-forwarded-for", "forwarded", "x-forwarded-host"})
MIN_SECRET_LENGTH = 43  # token_urlsafe(32); new-secret prints 64 characters
MIN_SECRET_DISTINCT = 16
MAX_COOKIE_LENGTH = 512
MAX_REDEEM_BODY_BYTES = 512
CLOCK_SKEW_SECONDS = 60


class AccessConfigError(ValueError):
    """Access settings that must not be used (raised at startup or per request)."""


@dataclass(frozen=True)
class AccessCode:
    code_id: str
    digest: str
    expires_at: float | None  # Unix time, exclusive; None: no expiry


@dataclass(frozen=True)
class AccessGrant:
    code_id: str
    expires_at: int  # Unix time the cookie expires


@dataclass(frozen=True)
class AccessConfig:
    codes: tuple[AccessCode, ...]
    secret: bytes
    previous_secrets: tuple[bytes, ...]
    max_age_seconds: int
    secure: bool
    client_ip_header: str

    @property
    def cookie_name(self) -> str:
        return cookie_name(self.secure)

    @classmethod
    def from_settings(cls, settings) -> "AccessConfig":
        """Validated access configuration; raises ``AccessConfigError``."""
        secret = _check_secret(settings.access_cookie_secret.get_secret_value(), "ACCESS_COOKIE_SECRET")
        previous = tuple(
            _check_secret(item, "ACCESS_COOKIE_PREVIOUS_SECRETS")
            for item in _split_list(settings.access_cookie_previous_secrets.get_secret_value())
        )
        if secret in previous:
            raise AccessConfigError("ACCESS_COOKIE_PREVIOUS_SECRETS must not repeat the current secret.")
        codes = parse_code_entries(settings.access_code_hashes)
        if not codes:
            raise AccessConfigError("ACCESS_CODE_HASHES must list at least one code.")
        if settings.access_cookie_secure is not True and settings.app_env != "development":
            raise AccessConfigError(
                "ACCESS_COOKIE_SECURE=false is allowed only with APP_ENV=development (localhost)."
            )
        header = check_client_ip_header(settings.access_client_ip_header)
        trust_socket = settings.access_trust_socket_peer is True
        if header and trust_socket:
            raise AccessConfigError(
                "Set either ACCESS_CLIENT_IP_HEADER or ACCESS_TRUST_SOCKET_PEER, not both."
            )
        if settings.app_env == "production" and not (header or trust_socket):
            raise AccessConfigError(
                "In production, access redemption needs a trusted client identity: set "
                "ACCESS_CLIENT_IP_HEADER to a header verified to be set by the platform, or "
                "ACCESS_TRUST_SOCKET_PEER=true if clients connect directly."
            )
        return cls(
            codes=codes,
            secret=secret,
            previous_secrets=previous,
            max_age_seconds=int(settings.access_cookie_max_age_seconds),
            secure=bool(settings.access_cookie_secure),
            client_ip_header=header,
        )


# ---------------------------------------------------------------------------
# Codes
# ---------------------------------------------------------------------------

def generate_code() -> str:
    """A new random access code (192 bits)."""
    return secrets.token_urlsafe(CODE_BYTES)


def hash_code(code: str) -> str:
    return hashlib.sha256(code.encode("ascii")).hexdigest()


def _split_list(raw: str) -> list[str]:
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


def _end_of_utc_day(text: str) -> float:
    try:
        day = date.fromisoformat(text)
    except ValueError:
        raise AccessConfigError("An access code expiry must be a date (YYYY-MM-DD).") from None
    end = datetime(day.year, day.month, day.day, tzinfo=timezone.utc) + timedelta(days=1)
    return end.timestamp()


def parse_code_entries(raw: str) -> tuple[AccessCode, ...]:
    """Parse ``id:digest[:YYYY-MM-DD]`` entries separated by commas (strict)."""
    codes, ids, digests = [], set(), set()
    for entry in _split_list(raw):
        parts = entry.split(":")
        if len(parts) not in (2, 3):
            raise AccessConfigError("Each ACCESS_CODE_HASHES entry must be id:sha256[:YYYY-MM-DD].")
        code_id, digest = parts[0], parts[1]
        if not CODE_ID_PATTERN.fullmatch(code_id):
            raise AccessConfigError("An access code id must be 1-32 characters: a-z, 0-9, '-' or '_'.")
        if not DIGEST_PATTERN.fullmatch(digest):
            raise AccessConfigError("An access code digest must be 64 lowercase hex characters (SHA-256).")
        if code_id in ids or digest in digests:
            raise AccessConfigError("Access code ids and digests must be unique.")
        ids.add(code_id)
        digests.add(digest)
        codes.append(AccessCode(code_id, digest, _end_of_utc_day(parts[2]) if len(parts) == 3 else None))
    return tuple(codes)


def _check_secret(value: str, name: str) -> bytes:
    if len(value) < MIN_SECRET_LENGTH or len(set(value)) < MIN_SECRET_DISTINCT:
        raise AccessConfigError(
            f"{name} must be at least {MIN_SECRET_LENGTH} characters of random text "
            "(use: python -m app.services.access new-secret)."
        )
    return value.encode("utf-8")


def check_client_ip_header(value: str) -> str:
    """The optional trusted client-IP header name, normalised; spoofable headers refused."""
    name = (value or "").strip().lower()
    if not name:
        return ""
    if not HEADER_NAME_PATTERN.fullmatch(name):
        raise AccessConfigError("ACCESS_CLIENT_IP_HEADER must be a header name.")
    if name in SPOOFABLE_HEADERS:
        raise AccessConfigError("ACCESS_CLIENT_IP_HEADER must not be a header clients can forge.")
    return name


def match_code(config: AccessConfig, candidate, now: float) -> AccessCode | None:
    """The configured, unexpired code for ``candidate``, or None."""
    if not isinstance(candidate, str) or not CODE_PATTERN.fullmatch(candidate):
        return None
    digest = hash_code(candidate)
    found = None
    for code in config.codes:  # every entry is compared: no early exit
        if hmac.compare_digest(code.digest, digest):
            found = code
    if found is None or (found.expires_at is not None and now >= found.expires_at):
        return None
    return found


# ---------------------------------------------------------------------------
# Cookies
# ---------------------------------------------------------------------------

def cookie_name(secure: bool) -> str:
    return SECURE_COOKIE_NAME if secure else COOKIE_NAME


def code_fingerprint(code: AccessCode) -> str:
    """Binds a cookie to one configured code (its digest), not just its id."""
    return _b64encode(hashlib.sha256(b"bp-access-code:" + code.digest.encode("ascii")).digest()[:16])


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _signature(secret: bytes, payload: str) -> str:
    message = f"v{COOKIE_VERSION}.".encode("ascii") + payload.encode("ascii")
    return _b64encode(hmac.new(secret, message, hashlib.sha256).digest())


def issue_cookie(config: AccessConfig, code: AccessCode, now: float) -> tuple[str, int]:
    """A signed cookie value for ``code`` and its expiry (Unix time)."""
    expires_at = now + config.max_age_seconds
    if code.expires_at is not None:
        expires_at = min(expires_at, code.expires_at)
    payload = {
        "v": COOKIE_VERSION,
        "cid": code.code_id,
        "cfp": code_fingerprint(code),
        "iat": int(now),
        "exp": int(expires_at),
    }
    encoded = _b64encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("ascii"))
    return f"{encoded}.{_signature(config.secret, encoded)}", int(expires_at)


def verify_cookie(config: AccessConfig, value, now: float) -> AccessGrant | None:
    """The grant a cookie carries, or None if it is malformed, forged, expired or revoked."""
    if not isinstance(value, str) or len(value) > MAX_COOKIE_LENGTH or value.count(".") != 1:
        return None
    encoded, signature = value.split(".")
    if not COOKIE_PART_PATTERN.fullmatch(encoded) or not COOKIE_PART_PATTERN.fullmatch(signature):
        return None
    valid = False
    for secret in (config.secret, *config.previous_secrets):  # all are checked
        if hmac.compare_digest(_signature(secret, encoded), signature):
            valid = True
    if not valid:
        return None
    try:
        payload = json.loads(_b64decode(encoded))
    except ValueError:
        return None
    if not isinstance(payload, dict) or set(payload) != {"v", "cid", "cfp", "iat", "exp"}:
        return None
    code_id, fingerprint = payload["cid"], payload["cfp"]
    issued, expires = payload["iat"], payload["exp"]
    if type(payload["v"]) is not int or payload["v"] != COOKIE_VERSION:
        return None
    if not isinstance(code_id, str) or not CODE_ID_PATTERN.fullmatch(code_id):
        return None
    if not isinstance(fingerprint, str) or not FINGERPRINT_PATTERN.fullmatch(fingerprint):
        return None
    if type(issued) is not int or type(expires) is not int:
        return None
    if issued > now + CLOCK_SKEW_SECONDS or expires <= now:
        return None
    if expires - issued > config.max_age_seconds + CLOCK_SKEW_SECONDS:
        return None  # longer than the current policy allows (for example, after it was shortened)
    code = next((item for item in config.codes if item.code_id == code_id), None)
    if code is None or not hmac.compare_digest(code_fingerprint(code), fingerprint):
        return None  # removed, or replaced under the same id, since the cookie was issued
    if code.expires_at is not None and now >= code.expires_at:
        return None  # expired since the cookie was issued
    return AccessGrant(code_id=code_id, expires_at=expires)


def code_reference(config: AccessConfig, code_id: str) -> str:
    """A short pseudonym for logs; the code id itself is never logged."""
    return hmac.new(config.secret, b"ref:" + code_id.encode("ascii"), hashlib.sha256).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Client identity and failed-attempt throttling
# ---------------------------------------------------------------------------

def _address_key(raw: str) -> str | None:
    """An IP address as a throttling key: IPv4 as is, IPv6 grouped by /64."""
    try:
        address = ipaddress.ip_address(raw.strip())
    except ValueError:
        return None
    if address.version == 6:
        if address.ipv4_mapped is not None:
            return str(address.ipv4_mapped)
        return str(ipaddress.ip_network(f"{address}/64", strict=False))
    return str(address)


def client_key(peer_host: str | None, headers, header_name: str) -> str | None:
    """Who is trying, for throttling only; None when it can't be established.

    With a configured platform header, it must be sent exactly once and hold
    exactly one IP address (repeated headers, lists and junk are refused).
    Without one, the socket peer reported by the ASGI server is used;
    forwarded headers are never read here. IPv6 is grouped by /64.
    """
    if header_name:
        values = headers.getlist(header_name)
        if len(values) != 1:
            return None
        return _address_key(values[0])
    if not peer_host:
        return None
    return _address_key(peer_host) or peer_host  # a non-IP peer (test clients) is used as is


class FailureThrottle:
    """Sliding-window limits on redemption attempts, per client and in total.

    ``attempt()`` checks and reserves an attempt atomically, so concurrent
    attempts can't slip past the limit; a reservation counts as a failure
    unless ``succeeded()`` releases it; a success never clears earlier
    failures. Memory is bounded: at most ``max_failures`` stamps per client,
    ``max_global_failures`` in total and ``max_tracked`` clients. In memory,
    for one process; it resets when the process restarts. When it can't track
    another client it refuses (fails closed).
    """

    def __init__(self, max_failures, window_seconds, max_global_failures, *, max_tracked=10_000,
                 clock=None):
        self.max_failures = max_failures
        self.window_seconds = window_seconds
        self.max_global_failures = max_global_failures
        self.max_tracked = max_tracked
        self._clock = clock or (lambda: time.monotonic())  # looked up per call (a seam for tests)
        self._lock = threading.Lock()
        self._clients: dict[str, deque] = {}
        self._global: deque = deque()

    def _drop_old(self, times: deque, now: float) -> None:
        cutoff = now - self.window_seconds
        while times and times[0] <= cutoff:
            times.popleft()

    def _prune(self, now: float) -> None:
        for key in list(self._clients):
            times = self._clients[key]
            self._drop_old(times, now)
            if not times:
                del self._clients[key]

    def attempt(self, client: str) -> tuple[float | None, float | None]:
        """``(retry_after, None)`` if blocked; otherwise ``(None, stamp)`` with the attempt reserved."""
        with self._lock:
            now = self._clock()
            self._drop_old(self._global, now)
            if len(self._global) >= self.max_global_failures:
                return max(1.0, self._global[0] + self.window_seconds - now), None
            times = self._clients.get(client)
            if times is not None:
                self._drop_old(times, now)
                if len(times) >= self.max_failures:
                    return max(1.0, times[0] + self.window_seconds - now), None
            elif len(self._clients) >= self.max_tracked:
                self._prune(now)
                if len(self._clients) >= self.max_tracked:
                    return float(self.window_seconds), None  # can't track this client: fail closed
            self._global.append(now)
            self._clients.setdefault(client, deque()).append(now)
            return None, now

    def succeeded(self, client: str, stamp: float) -> None:
        """Release only this attempt's reservation; earlier failures stay counted."""
        with self._lock:
            times = self._clients.get(client)
            if times is not None:
                try:
                    times.remove(stamp)
                except ValueError:
                    pass  # already outside the window
                if not times:
                    del self._clients[client]
            try:
                self._global.remove(stamp)
            except ValueError:
                pass  # already outside the window


_throttle: FailureThrottle | None = None
_throttle_lock = threading.Lock()


def get_throttle(settings) -> FailureThrottle:
    """The process-wide throttle, rebuilt if its limits changed."""
    global _throttle
    limits = (
        settings.access_max_failed_attempts,
        settings.access_failure_window_seconds,
        settings.access_max_global_failed_attempts,
    )
    with _throttle_lock:
        if _throttle is None or (
            _throttle.max_failures, _throttle.window_seconds, _throttle.max_global_failures
        ) != limits:
            _throttle = FailureThrottle(*limits)
        return _throttle


def reset_throttle() -> None:
    global _throttle
    with _throttle_lock:
        _throttle = None


def wall_clock() -> float:
    """Unix time for cookie and code expiry (a seam for tests)."""
    return time.time()


# ---------------------------------------------------------------------------
# Operator commands
# ---------------------------------------------------------------------------

def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.services.access")
    commands = parser.add_subparsers(dest="command", required=True)
    new_code = commands.add_parser("new-code", help="create a code for one recipient")
    new_code.add_argument("code_id")
    new_code.add_argument("--expires", help="last valid UTC date, YYYY-MM-DD")
    commands.add_parser("new-secret", help="create a cookie-signing secret")
    args = parser.parse_args(argv)

    if args.command == "new-secret":
        print(secrets.token_urlsafe(48))  # 64 characters
        return 0
    if not CODE_ID_PATTERN.fullmatch(args.code_id):
        parser.error("the id must be 1-32 characters: a-z, 0-9, '-' or '_'")
    if args.expires:
        try:
            _end_of_utc_day(args.expires)
        except AccessConfigError as exc:
            parser.error(str(exc))
    code = generate_code()
    entry = f"{args.code_id}:{hash_code(code)}" + (f":{args.expires}" if args.expires else "")
    print(f"Code (give it to the recipient; it is not stored): {code}")
    print(f"ACCESS_CODE_HASHES entry (add it to the server's environment): {entry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
