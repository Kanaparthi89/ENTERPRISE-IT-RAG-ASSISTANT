"""Security S1-5 Phase C: admin ingestion authorised before the upload is read.

Real ASGI calls into both FastAPI entry points (``app.main`` and
``app.api.main``), with a counting ``receive()``: refusals must happen
before a single body byte is read, with no temporary file created and no
indexing. Indexing (embeddings + Pinecone) is a fake; nothing here reaches a
paid API.
"""

import asyncio
import gc
import logging
import threading
import time
import weakref

import pytest
from pydantic import SecretStr
from starlette import formparsers

import app.main
from app.api import routes
from app.services import admin_auth

KEY = "admin-security-test-key-0123456789abcdef"  # 40 characters
GUARD = 10.0


def api_main_app():
    import app.api.main as api_main

    return api_main.app


APPS = {"app.main": lambda: app.main.app, "app.api.main": api_main_app}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def admin(monkeypatch, settings):
    monkeypatch.setattr(settings, "admin_api_key", SecretStr(KEY))
    admin_auth.reset_admin_throttle()
    yield settings
    for thread in admin_auth.worker_threads():
        thread.join(GUARD)
    assert not admin_auth.ingest_slot_taken(), "the ingest slot leaked"
    admin_auth.reset_admin_throttle()


class Indexer:
    """Stands in for add_documents; can be held open with ``gate``."""

    def __init__(self):
        self.calls = 0
        self.gate = None
        self.entered = threading.Event()
        self.error = None

    def __call__(self, chunks, ids=None):
        self.calls += 1
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(GUARD), "the test never released the gate"
        if self.error is not None:
            raise self.error
        return [f"id-{index}" for index in range(len(chunks))]


@pytest.fixture
def indexer(monkeypatch):
    fake = Indexer()
    monkeypatch.setattr(routes, "add_documents", fake)
    return fake


@pytest.fixture
def spool(monkeypatch):
    """Weak references to every temporary file Starlette's multipart parser creates.

    Weak, so that this record never keeps a file alive (and open) itself.
    """
    created = []
    real = formparsers.SpooledTemporaryFile

    class Spy(real):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(weakref.ref(self))

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", Spy)
    return created


# ---------------------------------------------------------------------------
# Raw ASGI
# ---------------------------------------------------------------------------

BOUNDARY = "AdminTestBoundary7MA4"


def multipart(content=b"# VPN\n\nInstall the VPN client.\n", filename="vpn.md", parts=1, close=True):
    body = b""
    for index in range(parts):
        name = filename if parts == 1 else f"{index}-{filename}"
        body += (f"--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
                 "Content-Type: application/octet-stream\r\n\r\n").encode() + content + b"\r\n"
    return body + (f"--{BOUNDARY}--\r\n".encode() if close else b"")


class Result:
    def __init__(self):
        self.status = None
        self.headers = {}
        self.body = b""
        self.receive_calls = 0
        self.bytes_received = 0

    def json(self):
        import json

        return json.loads(self.body)

    @property
    def code(self):
        detail = self.json().get("detail")
        return detail.get("code") if isinstance(detail, dict) else detail


def headers_for(key=KEY, body=b"", extra=(), content_type=None, length=True):
    headers = [("host", "testserver"),
               ("content-type", content_type or f"multipart/form-data; boundary={BOUNDARY}")]
    if length is True:
        headers.append(("content-length", str(len(body))))
    elif length is not None:
        headers.append(("content-length", str(length)))
    if key is not None:
        headers.append(("x-admin-key", key))
    return headers + list(extra)


async def asgi(application, headers, body=b"", *, stall_after=None, client=("198.51.100.20", 4000), chunk=64 * 1024):
    """One raw ASGI request. ``stall_after``: chunks sent before the client goes silent."""
    result = Result()
    pieces = [body[i:i + chunk] for i in range(0, len(body), chunk)] or [b""]

    async def receive():
        result.receive_calls += 1
        if stall_after is not None and result.receive_calls > stall_after:
            await asyncio.sleep(3600)
        if pieces:
            piece = pieces.pop(0)
            result.bytes_received += len(piece)
            return {"type": "http.request", "body": piece, "more_body": bool(pieces) or stall_after is not None}
        await asyncio.sleep(3600)

    async def send(message):
        if message["type"] == "http.response.start":
            result.status = message["status"]
            result.headers = {k.decode().lower(): v.decode() for k, v in message["headers"]}
        elif message["type"] == "http.response.body":
            result.body += message.get("body", b"")

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
             "path": "/api/ingest", "raw_path": b"/api/ingest", "query_string": b"", "root_path": "",
             "headers": [(k.encode("latin-1"), v.encode("latin-1")) for k, v in headers],
             "client": client, "server": ("testserver", 80), "state": {}}
    await application(scope, receive, send)  # an unhandled exception would propagate here
    return result


def run(coroutine, timeout=GUARD):
    return asyncio.run(asyncio.wait_for(coroutine, timeout))


def assert_admin_error(result, status, code):
    assert result.status == status, (result.status, result.body[:200])
    assert result.code == code
    assert result.headers.get("cache-control") == "no-store"
    if status == 401:
        assert result.headers["www-authenticate"].startswith("AdminKey")
    if status == 429 or code == "ingest_busy":
        assert int(result.headers["retry-after"]) >= 1


# ===========================================================================
# Without a valid key: nothing of the body is read, parsed or stored
# ===========================================================================

BAD_KEYS = {
    "missing": [],
    "empty": [("x-admin-key", "")],
    "wrong": [("x-admin-key", "wrong-key")],
    "almost": [("x-admin-key", KEY[:-1] + "x")],
    "prefix": [("x-admin-key", KEY[:-1])],
    "duplicated": [("x-admin-key", KEY), ("x-admin-key", KEY)],
    "too-long": [("x-admin-key", KEY + "x" * 300)],
    "non-ascii": [("x-admin-key", "café-" + KEY)],
    "with-space": [("x-admin-key", KEY[:10] + " " + KEY[10:])],
}


@pytest.mark.parametrize("entry", list(APPS))
@pytest.mark.parametrize("case", list(BAD_KEYS))
def test_bad_keys_get_401_before_any_body_is_read(entry, case, indexer, spool):
    body = multipart()
    result = run(asgi(APPS[entry](), headers_for(None, body) + BAD_KEYS[case], body))
    assert_admin_error(result, 401, "admin_auth_required")
    assert result.receive_calls == 0 and spool == [] and indexer.calls == 0


@pytest.mark.parametrize("entry", list(APPS))
@pytest.mark.parametrize("shape", ["25 MiB", "200 files", "10 GiB declared, stalls", "malformed"])
def test_hostile_bodies_without_a_key_are_never_read(entry, shape, indexer, spool):
    if shape == "25 MiB":
        body, extra = multipart(b"a" * (25 * 1024 * 1024)), {}
    elif shape == "200 files":
        body, extra = multipart(b"a" * (2 * 1024 * 1024), parts=200), {}
    elif shape == "malformed":
        body, extra = b"--" + BOUNDARY.encode() + b"\r\ngarbage", {}
    else:
        body, extra = multipart(b"a" * 1024, close=False), {"stall_after": 1}
    headers = headers_for(None, body, length=10 * 2**30 if "GiB" in shape else True)
    started = time.monotonic()
    result = run(asgi(APPS[entry](), headers, body, **extra))
    assert_admin_error(result, 401, "admin_auth_required")
    assert result.receive_calls == 0 and result.bytes_received == 0
    assert spool == [] and indexer.calls == 0
    assert time.monotonic() - started < 2.0


# ===========================================================================
# Key configuration
# ===========================================================================

@pytest.mark.parametrize("configured", ["", "x" * 31, KEY[:31]])
def test_a_missing_or_short_key_disables_ingestion_only(admin, monkeypatch, indexer, spool, caplog, configured):
    monkeypatch.setattr(admin, "admin_api_key", SecretStr(configured))
    body = multipart()
    for presented in (configured or "anything", KEY):
        result = run(asgi(APPS["app.main"](), headers_for(presented, body), body))
        assert_admin_error(result, 503, "ingest_unavailable")
        assert result.receive_calls == 0 and indexer.calls == 0
    if configured:
        assert configured not in caplog.text
        assert "shorter than 32 characters" in caplog.text
    # The rest of the application is unaffected.
    from starlette.testclient import TestClient

    assert TestClient(app.main.app).get("/api/health").status_code == 200


def test_a_32_character_key_is_accepted(admin, monkeypatch, indexer):
    monkeypatch.setattr(admin, "admin_api_key", SecretStr(KEY[:32]))
    body = multipart()
    result = run(asgi(APPS["app.main"](), headers_for(KEY[:32], body), body))
    assert result.status == 200 and indexer.calls == 1


def test_the_key_is_never_logged(indexer, caplog):
    caplog.set_level(logging.DEBUG)
    body = multipart()
    run(asgi(APPS["app.main"](), headers_for("wrong-key-attempt-zzzz", body), body))
    run(asgi(APPS["app.main"](), headers_for(KEY, body), body))
    assert KEY not in caplog.text and "wrong-key-attempt-zzzz" not in caplog.text
    assert "Admin key rejected (invalid)." in caplog.text


# ===========================================================================
# Failed-attempt throttle
# ===========================================================================

def _fail(client=("198.51.100.20", 4000), extra=()):
    body = multipart()
    return run(asgi(APPS["app.main"](), headers_for("wrong", body, extra), body, client=client))


def _good(client=("198.51.100.20", 4000), extra=()):
    body = multipart()
    return run(asgi(APPS["app.main"](), headers_for(KEY, body, extra), body, client=client))


def test_ten_failures_then_429_without_comparing_the_key(admin, indexer, monkeypatch):
    assert [_fail().status for _ in range(10)] == [401] * 10
    compared = []
    real = admin_auth.keys_match
    monkeypatch.setattr(admin_auth, "keys_match", lambda *a: compared.append(1) or real(*a))
    throttled = _good()  # even the right key, while throttled
    assert_admin_error(throttled, 429, "admin_auth_throttled")
    assert throttled.receive_calls == 0 and compared == [] and indexer.calls == 0


def test_the_window_expires(admin, indexer):
    clock = [1000.0]
    throttle = admin_auth.get_admin_throttle(admin)
    throttle._clock = lambda: clock[0]
    for _ in range(10):
        _fail()
    assert _good().status == 429
    clock[0] += admin.admin_failure_window_seconds + 1
    assert _good().status == 200


def test_a_success_does_not_clear_earlier_failures(indexer):
    for _ in range(9):
        _fail()
    assert _good().status == 200
    assert _fail().status == 401  # the 10th failure
    assert _good().status == 429


def test_clients_are_throttled_separately_and_forwarded_headers_are_ignored(indexer):
    for _ in range(10):
        _fail(client=("198.51.100.20", 1))
    assert _good(client=("198.51.100.20", 2)).status == 429
    spoofed = [("x-forwarded-for", "203.0.113.99"), ("forwarded", "for=203.0.113.99"), ("x-real-ip", "203.0.113.99")]
    assert _good(client=("198.51.100.20", 3), extra=spoofed).status == 429  # still the same client
    assert _good(client=("198.51.100.21", 1)).status == 200  # another client


def test_the_global_limit(admin, monkeypatch, indexer):
    monkeypatch.setattr(admin, "admin_max_global_failed_attempts", 5)
    for index in range(5):
        assert _fail(client=(f"198.51.100.{index}", 1)).status == 401
    assert _good(client=("198.51.100.200", 1)).status == 429


def test_a_verified_platform_header_identifies_clients(admin, monkeypatch, indexer):
    monkeypatch.setattr(admin, "access_client_ip_header", "do-connecting-ip")
    for _ in range(10):
        _fail(extra=[("do-connecting-ip", "203.0.113.5")])
    assert _good(extra=[("do-connecting-ip", "203.0.113.5")]).status == 429
    assert _good(extra=[("do-connecting-ip", "203.0.113.6")]).status == 200
    # Without the header (or with it twice), clients share one stricter bucket.
    for _ in range(10):
        _fail()
    assert _good(extra=[("do-connecting-ip", "203.0.113.7"), ("do-connecting-ip", "203.0.113.8")]).status == 429


def test_concurrent_failures_never_exceed_the_limit(indexer):
    async def burst():
        body = multipart()
        application = APPS["app.main"]()
        return await asyncio.gather(*(asyncio.create_task(asgi(application, headers_for("wrong", body), body))
                                      for _ in range(50)))

    statuses = sorted(r.status for r in run(burst()))
    assert statuses.count(401) == 10 and statuses.count(429) == 40


# ===========================================================================
# Authenticated uploads: framing, size, stalls and parsing
# ===========================================================================

@pytest.mark.parametrize("entry", list(APPS))
@pytest.mark.parametrize("content_type", ["application/json", "text/plain", "multipart/form-data",
                                          "application/x-www-form-urlencoded"])
def test_other_content_types_are_415_before_reading(entry, content_type, indexer, spool):
    body = multipart()
    result = run(asgi(APPS[entry](), headers_for(KEY, body, content_type=content_type), body))
    assert_admin_error(result, 415, "unsupported_media_type")
    assert result.receive_calls == 0 and spool == []


@pytest.mark.parametrize("entry", list(APPS))
@pytest.mark.parametrize("length", [None, "abc", "-1"])
def test_a_missing_or_invalid_length_is_411(entry, length, indexer, spool):
    body = multipart()
    result = run(asgi(APPS[entry](), headers_for(KEY, body, length=length), body))
    assert_admin_error(result, 411, "length_required")
    assert result.receive_calls == 0 and spool == []


@pytest.mark.parametrize("entry", list(APPS))
def test_a_declared_length_over_the_limit_is_413_before_reading(entry, indexer, spool):
    body = multipart(b"a" * (11 * 1024 * 1024))
    result = run(asgi(APPS[entry](), headers_for(KEY, body), body))
    assert_admin_error(result, 413, "upload_too_large")
    assert result.receive_calls == 0 and spool == [] and indexer.calls == 0


@pytest.mark.parametrize("entry", list(APPS))
def test_more_bytes_than_declared_stop_at_the_cap_with_413(entry, indexer, spool):
    body = multipart(b"a" * (30 * 1024 * 1024))
    result = run(asgi(APPS[entry](), headers_for(KEY, body, length=1000), body))  # declares far less
    assert_admin_error(result, 413, "upload_too_large")
    limit = routes.MAX_UPLOAD_SIZE + admin_auth.MULTIPART_OVERHEAD_BYTES
    assert limit < result.bytes_received <= limit + 64 * 1024  # stopped within one chunk of the cap
    assert indexer.calls == 0
    del result
    gc.collect()
    # Starlette closes its temp files itself only on its own parse errors; after
    # our 413 they are released with the request (anonymous files: nothing remains).
    assert spool and all(ref() is None for ref in spool)


@pytest.mark.parametrize("entry", list(APPS))
def test_a_stalled_authenticated_upload_gets_408_within_the_idle_timeout(entry, admin, monkeypatch, indexer, spool):
    monkeypatch.setattr(admin, "admin_upload_idle_timeout_seconds", 0.3)
    body = multipart(b"a" * 4096, close=False)
    started = time.monotonic()
    result = run(asgi(APPS[entry](), headers_for(KEY, body, length=len(body) + 1000), body, stall_after=1, chunk=1024))
    elapsed = time.monotonic() - started
    assert_admin_error(result, 408, "upload_timeout")
    assert 0.3 <= elapsed < 0.3 + 1.5 and indexer.calls == 0


@pytest.mark.parametrize("entry", list(APPS))
def test_malformed_multipart_from_an_admin_is_a_controlled_4xx(entry, indexer):
    for body in (b"--" + BOUNDARY.encode() + b"\r\ngarbage", b"no multipart here at all", b""):
        result = run(asgi(APPS[entry](), headers_for(KEY, body), body))
        assert result.status in (400, 422), (body, result.status)
    assert indexer.calls == 0


@pytest.mark.parametrize("entry", list(APPS))
def test_a_valid_admin_upload_is_indexed_as_before(entry, indexer, settings):
    body = multipart()
    result = run(asgi(APPS[entry](), headers_for(KEY, body), body))
    assert result.status == 200
    assert result.json() == {"message": "Document indexed successfully.", "file": "vpn.md",
                             "chunks": result.json()["chunks"], "ids_created": result.json()["chunks"]}
    assert indexer.calls == 1 and result.json()["chunks"] > 0


# ===========================================================================
# One ingestion at a time
# ===========================================================================

def _start_ingest(indexer, application=None):
    """Start an authenticated ingestion held inside the (fake) indexer; returns its task and loop thread."""
    indexer.gate = threading.Event()
    holder = {}

    def runner():
        async def main():
            body = multipart()
            holder["task"] = asyncio.current_task()
            try:
                holder["result"] = await asgi(application or APPS["app.main"](), headers_for(KEY, body), body)
            except asyncio.CancelledError:
                holder["cancelled"] = True

        asyncio.run(main())

    thread = threading.Thread(target=runner)
    thread.start()
    assert indexer.entered.wait(GUARD)
    return holder, thread


@pytest.mark.parametrize("entry", list(APPS))
def test_a_second_ingestion_is_refused_at_once_while_one_runs(entry, indexer, spool):
    holder, thread = _start_ingest(indexer, APPS[entry]())
    try:
        body = multipart()
        started = time.monotonic()
        busy = run(asgi(APPS[entry](), headers_for(KEY, body), body))
        assert_admin_error(busy, 503, "ingest_busy")
        assert time.monotonic() - started < 1.0  # never waits for the slot
        assert busy.receive_calls == 0
    finally:
        indexer.gate.set()
        thread.join(GUARD)
    assert holder["result"].status == 200
    assert run(asgi(APPS[entry](), headers_for(KEY, multipart()), multipart())).status == 200


def test_the_slot_is_claimed_atomically_under_concurrency(indexer):
    indexer.gate = threading.Event()

    async def burst():
        body = multipart()
        application = APPS["app.main"]()
        tasks = [asyncio.create_task(asgi(application, headers_for(KEY, body), body)) for _ in range(20)]
        await asyncio.sleep(0.5)
        indexer.gate.set()
        return await asyncio.gather(*tasks)

    statuses = sorted(r.status for r in run(burst()))
    assert statuses.count(200) == 1 and statuses.count(503) == 19 and indexer.calls == 1


def test_a_cancelled_request_keeps_the_slot_until_its_worker_finishes(indexer):
    holder, thread = _start_ingest(indexer)
    holder["task"].get_loop().call_soon_threadsafe(holder["task"].cancel)  # the client is gone
    thread.join(GUARD)
    assert holder.get("cancelled") is True
    assert admin_auth.ingest_slot_taken()  # the worker is still indexing: still busy
    body = multipart()
    assert run(asgi(APPS["app.main"](), headers_for(KEY, body), body)).code == "ingest_busy"
    indexer.gate.set()
    for worker in admin_auth.worker_threads():
        worker.join(GUARD)
    assert not admin_auth.ingest_slot_taken() and indexer.calls == 1
    assert run(asgi(APPS["app.main"](), headers_for(KEY, body), body)).status == 200


def test_a_client_disconnect_mid_ingestion_keeps_the_slot_until_the_worker_finishes(indexer):
    indexer.gate = threading.Event()

    async def disconnecting_client():
        body = multipart()
        sent = []

        async def receive():
            if not sent:
                sent.append(1)
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}  # the client leaves while indexing runs

        async def send(message):
            pass

        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
                 "scheme": "http", "path": "/api/ingest", "raw_path": b"/api/ingest", "query_string": b"",
                 "root_path": "", "headers": [(k.encode(), v.encode()) for k, v in headers_for(KEY, body)],
                 "client": ("198.51.100.20", 1), "server": ("testserver", 80), "state": {}}
        task = asyncio.create_task(app.main.app(scope, receive, send))
        await asyncio.to_thread(indexer.entered.wait, GUARD)
        assert admin_auth.ingest_slot_taken()
        task.cancel()  # what a server does when the connection is gone
        try:
            await task
        except asyncio.CancelledError:
            pass

    run(disconnecting_client())
    assert admin_auth.ingest_slot_taken()  # the worker thread still owns it
    indexer.gate.set()
    for worker in admin_auth.worker_threads():
        worker.join(GUARD)
    assert not admin_auth.ingest_slot_taken()


def test_failures_inside_and_before_the_worker_release_the_slot(indexer, monkeypatch):
    indexer.error = RuntimeError("pinecone down: pc-secret")
    body = multipart()
    failed = run(asgi(APPS["app.main"](), headers_for(KEY, body), body))
    assert failed.status == 500 and b"pc-secret" not in failed.body
    for worker in admin_auth.worker_threads():
        worker.join(GUARD)
    assert not admin_auth.ingest_slot_taken()
    unsupported = multipart(filename="malware.exe")
    assert run(asgi(APPS["app.main"](), headers_for(KEY, unsupported), unsupported)).status == 400
    assert not admin_auth.ingest_slot_taken()
    indexer.error = None
    assert run(asgi(APPS["app.main"](), headers_for(KEY, body), body)).status == 200


def test_the_handler_refuses_without_the_route_class():
    from types import SimpleNamespace

    class Upload:
        filename = "vpn.md"

        async def read(self, size=-1):
            raise AssertionError("must not be read")

        async def close(self):
            pass

    with pytest.raises(Exception) as caught:
        asyncio.run(routes.ingest(SimpleNamespace(state=SimpleNamespace()), Upload()))
    assert getattr(caught.value, "status_code", None) == 503


# ===========================================================================
# C2-F1: strict Content-Length parsing (raw ASGI: real HTTP parsers would
# reject most of these before the application, so they are sent directly)
# ===========================================================================

INVALID_LENGTHS = {
    "superscript-2": "²",
    "superscripts-13": "¹³",
    "5000-digits": "9" * 5000,
    "21-digits": "1" * 21,
    "empty": "",
    "negative": "-1",
    "plus-signed": "+120",
    "leading-space": " 120",
    "trailing-space": "120 ",
    "inner-space": "1 20",
    "tab": "120\t",
    "decimal-point": "120.0",
    "hex": "0x78",
    "exponent": "1e3",
    "underscore": "1_000",
    "letters": "abc",
}


@pytest.mark.parametrize("entry", list(APPS))
@pytest.mark.parametrize("case", list(INVALID_LENGTHS))
def test_malformed_content_lengths_are_411_without_reading(entry, case, indexer, spool):
    body = multipart()
    result = run(asgi(APPS[entry](), headers_for(KEY, body, length=INVALID_LENGTHS[case]), body))
    assert_admin_error(result, 411, "length_required")  # a controlled response, never an exception
    assert result.receive_calls == 0 and spool == [] and indexer.calls == 0
    assert not admin_auth.ingest_slot_taken()


@pytest.mark.parametrize("entry", list(APPS))
def test_a_20_digit_length_is_parsed_and_refused_as_too_large(entry, indexer, spool):
    body = multipart()
    result = run(asgi(APPS[entry](), headers_for(KEY, body, length="9" * 20), body))
    assert_admin_error(result, 413, "upload_too_large")
    assert result.receive_calls == 0 and spool == []


@pytest.mark.parametrize("entry", list(APPS))
def test_zero_and_normal_decimal_lengths_are_accepted(entry, indexer):
    zero = run(asgi(APPS[entry](), headers_for(KEY, b"", length="0"), b""))
    assert zero.status in (400, 422) and zero.status != 411  # framing accepted; the empty form is invalid
    body = multipart()
    normal = run(asgi(APPS[entry](), headers_for(KEY, body), body))
    padded = run(asgi(APPS[entry](), headers_for(KEY, body, length=f"{len(body):08d}"), body))  # leading zeros
    assert normal.status == padded.status == 200 and indexer.calls == 2


@pytest.mark.parametrize("entry", list(APPS))
@pytest.mark.parametrize("pair", [("120", "120"), ("120", "130"), ("120", "²"), ("120", "")])
def test_duplicate_content_length_headers_are_411(entry, pair, indexer, spool):
    body = multipart()
    result = run(asgi(APPS[entry](), headers_for(KEY, body, length=pair[0], extra=[("content-length", pair[1])]), body))
    assert_admin_error(result, 411, "length_required")
    assert result.receive_calls == 0 and spool == []


@pytest.mark.parametrize("entry", list(APPS))
def test_a_valid_declared_length_is_not_the_only_size_limit(entry, indexer):
    body = multipart(b"a" * (30 * 1024 * 1024))
    result = run(asgi(APPS[entry](), headers_for(KEY, body, length="100"), body))  # valid, but dishonest
    assert_admin_error(result, 413, "upload_too_large")
    limit = routes.MAX_UPLOAD_SIZE + admin_auth.MULTIPART_OVERHEAD_BYTES
    assert limit < result.bytes_received <= limit + 64 * 1024
    assert indexer.calls == 0
