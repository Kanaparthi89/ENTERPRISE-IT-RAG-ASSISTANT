"""Test-only support for the Phase A1 streaming work.

Contents:

* ``ChunkScript`` / ``ScriptedChatModel``: a chat model that emits scripted
  model output *chunks* through LangChain callbacks. A chunk is one
  ``on_llm_new_token`` callback; it may hold zero, one, or several tokenizer
  tokens. Scripts can pause (gate), fail after N chunks, emit list content,
  or generate very large outputs lazily.
* ``parse_sse`` / ``assert_stream_grammar``: a strict Server-Sent Events
  parser and the approved event-order contract.
* ``ASGIHarness``: drives an ASGI app directly (no sockets) with scripted
  ``receive``/``send``, for ASGI spec 2.3 (``http.disconnect`` via receive)
  and 2.4 (``OSError`` from send), per-message send gates, and a
  uvicorn-faithful mode in which sends after a disconnect return silently.
* ``FakeClock``: a manually advanced monotonic clock.
* ``LoopbackServer``: a real uvicorn server on 127.0.0.1 in a thread, forced
  to the standard asyncio loop and the h11 protocol.

This module must not import application modules: it is imported by
``tests/conftest.py`` after isolated settings are installed, and by tests.
"""

import asyncio
import json
import re
import socket
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    generate_from_stream,
)
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

# How long a scripted pause waits before failing the test instead of hanging.
PAUSE_GUARD_SECONDS = 10.0


# -------------------------------------------------------------------
# Scripted chunk-emitting chat model
# -------------------------------------------------------------------

class ChunkScript:
    """Scripted model output, emitted as chunks.

    ``chunks`` lists the chunk texts explicitly. Alternatively, ``count`` with
    ``chunk_text(i)`` generates chunks lazily, so large outputs do not need to
    be materialized up front.

    ``pause_after=k`` blocks the model thread after ``k`` chunks until
    ``resume`` is set; ``paused`` is set while blocked. ``fail_after=k``
    raises ``error`` after ``k`` chunks.

    The script does not emit a trailing empty chunk: langchain-core adds one
    itself (``chunk_position="last"``) for every streaming model. List
    content blocks carry ``"index": 0`` so they merge into one text block, as
    provider content blocks do.
    """

    def __init__(
        self,
        chunks=None,
        *,
        count=None,
        chunk_text=None,
        pause_after=None,
        fail_after=None,
        error=None,
        list_content=False,
    ):
        if (chunks is None) == (count is None):
            raise ValueError("Provide exactly one of 'chunks' or 'count'.")
        self._chunks = list(chunks) if chunks is not None else None
        self.count = len(self._chunks) if chunks is not None else count
        self._chunk_text = chunk_text or (lambda index: f"c{index} ")
        self.pause_after = pause_after
        self.fail_after = fail_after
        self.error = error or RuntimeError("Scripted model failure.")
        self.list_content = list_content

        self.paused = threading.Event()
        self.resume = threading.Event()
        self.closed = False
        self._lock = threading.Lock()
        self._produced = 0

    @classmethod
    def from_text(cls, text, **kwargs):
        """Split text into whitespace and non-whitespace chunks (lossless)."""
        return cls(re.findall(r"\s+|\S+", text), **kwargs)

    @property
    def produced(self) -> int:
        """Number of non-empty chunks emitted so far."""
        with self._lock:
            return self._produced

    def text(self, index: int) -> str:
        if self._chunks is not None:
            return self._chunks[index]
        return self._chunk_text(index)

    def full_text(self) -> str:
        return "".join(self.text(index) for index in range(self.count))

    def _wait_for_resume(self):
        self.paused.set()
        if not self.resume.wait(PAUSE_GUARD_SECONDS):
            raise TimeoutError("ChunkScript pause was never released.")

    def _chunk(self, text: str) -> ChatGenerationChunk:
        content = [{"type": "text", "text": text, "index": 0}] if self.list_content else text
        return ChatGenerationChunk(message=AIMessageChunk(content=content))

    def iter_chunks(self) -> Iterator[ChatGenerationChunk]:
        # langchain-core calls _stream() without a run manager and emits
        # on_llm_new_token(chunk.message.content, chunk=chunk) itself for each
        # chunk, followed by a final empty chunk.
        try:
            for index in range(self.count):
                if index == self.pause_after:
                    self._wait_for_resume()
                if index == self.fail_after:
                    raise self.error
                chunk = self._chunk(self.text(index))
                with self._lock:
                    self._produced += 1
                yield chunk

            if self.fail_after == self.count:
                raise self.error
        finally:
            self.closed = True


class ScriptedChatModel(BaseChatModel):
    """A chat model whose output comes from a ``ChunkScript``.

    Like ``ChatOpenAI``, it streams only when LangChain decides to stream
    (for example, when LangGraph's ``messages`` mode attaches its handler,
    or when constructed with ``streaming=True``).
    """

    script: Any
    streaming: bool = False

    @property
    def _llm_type(self) -> str:
        return "scripted-chunks"

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        yield from self.script.iter_chunks()

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return generate_from_stream(self.script.iter_chunks())


# -------------------------------------------------------------------
# Strict SSE parsing
# -------------------------------------------------------------------

STREAM_EVENTS = ("start", "trace", "delta", "done", "error")
TERMINAL_EVENTS = ("done", "error")
# A2-6: session streams may add one non-terminal ``session`` event after ``done``.
SESSION_STREAM_EVENTS = STREAM_EVENTS + ("session",)


class SSEProtocolError(AssertionError):
    """The byte stream violates the SSE framing or the event contract."""


@dataclass(frozen=True)
class SSEEvent:
    event: str
    data: Any


@dataclass
class ParsedStream:
    events: list[SSEEvent] = field(default_factory=list)
    keepalives: int = 0

    @property
    def names(self) -> list[str]:
        return [event.event for event in self.events]


def parse_sse(raw: bytes, *, allowed=STREAM_EVENTS) -> ParsedStream:
    """Parse an SSE body strictly.

    Each frame must be either comment-only lines (heartbeats) or exactly one
    ``event: <name>`` line followed by one ``data: <json>`` line. Frames end
    with a blank line, the body may not end mid-frame, and CR characters are
    rejected.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SSEProtocolError(f"Body is not valid UTF-8: {exc}") from exc

    if "\r" in text:
        raise SSEProtocolError("CR characters are not allowed.")
    if text and not text.endswith("\n\n"):
        raise SSEProtocolError(f"Incomplete final frame: {text[-80:]!r}")

    parsed = ParsedStream()
    for frame in text.split("\n\n")[:-1] if text else []:
        if not frame:
            raise SSEProtocolError("Empty frame.")
        lines = frame.split("\n")

        if all(line.startswith(":") for line in lines):
            parsed.keepalives += 1
            continue

        if (
            len(lines) != 2
            or not lines[0].startswith("event: ")
            or not lines[1].startswith("data: ")
        ):
            raise SSEProtocolError(f"Malformed frame: {frame!r}")

        name = lines[0][len("event: "):]
        if allowed is not None and name not in allowed:
            raise SSEProtocolError(f"Unknown event name: {name!r}")

        try:
            data = json.loads(lines[1][len("data: "):])
        except json.JSONDecodeError as exc:
            raise SSEProtocolError(f"Invalid JSON in {name!r} event: {exc}") from exc

        parsed.events.append(SSEEvent(name, data))

    return parsed


def assert_stream_grammar(parsed: ParsedStream, *, connected: bool, session: bool = False) -> None:
    """Check the approved order.

    connected:     start ( trace | delta )* ( done | error )
    disconnected:  start? ( trace | delta )*        (no terminal event)

    With ``session=True`` (A2-6) a connected stream that ended with ``done``
    may be followed by exactly one ``session`` event, which is then last.
    Without it, any ``session`` event is a violation.
    """
    names = list(parsed.names)
    if "session" in names:
        if not session:
            raise SSEProtocolError(f"'session' event in a stateless stream: {names}")
        if names.count("session") != 1 or names[-1] != "session" or len(names) < 2 or names[-2] != "done":
            raise SSEProtocolError(f"'session' must appear once, right after 'done', last: {names}")
        names = names[:-1]
    terminals = [name for name in names if name in TERMINAL_EVENTS]

    if names.count("start") > 1 or ("start" in names and names[0] != "start"):
        raise SSEProtocolError(f"'start' must appear once, first: {names}")

    if connected:
        if not names or names[0] != "start":
            raise SSEProtocolError(f"Connected stream must begin with 'start': {names}")
        if len(terminals) != 1 or names[-1] not in TERMINAL_EVENTS:
            raise SSEProtocolError(
                f"Connected stream must end with exactly one terminal event: {names}"
            )
    elif terminals:
        raise SSEProtocolError(f"Disconnected stream must not have a terminal event: {names}")


# -------------------------------------------------------------------
# Fake clock
# -------------------------------------------------------------------

class FakeClock:
    """A monotonic clock that only moves when the test advances it."""

    def __init__(self, start: float = 1_000.0):
        self._now = start
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("A monotonic clock cannot go backwards.")
        with self._lock:
            self._now += seconds


# -------------------------------------------------------------------
# Event-loop proxy for the "closed between check and schedule" race
# -------------------------------------------------------------------

class LoopClosingOnSchedule:
    """Event-loop proxy that becomes unavailable exactly when a wake-up is scheduled.

    Before ``arm()`` it delegates to the real loop. Once armed, the next
    ``call_soon_threadsafe`` marks the proxy closed and raises
    ``RuntimeError("Event loop is closed")``, so a caller's earlier
    ``is_closed()`` check returned False but scheduling fails, which is the
    window between the check and the call. The real loop is never closed.
    """

    def __init__(self, loop):
        self._loop = loop
        self._armed = False
        self.closed = False
        self.attempts = 0

    def arm(self) -> None:
        self._armed = True

    def is_closed(self) -> bool:
        return self.closed or self._loop.is_closed()

    def call_soon_threadsafe(self, callback, *args):
        if self._armed:
            self.attempts += 1
            self.closed = True
            raise RuntimeError("Event loop is closed")
        return self._loop.call_soon_threadsafe(callback, *args)


# -------------------------------------------------------------------
# ASGI harness
# -------------------------------------------------------------------

def is_response_start(message) -> bool:
    return message["type"] == "http.response.start"


def body_contains(fragment: bytes) -> Callable[[dict], bool]:
    def predicate(message) -> bool:
        return message["type"] == "http.response.body" and fragment in message.get("body", b"")

    return predicate


def is_end_of_body(message) -> bool:
    return message["type"] == "http.response.body" and not message.get("more_body", False)


@dataclass
class SendGate:
    """Blocks the first send matching ``predicate`` until ``release`` is set."""

    predicate: Callable[[dict], bool]
    reached: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    used: bool = False


@dataclass
class SentMessage:
    message: dict
    dropped: bool  # True when sent after a disconnect (uvicorn returns silently)


@dataclass
class HarnessResult:
    sent: list[SentMessage]
    exception: BaseException | None

    @property
    def delivered(self) -> list[dict]:
        return [item.message for item in self.sent if not item.dropped]

    @property
    def status(self) -> int | None:
        for message in self.delivered:
            if message["type"] == "http.response.start":
                return message["status"]
        return None

    @property
    def headers(self) -> dict[str, str]:
        for message in self.delivered:
            if message["type"] == "http.response.start":
                return {
                    key.decode("latin-1").lower(): value.decode("latin-1")
                    for key, value in message.get("headers", [])
                }
        return {}

    @property
    def body(self) -> bytes:
        return b"".join(
            message.get("body", b"")
            for message in self.delivered
            if message["type"] == "http.response.body"
        )

    @property
    def complete(self) -> bool:
        return any(is_end_of_body(message) for message in self.delivered)


class ASGIHarness:
    """Run one HTTP request against an ASGI app with scripted transport behaviour.

    * ``spec_version`` is reported in the scope (Starlette changes its
      disconnect handling at 2.4).
    * ``disconnect()`` makes ``receive`` return ``http.disconnect`` and, in
      uvicorn-faithful mode, makes every later ``send`` return silently
      without delivering anything (uvicorn ``send`` after ``connection_lost``).
      A send blocked on a gate is released by a disconnect, as uvicorn's
      ``drain()`` is released by ``connection_lost``.
    * ``gate()``, ``fail_send()`` and ``disconnect_on()`` script individual
      sends; ``wait_for_send()`` synchronizes the test with the app.
    """

    def __init__(self, app, *, spec_version: str = "2.3", uvicorn_faithful: bool = True):
        self.app = app
        self.spec_version = spec_version
        self.uvicorn_faithful = uvicorn_faithful
        self.sent: list[SentMessage] = []
        self.attempted: list[dict] = []
        self.disconnected = False
        self._disconnect_event = asyncio.Event()
        self._attempt_event = asyncio.Event()
        self._gates: list[SendGate] = []
        self._failures: list[tuple[Callable, Callable]] = []
        self._disconnect_triggers: list[Callable] = []
        self._request_body = b""
        self._request_delivered = False

    # -- scripting ---------------------------------------------------

    def gate(self, predicate) -> SendGate:
        gate = SendGate(predicate)
        self._gates.append(gate)
        return gate

    def fail_send(self, predicate, exc_factory=lambda: OSError("send failed")) -> None:
        self._failures.append((predicate, exc_factory))

    def disconnect_on(self, predicate) -> None:
        self._disconnect_triggers.append(predicate)

    def disconnect(self) -> None:
        self.disconnected = True
        self._disconnect_event.set()

    async def wait_for_send(self, predicate) -> dict:
        while True:
            for message in self.attempted:
                if predicate(message):
                    return message
            self._attempt_event.clear()
            await self._attempt_event.wait()

    # -- ASGI callables ----------------------------------------------

    async def _receive(self):
        if not self._request_delivered:
            self._request_delivered = True
            return {"type": "http.request", "body": self._request_body, "more_body": False}
        await self._disconnect_event.wait()
        return {"type": "http.disconnect"}

    async def _send(self, message):
        self.attempted.append(message)
        self._attempt_event.set()

        for gate in self._gates:
            if not gate.used and gate.predicate(message):
                gate.used = True
                gate.reached.set()
                await self._wait_for_release_or_disconnect(gate.release)
                break

        for predicate, exc_factory in self._failures:
            if predicate(message):
                raise exc_factory()

        if self.disconnected:
            if self.uvicorn_faithful:
                self.sent.append(SentMessage(message, dropped=True))
                return
            raise OSError("Client disconnected.")

        self.sent.append(SentMessage(message, dropped=False))

        if any(trigger(message) for trigger in self._disconnect_triggers):
            self.disconnect()

    async def _wait_for_release_or_disconnect(self, release: asyncio.Event) -> None:
        release_task = asyncio.ensure_future(release.wait())
        disconnect_task = asyncio.ensure_future(self._disconnect_event.wait())
        try:
            await asyncio.wait(
                {release_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            for task in (release_task, disconnect_task):
                task.cancel()

    # -- running -----------------------------------------------------

    def _scope(self, method: str, path: str, headers) -> dict:
        raw_headers = [(b"host", b"testserver")]
        raw_headers += [
            (key.lower().encode("latin-1"), value.encode("latin-1"))
            for key, value in (headers or {}).items()
        ]
        return {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": self.spec_version},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("latin-1"),
            "query_string": b"",
            "root_path": "",
            "headers": raw_headers,
            "client": ("127.0.0.1", 50000),
            "server": ("testserver", 80),
            "state": {},
        }

    async def request(self, method: str, path: str, *, json_body=None, headers=None) -> HarnessResult:
        headers = dict(headers or {})
        if json_body is not None:
            self._request_body = json.dumps(json_body).encode("utf-8")
            headers.setdefault("content-type", "application/json")
        headers.setdefault("content-length", str(len(self._request_body)))

        exception = None
        try:
            await self.app(self._scope(method, path, headers), self._receive, self._send)
        except Exception as exc:  # noqa: BLE001 - recorded for assertions
            exception = exc
        return HarnessResult(sent=list(self.sent), exception=exception)


# -------------------------------------------------------------------
# Loopback uvicorn server
# -------------------------------------------------------------------

# H3 runtimes: (event loop, HTTP protocol). "A" is the baseline the suite has
# always used; "P" is what production uvicorn selects ("auto" with
# uvicorn[standard] installed). D1/D2 isolate a difference between A and P.
RUNTIMES = {
    "A": ("asyncio", "h11"),
    "P": ("uvloop", "httptools"),
    "D1": ("asyncio", "httptools"),
    "D2": ("uvloop", "h11"),
}


def _http_protocol_class(name: str) -> type:
    if name == "h11":
        from uvicorn.protocols.http.h11_impl import H11Protocol

        return H11Protocol
    from uvicorn.protocols.http.httptools_impl import HttpToolsProtocol

    return HttpToolsProtocol


class LoopbackServer:
    """Real uvicorn on 127.0.0.1 in a background thread.

    ``runtime`` selects the event loop and HTTP protocol (``RUNTIMES``);
    the default is the baseline, asyncio + h11. The harness creates the
    event loop itself (``serve()`` does not run uvicorn's loop setup): the
    standard asyncio loop, or for uvloop ``guarded_uvloop_class()`` from the
    conftest, so the network guard stays effective. Startup fails unless the
    loop and protocol actually in use are the requested ones. The listening
    socket is bound before the server starts, so no host name is resolved.
    """

    def __init__(
        self,
        app,
        *,
        runtime: str = "A",
        lifespan: str = "off",
        startup_timeout: float = 10.0,
        graceful_shutdown_timeout: float | None = 5.0,
        sndbuf: int | None = None,
    ):
        self.app = app
        self.runtime = runtime
        self.loop_name, self.http_name = RUNTIMES[runtime]
        self.lifespan = lifespan
        self.startup_timeout = startup_timeout
        # Bounds shutdown if a test leaves a streaming connection open.
        self.graceful_shutdown_timeout = graceful_shutdown_timeout
        # H3-A: optional SO_SNDBUF for the listening socket, inherited by the
        # connections it accepts. Small values make a slow reader fill the
        # kernel buffers after little data instead of the OS's auto-tuned
        # maximum (several MB), so transport flow control engages promptly.
        # None (the default) leaves the OS default untouched.
        self.sndbuf = sndbuf
        self.loop_class: type | None = None  # recorded inside the server task
        self.loop: asyncio.AbstractEventLoop | None = None  # the server's own loop
        self._error: BaseException | None = None

    def _loop_factory(self):
        if self.loop_name == "uvloop":
            from tests.conftest import guarded_uvloop_class

            return guarded_uvloop_class()
        return None  # asyncio's default loop, as asyncio.run() uses

    def __enter__(self) -> "LoopbackServer":
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if self.sndbuf is not None:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.sndbuf)
        self.sock.bind(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]

        config = uvicorn.Config(
            self.app,
            loop="none",  # the harness creates the loop (see _run)
            http=self.http_name,
            ws="none",
            lifespan=self.lifespan,
            log_config=None,
            access_log=False,
            timeout_graceful_shutdown=self.graceful_shutdown_timeout,
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self._run, name="loopback-uvicorn", daemon=True)
        self.thread.start()

        deadline = time.monotonic() + self.startup_timeout
        while not self.server.started:
            if self._error is not None or not self.thread.is_alive():
                raise RuntimeError("Loopback server failed to start.") from self._error
            if time.monotonic() > deadline:
                raise TimeoutError("Loopback server did not start in time.")
            time.sleep(0.01)
        self._check_runtime()
        return self

    def _check_runtime(self) -> None:
        """Fail fast unless the requested loop and protocol are the ones in use."""
        wanted_protocol = _http_protocol_class(self.http_name)
        problems = []
        if self.loop_name == "uvloop":
            from tests.conftest import guarded_uvloop_class

            if self.loop_class is not guarded_uvloop_class():
                problems.append(f"loop {self.loop_class!r}, expected the guarded uvloop loop")
        elif not issubclass(self.loop_class, asyncio.BaseEventLoop):  # uvloop.Loop is not one
            problems.append(f"loop {self.loop_class!r}, expected the standard asyncio loop")
        if self.server.config.http_protocol_class is not wanted_protocol:
            problems.append(
                f"protocol {self.server.config.http_protocol_class!r}, expected {wanted_protocol!r}"
            )
        if problems:
            self.server.should_exit = True
            self.thread.join(10)
            self.sock.close()
            raise RuntimeError(f"Loopback runtime {self.runtime} not in use: " + "; ".join(problems))

    def _run(self) -> None:
        async def main():
            self.loop = asyncio.get_running_loop()
            self.loop_class = type(self.loop)  # inside the server task
            await self.server.serve(sockets=[self.sock])

        try:
            # asyncio.Runner (Python 3.11+) is what asyncio.run() uses; it also
            # accepts the guarded uvloop loop for the production runtime.
            with asyncio.Runner(loop_factory=self._loop_factory()) as runner:
                runner.run(main())
        except BaseException as exc:  # noqa: BLE001 - surfaced in __enter__
            self._error = exc

    def call_in_loop(self, fn: Callable[[], Any], timeout: float = 5.0) -> Any:
        """Run ``fn`` on the server's event-loop thread and return its result.

        For reading per-connection state that the loop owns (H3-A) without
        racing it. Fails if the loop does not run ``fn`` within ``timeout``,
        which also shows the loop is not blocked when it succeeds.
        """
        assert self.loop is not None, "the server loop is not running"
        done: Future = Future()

        def run():
            try:
                done.set_result(fn())
            except BaseException as exc:  # noqa: BLE001 - re-raised by result()
                done.set_exception(exc)

        self.loop.call_soon_threadsafe(run)
        return done.result(timeout)

    def connection_flow(self) -> list[dict]:
        """Flow-control state of each open connection, read on the server loop (H3-A).

        ``write_paused`` is uvicorn's ``FlowControl`` flag, set when the
        transport's write buffer passed its high-water mark and the transport
        called ``protocol.pause_writing()``; ``sndbuf`` is the accepted
        socket's actual SO_SNDBUF (``None`` if the transport does not expose
        a socket).
        """

        def read():
            states = []
            for protocol in list(self.server.server_state.connections):
                flow = getattr(protocol, "flow", None)
                transport = getattr(protocol, "transport", None)
                sock = transport.get_extra_info("socket") if transport is not None else None
                states.append(
                    {
                        "write_paused": None if flow is None else flow.write_paused,
                        "write_buffer": None if transport is None else transport.get_write_buffer_size(),
                        "sndbuf": None if sock is None else sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF),
                    }
                )
            return states

        return self.call_in_loop(read)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, *exc_info) -> None:
        self.server.should_exit = True
        self.thread.join(10)
        self.sock.close()
        if self.thread.is_alive():
            raise RuntimeError("Loopback server did not stop.")


def assert_run_cleaned_up(service, run, server: "LoopbackServer | None" = None) -> None:
    """The run's own cleanup (H3): its worker exited, its lease and capacity
    slot were released, it wrote exactly one audit row, and the loopback
    server (if given) stopped. Waits are the callers' (bounded) waits."""
    assert run.hooks["worker_exited"].is_set(), "the run's worker has not exited"
    assert run.lease.released, "the run's lease was not released"
    assert service.capacity.in_use == 0
    assert service.capacity.acquired_total == service.capacity.released_total
    assert run.audit_writes == 1
    assert not run.defensive_claim_fired
    if server is not None:
        assert not server.thread.is_alive(), "the loopback server did not stop"


def read_sse_until(chunks: Iterator[bytes], buffer: bytearray, fragment: bytes) -> None:
    """Read raw response chunks into ``buffer`` until it contains ``fragment``.

    For loopback tests: ``chunks`` is ``httpx.Response.iter_raw()``. Raises
    ``AssertionError`` if the stream ends first; the client's read timeout
    bounds the wait.
    """
    while fragment not in buffer:
        try:
            buffer += next(chunks)
        except StopIteration:
            raise AssertionError(f"stream ended before {fragment!r}") from None
