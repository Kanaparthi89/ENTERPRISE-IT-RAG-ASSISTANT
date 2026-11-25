"""H3: the loopback harness runs the requested uvicorn runtime.

Runtime A (asyncio + h11) is the baseline the suite has always used;
runtime P (uvloop + httptools) is what production uvicorn selects. These
tests check, from inside the running server, which event loop and which HTTP
protocol actually serve a request, so the later H3 scenarios cannot pass on
the wrong runtime. Network access stays guarded (the uvloop loop is the
conftest's guarded subclass).
"""

import asyncio
import threading

import httpx
import pytest
from uvicorn.protocols.http.h11_impl import H11Protocol
from uvicorn.protocols.http.httptools_impl import HttpToolsProtocol

from tests import streaming_support
from tests.conftest import guarded_uvloop_class
from tests.streaming_support import LoopbackServer

GUARD = 5.0

EXPECTED_PROTOCOL = {"A": H11Protocol, "P": HttpToolsProtocol}


class RecordingApp:
    """A minimal ASGI app that records the runtime serving each request."""

    def __init__(self):
        self.server = None  # set once the loopback server is running
        self.requests = []

    async def __call__(self, scope, receive, send):
        assert scope["type"] == "http"
        connections = self.server.server.server_state.connections  # the request's own is open
        self.requests.append(
            {
                "loop": type(asyncio.get_running_loop()),
                "protocols": {type(connection) for connection in connections},
                "spec_version": scope["asgi"]["spec_version"],
                "thread": threading.current_thread(),
            }
        )
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": b"ok"})


@pytest.mark.parametrize("runtime", ["A", "P"])
def test_the_loopback_server_runs_the_requested_runtime(runtime):
    app = RecordingApp()

    with LoopbackServer(app, runtime=runtime) as server:
        app.server = server
        with httpx.Client(base_url=server.base_url, trust_env=False, timeout=GUARD) as client:
            response = client.get("/")
        server_thread = server.thread

    assert not server_thread.is_alive()  # this test's server stopped
    assert response.status_code == 200 and response.text == "ok"
    [request] = app.requests
    assert request["thread"] is server_thread  # served on the server's own loop thread
    assert request["loop"] is server.loop_class  # the same loop as the server task
    assert request["protocols"] == {EXPECTED_PROTOCOL[runtime]}  # the live connection's protocol
    assert server.server.config.http_protocol_class is EXPECTED_PROTOCOL[runtime]
    assert request["spec_version"] == "2.3"  # both protocols: SSEResponse takes the same branch
    if runtime == "P":
        import uvloop

        assert request["loop"] is guarded_uvloop_class()
        assert issubclass(request["loop"], uvloop.Loop)
    else:
        assert issubclass(request["loop"], asyncio.BaseEventLoop)  # the standard asyncio loop


def test_the_default_runtime_is_the_baseline():
    assert LoopbackServer(RecordingApp()).runtime == "A"
    assert streaming_support.RUNTIMES["A"] == ("asyncio", "h11")
    assert streaming_support.RUNTIMES["P"] == ("uvloop", "httptools")


def test_the_harness_refuses_a_runtime_it_did_not_get(monkeypatch):
    # If uvicorn served with another protocol than requested, startup fails
    # instead of letting a test pass on the wrong runtime.
    monkeypatch.setattr(streaming_support, "_http_protocol_class", lambda name: HttpToolsProtocol)
    server = LoopbackServer(RecordingApp(), runtime="A")

    with pytest.raises(RuntimeError, match="Loopback runtime A not in use: protocol"):
        server.__enter__()

    assert not server.thread.is_alive()  # the server it started was stopped
