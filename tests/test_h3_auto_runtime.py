"""H3-B: Uvicorn's own loop="auto" and http="auto" selection.

The loopback harness in ``tests/streaming_support.py`` creates its event loop
itself (``loop="none"``) and names the HTTP protocol explicitly, so the
runtime "P" tests run *under* uvloop + httptools but never exercise how
Uvicorn *chooses* them. This module does: each test starts Uvicorn's real
CLI (``python -m uvicorn``, the same entry point as the Dockerfile's
``uvicorn`` command, which ends in ``Server.run()``) in a separate process
and asks ``tests/h3_probe_app.py`` which event loop, loop policy, and HTTP
protocol served the request.

Why a subprocess: selecting uvloop installs a process-wide event-loop policy
(``uvicorn.loops.uvloop.uvloop_setup``), and the pytest process must stay on
the standard asyncio policy (``tests/test_isolation.py``).

Scope and limits:

* The probe imports nothing from ``app``: no settings, ``.env``, databases,
  or providers. It is served on a Unix-domain socket, so no TCP port is
  used, and it opens no connections. The conftest network guard does not
  apply inside the child process; the probe has nothing to guard.
* These tests verify this machine's installation (the local virtualenv).
  They do not verify the production Docker image (Linux, Python 3.11); that
  is a separate, optional check.
* ``UVICORN_LOOP`` / ``UVICORN_HTTP`` environment variables change the
  selection. The Dockerfile and ``compose.yaml`` are checked statically
  for them; ``.env`` (which ``compose.yaml`` loads into the container) is
  never read by tests, so it is not checked.
"""

import importlib.util
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_APP = "tests.h3_probe_app:app"

STARTUP_TIMEOUT = 20.0  # interpreter start, imports, and the first bind
REQUEST_TIMEOUT = 5.0
STOP_TIMEOUT = 5.0
SUN_PATH_LIMIT = 100  # macOS allows 104 bytes for a Unix socket path, Linux 108


class ProbeServer:
    """One ``python -m uvicorn tests.h3_probe_app:app`` child process."""

    def __init__(self, args, env_overrides):
        self.dir = tempfile.mkdtemp(prefix="h3-")
        self.socket_path = os.path.join(self.dir, "probe.sock")
        assert len(self.socket_path) < SUN_PATH_LIMIT, f"socket path too long: {self.socket_path}"
        self.log_path = os.path.join(self.dir, "uvicorn.log")
        self.process = None
        self._log = None

        env = {key: value for key, value in os.environ.items() if not key.startswith("UVICORN_")}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.update(env_overrides)
        self.command = [
            sys.executable, "-m", "uvicorn", PROBE_APP,
            "--uds", self.socket_path,
            "--lifespan", "off",
            "--log-level", "warning",
            *args,
        ]
        self.env = env

    def start(self) -> "ProbeServer":
        self._log = open(self.log_path, "wb")  # a file, not a pipe: the child can never block on output
        self.process = subprocess.Popen(
            self.command,
            cwd=REPO_ROOT,
            env=self.env,
            stdin=subprocess.DEVNULL,
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while True:
            if self.process.poll() is not None:
                pytest.fail(f"uvicorn exited with {self.process.returncode} before serving:\n{self.output()}")
            if os.path.exists(self.socket_path):
                try:
                    self.fetch()
                    return self
                except httpx.TransportError:
                    pass  # bound but not accepting yet
            if time.monotonic() > deadline:
                pytest.fail(f"uvicorn did not serve within {STARTUP_TIMEOUT} s:\n{self.output()}")
            time.sleep(0.05)

    def fetch(self) -> dict:
        transport = httpx.HTTPTransport(uds=self.socket_path)
        with httpx.Client(
            transport=transport, base_url="http://h3-probe", timeout=REQUEST_TIMEOUT, trust_env=False
        ) as client:
            response = client.get("/")
        assert response.status_code == 200, response.text
        return response.json()

    def output(self) -> str:
        if self._log is not None:
            self._log.flush()
        try:
            with open(self.log_path, "rb") as log:
                return log.read().decode(errors="replace")
        except OSError:
            return ""

    def stop(self) -> None:
        """Always safe to call; ends the child and removes its directory."""
        try:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(STOP_TIMEOUT)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(STOP_TIMEOUT)
        finally:
            if self._log is not None:
                self._log.close()
            shutil.rmtree(self.dir, ignore_errors=True)


@pytest.fixture
def probe():
    """Start probe servers; every one is stopped and cleaned up after the test, pass or fail."""
    servers = []

    def run(*args, env=None) -> dict:
        server = ProbeServer(args, env or {})
        servers.append(server)
        runtime = server.start().fetch()
        assert runtime["pid"] == server.process.pid != os.getpid()  # served by the child
        return runtime

    yield run

    for server in servers:
        server.stop()
    for server in servers:
        assert server.process is None or server.process.poll() is not None, "a probe server is still running"
        assert not os.path.exists(server.dir), "a probe directory was left behind"


UVLOOP = "uvloop.Loop"
HTTPTOOLS = "httptools_impl.py"
H11 = "h11_impl.py"


def is_asyncio_loop(name: str) -> bool:
    return name.startswith("asyncio.") and "uvloop" not in name


# ---------------------------------------------------------------------------
# Uvicorn's real selection, in a subprocess
# ---------------------------------------------------------------------------

def test_defaults_select_uvloop_and_httptools(probe):
    runtime = probe()

    assert runtime["loop"] == UVLOOP  # uvloop's own loop, not the tests' guarded subclass
    assert runtime["policy"].startswith("uvloop")  # installed by uvloop_setup (loop="auto")
    assert runtime["cycle_file"] == HTTPTOOLS  # AutoHTTPProtocol resolved to httptools
    assert runtime["cycle_name"] == "RequestResponseCycle.run_asgi"
    assert runtime["app_modules"] == []  # isolated from application startup


def test_explicit_asyncio_and_h11_are_honored(probe):
    # Negative control: the probe tells the runtimes apart.
    runtime = probe("--loop", "asyncio", "--http", "h11")

    assert is_asyncio_loop(runtime["loop"]), runtime["loop"]
    assert "uvloop" not in runtime["policy"]
    assert runtime["cycle_file"] == H11


@pytest.mark.parametrize(
    "env, loop_is_uvloop, cycle_file",
    [
        ({"UVICORN_LOOP": "asyncio"}, False, HTTPTOOLS),
        ({"UVICORN_HTTP": "h11"}, True, H11),
    ],
)
def test_uvicorn_environment_variables_override_auto(probe, env, loop_is_uvloop, cycle_file):
    # The CLI reads UVICORN_* variables (auto_envvar_prefix), so a deployment
    # environment can switch the runtime without any command-line flag.
    runtime = probe(env=env)

    if loop_is_uvloop:
        assert runtime["loop"] == UVLOOP
    else:
        assert is_asyncio_loop(runtime["loop"]), runtime["loop"]
    assert runtime["cycle_file"] == cycle_file


def test_inherited_uvicorn_variables_do_not_leak_into_the_probe(probe, monkeypatch):
    # The default-selection test must see "auto", even if the test run itself
    # was started with a UVICORN_* variable set.
    monkeypatch.setenv("UVICORN_LOOP", "asyncio")
    monkeypatch.setenv("UVICORN_HTTP", "h11")

    runtime = probe()

    assert runtime["loop"] == UVLOOP
    assert runtime["cycle_file"] == HTTPTOOLS


# ---------------------------------------------------------------------------
# Static: what makes "auto" the production choice
# ---------------------------------------------------------------------------

def test_uvicorn_cli_and_config_default_to_auto():
    import uvicorn
    from uvicorn.main import main

    options = {param.name: param.default for param in main.params}
    assert (options["loop"], options["http"]) == ("auto", "auto")
    assert main.context_settings.get("auto_envvar_prefix") == "UVICORN"  # see the override test
    defaults = inspect.signature(uvicorn.Config).parameters
    assert (defaults["loop"].default, defaults["http"].default) == ("auto", "auto")


def test_uvloop_and_httptools_are_importable_here():
    # "auto" picks them only when they import (uvicorn.loops.auto, protocols.http.auto).
    assert importlib.util.find_spec("uvloop") is not None
    assert importlib.util.find_spec("httptools") is not None


def test_requirements_install_uvicorn_standard():
    requirements = (REPO_ROOT / "requirements.txt").read_text()

    assert re.search(r"^uvicorn\[standard\]==", requirements, re.MULTILINE)


def test_docker_and_compose_leave_loop_and_http_on_auto():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()
    compose = (REPO_ROOT / "compose.yaml").read_text()

    [cmd] = [line for line in dockerfile.splitlines() if line.startswith("CMD ")]
    assert "uvicorn app.main:app" in cmd
    for text in (dockerfile, compose):
        assert "--loop" not in text and "--http" not in text
        assert "UVICORN_" not in text
