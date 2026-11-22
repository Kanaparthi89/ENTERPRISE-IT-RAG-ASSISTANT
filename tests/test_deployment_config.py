"""Security S1-5 Phases D2 and E2A: the container's uvicorn command.

D2: it trusts no forwarded headers. E2A: it runs exactly one worker process,
which no environment variable (UVICORN_WORKERS, WEB_CONCURRENCY) can change,
because the in-memory throttles, quotas and the ingest slot are per process.
E2A-R1: the ASGI lifespan is forced on (UVICORN_LIFESPAN cannot turn it off),
so the app's startup and shutdown hooks always run: init_db() and the
sessions lifespan (startup cleanup, lease keeper, graceful session shutdown).
E2B: scripts/check_runtime_env.py runs first (``python -I ... && exec uvicorn``)
and refuses to start while any UVICORN_* variable is set, so no option the
command line leaves unpinned can be changed from the environment.
E2B-R1: uvicorn itself runs as ``python -I -u -B -m uvicorn``: PYTHONPATH (and so a
planted sitecustomize.py) and every other PYTHON* variable are ignored, so code
cannot set a UVICORN_* variable inside the server after the guard has passed.
E2B-R2: the port is passed as ``--port "${PORT:-8080}"``: quoted, so a PORT value is
always one argument (no word splitting into extra uvicorn options, no glob expansion).
One worker per container does not mean one DigitalOcean instance: the
instance count must be checked on the platform.

The Dockerfile's CMD is parsed exactly as uvicorn's own command line parses
it, turned into the effective ``uvicorn.Config`` and loaded (offline, no
server, no network). With ``--no-proxy-headers``, X-Forwarded-For and
X-Forwarded-Proto can never replace the socket peer or the scheme the app
sees, whatever FORWARDED_ALLOW_IPS says and even when the peer is loopback
(which uvicorn trusts by default). The only trusted client-IP input stays the
verified ACCESS_CLIENT_IP_HEADER (app/services/access.py).
"""

import asyncio
import http.client
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import uvicorn
from uvicorn.main import main as uvicorn_cli
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

ROOT = Path(__file__).resolve().parents[1]
# Variables that can change uvicorn's effective configuration from the environment.
AMBIENT = ("WEB_CONCURRENCY", "FORWARDED_ALLOW_IPS")


@pytest.fixture(autouse=True)
def isolated_uvicorn_environment(monkeypatch):
    """No ambient UVICORN_*, WEB_CONCURRENCY or FORWARDED_ALLOW_IPS: tests set them explicitly."""
    for name in list(os.environ):
        if name.startswith("UVICORN_") or name in AMBIENT:
            monkeypatch.delenv(name, raising=False)


SPOOF = [(b"x-forwarded-for", b"6.6.6.6, 203.0.113.50"), (b"x-forwarded-proto", b"https"),
         (b"x-forwarded-host", b"evil.example"), (b"forwarded", b"for=7.7.7.7;proto=https;host=evil.example")]


# ---------------------------------------------------------------------------
# The command, as the container runs it
# ---------------------------------------------------------------------------

def dockerfile_cmd() -> list:
    """The exec-form CMD of the Dockerfile (the last CMD instruction, not HEALTHCHECK's)."""
    commands = [line for line in (ROOT / "Dockerfile").read_text().splitlines() if line.startswith("CMD ")]
    assert len(commands) == 1, commands
    return json.loads(commands[0][len("CMD "):])


GUARD = "python -I scripts/check_runtime_env.py"
UVICORN_EXEC = "exec python -I -u -B -m uvicorn "  # E2B-R1
UVICORN_EXEC_TOKENS = ["exec", "python", "-I", "-u", "-B", "-m", "uvicorn"]
GUARD_SCRIPT = ROOT / "scripts" / "check_runtime_env.py"


def split_cmd_script(script: str) -> tuple[str, str]:
    """(guard, uvicorn command) of ``<guard> && exec uvicorn ...``: exactly one ``&&``, nothing else."""
    for forbidden in (";", "||", "|", "\n", "`", "$("):
        assert forbidden not in script.replace("&&", ""), f"{forbidden!r} in the CMD script"
    parts = script.split(" && ")
    assert len(parts) == 2, script
    assert script.count("&") == 2, script  # the one "&&": no background "&"
    return parts[0], parts[1]


def pins_only_script() -> str:
    """The CMD's uvicorn command without the guard: proves the CLI pins hold on their own (defence in depth)."""
    _, _, script = dockerfile_cmd()
    return split_cmd_script(script)[1]


def uvicorn_argv(port: str = "8080") -> list:
    """The uvicorn arguments, with ${PORT:-8080} expanded as sh would for PORT=``port``."""
    shell, flag, script = dockerfile_cmd()
    assert (shell, flag) == ("sh", "-c")
    guard, command = split_cmd_script(script)
    assert guard == GUARD, guard
    assert command.startswith(UVICORN_EXEC), command  # exec: uvicorn is PID 1's process and receives SIGTERM
    tokens = shlex.split(re.sub(r"\$\{PORT:-8080\}", port, command))
    assert tokens[:7] == UVICORN_EXEC_TOKENS, tokens
    return tokens[7:]


def effective_params(port: str = "8080") -> dict:
    return uvicorn_cli.make_context("uvicorn", uvicorn_argv(port)).params


def effective_config(app_target, monkeypatch, *, forwarded_allow_ips_env=None, argv=None) -> uvicorn.Config:
    """The uvicorn.Config the container would build (log_config=None: never reconfigure test logging)."""
    if forwarded_allow_ips_env is None:
        monkeypatch.delenv("FORWARDED_ALLOW_IPS", raising=False)
    else:
        monkeypatch.setenv("FORWARDED_ALLOW_IPS", forwarded_allow_ips_env)
    # A copy: click consumes the argument list it is given.
    params = uvicorn_cli.make_context("uvicorn", list(argv) if argv is not None else uvicorn_argv()).params
    return uvicorn.Config(
        app_target,
        host=params["host"],
        port=params["port"],
        proxy_headers=params["proxy_headers"],
        forwarded_allow_ips=params["forwarded_allow_ips"],
        workers=params["workers"],
        timeout_graceful_shutdown=params["timeout_graceful_shutdown"],
        log_config=None,
    )


def middleware_chain(app):
    seen = []
    while app is not None and len(seen) < 20:
        seen.append(app)
        app = getattr(app, "app", None)
    return seen


# ---------------------------------------------------------------------------
# A probe app: reports what an application sees
# ---------------------------------------------------------------------------

async def echo_app(scope, receive, send):
    body = json.dumps({"client": scope.get("client", [None])[0], "scheme": scope.get("scheme")}).encode()
    await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
    await send({"type": "http.response.body", "body": body})


def seen_by(loaded_app, peer, headers=SPOOF, path="/"):
    out = {}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            out["status"] = message["status"]
            out["location"] = dict(message["headers"]).get(b"location", b"").decode()
        elif message["type"] == "http.response.body":
            out["body"] = out.get("body", b"") + message.get("body", b"")

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET", "scheme": "http",
             "path": path, "raw_path": path.encode(), "query_string": b"", "root_path": "",
             "headers": [(b"host", b"bluepeak.example"), *headers], "client": (peer, 40000), "server": ("10.0.0.5", 8080)}
    asyncio.run(loaded_app(scope, receive, send))
    return out


# ===========================================================================
# The effective configuration
# ===========================================================================

def test_the_command_runs_the_production_app_unchanged_except_proxy_headers():
    params = effective_params()
    assert params["app"] == "app.main:app"
    assert (params["host"], params["port"]) == ("0.0.0.0", 8080)
    assert params["proxy_headers"] is False
    assert params["forwarded_allow_ips"] is None
    assert params["workers"] == 1  # E2A: pinned (in-memory limits assume one process)
    assert params["timeout_graceful_shutdown"] == 8
    assert params["reload"] is False
    assert params["lifespan"] == "on"  # E2A-R1


def test_the_port_still_comes_from_the_environment():
    assert effective_params(port="9123")["port"] == 9123


def test_no_flag_reenables_or_widens_proxy_trust():
    argv = uvicorn_argv()
    assert "--no-proxy-headers" in argv
    assert "--proxy-headers" not in argv
    assert not any(arg.startswith("--forwarded-allow-ips") for arg in argv)
    assert not any(arg.startswith(("--reload", "--root-path")) for arg in argv)


def test_exactly_one_workers_flag_set_to_one():
    argv = uvicorn_argv()
    workers = [index for index, arg in enumerate(argv) if arg == "--workers" or arg.startswith("--workers=")]
    assert len(workers) == 1, "exactly one --workers flag (a duplicate or conflicting one is refused)"
    assert argv[workers[0]] == "--workers" and argv[workers[0] + 1] == "1"


def test_exactly_one_lifespan_flag_set_to_on():
    argv = uvicorn_argv()
    lifespan = [index for index, arg in enumerate(argv) if arg == "--lifespan" or arg.startswith("--lifespan=")]
    assert len(lifespan) == 1, "exactly one --lifespan flag (a duplicate or conflicting one is refused)"
    assert argv[lifespan[0]] == "--lifespan" and argv[lifespan[0] + 1] == "on"


def test_the_rest_of_the_dockerfile_is_unchanged():
    text = (ROOT / "Dockerfile").read_text()
    assert "FROM python:3.11-slim" in text and "USER appuser" in text and "EXPOSE 8080" in text
    assert "urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)" in text  # HEALTHCHECK


@pytest.mark.parametrize("env", [None, "*", "127.0.0.1", "0.0.0.0/0,::/0"])
def test_the_loaded_app_has_no_proxy_headers_middleware(monkeypatch, env):
    config = effective_config("tests.test_deployment_config:echo_app", monkeypatch, forwarded_allow_ips_env=env)
    config.load()
    assert config.proxy_headers is False
    assert not any(isinstance(layer, ProxyHeadersMiddleware) for layer in middleware_chain(config.loaded_app))


# ===========================================================================
# Forwarded headers cannot override the socket identity or the scheme
# ===========================================================================

@pytest.mark.parametrize("env", [None, "*", "127.0.0.1", "0.0.0.0/0,::/0"])
@pytest.mark.parametrize("peer", ["127.0.0.1", "::1", "10.244.0.7", "203.0.113.9"])
def test_spoofed_forwarded_headers_change_nothing(monkeypatch, env, peer):
    config = effective_config("tests.test_deployment_config:echo_app", monkeypatch, forwarded_allow_ips_env=env)
    config.load()
    seen = json.loads(seen_by(config.loaded_app, peer)["body"])
    assert seen == {"client": peer, "scheme": "http"}


def test_control_without_the_flag_the_same_request_is_spoofable(monkeypatch):
    # Shows the tests above detect the risk: uvicorn's default trusts a loopback
    # peer, and FORWARDED_ALLOW_IPS='*' trusts everyone.
    argv = [arg for arg in uvicorn_argv() if arg != "--no-proxy-headers"]
    # Default trust, loopback peer: the right-most untrusted X-Forwarded-For entry.
    # FORWARDED_ALLOW_IPS='*': the left-most entry, which the client chooses outright.
    for env, peer, spoofed in ((None, "127.0.0.1", "203.0.113.50"), ("*", "10.244.0.7", "6.6.6.6")):
        config = effective_config("tests.test_deployment_config:echo_app", monkeypatch,
                                  forwarded_allow_ips_env=env, argv=argv)
        config.load()
        seen = json.loads(seen_by(config.loaded_app, peer)["body"])
        assert seen == {"client": spoofed, "scheme": "https"}, (env, peer)
        assert seen["client"] != peer


@pytest.mark.parametrize("env", [None, "*"])
@pytest.mark.parametrize("peer", ["127.0.0.1", "10.244.0.7"])
def test_the_real_app_sees_the_socket_scheme_not_x_forwarded_proto(monkeypatch, env, peer):
    config = effective_config("app.main:app", monkeypatch, forwarded_allow_ips_env=env)
    config.load()
    result = seen_by(config.loaded_app, peer, path="/api/health/")  # a trailing-slash redirect reveals the scheme
    assert result["status"] == 307
    assert result["location"] == "http://bluepeak.example/api/health"  # X-Forwarded-Proto: https was ignored


@pytest.mark.parametrize("env", [None, "*"])
def test_the_throttle_identity_is_the_socket_peer(monkeypatch, settings, env):
    """What the access and admin throttles key on: never an X-Forwarded-For entry."""
    from starlette.requests import Request

    from app.services import access, admin_auth

    captured = {}

    async def identity_app(scope, receive, send):
        request = Request(scope)
        captured["access"] = access.client_key(request.client.host, request.headers, "")
        captured["admin"] = admin_auth.client_identity(request, settings)
        await echo_app(scope, receive, send)

    import tests.test_deployment_config as this_module

    monkeypatch.setattr(this_module, "identity_app", identity_app, raising=False)
    monkeypatch.setattr(settings, "access_client_ip_header", "")
    config = effective_config("tests.test_deployment_config:identity_app", monkeypatch, forwarded_allow_ips_env=env)
    config.load()
    seen_by(config.loaded_app, "127.0.0.1")
    assert captured == {"access": "127.0.0.1", "admin": "127.0.0.1"}


# ===========================================================================
# E2A: one worker, whatever the environment says
# ===========================================================================

OVERRIDES = {
    "UVICORN_WORKERS=4": {"UVICORN_WORKERS": "4"},
    "WEB_CONCURRENCY=4": {"WEB_CONCURRENCY": "4"},
    "both": {"UVICORN_WORKERS": "4", "WEB_CONCURRENCY": "3"},
}


def resolved_workers(monkeypatch, env, argv=None):
    """(parsed --workers, effective Config.workers) for the command under ``env``."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    params = uvicorn_cli.make_context("uvicorn", list(argv) if argv is not None else uvicorn_argv()).params
    config = uvicorn.Config("tests.test_deployment_config:echo_app", workers=params["workers"], log_config=None)
    return params["workers"], config.workers


def test_without_overrides_the_effective_worker_count_is_one(monkeypatch):
    assert resolved_workers(monkeypatch, {}) == (1, 1)


@pytest.mark.parametrize("case", list(OVERRIDES))
def test_the_environment_cannot_raise_the_worker_count(monkeypatch, case):
    assert resolved_workers(monkeypatch, OVERRIDES[case]) == (1, 1)  # single server.run(): no supervisor


@pytest.mark.parametrize("case", list(OVERRIDES))
def test_control_without_the_flag_the_overrides_take_effect(monkeypatch, case):
    argv = uvicorn_argv()
    index = argv.index("--workers")
    unpinned = argv[:index] + argv[index + 2:]
    expected = 4 if "UVICORN_WORKERS" in OVERRIDES[case] else int(OVERRIDES[case]["WEB_CONCURRENCY"])
    parsed, effective = resolved_workers(monkeypatch, OVERRIDES[case], argv=unpinned)
    assert effective == expected > 1  # what the pin prevents
    if "UVICORN_WORKERS" not in OVERRIDES[case]:
        assert parsed is None  # WEB_CONCURRENCY is invisible to the CLI: only Config shows it


# ---------------------------------------------------------------------------
# A real uvicorn process, started with the Dockerfile's exact command
# ---------------------------------------------------------------------------

async def lifespan_echo_app(scope, receive, send):
    """A minimal app that also speaks the lifespan protocol, so shutdown is observable."""
    if scope["type"] == "lifespan":
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return
    if scope["path"] == "/pid":  # E2B: which process answered
        body = json.dumps({"pid": os.getpid()}).encode()
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})
        return
    await echo_app(scope, receive, send)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by someone else (cannot be ours)
        return True
    return True


def start_group(argv, env, cwd=ROOT):
    """Start ``argv`` in a new session: its whole process tree is one group we can kill (E2B)."""
    return subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)


def stop_group(process, timeout: float) -> str:
    """SIGTERM the leader; after ``timeout`` SIGKILL the whole group. Returns the output; no process survives."""
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
    # Workers or other children may outlive a leader that exited on its own: never leave them behind.
    # Right after a SIGKILL, macOS can answer EPERM for a group whose last members are still being
    # reaped; retry until the group is gone (ProcessLookupError), never treating EPERM as success.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            break
        except PermissionError:
            pass
        time.sleep(0.05)
    assert not process_group_alive(process.pid), "a child process outlived the test"
    return output


def real_server_env(port: int, env_extra) -> dict:
    return {
        "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
        "PORT": str(port),
        "PYTHONPATH": str(ROOT),
        "HOME": str(ROOT / "data"),
        "PYTHONDONTWRITEBYTECODE": "1",
        **env_extra,
    }


def run_real_server(env_extra, *, script=None, timeout=30.0, path="/", details=False, port_template=None):
    """Start ``sh -c <CMD>`` (app target swapped for lifespan_echo_app), make one request, SIGTERM it.

    Returns (status of GET ``path``, the process's log lines, its exit code); with
    ``details``, also the response body and the launched process's PID. ``port_template``
    (for example ``"{port} --reload"``) sets PORT to a hostile value built around the probe port.
    """
    shell, flag, cmd_script = dockerfile_cmd()
    assert (shell, flag) == ("sh", "-c")
    port = _free_port()
    script = (script or cmd_script).replace("app.main:app", "tests.test_deployment_config:lifespan_echo_app")
    env = real_server_env(port, env_extra)
    if port_template is not None:
        env["PORT"] = port_template.format(port=port)
    process = start_group([shell, flag, script], env)
    status, body = None, None
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and process.poll() is None:
            try:
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
                connection.request("GET", path)
                response = connection.getresponse()
                status, body = response.status, response.read()
                connection.close()
                break
            except OSError:
                time.sleep(0.1)
        time.sleep(0.3)  # let any further worker processes start and log
    finally:
        output = stop_group(process, timeout)
    log = [line.strip() for line in output.splitlines() if line.strip()]
    if details:
        return status, log, process.returncode, body, process.pid
    return status, log, process.returncode


GRACEFUL = ("Shutting down", "Waiting for application shutdown.", "Application shutdown complete.", "Finished server process")


def test_a_real_process_runs_one_server_under_both_overrides_and_shuts_down_gracefully():
    # Pins only: with the full CMD the guard refuses UVICORN_WORKERS first (E2B tests below).
    status, log, returncode = run_real_server({"UVICORN_WORKERS": "4", "WEB_CONCURRENCY": "4", "FORWARDED_ALLOW_IPS": "*"},
                                              script=pins_only_script())
    assert status == 200, log
    # uvicorn re-raises the captured SIGTERM after a complete graceful shutdown
    # (Server.capture_signals): exit by that signal, not a crash or a SIGKILL.
    assert returncode == -signal.SIGTERM, (returncode, log)
    assert sum("Started server process" in line for line in log) == 1, log
    assert not any("Started parent process" in line for line in log), log
    for marker in GRACEFUL:
        assert any(marker in line for line in log), (marker, log)


def test_control_a_real_process_without_the_flag_becomes_multi_process():
    _, _, script = dockerfile_cmd()
    unpinned = script.replace(" --workers 1", "")
    assert unpinned != script
    status, log, _ = run_real_server({"WEB_CONCURRENCY": "2"}, script=unpinned)
    assert status == 200, log
    assert any("Started parent process" in line for line in log), log  # what the pin prevents


# ===========================================================================
# E2A: .env.example is an accurate, read-only development template
# ===========================================================================

ENV_EXAMPLE = ROOT / ".env.example"
ASSIGNMENT = re.compile(r"^([A-Z][A-Z0-9_]*)=(.*)$")
DOCUMENTED = re.compile(r"\b([A-Z][A-Z0-9_]{2,})=")


def template_settings_class():
    """Settings that read ONLY .env.example: no real .env, no environment variables, no secret files."""
    from pydantic_settings import SettingsConfigDict

    from app.core import config

    class TemplateSettings(config.Settings):
        model_config = SettingsConfigDict(**{**config.Settings.model_config, "env_file": str(ENV_EXAMPLE)})

        @classmethod
        def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings,
                                       file_secret_settings):
            return (init_settings, dotenv_settings)

    return TemplateSettings


def uncommented_assignments() -> dict:
    found = {}
    for line in ENV_EXAMPLE.read_text().splitlines():
        match = ASSIGNMENT.match(line)
        if match:
            assert match.group(1) not in found, f"{match.group(1)} assigned twice"
            found[match.group(1)] = match.group(2)
    return found


def test_the_template_parses_with_the_real_settings_and_ignores_the_environment(monkeypatch):
    # Hostile environment: proves only the file is read.
    for name, value in (("ACCESS_ENABLED", "true"), ("QUOTA_ENABLED", "true"), ("APP_ENV", "production"),
                        ("OPENAI_API_KEY", "sk-from-the-environment")):
        monkeypatch.setenv(name, value)
    template = template_settings_class()()
    assert template.access_enabled is False and template.quota_enabled is False
    assert template.app_env == "development"
    assert template.openai_api_key.get_secret_value() == ""


def test_every_name_in_the_template_is_a_real_setting():
    from app.core.config import Settings

    fields = {name.upper() for name in Settings.model_fields}
    uncommented = set(uncommented_assignments())
    documented = set(DOCUMENTED.findall(ENV_EXAMPLE.read_text()))
    assert uncommented <= fields, uncommented - fields
    assert documented <= fields, documented - fields  # commented examples are accurate too


def test_every_uncommented_value_is_a_line_of_its_own():
    # No inline comments after values: a value like "false  # note" would be read differently by tools.
    for name, value in uncommented_assignments().items():
        assert "#" not in value, name


def test_the_template_keeps_every_paid_or_risky_feature_off():
    from app.services import access_policy, admin_auth

    template = template_settings_class()()
    for flag in ("access_enabled", "quota_enabled", "sessions_enabled", "stream_enabled", "voice_enabled",
                 "voice_transcription_enabled", "api_docs_enabled"):
        assert getattr(template, flag) is False, flag
    assert template.public_origins == ""
    assert access_policy.configuration_problem(template) is not None  # read-only: paid routes answer 503
    assert admin_auth.key_status(template) == admin_auth.MISSING  # ingestion off


# Provider key prefixes, a hex digest, or a long token mixing letters and digits (as generated secrets do).
SECRET_LIKE = re.compile(
    r"sk-[A-Za-z0-9_-]{8,}|pcsk_|tvly-|\b[0-9a-f]{64}\b"
    r"|(?=[A-Za-z0-9_-]*[0-9])(?=[A-Za-z0-9_-]*[a-z])(?=[A-Za-z0-9_-]*[A-Z])[A-Za-z0-9_-]{32,}"
)


def test_the_secret_scan_would_catch_a_planted_secret():
    for planted in ("Zq8Wm3Kx7Pv2Ty9Lr4Nb6Hc1Jd5Fg0SeQ", "sk-proj-abcdef0123456789", "a" * 32 + "0" * 32,
                    "D1-origin-secret-Zq8Wm3Kx7Pv2Ty9Lr4Nb6Hc1Jd5Fg0Se", "pcsk_example", "tvly-example"):
        assert SECRET_LIKE.search(f"ADMIN_API_KEY={planted}"), planted
    assert not SECRET_LIKE.search("# " + "-" * 75 + "\nQUOTA_ENABLED=false\n")


def test_no_secret_has_a_value_in_the_template():
    from pydantic import SecretStr

    from app.core.config import Settings

    template = template_settings_class()()
    secrets_fields = [name for name, field in Settings.model_fields.items() if field.annotation is SecretStr]
    assert set(secrets_fields) >= {"openai_api_key", "pinecone_api_key", "tavily_api_key", "admin_api_key",
                                   "access_cookie_secret", "access_cookie_previous_secrets"}
    for name in secrets_fields:
        assert getattr(template, name).get_secret_value() == "", name
    assert template.access_code_hashes == ""
    text = ENV_EXAMPLE.read_text()
    assert not SECRET_LIKE.search(text)


def test_the_documented_quota_deadlines_actually_satisfy_the_quota_checks():
    from app.services import quota_runtime

    text = ENV_EXAMPLE.read_text()
    documented = {name: value for name, value in re.findall(r"\b([A-Z][A-Z0-9_]+)=(\d+)\b", text)}
    required = ("CHAT_DEADLINE_SECONDS", "STREAM_TIMEOUT_SECONDS", "OPENAI_TIMEOUT_SECONDS", "OPENAI_MAX_RETRIES",
                "SESSION_TURN_MAX_SECONDS", "PINECONE_RETRY_AFTER_MAX_SECONDS")
    assert set(required) <= set(documented), set(required) - set(documented)
    template = template_settings_class()(quota_enabled=True, **{name.lower(): int(documented[name]) for name in required})
    quota_runtime.check_settings(template)  # raises if the documented values were wrong


def test_the_template_documents_the_deployment_requirements():
    text = ENV_EXAMPLE.read_text()
    lowered = text.lower()
    assert "development" in lowered and "not production-safe" in lowered
    assert "one worker" in lowered and "one instance" in lowered
    assert "ACCESS_TRUST_SOCKET_PEER" in text and "never true behind" in lowered
    assert "no authentication or rate limit" not in lowered  # the stale transcription warning is gone
    assert "do-connecting-ip" in text and "verif" in lowered
    # E2A-R1: every uvicorn environment variable is warned against, and the reset on restart is explained.
    assert "NO UVICORN_* and NO WEB_CONCURRENCY" in text
    assert "worker variables" not in lowered  # the narrower, earlier wording
    assert "reset" in lowered and "restart" in lowered and "redeploy" in lowered
    # Session caps are SQLite row counts (app/services/sessions.py), not in-memory counters.
    in_memory = text[text.index("are kept in\n# memory, PER PROCESS") - 80:text.index("are kept in\n# memory, PER PROCESS")]
    assert "session" not in in_memory.lower()
    assert "Sessions and their caps live in a local SQLite file" in text


# ===========================================================================
# E2A-R1: the lifespan is always on, whatever the environment says
# ===========================================================================

def resolved_lifespan(monkeypatch, env, argv=None):
    """(parsed --lifespan, effective Config.lifespan) for the command under ``env``."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    params = uvicorn_cli.make_context("uvicorn", list(argv) if argv is not None else uvicorn_argv()).params
    config = uvicorn.Config("tests.test_deployment_config:echo_app", lifespan=params["lifespan"], log_config=None)
    return params["lifespan"], config.lifespan


def test_the_effective_configuration_enables_the_lifespan(monkeypatch):
    assert resolved_lifespan(monkeypatch, {}) == ("on", "on")


@pytest.mark.parametrize("value", ["off", "auto"])
def test_uvicorn_lifespan_cannot_override_the_flag(monkeypatch, value):
    assert resolved_lifespan(monkeypatch, {"UVICORN_LIFESPAN": value}) == ("on", "on")


def test_control_without_the_flag_uvicorn_lifespan_off_takes_effect(monkeypatch):
    argv = uvicorn_argv()
    index = argv.index("--lifespan")
    unpinned = argv[:index] + argv[index + 2:]
    assert resolved_lifespan(monkeypatch, {"UVICORN_LIFESPAN": "off"}, argv=unpinned) == ("off", "off")


LIFESPAN_STARTUP = ("Waiting for application startup.", "Application startup complete.")
HOSTILE = {"UVICORN_WORKERS": "4", "WEB_CONCURRENCY": "4", "UVICORN_LIFESPAN": "off", "FORWARDED_ALLOW_IPS": "*"}


def test_a_real_process_runs_the_lifespan_hooks_under_a_hostile_environment():
    status, log, returncode = run_real_server(HOSTILE, script=pins_only_script())  # pins only (see above)
    assert status == 200, log
    assert returncode == -signal.SIGTERM, (returncode, log)
    assert sum("Started server process" in line for line in log) == 1, log
    for marker in LIFESPAN_STARTUP + GRACEFUL:
        assert any(marker in line for line in log), (marker, log)
    # Order: startup completes before the request, shutdown completes after "Shutting down".
    index = {marker: next(i for i, line in enumerate(log) if marker in line) for marker in LIFESPAN_STARTUP + GRACEFUL}
    assert index["Application startup complete."] < index["Shutting down"] < index["Application shutdown complete."]


def test_control_a_real_process_without_the_flag_skips_the_lifespan():
    script = pins_only_script()
    unpinned = script.replace(" --lifespan on", "")
    assert unpinned != script
    status, log, _ = run_real_server({"UVICORN_LIFESPAN": "off"}, script=unpinned)
    assert status == 200, log
    for marker in LIFESPAN_STARTUP + ("Waiting for application shutdown.", "Application shutdown complete."):
        assert not any(marker in line for line in log), (marker, log)  # what the pin prevents


# ===========================================================================
# E2B: the runtime environment guard (scripts/check_runtime_env.py)
# ===========================================================================

REFUSAL = "check_runtime_env: refusing to start; unset these environment variables: "
UVICORN_OPTION_VARIABLES = sorted("UVICORN_" + param.name.upper() for param in uvicorn_cli.params)
INERT_VARIABLES = ("WEB_CONCURRENCY", "FORWARDED_ALLOW_IPS")


def load_guard():
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_runtime_env_under_test", GUARD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def guard_argv() -> list:
    """The guard command exactly as the Dockerfile CMD runs it (``python`` resolved from this venv)."""
    _, _, script = dockerfile_cmd()
    guard, _ = split_cmd_script(script)
    tokens = shlex.split(guard)
    assert tokens[0] == "python"
    return [sys.executable, *tokens[1:]]


def run_guard(env_extra, *, extra_pythonpath=None):
    """Run the Dockerfile's guard command in a minimal environment: (exit status, stdout, stderr)."""
    env = {"PATH": "/usr/bin:/bin", "HOME": str(ROOT / "data"), **env_extra}
    if extra_pythonpath:
        env["PYTHONPATH"] = str(extra_pythonpath)
    completed = subprocess.run(guard_argv(), cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    return completed.returncode, completed.stdout, completed.stderr


def test_uvicorn_option_variables_are_discovered_from_uvicorn_itself():
    # Generated from uvicorn's own CLI, so a new uvicorn option is covered automatically.
    assert len(UVICORN_OPTION_VARIABLES) >= 45
    for name in ("UVICORN_RELOAD", "UVICORN_APP_DIR", "UVICORN_LIMIT_MAX_REQUESTS", "UVICORN_ENV_FILE",
                 "UVICORN_UDS", "UVICORN_FD", "UVICORN_ROOT_PATH", "UVICORN_LIMIT_CONCURRENCY", "UVICORN_FACTORY",
                 "UVICORN_WORKERS", "UVICORN_LIFESPAN", "UVICORN_PROXY_HEADERS", "UVICORN_FORWARDED_ALLOW_IPS",
                 "UVICORN_H11_MAX_INCOMPLETE_EVENT_SIZE", "UVICORN_SSL_KEYFILE_PASSWORD"):
        assert name in UVICORN_OPTION_VARIABLES, name


def test_a_clean_environment_starts_silently():
    assert run_guard({}) == (0, "", "")
    assert load_guard().check([]) == (0, [])


@pytest.mark.parametrize("name", UVICORN_OPTION_VARIABLES)
def test_every_uvicorn_option_variable_is_refused(name):
    status, messages = load_guard().check([name, "PATH", "HOME"])
    assert (status, messages) == (78, [REFUSAL + name])


@pytest.mark.parametrize("name", ["UVICORN_FUTURE_OPTION", "UVICORN_", "UVICORN_X", "UVICORN_RELOAD_EVERYTHING"])
def test_unknown_and_future_uvicorn_names_are_refused(name):
    assert run_guard({name: "1"}) == (78, "", REFUSAL + name + "\n")


def test_several_are_refused_in_one_sorted_line():
    hostile = {"UVICORN_ROOT_PATH": "/x", "UVICORN_APP_DIR": "/tmp", "UVICORN_RELOAD": "true", "UVICORN_FD": "3"}
    status, stdout, stderr = run_guard(hostile)
    assert (status, stdout) == (78, "")
    assert stderr == REFUSAL + "UVICORN_APP_DIR, UVICORN_FD, UVICORN_RELOAD, UVICORN_ROOT_PATH\n"


def test_values_are_never_printed():
    sentinels = {name: f"SENTINEL-VALUE-{index:03d}-do-not-print" for index, name in enumerate(UVICORN_OPTION_VARIABLES)}
    sentinels.update({"WEB_CONCURRENCY": "SENTINEL-WEB-CONCURRENCY", "FORWARDED_ALLOW_IPS": "SENTINEL-FORWARDED",
                      "ADMIN_API_KEY": "SENTINEL-ADMIN-KEY"})
    status, stdout, stderr = run_guard(sentinels)
    assert status == 78
    assert "SENTINEL" not in stdout + stderr
    assert stderr == REFUSAL + ", ".join(UVICORN_OPTION_VARIABLES) + "\n"
    # Inert variables alone: warnings name them, never their values.
    status, stdout, stderr = run_guard({"WEB_CONCURRENCY": "SENTINEL-WEB", "FORWARDED_ALLOW_IPS": "SENTINEL-FWD"})
    assert status == 0 and "SENTINEL" not in stdout + stderr


def test_unprintable_names_stay_on_one_escaped_line():
    status, [message] = load_guard().check(["UVICORN_X\nFORGED LOG LINE"])
    assert status == 78 and "\n" not in message and "\\n" in message


@pytest.mark.parametrize("names", [("WEB_CONCURRENCY",), ("FORWARDED_ALLOW_IPS",), INERT_VARIABLES])
def test_inert_variables_only_warn(names):
    status, stdout, stderr = run_guard({name: "4" for name in names})
    assert (status, stdout) == (0, "")
    expected = [f"check_runtime_env: warning: {name} is set but has no effect (the command line pins it)"
                for name in INERT_VARIABLES if name in names]
    assert stderr.splitlines() == expected


@pytest.mark.parametrize("name", ["uvicorn_reload", "Uvicorn_Reload", "uvicorn_app_dir", "web_concurrency",
                                  "forwarded_allow_ips", "XUVICORN_RELOAD", "MY_UVICORN_RELOAD"])
def test_lowercase_and_lookalike_names_are_ignored(name):
    assert run_guard({name: "1"}) == (0, "", "")


@pytest.mark.parametrize("bypass", ["CHECK_RUNTIME_ENV_SKIP", "ALLOW_UNSAFE_UVICORN_ENV", "BLUEPEAK_ALLOW_UVICORN",
                                    "APP_ENV", "SKIP_GUARD"])
def test_there_is_no_bypass(bypass):
    assert run_guard({bypass: "1", "UVICORN_RELOAD": "true"})[0] == 78
    assert run_guard({bypass: "development", "UVICORN_RELOAD": "true"})[0] == 78


def test_the_guard_runs_isolated_with_python_dash_i(tmp_path):
    # A sitecustomize on PYTHONPATH runs in a normal interpreter; python -I ignores PYTHONPATH.
    marker = tmp_path / "sitecustomize-ran"
    (tmp_path / "sitecustomize.py").write_text(
        f"import os\nopen({str(marker)!r}, 'w').close()\nos.environ.pop('UVICORN_RELOAD', None)\n")
    assert run_guard({"UVICORN_RELOAD": "true"}, extra_pythonpath=tmp_path)[0] == 78
    assert not marker.exists()
    # Control: the same planted code does run (and could hide the variable) without -I.
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(tmp_path), "UVICORN_RELOAD": "true"}
    control = subprocess.run([sys.executable, str(GUARD_SCRIPT)], cwd=ROOT, env=env, capture_output=True, timeout=30)
    assert marker.exists() and control.returncode == 0


def test_the_guard_is_standard_library_only_and_reads_nothing_else():
    import ast

    text = GUARD_SCRIPT.read_text()
    tree = ast.parse(text)
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imported == {"os", "sys"}
    code = ast.unparse(tree)  # without docstrings' influence on the checks below
    for forbidden in ("open(", "dotenv", "Settings", "getenv", "environ.get", "environ["):
        assert forbidden not in code, forbidden
    assert "os.environ.keys()" in code  # names only


def test_the_guard_script_is_copied_into_the_image():
    import fnmatch

    patterns = [line.strip() for line in (ROOT / ".dockerignore").read_text().splitlines()
                if line.strip() and not line.startswith("#")]
    for candidate in ("scripts", "scripts/check_runtime_env.py", "check_runtime_env.py"):
        assert not any(fnmatch.fnmatch(candidate, pattern.rstrip("/")) for pattern in patterns if not pattern.startswith("!")), candidate


# ---------------------------------------------------------------------------
# The CMD: guard first, joined by && to exec uvicorn
# ---------------------------------------------------------------------------

def test_the_cmd_runs_the_guard_first_and_uvicorn_only_on_success():
    _, _, script = dockerfile_cmd()
    guard, command = split_cmd_script(script)
    assert guard == "python -I scripts/check_runtime_env.py"
    assert command.startswith(UVICORN_EXEC + "app.main:app ")
    assert script.index("check_runtime_env") < script.index(UVICORN_EXEC)
    assert script.count("check_runtime_env") == 1 and script.count("exec ") == 1


def test_the_existing_pins_are_unchanged_after_the_guard():
    argv = uvicorn_argv()
    for flag, value in (("--workers", "1"), ("--lifespan", "on"), ("--timeout-graceful-shutdown", "8"),
                        ("--host", "0.0.0.0"), ("--port", "8080")):
        assert argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value, flag
    assert argv.count("--no-proxy-headers") == 1
    assert "UVICORN_" not in (ROOT / "Dockerfile").read_text()


# ---------------------------------------------------------------------------
# Real processes with the exact CMD
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hostile", [
    {"UVICORN_RELOAD": "true"},
    {"UVICORN_APP_DIR": "/tmp"},
    {"UVICORN_LIMIT_MAX_REQUESTS": "2"},
    {"UVICORN_ENV_FILE": "/dev/null"},
    {"UVICORN_UDS": "/tmp/bluepeak.sock", "UVICORN_LIMIT_CONCURRENCY": "1"},
    {"UVICORN_FUTURE_OPTION": "x"},
])
def test_a_real_process_is_refused_before_uvicorn_starts(hostile):
    status, log, returncode = run_real_server(hostile, timeout=15)
    assert returncode == 78, (returncode, log)
    assert status is None  # nothing ever listened
    assert log == [REFUSAL + ", ".join(sorted(hostile))], log  # the guard's line only: uvicorn never ran


def test_a_safe_real_process_keeps_its_pid_and_shuts_down_gracefully():
    status, log, returncode, body, pid = run_real_server({}, path="/pid", details=True)
    assert status == 200, log
    assert json.loads(body) == {"pid": pid}  # sh exec'd uvicorn: the launched process serves
    assert returncode == -signal.SIGTERM, (returncode, log)
    assert not any("check_runtime_env" in line for line in log), log
    for marker in LIFESPAN_STARTUP + GRACEFUL:
        assert any(marker in line for line in log), (marker, log)


@pytest.mark.parametrize("attempt", range(10))
def test_the_inert_variables_start_normally_every_time(attempt):
    status, log, returncode, body, pid = run_real_server(
        {"WEB_CONCURRENCY": "4", "FORWARDED_ALLOW_IPS": "*"}, path="/pid", details=True)
    assert status == 200 and json.loads(body) == {"pid": pid}, log
    assert returncode == -signal.SIGTERM, (returncode, log)
    assert not any("Traceback" in line or "Error" in line for line in log), log
    assert [line for line in log if line.startswith("check_runtime_env")] == [
        "check_runtime_env: warning: WEB_CONCURRENCY is set but has no effect (the command line pins it)",
        "check_runtime_env: warning: FORWARDED_ALLOW_IPS is set but has no effect (the command line pins it)",
    ]
    assert sum("Started server process" in line for line in log) == 1, log
    for marker in LIFESPAN_STARTUP + GRACEFUL:
        assert any(marker in line for line in log), (marker, log)


def test_no_child_process_survives_a_timeout():
    # A tree that ignores SIGTERM: the timeout path must kill the whole group.
    process = start_group(["sh", "-c", "trap '' TERM; (trap '' TERM; sleep 60) & sleep 60"],
                          {"PATH": "/usr/bin:/bin"})
    time.sleep(0.3)
    started = time.monotonic()
    stop_group(process, timeout=1.0)
    elapsed = time.monotonic() - started
    # E2B-R1 (audit A3): the escalation path really ran. SIGTERM was ignored for the whole
    # 1 s timeout, and the leader was then killed by SIGKILL, not by the SIGTERM.
    assert 1.0 <= elapsed < 10, elapsed
    assert process.returncode == -signal.SIGKILL, process.returncode
    assert not process_group_alive(process.pid)


# ===========================================================================
# E2B-R1: a planted sitecustomize.py on PYTHONPATH cannot reach the server
# ===========================================================================

def plant_sitecustomize(directory: Path, marker: Path) -> None:
    """Code that, if it ever runs, sets UVICORN_RELOAD inside the interpreter after the guard passed."""
    (directory / "sitecustomize.py").write_text(
        "import os\n"
        "os.environ['UVICORN_RELOAD'] = 'true'\n"
        f"open({str(marker)!r}, 'a').write('ran %d\\n' % os.getpid())\n")


def test_a_planted_sitecustomize_cannot_inject_uvicorn_reload(tmp_path):
    marker = tmp_path / "sitecustomize-ran"
    plant_sitecustomize(tmp_path, marker)
    status, log, returncode, body, pid = run_real_server(
        {"PYTHONPATH": f"{tmp_path}:{ROOT}"}, path="/pid", details=True)
    assert status == 200 and json.loads(body) == {"pid": pid}, log  # one server, the launched process
    assert not marker.exists(), marker.read_text()  # neither the guard nor uvicorn ran the planted code
    assert not any("reloader" in line for line in log), log
    assert returncode == -signal.SIGTERM, (returncode, log)
    for marker_line in LIFESPAN_STARTUP + GRACEFUL:
        assert any(marker_line in line for line in log), (marker_line, log)


def test_control_without_isolation_the_planted_sitecustomize_starts_a_reloader(tmp_path):
    # What E2B-R1 closes: the E2B command ran uvicorn as a plain ``uvicorn`` (no -I).
    _, _, script = dockerfile_cmd()
    unisolated = script.replace(UVICORN_EXEC, "exec uvicorn ")
    assert unisolated != script
    marker = tmp_path / "sitecustomize-ran"
    plant_sitecustomize(tmp_path, marker)
    status, log, _ = run_real_server({"PYTHONPATH": f"{tmp_path}:{ROOT}"}, script=unisolated)
    assert status == 200, log
    assert marker.exists()  # the planted code ran inside uvicorn, after the guard had passed
    assert any("Started reloader process" in line for line in log), log


# ===========================================================================
# E2B-R2: PORT is always exactly one argument
# ===========================================================================

PORT_INJECTIONS = ["{port} --reload", "{port} --limit-max-requests 1", "{port} --root-path /evil",
                   "{port} --app-dir /tmp", "*", "{port} *"]


def shell_arguments_for(port_env) -> list:
    """The arguments sh would pass to uvicorn: the Dockerfile's real uvicorn command, with the
    ``exec python ... -m uvicorn`` prefix replaced by printf. ``port_env`` None leaves PORT unset."""
    _, _, script = dockerfile_cmd()
    command = split_cmd_script(script)[1]
    assert command.startswith(UVICORN_EXEC)
    printed = "printf '%s\\n' " + command[len(UVICORN_EXEC):]
    env = {"PATH": "/usr/bin:/bin"}
    if port_env is not None:
        env["PORT"] = port_env
    completed = subprocess.run(["sh", "-c", printed], cwd=ROOT, env=env, capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.splitlines()


def expected_arguments(port_value: str) -> list:
    return ["app.main:app", "--host", "0.0.0.0", "--port", port_value, "--no-proxy-headers",
            "--workers", "1", "--lifespan", "on", "--timeout-graceful-shutdown", "8"]


def test_the_port_expansion_is_double_quoted_in_the_raw_cmd():
    [line] = [line for line in (ROOT / "Dockerfile").read_text().splitlines() if line.startswith("CMD ")]
    assert '--port \\"${PORT:-8080}\\"' in line  # JSON-escaped quotes in the Dockerfile itself
    _, _, script = dockerfile_cmd()
    assert '--port "${PORT:-8080}"' in script
    assert script.count("$") == 1  # the quoted PORT expansion is the only expansion in the command


@pytest.mark.parametrize("port_env, expected", [(None, "8080"), ("", "8080"), ("9123", "9123"), ("80", "80")])
def test_numeric_and_default_ports_reach_uvicorn_unchanged(port_env, expected):
    assert shell_arguments_for(port_env) == expected_arguments(expected)


@pytest.mark.parametrize("template", PORT_INJECTIONS)
def test_a_hostile_port_stays_one_argument(template):
    value = template.format(port=9123)
    assert shell_arguments_for(value) == expected_arguments(value)  # never split, never glob-expanded


@pytest.mark.parametrize("template", PORT_INJECTIONS)
def test_a_real_process_with_a_hostile_port_fails_closed(template):
    status, log, returncode = run_real_server({}, port_template=template, timeout=15)
    assert returncode == 2, (returncode, log)  # uvicorn's usage error: the whole value is not a port
    assert status is None  # nothing ever listened
    assert not any("Started server process" in line or "reloader" in line or "parent process" in line
                   for line in log), log
    port_value = template.replace("{port}", "")  # the probe port differs per run: match the rest
    assert any("is not a valid integer" in line and port_value.strip() in line for line in log), log


def test_the_glob_is_not_expanded_into_file_names():
    status, log, returncode = run_real_server({}, port_template="*", timeout=15)
    assert returncode == 2 and status is None, log
    assert any("'*' is not a valid integer" in line for line in log), log  # not 'Dockerfile' or another name


def test_a_hostile_port_and_a_uvicorn_variable_are_refused_by_the_guard_first():
    status, log, returncode = run_real_server({"UVICORN_RELOAD": "true"}, port_template="{port} --reload", timeout=15)
    assert returncode == 78 and status is None, log
    assert log == [REFUSAL + "UVICORN_RELOAD"], log


def test_control_an_unquoted_port_injects_a_reloader():
    # What E2B-R2 closes: with ``--port ${PORT:-8080}`` unquoted, the value split into extra options.
    _, _, script = dockerfile_cmd()
    unquoted = script.replace('--port "${PORT:-8080}"', "--port ${PORT:-8080}")
    assert unquoted != script
    status, log, _ = run_real_server({}, script=unquoted, port_template="{port} --reload")
    assert status == 200, log
    assert any("Started reloader process" in line for line in log), log
