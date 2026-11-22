"""Security S1-5 Phase D2: the container's uvicorn command trusts no forwarded headers.

The Dockerfile's CMD is parsed exactly as uvicorn's own command line parses
it, turned into the effective ``uvicorn.Config`` and loaded (offline, no
server, no network). With ``--no-proxy-headers``, X-Forwarded-For and
X-Forwarded-Proto can never replace the socket peer or the scheme the app
sees, whatever FORWARDED_ALLOW_IPS says and even when the peer is loopback
(which uvicorn trusts by default). The only trusted client-IP input stays the
verified ACCESS_CLIENT_IP_HEADER (app/services/access.py).
"""

import asyncio
import json
import re
import shlex
from pathlib import Path

import pytest
import uvicorn
from uvicorn.main import main as uvicorn_cli
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

ROOT = Path(__file__).resolve().parents[1]
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


def uvicorn_argv(port: str = "8080") -> list:
    """The uvicorn arguments, with ${PORT:-8080} expanded as sh would for PORT=``port``."""
    shell, flag, script = dockerfile_cmd()
    assert (shell, flag) == ("sh", "-c")
    assert script.startswith("exec uvicorn "), script  # exec: uvicorn is PID 1's process and receives SIGTERM
    tokens = shlex.split(re.sub(r"\$\{PORT:-8080\}", port, script))
    assert tokens[:2] == ["exec", "uvicorn"]
    return tokens[2:]


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
    assert params["workers"] is None  # one process, as before (in-memory limits assume it)
    assert params["timeout_graceful_shutdown"] == 8
    assert params["reload"] is False


def test_the_port_still_comes_from_the_environment():
    assert effective_params(port="9123")["port"] == 9123


def test_no_flag_reenables_or_widens_proxy_trust():
    argv = uvicorn_argv()
    assert "--no-proxy-headers" in argv
    assert "--proxy-headers" not in argv
    assert not any(arg.startswith("--forwarded-allow-ips") for arg in argv)
    assert not any(arg.startswith(("--workers", "--reload", "--root-path")) for arg in argv)


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
