"""H3-B: a minimal ASGI app that reports the runtime Uvicorn chose.

Served by ``tests/test_h3_auto_runtime.py`` in a separate process through
Uvicorn's real CLI (``python -m uvicorn tests.h3_probe_app:app``), so the
event loop and HTTP protocol are the ones Uvicorn itself selects
(``Server.run()`` → ``Config.setup_event_loop()`` and ``Config.load()``).

It is isolated from the application on purpose: it imports nothing from
``app`` (no settings, no ``.env``, no databases, no providers), opens no
connections, and only answers the request it is given.

Per request it reports:

* ``loop``: the running event loop's class (module and name);
* ``policy``: the module of the global event-loop policy, which
  ``uvicorn.loops.uvloop.uvloop_setup`` replaces when it selects uvloop;
* ``cycle_file``: the file of the coroutine serving this request, Uvicorn's
  ``RequestResponseCycle.run_asgi`` in ``h11_impl.py`` or
  ``httptools_impl.py``, so the HTTP protocol actually used;
* ``app_modules``: any ``app`` modules loaded in this process (expected: none).
"""

import asyncio
import json
import os
import sys


def _runtime() -> dict:
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    coro = task.get_coro() if task is not None else None
    return {
        "loop": f"{type(loop).__module__}.{type(loop).__qualname__}",
        "policy": type(asyncio.get_event_loop_policy()).__module__,
        "cycle_file": os.path.basename(coro.cr_code.co_filename) if coro is not None else None,
        "cycle_name": coro.cr_code.co_qualname if coro is not None else None,
        "app_modules": sorted(name for name in sys.modules if name == "app" or name.startswith("app.")),
        "pid": os.getpid(),
    }


async def app(scope, receive, send):
    if scope["type"] != "http":  # run with --lifespan off; nothing else is expected
        return
    body = json.dumps(_runtime()).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
