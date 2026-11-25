"""Container start-up guard: refuse to start when any UVICORN_* variable is set (Phase E2B).

Run by the Dockerfile CMD as ``python -I scripts/check_runtime_env.py && exec uvicorn ...``.

uvicorn reads a UVICORN_<OPTION> environment variable for every option its
command line does not set. Only a few options are pinned on the command line;
the rest (reload, app directory, request limits, sockets, env files, ...) could
change how the server runs. So every name starting with UVICORN_ is refused,
including unknown and future ones, before uvicorn starts.

* Only variable NAMES are inspected and printed, never values (a value may be
  a secret).
* Nothing else is read: no .env file, no application settings, no APP_ENV.
* There is deliberately no bypass: unset the variable instead.
* WEB_CONCURRENCY and FORWARDED_ALLOW_IPS have no effect with the pinned
  command line; they only produce a warning.

Standard library only; run with ``python -I`` (ignores PYTHON* variables and
user site-packages). Exit status 0 to start, 78 (EX_CONFIG) to refuse.
"""

import os
import sys

REFUSED_PREFIX = "UVICORN_"
INERT_NAMES = ("WEB_CONCURRENCY", "FORWARDED_ALLOW_IPS")
EX_CONFIG = 78


def _shown(name: str) -> str:
    """The name as printed: escaped if it contains unprintable characters (one line per message)."""
    return name if name.isprintable() else ascii(name)


def check(names) -> tuple[int, list[str]]:
    """(exit status, messages) for a collection of environment variable names."""
    names = set(names)
    refused = sorted(name for name in names if name.startswith(REFUSED_PREFIX))
    if refused:
        return EX_CONFIG, [
            "check_runtime_env: refusing to start; unset these environment variables: "
            + ", ".join(_shown(name) for name in refused)
        ]
    warnings = [
        f"check_runtime_env: warning: {name} is set but has no effect (the command line pins it)"
        for name in INERT_NAMES
        if name in names
    ]
    return 0, warnings


def main() -> int:
    status, messages = check(os.environ.keys())
    for message in messages:
        print(message, file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
