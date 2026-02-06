"""Minimal local .env loader.

This avoids an external dependency while supporting common KEY=VALUE files
for local development and secret management.
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

_default_env_loaded = False
_load_lock = Lock()


def load_local_env(env_path: str | Path = ".env", override: bool = False) -> bool:
    """Load variables from a .env-style file into process environment.

    Args:
        env_path: File path to load. Defaults to project-local ".env".
        override: If True, overwrite existing environment variables.

    Returns:
        True if at least one variable was set from file, else False.
    """
    global _default_env_loaded

    path = Path(env_path)
    is_default = path == Path(".env")

    with _load_lock:
        if is_default and _default_env_loaded:
            return False

        if not path.exists() or not path.is_file():
            return False

        loaded = False
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("export "):
                line = line[7:].strip()

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if not key:
                continue

            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            elif " #" in value:
                value = value.split(" #", 1)[0].rstrip()

            if override or key not in os.environ:
                os.environ[key] = value
                loaded = True

        if is_default:
            _default_env_loaded = True

        return loaded
