"""Runtime profile loader for environment-specific defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """Resolved runtime defaults for dashboard/pipeline/demo commands."""

    name: str = "dev"
    source: str = ""
    env_file: str = ".env"

    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8050
    dashboard_debug: bool = True

    refresh_interval_ms: int = 3000

    pipeline_top: int = 20
    pipeline_start: str = "2020-01-01"

    paper_symbol: str = "AAPL"
    paper_steps: int = 120


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_profile_path(profile: str, profile_dir: str | Path | None = None) -> Path:
    profile_name = profile.strip()
    if not profile_name:
        profile_name = "dev"

    explicit = Path(profile_name)
    if explicit.exists() and explicit.is_file():
        return explicit.resolve()

    if explicit.suffix in {".yaml", ".yml"}:
        filename = explicit.name
    else:
        filename = f"{profile_name}.yaml"

    candidates: list[Path] = []
    if profile_dir:
        candidates.append(Path(profile_dir) / filename)

    # Prefer caller's working tree, then fallback to package-relative root.
    candidates.append(Path.cwd() / "config" / "profiles" / filename)
    project_root = Path(__file__).resolve().parents[2]
    candidates.append(project_root / "config" / "profiles" / filename)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    search_paths = ", ".join(str(path.parent) for path in candidates)
    raise FileNotFoundError(
        f"Runtime profile '{profile}' not found. Searched: {search_paths}"
    )


def load_runtime_profile(
    profile: str | None = None,
    profile_dir: str | Path | None = None,
) -> RuntimeProfile:
    """Load runtime profile from YAML.

    Args:
        profile: Profile name (e.g., dev/paper/live) or direct YAML file path.
            If omitted, uses AQ_PROFILE env var, then falls back to "dev".
        profile_dir: Optional directory override for profile lookup.

    Returns:
        RuntimeProfile with parsed defaults.
    """

    selected = (profile or os.getenv("AQ_PROFILE") or "dev").strip()
    path = _resolve_profile_path(selected, profile_dir=profile_dir)

    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid runtime profile format: {path}")

    dashboard = loaded.get("dashboard", {}) or {}
    realtime = loaded.get("realtime", {}) or {}
    pipeline = loaded.get("pipeline", {}) or {}
    paper_demo = loaded.get("paper_demo", {}) or {}

    return RuntimeProfile(
        name=str(loaded.get("name") or path.stem),
        source=str(path),
        env_file=str(loaded.get("env_file") or ".env"),
        dashboard_host=str(dashboard.get("host") or "127.0.0.1"),
        dashboard_port=_to_int(dashboard.get("port"), 8050),
        dashboard_debug=_to_bool(dashboard.get("debug"), True),
        refresh_interval_ms=_to_int(realtime.get("refresh_interval_ms"), 3000),
        pipeline_top=_to_int(pipeline.get("top"), 20),
        pipeline_start=str(pipeline.get("start") or "2020-01-01"),
        paper_symbol=str(paper_demo.get("symbol") or "AAPL").upper(),
        paper_steps=_to_int(paper_demo.get("steps"), 120),
    )
