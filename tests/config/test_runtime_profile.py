"""Tests for runtime profile loading."""

from __future__ import annotations

from pathlib import Path

from src.config import load_runtime_profile


def test_load_runtime_profile_from_default_directory() -> None:
    profile = load_runtime_profile("dev")

    assert profile.name == "dev"
    assert profile.dashboard_port == 8050
    assert profile.env_file == ".env"


def test_load_runtime_profile_from_explicit_file(tmp_path: Path) -> None:
    profile_file = tmp_path / "custom.yaml"
    profile_file.write_text(
        """
name: custom
env_file: .env.custom
dashboard:
  host: 0.0.0.0
  port: 9999
  debug: false
realtime:
  refresh_interval_ms: 777
pipeline:
  top: 12
  start: "2024-01-01"
paper_demo:
  symbol: tsla
  steps: 42
""".strip(),
        encoding="utf-8",
    )

    profile = load_runtime_profile(str(profile_file))

    assert profile.name == "custom"
    assert profile.dashboard_host == "0.0.0.0"
    assert profile.dashboard_port == 9999
    assert profile.dashboard_debug is False
    assert profile.refresh_interval_ms == 777
    assert profile.pipeline_top == 12
    assert profile.pipeline_start == "2024-01-01"
    assert profile.paper_symbol == "TSLA"
    assert profile.paper_steps == 42
