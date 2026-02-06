"""Tests for local .env loader."""

from __future__ import annotations

import os

import src.env as env_module


def _reset_default_env_cache() -> None:
    env_module._default_env_loaded = False


def test_load_local_env_sets_new_values_without_overriding_existing(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    _reset_default_env_cache()

    (tmp_path / ".env").write_text(
        "NEW_KEY=new_value\nEXISTING_KEY=file_value\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EXISTING_KEY", "process_value")
    monkeypatch.delenv("NEW_KEY", raising=False)

    loaded = env_module.load_local_env()

    assert loaded is True
    assert os.getenv("NEW_KEY") == "new_value"
    assert os.getenv("EXISTING_KEY") == "process_value"


def test_load_local_env_parses_export_quotes_and_inline_comments(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    _reset_default_env_cache()

    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "export A=1",
                "B='hello world'",
                'C="quoted value"',
                "D=plain_value # trailing comment",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for key in ("A", "B", "C", "D"):
        monkeypatch.delenv(key, raising=False)

    loaded = env_module.load_local_env()

    assert loaded is True
    assert os.getenv("A") == "1"
    assert os.getenv("B") == "hello world"
    assert os.getenv("C") == "quoted value"
    assert os.getenv("D") == "plain_value"


def test_missing_default_env_can_be_loaded_after_file_is_created(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    _reset_default_env_cache()
    monkeypatch.delenv("LATE_KEY", raising=False)

    first = env_module.load_local_env()
    assert first is False

    (tmp_path / ".env").write_text("LATE_KEY=arrived\n", encoding="utf-8")
    second = env_module.load_local_env()

    assert second is True
    assert os.getenv("LATE_KEY") == "arrived"
