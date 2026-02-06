"""Tests for unified CLI entrypoint."""

from __future__ import annotations

import src.cli as cli


def test_parser_accepts_subcommands() -> None:
    parser = cli.build_parser()
    args, passthrough = parser.parse_known_args(["dashboard", "--profile", "live"])

    assert args.command == "dashboard"
    assert passthrough == ["--profile", "live"]


def test_main_delegates_to_runner(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_runner(command: str, passthrough: list[str]) -> int:
        observed["command"] = command
        observed["passthrough"] = passthrough
        return 0

    monkeypatch.setattr(cli, "_run_script", fake_runner)

    code = cli.main(["pipeline", "--profile", "paper", "--top", "10"])

    assert code == 0
    assert observed["command"] == "pipeline"
    assert observed["passthrough"] == ["--profile", "paper", "--top", "10"]
