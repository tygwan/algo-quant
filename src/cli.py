"""Unified command-line entrypoint for algo-quant workflows."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


_SCRIPT_MAP = {
    "dashboard": "scripts/run_dashboard.py",
    "pipeline": "scripts/run_pipeline.py",
    "paper-demo": "scripts/demo_paper_trading.py",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_script(command: str, passthrough: list[str]) -> int:
    script_rel = _SCRIPT_MAP[command]
    script_path = _repo_root() / script_rel

    if not script_path.exists():
        print(f"Script not found: {script_path}", file=sys.stderr)
        return 1

    cmd = [sys.executable, str(script_path), *passthrough]
    completed = subprocess.run(cmd, cwd=str(_repo_root()))
    return int(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aq",
        description="algo-quant unified CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("dashboard", help="Run Dash dashboard")
    subparsers.add_parser("pipeline", help="Run quant pipeline")
    subparsers.add_parser("paper-demo", help="Run offline paper-trading demo")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args(argv)
    if passthrough[:1] == ["--"]:
        passthrough = passthrough[1:]
    return _run_script(args.command, passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
