#!/usr/bin/env python3
"""Run the algo-quant dashboard with runtime profile defaults."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from src.config import load_runtime_profile
from src.env import load_local_env
from src.ui.dash_app import run_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Run algo-quant dashboard")
    parser.add_argument(
        "--profile",
        default=None,
        help="Runtime profile name or YAML path (default: AQ_PROFILE or dev)",
    )
    parser.add_argument(
        "--refresh-ms",
        type=int,
        default=None,
        help="Live analyzer refresh interval in milliseconds",
    )
    parser.add_argument("--host", default=None, help="Host override")
    parser.add_argument("--port", type=int, default=None, help="Port override")
    debug_group = parser.add_mutually_exclusive_group()
    debug_group.add_argument("--debug", dest="debug", action="store_true", help="Enable debug mode")
    debug_group.add_argument("--no-debug", dest="debug", action="store_false", help="Disable debug mode")
    parser.set_defaults(debug=None)

    args = parser.parse_args()

    profile = load_runtime_profile(args.profile)
    load_local_env(profile.env_file)

    host = args.host or profile.dashboard_host
    port = int(args.port) if args.port is not None else int(profile.dashboard_port)
    debug = profile.dashboard_debug if args.debug is None else bool(args.debug)
    refresh_ms = (
        int(args.refresh_ms)
        if args.refresh_ms is not None
        else int(profile.refresh_interval_ms)
    )

    # Propagate runtime refresh configuration into Dash layout callbacks.
    os.environ["AQ_REFRESH_INTERVAL_MS"] = str(refresh_ms)

    print(
        f"""
╔══════════════════════════════════════════════════════════╗
║           algo-quant Dashboard                           ║
╠══════════════════════════════════════════════════════════╣
║  Profile: {profile.name:<47}║
║  Local:   http://localhost:{port:<27}║
║  Host:    http://{host}:{port:<26}║
║  Debug:   {str(debug):<47}║
║  Refresh: {str(refresh_ms) + ' ms':<47}║
╠══════════════════════════════════════════════════════════╣
║  Pages:                                                  ║
║    /                  - Dashboard overview               ║
║    /live-analyzer     - Enter tickers for analysis      ║
║    /data-explorer     - Explore market data             ║
║    /factor-analysis   - Fama-French factor analysis     ║
║    /regime-monitor    - Market regime classification    ║
║    /backtest          - Strategy backtesting            ║
║    /portfolio         - Portfolio management            ║
╚══════════════════════════════════════════════════════════╝
    """
    )

    run_dashboard(debug=debug, port=port, host=host)


if __name__ == "__main__":
    main()
