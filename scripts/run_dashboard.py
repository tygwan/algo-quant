#!/usr/bin/env python3
"""Run the algo-quant dashboard.

Usage:
    python scripts/run_dashboard.py
    python scripts/run_dashboard.py --port 8080
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui.dash_app import run_dashboard


def main():
    parser = argparse.ArgumentParser(description="Run algo-quant dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Port to run on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug mode")

    args = parser.parse_args()

    debug = True if args.debug else (False if args.no_debug else True)

    print(f"""
╔══════════════════════════════════════════════════════════╗
║           algo-quant Dashboard                           ║
╠══════════════════════════════════════════════════════════╣
║  Local:   http://localhost:{args.port}                        ║
║  Network: http://0.0.0.0:{args.port}                          ║
╠══════════════════════════════════════════════════════════╣
║  Pages:                                                  ║
║    /                  - Dashboard overview               ║
║    /live-analyzer     - Enter tickers for analysis       ║
║    /data-explorer     - Explore market data              ║
║    /factor-analysis   - Fama-French factor analysis      ║
║    /regime-monitor    - Market regime classification     ║
║    /backtest          - Strategy backtesting             ║
║    /portfolio         - Portfolio management             ║
╚══════════════════════════════════════════════════════════╝
    """)

    run_dashboard(debug=debug, port=args.port)


if __name__ == "__main__":
    main()
