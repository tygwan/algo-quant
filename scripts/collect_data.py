#!/usr/bin/env python3
"""Data collection CLI script.

Usage:
    python scripts/collect_data.py --help
    python scripts/collect_data.py prices --symbols AAPL MSFT GOOGL
    python scripts/collect_data.py macro
    python scripts/collect_data.py all
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataCollector, CollectionConfig, SP500_TOP_100

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_api_keys(config_path: str = "config/api_keys.yaml", required: bool = True) -> dict:
    """Load API keys from config file.

    Args:
        config_path: Path to API keys config file
        required: If True, exit if file not found. If False, return empty dict.
    """
    path = Path(config_path)

    if not path.exists():
        if required:
            logger.error(f"API keys file not found: {config_path}")
            logger.error("Please copy config/api_keys.yaml.example to config/api_keys.yaml and add your keys")
            sys.exit(1)
        else:
            logger.warning(f"API keys file not found: {config_path}. Using free data sources only.")
            return {}

    with open(path) as f:
        config = yaml.safe_load(f)

    return config or {}


def cmd_prices(args, collector: DataCollector):
    """Collect stock prices."""
    symbols = args.symbols if args.symbols else None

    if args.top:
        symbols = SP500_TOP_100[:args.top]
        logger.info(f"Using top {args.top} S&P 500 stocks")

    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None
    source = args.source if hasattr(args, 'source') else "auto"

    prices = collector.collect_stock_prices(symbols=symbols, start=start, end=end, source=source)

    logger.info(f"Collected prices for {len(prices)} stocks")

    return prices


def cmd_index(args, collector: DataCollector):
    """Collect market index data using Yahoo Finance (free)."""
    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None
    indexes = args.indexes if args.indexes else None

    index_data = collector.collect_index_data(indexes=indexes, start=start, end=end)
    logger.info(f"Collected {len(index_data)} market indexes")

    return index_data


def cmd_sectors(args, collector: DataCollector):
    """Collect sector ETF data using Yahoo Finance (free)."""
    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None

    sector_data = collector.collect_sector_etfs(start=start, end=end)
    logger.info(f"Collected {len(sector_data)} sector ETFs")

    return sector_data


def cmd_macro(args, collector: DataCollector):
    """Collect macro indicators."""
    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None

    indicators = collector.collect_macro_indicators(start=start, end=end)

    logger.info(f"Collected {len(indicators)} macro indicators")

    # Also create aligned regime indicators
    if indicators:
        regime_df = collector.collect_regime_indicators(start=start, end=end)
        logger.info(f"Created regime indicators DataFrame: {regime_df.shape}")

    return indicators


def cmd_financials(args, collector: DataCollector):
    """Collect financial statements."""
    symbols = args.symbols if args.symbols else None

    if args.top:
        symbols = SP500_TOP_100[:args.top]

    financials = collector.collect_financials(symbols=symbols)

    logger.info(f"Collected financials for {len(financials)} stocks")

    return financials


def cmd_metrics(args, collector: DataCollector):
    """Collect key metrics."""
    symbols = args.symbols if args.symbols else None

    if args.top:
        symbols = SP500_TOP_100[:args.top]

    metrics = collector.collect_key_metrics(symbols=symbols)

    logger.info(f"Collected metrics for {len(metrics)} stocks")

    return metrics


def cmd_vix(args, collector: DataCollector):
    """Collect VIX data."""
    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None

    vix = collector.collect_vix_data(start=start, end=end)
    logger.info(f"Collected VIX data: {len(vix)} rows")

    return vix


def cmd_factors(args, collector: DataCollector):
    """Collect Fama-French factors."""
    frequency = args.frequency or "daily"
    include_mom = not args.no_momentum

    factors = collector.collect_ff_factors(frequency=frequency, include_momentum=include_mom)
    logger.info(f"Collected FF factors: {len(factors)} rows")

    return factors


def cmd_calendar(args, collector: DataCollector):
    """Collect economic calendar."""
    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None

    calendar = collector.collect_economic_calendar(start=start, end=end)
    logger.info(f"Collected economic calendar: {len(calendar)} events")

    return calendar


def cmd_all(args, collector: DataCollector):
    """Collect all data."""
    results = collector.collect_all()

    print("\n=== Collection Summary ===")
    print(f"Timestamp: {results['timestamp']}")
    print(f"Date range: {results['config']['start_date']} to {results['config']['end_date']}")
    print(f"Prices: {results['prices'].get('count', 0)} stocks")
    print(f"Macro: {results['macro'].get('count', 0)} indicators")
    print(f"VIX: {results['vix'].get('count', 0)} rows")
    print(f"FF Factors: {results['factors'].get('count', 0)} rows")
    print(f"Calendar: {results['calendar'].get('count', 0)} events")
    print(f"Financials: {results['financials'].get('count', 0)} stocks")
    print(f"Metrics: {results['metrics'].get('count', 0)} stocks")

    return results


def cmd_status(args, collector: DataCollector):
    """Show collection status."""
    status = collector.get_collection_status()

    print("\n=== Data Collection Status ===")
    print(f"Prices: {status['prices']['count']} stocks")
    if status['prices']['symbols']:
        print(f"  Sample: {', '.join(status['prices']['symbols'][:5])}...")

    print(f"Financials: {status['financials']['count']} files")
    print(f"Metrics: {status['metrics']['count']} files")
    print(f"Macro: {status['macro']['count']} indicators")
    if status['macro']['indicators']:
        print(f"  Indicators: {', '.join(status['macro']['indicators'])}")

    return status


def main():
    parser = argparse.ArgumentParser(
        description="Data Collection CLI for algo-quant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global options
    parser.add_argument(
        "--config", default="config/api_keys.yaml",
        help="Path to API keys config file"
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Data storage directory"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # prices command
    prices_parser = subparsers.add_parser("prices", help="Collect stock prices")
    prices_parser.add_argument("--symbols", nargs="+", help="Stock symbols to collect")
    prices_parser.add_argument("--top", type=int, help="Collect top N S&P 500 stocks")
    prices_parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    prices_parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    prices_parser.add_argument("--source", choices=["auto", "fmp", "yfinance"], default="auto",
                               help="Data source (auto uses yfinance if fmp unavailable)")

    # macro command
    macro_parser = subparsers.add_parser("macro", help="Collect macro indicators")
    macro_parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    macro_parser.add_argument("--end", help="End date (YYYY-MM-DD)")

    # financials command
    fin_parser = subparsers.add_parser("financials", help="Collect financial statements")
    fin_parser.add_argument("--symbols", nargs="+", help="Stock symbols")
    fin_parser.add_argument("--top", type=int, help="Collect top N S&P 500 stocks")

    # metrics command
    metrics_parser = subparsers.add_parser("metrics", help="Collect key metrics")
    metrics_parser.add_argument("--symbols", nargs="+", help="Stock symbols")
    metrics_parser.add_argument("--top", type=int, help="Collect top N S&P 500 stocks")

    # vix command
    vix_parser = subparsers.add_parser("vix", help="Collect VIX volatility data")
    vix_parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    vix_parser.add_argument("--end", help="End date (YYYY-MM-DD)")

    # factors command
    factors_parser = subparsers.add_parser("factors", help="Collect Fama-French factors")
    factors_parser.add_argument("--frequency", choices=["daily", "monthly"], default="daily")
    factors_parser.add_argument("--no-momentum", action="store_true", help="Exclude momentum factor")

    # calendar command
    calendar_parser = subparsers.add_parser("calendar", help="Collect economic calendar")
    calendar_parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    calendar_parser.add_argument("--end", help="End date (YYYY-MM-DD)")

    # index command (Yahoo Finance - free)
    index_parser = subparsers.add_parser("index", help="Collect market index data (free, no API key)")
    index_parser.add_argument("--indexes", nargs="+", choices=["sp500", "dow", "nasdaq", "russell2000", "vix"],
                              help="Index names to collect (default: all)")
    index_parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    index_parser.add_argument("--end", help="End date (YYYY-MM-DD)")

    # sectors command (Yahoo Finance - free)
    sectors_parser = subparsers.add_parser("sectors", help="Collect sector ETF data (free, no API key)")
    sectors_parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    sectors_parser.add_argument("--end", help="End date (YYYY-MM-DD)")

    # all command
    all_parser = subparsers.add_parser("all", help="Collect all data")

    # status command
    status_parser = subparsers.add_parser("status", help="Show collection status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Commands that work without API keys (free data sources)
    free_commands = {"index", "sectors", "vix", "factors", "status"}
    api_keys_required = args.command not in free_commands

    # Load API keys (not required for free commands)
    api_keys = load_api_keys(args.config, required=api_keys_required)

    # Create collector config
    config = CollectionConfig(
        data_dir=Path(args.data_dir),
    )

    # Create collector
    collector = DataCollector(
        fmp_key=api_keys.get("fmp", {}).get("api_key"),
        fred_key=api_keys.get("fred", {}).get("api_key"),
        finnhub_key=api_keys.get("finnhub", {}).get("api_key"),
        config=config,
    )

    # Execute command
    commands = {
        "prices": cmd_prices,
        "macro": cmd_macro,
        "financials": cmd_financials,
        "metrics": cmd_metrics,
        "vix": cmd_vix,
        "factors": cmd_factors,
        "calendar": cmd_calendar,
        "index": cmd_index,
        "sectors": cmd_sectors,
        "all": cmd_all,
        "status": cmd_status,
    }

    cmd_func = commands.get(args.command)
    if cmd_func:
        cmd_func(args, collector)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
