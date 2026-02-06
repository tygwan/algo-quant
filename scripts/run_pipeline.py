#!/usr/bin/env python3
"""End-to-end pipeline for quantitative investing system.

This script runs the complete workflow:
1. Data Collection (free sources)
2. Factor Analysis (FF5)
3. Regime Classification (HMM)
4. Strategy Execution
5. Backtesting
6. Results Display

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --symbols AAPL MSFT GOOGL
    python scripts/run_pipeline.py --top 20 --start 2020-01-01
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_runtime_profile
from src.data.yfinance_client import YFinanceClient
from src.data.collector import DataCollector, CollectionConfig, SP500_TOP_100
from src.env import load_local_env
from src.factors.ff5 import FamaFrench5
from src.factors.ff_data import FamaFrenchDataLoader
from src.regime.hmm_classifier import HMMClassifier
from src.regime.rule_based import RuleBasedClassifier
from src.strategy.factor_strategy import FactorStrategy, FactorConfig, FactorWeightMethod
from src.backtest.engine import BacktestEngine, BacktestConfig, VectorizedBacktester
from src.backtest.metrics import calculate_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def collect_data(symbols: list[str], start_date: str, end_date: str) -> dict:
    """Collect all required data.

    Args:
        symbols: Stock symbols to collect
        start_date: Start date
        end_date: End date

    Returns:
        Dictionary with collected data
    """
    logger.info("=" * 50)
    logger.info("STEP 1: Data Collection")
    logger.info("=" * 50)

    data = {}

    # 1. Stock prices (Yahoo Finance - free)
    logger.info(f"Collecting prices for {len(symbols)} stocks...")
    yf_client = YFinanceClient()

    prices_dict = {}
    for symbol in symbols:
        try:
            df = yf_client.get_historical_prices(symbol, start=start_date, end=end_date)
            if not df.empty and 'close' in df.columns:
                prices_dict[symbol] = df['close']
        except Exception as e:
            logger.warning(f"Failed to get {symbol}: {e}")

    if prices_dict:
        data['prices'] = pd.DataFrame(prices_dict)
        data['returns'] = data['prices'].pct_change().dropna()
        logger.info(f"Collected prices: {data['prices'].shape}")

    # 2. Market index (S&P 500)
    logger.info("Collecting S&P 500 index...")
    try:
        sp500 = yf_client.get_index_data('sp500', start=start_date, end=end_date)
        if not sp500.empty:
            data['market'] = sp500['close']
            data['market_returns'] = data['market'].pct_change().dropna()
            logger.info(f"Collected market index: {len(data['market'])} rows")
    except Exception as e:
        logger.warning(f"Failed to get S&P 500: {e}")

    # 3. Fama-French factors (free)
    logger.info("Collecting Fama-French factors...")
    try:
        ff_loader = FamaFrenchDataLoader()
        ff_factors = ff_loader.load_factors_with_momentum(
            num_factors=5,
            frequency='daily',
            start_date=start_date,
            end_date=end_date,
        )
        if not ff_factors.empty:
            data['ff_factors'] = ff_factors
            logger.info(f"Collected FF factors: {ff_factors.shape}, columns: {list(ff_factors.columns)}")
    except Exception as e:
        logger.warning(f"Failed to get FF factors: {e}")

    # 4. VIX (free via yfinance)
    logger.info("Collecting VIX...")
    try:
        vix = yf_client.get_index_data('vix', start=start_date, end=end_date)
        if not vix.empty:
            data['vix'] = vix['close']
            logger.info(f"Collected VIX: {len(data['vix'])} rows")
    except Exception as e:
        logger.warning(f"Failed to get VIX: {e}")

    return data


def run_factor_analysis(data: dict) -> dict:
    """Run factor analysis on collected data.

    Args:
        data: Collected data dictionary

    Returns:
        Factor analysis results
    """
    logger.info("=" * 50)
    logger.info("STEP 2: Factor Analysis")
    logger.info("=" * 50)

    results = {}

    if 'returns' not in data or 'ff_factors' not in data:
        logger.warning("Missing data for factor analysis")
        return results

    returns = data['returns']
    ff_factors = data['ff_factors']

    # Run FF5 analysis for each stock
    ff5_model = FamaFrench5()
    factor_results = {}

    for symbol in returns.columns[:10]:  # Limit for demo
        try:
            stock_returns = returns[symbol].dropna()
            ff5_model.fit(stock_returns, ff_factors)
            result = ff5_model.result

            factor_results[symbol] = {
                'alpha': result.alpha,
                'alpha_pvalue': result.p_values.get('const', 1.0),
                'market_beta': result.loadings.get('Mkt-RF', 0),
                'smb': result.loadings.get('SMB', 0),
                'hml': result.loadings.get('HML', 0),
                'rmw': result.loadings.get('RMW', 0),
                'cma': result.loadings.get('CMA', 0),
                'r_squared': result.r_squared,
            }

            logger.info(f"{symbol}: alpha={result.alpha:.4f}, R²={result.r_squared:.3f}, "
                       f"β={result.loadings.get('Mkt-RF', 0):.2f}")
        except Exception as e:
            logger.warning(f"FF5 failed for {symbol}: {e}")

    if factor_results:
        results['ff5'] = pd.DataFrame(factor_results).T
        logger.info(f"\nFactor analysis completed for {len(factor_results)} stocks")

    return results


def classify_regimes(data: dict) -> dict:
    """Classify market regimes.

    Args:
        data: Collected data dictionary

    Returns:
        Regime classification results
    """
    logger.info("=" * 50)
    logger.info("STEP 3: Regime Classification")
    logger.info("=" * 50)

    results = {}

    # Create indicators for regime classification
    indicators = pd.DataFrame()

    if 'market_returns' in data:
        indicators['market_return'] = data['market_returns'].rolling(20).mean()
        indicators['market_vol'] = data['market_returns'].rolling(20).std()

    if 'vix' in data:
        indicators['vix'] = data['vix']
        indicators['vix_ma'] = data['vix'].rolling(20).mean()

    if indicators.empty:
        logger.warning("No indicators for regime classification")
        return results

    indicators = indicators.dropna()

    # Rule-based classification
    logger.info("Running rule-based classification...")
    rule_classifier = RuleBasedClassifier()
    rule_result = rule_classifier.classify(indicators)
    logger.info(f"Current regime (rule-based): {rule_result.regime.value} "
               f"(confidence: {rule_result.confidence:.2f})")
    results['rule_based'] = rule_result

    # HMM classification (if enough data)
    if len(indicators) >= 100:
        logger.info("Running HMM classification...")
        try:
            hmm_classifier = HMMClassifier(n_regimes=4)
            hmm_classifier.fit(indicators)

            history = hmm_classifier.classify_history(indicators)
            results['hmm_history'] = history

            # Current regime
            hmm_result = hmm_classifier.classify(indicators)
            logger.info(f"Current regime (HMM): {hmm_result.regime.value} "
                       f"(confidence: {hmm_result.confidence:.2f})")
            results['hmm'] = hmm_result

            # Transition matrix
            trans_matrix = hmm_classifier.get_transition_matrix()
            logger.info(f"\nRegime transition matrix:\n{trans_matrix.round(2)}")
            results['transition_matrix'] = trans_matrix

        except Exception as e:
            logger.warning(f"HMM classification failed: {e}")

    return results


def build_strategy(factor_results: dict, regime_results: dict) -> FactorStrategy:
    """Build factor-based strategy.

    Args:
        factor_results: Factor analysis results
        regime_results: Regime classification results

    Returns:
        Configured strategy
    """
    logger.info("=" * 50)
    logger.info("STEP 4: Strategy Construction")
    logger.info("=" * 50)

    # Define factors with regime-aware weights
    current_regime = regime_results.get('rule_based', {})
    regime_name = current_regime.regime.value if hasattr(current_regime, 'regime') else 'unknown'

    # Regime-based factor weights
    regime_weights = {
        'expansion': {'momentum': 1.5, 'value': 0.5, 'quality': 1.0},
        'contraction': {'momentum': 0.5, 'value': 1.5, 'quality': 1.5},
        'peak': {'momentum': 0.3, 'value': 1.0, 'quality': 1.5},
        'trough': {'momentum': 1.5, 'value': 1.0, 'quality': 0.5},
    }

    weights = regime_weights.get(regime_name, {'momentum': 1.0, 'value': 1.0, 'quality': 1.0})
    logger.info(f"Regime '{regime_name}' -> Factor weights: {weights}")

    # Create strategy
    factors = [
        FactorConfig(name='momentum', weight=weights['momentum'], lookback=12),
        FactorConfig(name='value', weight=weights['value']),
        FactorConfig(name='quality', weight=weights['quality']),
    ]

    strategy = FactorStrategy(
        factors=factors,
        weight_method=FactorWeightMethod.RANK,
        long_only=True,
        top_n=10,
    )

    logger.info(f"Strategy created: top {strategy.top_n} stocks, {len(factors)} factors")

    return strategy


def run_backtest(data: dict, strategy: FactorStrategy) -> dict:
    """Run backtest on the strategy.

    Args:
        data: Collected data
        strategy: Strategy to test

    Returns:
        Backtest results
    """
    logger.info("=" * 50)
    logger.info("STEP 5: Backtesting")
    logger.info("=" * 50)

    if 'returns' not in data:
        logger.warning("No returns data for backtesting")
        return {}

    returns = data['returns']

    # Calculate momentum scores (simplified)
    momentum_scores = {}
    for symbol in returns.columns:
        if len(returns[symbol].dropna()) >= 252:
            # 12-month momentum, skip last month
            mom = (1 + returns[symbol].iloc[-252:-21]).prod() - 1
            momentum_scores[symbol] = mom

    if not momentum_scores:
        logger.warning("Not enough data for momentum calculation")
        return {}

    scores = pd.Series(momentum_scores)

    # Generate initial weights
    weights = strategy.generate_weights(scores)
    logger.info(f"Initial portfolio weights:\n{weights.sort_values(ascending=False).head(10)}")

    # Create weight time series (monthly rebalance)
    dates = returns.index
    monthly_dates = pd.date_range(start=dates[0], end=dates[-1], freq='MS')

    weights_history = pd.DataFrame(index=dates, columns=returns.columns, dtype=float)
    weights_history = weights_history.fillna(0)

    for dt in monthly_dates:
        if dt in dates:
            # Recalculate scores up to this date
            hist_returns = returns.loc[:dt]
            if len(hist_returns) >= 252:
                mom_scores = {}
                for symbol in returns.columns:
                    if len(hist_returns[symbol].dropna()) >= 252:
                        mom = (1 + hist_returns[symbol].iloc[-252:-21]).prod() - 1
                        mom_scores[symbol] = mom

                if mom_scores:
                    new_weights = strategy.generate_weights(pd.Series(mom_scores))
                    for symbol, w in new_weights.items():
                        if symbol in weights_history.columns:
                            weights_history.loc[dt:, symbol] = w

    # Forward fill weights
    weights_history = weights_history.ffill()

    # Run vectorized backtest
    backtester = VectorizedBacktester(commission=0.001, slippage=0.0005)
    result = backtester.run(
        weights=weights_history,
        returns=returns,
        initial_capital=100000,
    )

    # Calculate metrics
    metrics = calculate_metrics(
        returns=result.returns,
        portfolio_values=result.portfolio_values,
        weights=weights_history,
    )

    results = {
        'portfolio_values': result.portfolio_values,
        'returns': result.returns,
        'weights': weights_history,
        'metrics': {
            'total_return': metrics.total_return,
            'cagr': metrics.cagr,
            'volatility': metrics.volatility,
            'sharpe_ratio': metrics.sharpe_ratio,
            'max_drawdown': metrics.max_drawdown,
            'sortino_ratio': metrics.sortino_ratio,
            'calmar_ratio': metrics.calmar_ratio,
        }
    }

    return results


def display_results(backtest_results: dict):
    """Display backtest results.

    Args:
        backtest_results: Backtest results dictionary
    """
    logger.info("=" * 50)
    logger.info("STEP 6: Results")
    logger.info("=" * 50)

    if not backtest_results:
        logger.warning("No backtest results to display")
        return

    metrics = backtest_results.get('metrics', {})
    portfolio = backtest_results.get('portfolio_values')

    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)

    if portfolio is not None and len(portfolio) > 0:
        print(f"Initial Capital:  ${100000:,.2f}")
        print(f"Final Value:      ${portfolio.iloc[-1]:,.2f}")
        print(f"Total Return:     {metrics.get('total_return', 0)*100:.2f}%")
        print(f"CAGR:             {metrics.get('cagr', 0)*100:.2f}%")
        print(f"Volatility:       {metrics.get('volatility', 0)*100:.2f}%")
        print(f"Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown:     {metrics.get('max_drawdown', 0)*100:.2f}%")
        print(f"Sortino Ratio:    {metrics.get('sortino_ratio', 0):.2f}")
        print(f"Calmar Ratio:     {metrics.get('calmar_ratio', 0):.2f}")

    print("\n" + "=" * 50)

    # Save results
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    if portfolio is not None:
        portfolio.to_csv(output_dir / "portfolio_values.csv")

    weights = backtest_results.get('weights')
    if weights is not None:
        weights.to_csv(output_dir / "weights_history.csv")

    logger.info(f"Results saved to {output_dir}")


def load_watchlist(name: str = "default") -> list[str]:
    """Load watchlist from config file.

    Args:
        name: Watchlist name (default, tech, etf, semiconductor, etc.)

    Returns:
        List of ticker symbols
    """
    watchlist_path = Path("config/watchlist.yaml")

    if not watchlist_path.exists():
        logger.warning(f"Watchlist file not found: {watchlist_path}")
        return []

    with open(watchlist_path) as f:
        watchlists = yaml.safe_load(f)

    if name not in watchlists:
        available = list(watchlists.keys())
        logger.warning(f"Watchlist '{name}' not found. Available: {available}")
        return []

    return watchlists[name]


def normalize_symbols(raw_symbols: list[str]) -> list[str]:
    """Normalize symbol arguments from space or comma-separated input."""
    normalized = []
    for item in raw_symbols:
        parts = [part.strip().upper() for part in item.split(",")]
        normalized.extend([part for part in parts if part])

    # Preserve input order while removing duplicates
    return list(dict.fromkeys(normalized))


def main():
    parser = argparse.ArgumentParser(
        description="Run algo-quant pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py --symbols AAPL TSLA NVDA
  python scripts/run_pipeline.py --watchlist tech --start 2020-01-01
  python scripts/run_pipeline.py --watchlist etf
  python scripts/run_pipeline.py --top 30
        """
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Runtime profile name or YAML path (default: AQ_PROFILE or dev)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Stock symbols (e.g., AAPL MSFT GOOGL or AAPL,MSFT,GOOGL)",
    )
    parser.add_argument("--watchlist", "-w", help="Watchlist name from config/watchlist.yaml (default, tech, etf, semiconductor, value, growth)")
    parser.add_argument("--top", type=int, default=None, help="Top N S&P 500 stocks (if no symbols/watchlist)")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=str(date.today()), help="End date (YYYY-MM-DD)")

    args = parser.parse_args()
    profile = load_runtime_profile(args.profile)
    load_local_env(profile.env_file)

    top_n = int(args.top) if args.top is not None else int(profile.pipeline_top)
    start_date = args.start or profile.pipeline_start

    # Get symbols (priority: --symbols > --watchlist > --top)
    if args.symbols:
        symbols = normalize_symbols(args.symbols)
        logger.info(f"Using command-line symbols: {symbols}")
    elif args.watchlist:
        symbols = load_watchlist(args.watchlist)
        if not symbols:
            logger.error(f"Failed to load watchlist: {args.watchlist}")
            sys.exit(1)
        logger.info(f"Using watchlist '{args.watchlist}': {symbols}")
    else:
        symbols = SP500_TOP_100[:top_n]
        logger.info(f"Using top {top_n} S&P 500 stocks")

    logger.info(f"Running pipeline with {len(symbols)} symbols")
    logger.info(f"Profile: {profile.name}")
    logger.info(f"Date range: {start_date} to {args.end}")

    # Run pipeline
    data = collect_data(symbols, start_date, args.end)

    factor_results = run_factor_analysis(data)

    regime_results = classify_regimes(data)

    strategy = build_strategy(factor_results, regime_results)

    backtest_results = run_backtest(data, strategy)

    display_results(backtest_results)

    logger.info("\nPipeline completed!")


if __name__ == "__main__":
    main()
