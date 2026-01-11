"""Data service for fetching and caching data.

This module provides a DataService class with built-in caching for market data,
returns, macroeconomic indicators, and factor data. Caching uses a TTL-like
mechanism that rounds timestamps to 5-minute intervals, ensuring data freshness
while minimizing redundant API calls.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
from functools import lru_cache
import time


def _cache_key(*args, ttl_minutes: int = 5, **kwargs) -> str:
    """Generate a cache key from parameters with TTL-like behavior.

    Creates a unique string key from the provided arguments, including a
    timestamp rounded to the specified TTL interval. This enables automatic
    cache invalidation as time passes.

    Args:
        *args: Positional arguments to include in the key.
        ttl_minutes: Time-to-live in minutes. The current timestamp is
            rounded to this interval (default: 5 minutes).
        **kwargs: Keyword arguments to include in the key.

    Returns:
        A string cache key combining all arguments and the rounded timestamp.

    Example:
        >>> _cache_key("AAPL", 252, ttl_minutes=5)
        "('AAPL', 252)_{}_{timestamp}"
    """
    # Round current time to ttl_minutes intervals
    current_time = time.time()
    rounded_time = int(current_time // (ttl_minutes * 60)) * (ttl_minutes * 60)

    # Convert list arguments to tuples for hashability
    processed_args = tuple(
        tuple(arg) if isinstance(arg, list) else arg for arg in args
    )

    return f"{processed_args}_{kwargs}_{rounded_time}"


# Module-level caches for class methods
_prices_cache: dict = {}
_returns_cache: dict = {}
_macro_cache: dict = {}
_factor_cache: dict = {}


class DataService:
    """Service for fetching market data with built-in caching.

    This class provides methods to fetch various types of financial data
    including prices, returns, macroeconomic indicators, and Fama-French
    factors. All data fetching methods support caching with a TTL-like
    mechanism that invalidates entries after approximately 5 minutes.

    Caching behavior:
        - get_prices(): Cached with maxsize=32, 5-minute TTL
        - get_returns(): Cached with maxsize=32, 5-minute TTL
        - get_macro_data(): Cached with maxsize=16, 5-minute TTL
        - get_factor_data(): Cached with maxsize=16, 5-minute TTL

    Use clear_cache() to manually invalidate all cached data.

    Attributes:
        demo_mode: If True, generates sample data instead of fetching real data.
    """

    # Class-level cache size limits
    _PRICES_CACHE_MAXSIZE = 32
    _RETURNS_CACHE_MAXSIZE = 32
    _MACRO_CACHE_MAXSIZE = 16
    _FACTOR_CACHE_MAXSIZE = 16

    def __init__(self, demo_mode: bool = True):
        self.demo_mode = demo_mode
        self._cache = {}

    def get_prices(
        self,
        symbols: list[str],
        periods: int = 252,
        start_date: str = "2023-01-01",
    ) -> pd.DataFrame:
        """Get price data for symbols with caching.

        Results are cached with a maximum of 32 entries and a TTL of
        approximately 5 minutes. Cache keys are generated from the
        symbols, periods, start_date, and a rounded timestamp.

        Args:
            symbols: List of ticker symbols.
            periods: Number of trading days.
            start_date: Start date for data.

        Returns:
            DataFrame with price data indexed by date, columns are symbols.

        Note:
            Cache is shared across all DataService instances.
        """
        global _prices_cache

        cache_key = _cache_key(symbols, periods, start_date, ttl_minutes=5)

        if cache_key in _prices_cache:
            return _prices_cache[cache_key]

        # Enforce maxsize by removing oldest entry if at capacity
        if len(_prices_cache) >= self._PRICES_CACHE_MAXSIZE:
            oldest_key = next(iter(_prices_cache))
            del _prices_cache[oldest_key]

        if self.demo_mode:
            result = self._generate_sample_prices(symbols, periods, start_date)
        else:
            # TODO: Implement real data fetching
            result = self._generate_sample_prices(symbols, periods, start_date)

        _prices_cache[cache_key] = result
        return result

    def get_returns(
        self,
        symbols: list[str],
        periods: int = 252,
    ) -> pd.DataFrame:
        """Get return data for symbols with caching.

        Results are cached with a maximum of 32 entries and a TTL of
        approximately 5 minutes. This method calculates percentage changes
        from the cached price data.

        Args:
            symbols: List of ticker symbols.
            periods: Number of trading days.

        Returns:
            DataFrame with daily return data, first row dropped due to pct_change.

        Note:
            Cache is shared across all DataService instances.
        """
        global _returns_cache

        cache_key = _cache_key(symbols, periods, ttl_minutes=5)

        if cache_key in _returns_cache:
            return _returns_cache[cache_key]

        # Enforce maxsize by removing oldest entry if at capacity
        if len(_returns_cache) >= self._RETURNS_CACHE_MAXSIZE:
            oldest_key = next(iter(_returns_cache))
            del _returns_cache[oldest_key]

        prices = self.get_prices(symbols, periods)
        result = prices.pct_change().dropna()

        _returns_cache[cache_key] = result
        return result

    def get_macro_data(self, periods: int = 48) -> pd.DataFrame:
        """Get macroeconomic indicator data with caching.

        Results are cached with a maximum of 16 entries and a TTL of
        approximately 5 minutes. Macroeconomic data includes GDP growth,
        unemployment, yield spread, and Fed Funds rate.

        Args:
            periods: Number of monthly periods.

        Returns:
            DataFrame with macroeconomic indicators indexed by month-end dates.

        Note:
            Cache is shared across all DataService instances.
        """
        global _macro_cache

        cache_key = _cache_key(periods, ttl_minutes=5)

        if cache_key in _macro_cache:
            return _macro_cache[cache_key]

        # Enforce maxsize by removing oldest entry if at capacity
        if len(_macro_cache) >= self._MACRO_CACHE_MAXSIZE:
            oldest_key = next(iter(_macro_cache))
            del _macro_cache[oldest_key]

        if self.demo_mode:
            result = self._generate_macro_data(periods)
        else:
            # TODO: Implement FRED data fetching
            result = self._generate_macro_data(periods)

        _macro_cache[cache_key] = result
        return result

    def get_factor_data(self, periods: int = 252) -> pd.DataFrame:
        """Get Fama-French factor data with caching.

        Results are cached with a maximum of 16 entries and a TTL of
        approximately 5 minutes. Factor data includes Mkt-RF, SMB, HML,
        RMW, and CMA factors.

        Args:
            periods: Number of trading days.

        Returns:
            DataFrame with Fama-French 5-factor data indexed by business day.

        Note:
            Cache is shared across all DataService instances.
        """
        global _factor_cache

        cache_key = _cache_key(periods, ttl_minutes=5)

        if cache_key in _factor_cache:
            return _factor_cache[cache_key]

        # Enforce maxsize by removing oldest entry if at capacity
        if len(_factor_cache) >= self._FACTOR_CACHE_MAXSIZE:
            oldest_key = next(iter(_factor_cache))
            del _factor_cache[oldest_key]

        if self.demo_mode:
            result = self._generate_factor_data(periods)
        else:
            # TODO: Implement FF data fetching
            result = self._generate_factor_data(periods)

        _factor_cache[cache_key] = result
        return result

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached data across all DataService instances.

        This method invalidates all entries in the prices, returns,
        macro, and factor caches. Use this when you need to force
        fresh data fetches, such as after market data updates or
        when switching between demo and live modes.

        Example:
            >>> service = DataService()
            >>> service.get_prices(["AAPL"], 100)  # Cached
            >>> DataService.clear_cache()  # Clear all caches
            >>> service.get_prices(["AAPL"], 100)  # Fresh fetch
        """
        global _prices_cache, _returns_cache, _macro_cache, _factor_cache

        _prices_cache.clear()
        _returns_cache.clear()
        _macro_cache.clear()
        _factor_cache.clear()

    def calculate_portfolio_metrics(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02,
    ) -> dict:
        """Calculate portfolio performance metrics.

        Args:
            returns: Portfolio return series
            risk_free_rate: Annual risk-free rate

        Returns:
            Dictionary of metrics
        """
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = (annual_return - risk_free_rate) / annual_vol

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0

        # Max drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_dd = drawdown.min()

        # Win rate
        win_rate = (returns > 0).mean()

        return {
            "total_return": cum_returns.iloc[-1] - 1,
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "num_trades": len(returns),
        }

    @staticmethod
    def _generate_sample_prices(
        symbols: list[str],
        periods: int,
        start_date: str,
    ) -> pd.DataFrame:
        """Generate sample price data."""
        np.random.seed(42)
        dates = pd.date_range(start_date, periods=periods, freq="B")

        data = {}
        for symbol in symbols:
            returns = np.random.normal(0.0005, 0.02, periods)
            prices = 100 * np.cumprod(1 + returns)
            data[symbol] = prices

        return pd.DataFrame(data, index=dates)

    @staticmethod
    def _generate_macro_data(periods: int) -> pd.DataFrame:
        """Generate sample macro data."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=periods, freq="ME")

        return pd.DataFrame({
            "GDP Growth (%)": np.random.normal(2.5, 1.0, periods).cumsum() / 10,
            "Unemployment (%)": 4 + np.random.normal(0, 0.5, periods).cumsum() / 5,
            "10Y-2Y Spread (%)": 1.5 + np.random.normal(0, 0.2, periods).cumsum() / 3,
            "Fed Funds Rate (%)": 4 + np.random.normal(0, 0.1, periods).cumsum() / 5,
        }, index=dates)

    @staticmethod
    def _generate_factor_data(periods: int) -> pd.DataFrame:
        """Generate sample factor return data."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=periods, freq="B")

        return pd.DataFrame({
            "Mkt-RF": np.random.normal(0.0004, 0.01, periods),
            "SMB": np.random.normal(0.0001, 0.005, periods),
            "HML": np.random.normal(0.0001, 0.006, periods),
            "RMW": np.random.normal(0.0001, 0.004, periods),
            "CMA": np.random.normal(0.0001, 0.004, periods),
        }, index=dates)
