"""Data service for fetching and caching data."""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
from functools import lru_cache


class DataService:
    """Service for fetching market data."""

    def __init__(self, demo_mode: bool = True):
        self.demo_mode = demo_mode
        self._cache = {}

    def get_prices(
        self,
        symbols: list[str],
        periods: int = 252,
        start_date: str = "2023-01-01",
    ) -> pd.DataFrame:
        """Get price data for symbols.

        Args:
            symbols: List of ticker symbols
            periods: Number of trading days
            start_date: Start date for data

        Returns:
            DataFrame with price data
        """
        if self.demo_mode:
            return self._generate_sample_prices(symbols, periods, start_date)

        # TODO: Implement real data fetching
        return self._generate_sample_prices(symbols, periods, start_date)

    def get_returns(
        self,
        symbols: list[str],
        periods: int = 252,
    ) -> pd.DataFrame:
        """Get return data for symbols."""
        prices = self.get_prices(symbols, periods)
        return prices.pct_change().dropna()

    def get_macro_data(self, periods: int = 48) -> pd.DataFrame:
        """Get macroeconomic indicator data."""
        if self.demo_mode:
            return self._generate_macro_data(periods)

        # TODO: Implement FRED data fetching
        return self._generate_macro_data(periods)

    def get_factor_data(self, periods: int = 252) -> pd.DataFrame:
        """Get Fama-French factor data."""
        if self.demo_mode:
            return self._generate_factor_data(periods)

        # TODO: Implement FF data fetching
        return self._generate_factor_data(periods)

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
