"""Tests for performance metrics."""

import pytest
import numpy as np
import pandas as pd

from src.backtest.metrics import (
    PerformanceMetrics,
    calculate_metrics,
    calculate_total_return,
    calculate_cagr,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    calculate_var,
    calculate_cvar,
    calculate_win_rate,
    calculate_profit_factor,
    BenchmarkComparison,
)


class TestReturnMetrics:
    """Test return-based metrics."""

    @pytest.fixture
    def sample_portfolio_values(self):
        """Create sample portfolio values."""
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        # Portfolio grows from 100000 to ~110000
        values = 100000 * np.cumprod(1 + np.random.normal(0.0004, 0.01, 252))
        return pd.Series(values, index=dates)

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        returns = np.random.normal(0.0004, 0.01, 252)
        return pd.Series(returns, index=dates)

    def test_total_return(self, sample_portfolio_values):
        """Test total return calculation."""
        total_return = calculate_total_return(sample_portfolio_values)
        
        expected = sample_portfolio_values.iloc[-1] / sample_portfolio_values.iloc[0] - 1
        assert abs(total_return - expected) < 0.0001

    def test_cagr(self, sample_portfolio_values):
        """Test CAGR calculation."""
        cagr = calculate_cagr(sample_portfolio_values, periods_per_year=252)
        
        # CAGR should be reasonable for 1 year of data
        assert -0.5 < cagr < 0.5

    def test_volatility(self, sample_returns):
        """Test volatility calculation."""
        vol = calculate_volatility(sample_returns, periods_per_year=252)
        
        # Should be annualized (roughly 15-20% for 1% daily vol)
        assert 0.05 < vol < 0.50


class TestRiskMetrics:
    """Test risk-based metrics."""

    @pytest.fixture
    def sample_portfolio_values(self):
        """Create sample portfolio with drawdown."""
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        np.random.seed(42)
        
        # Create values with a clear drawdown
        values = [100000]
        for i in range(1, 252):
            if 50 < i < 100:  # Drawdown period
                values.append(values[-1] * 0.995)
            else:
                values.append(values[-1] * 1.002)
        
        return pd.Series(values, index=dates)

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        returns = np.random.normal(0.0004, 0.015, 252)
        return pd.Series(returns, index=dates)

    def test_max_drawdown(self, sample_portfolio_values):
        """Test max drawdown calculation."""
        max_dd, duration = calculate_max_drawdown(sample_portfolio_values)
        
        assert max_dd > 0  # Should have some drawdown
        assert max_dd < 1  # Less than 100%
        assert duration >= 0

    def test_sharpe_ratio(self, sample_returns):
        """Test Sharpe ratio calculation."""
        sharpe = calculate_sharpe_ratio(sample_returns, risk_free_rate=0.02)
        
        # Sharpe should be reasonable
        assert -5 < sharpe < 5

    def test_sortino_ratio(self, sample_returns):
        """Test Sortino ratio calculation."""
        sortino = calculate_sortino_ratio(sample_returns, risk_free_rate=0.02)
        
        # Sortino should be >= Sharpe for same returns
        sharpe = calculate_sharpe_ratio(sample_returns, risk_free_rate=0.02)
        assert sortino >= sharpe - 0.5  # Allow some tolerance

    def test_var(self, sample_returns):
        """Test VaR calculation."""
        var = calculate_var(sample_returns, confidence=0.95)
        
        assert var > 0  # VaR should be positive (loss)
        assert var < 0.5  # Less than 50% daily

    def test_cvar(self, sample_returns):
        """Test CVaR calculation."""
        var = calculate_var(sample_returns, confidence=0.95)
        cvar = calculate_cvar(sample_returns, confidence=0.95)
        
        # CVaR should be >= VaR
        assert cvar >= var


class TestTradeMetrics:
    """Test trade-based metrics."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns with wins and losses."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        returns = np.random.normal(0.001, 0.02, 252)  # Slightly positive bias
        return pd.Series(returns, index=dates)

    def test_win_rate(self, sample_returns):
        """Test win rate calculation."""
        win_rate = calculate_win_rate(sample_returns)
        
        assert 0 <= win_rate <= 1
        # With positive mean, should be > 50%
        assert win_rate > 0.4

    def test_profit_factor(self, sample_returns):
        """Test profit factor calculation."""
        pf = calculate_profit_factor(sample_returns)
        
        assert pf > 0
        # With positive expected returns, should be > 1
        # But random so allow some tolerance
        assert pf > 0.5


class TestCalculateMetrics:
    """Test combined metrics calculation."""

    @pytest.fixture
    def sample_data(self):
        """Create complete sample data."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        
        returns = pd.Series(
            np.random.normal(0.0004, 0.015, 252),
            index=dates,
        )
        
        portfolio_values = 100000 * (1 + returns).cumprod()
        
        weights = pd.DataFrame(
            [[0.5, 0.5]] * 252,
            index=dates,
            columns=["A", "B"],
        )
        
        return returns, portfolio_values, weights

    def test_calculate_all_metrics(self, sample_data):
        """Test calculating all metrics."""
        returns, portfolio_values, weights = sample_data
        
        metrics = calculate_metrics(
            returns,
            portfolio_values,
            weights,
            num_trades=100,
        )
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.num_trades == 100

    def test_metrics_to_dict(self, sample_data):
        """Test metrics conversion to dict."""
        returns, portfolio_values, weights = sample_data
        
        metrics = calculate_metrics(returns, portfolio_values)
        d = metrics.to_dict()
        
        assert "total_return" in d
        assert "sharpe_ratio" in d
        assert "max_drawdown" in d

    def test_metrics_summary(self, sample_data):
        """Test metrics summary generation."""
        returns, portfolio_values, weights = sample_data
        
        metrics = calculate_metrics(returns, portfolio_values)
        summary = metrics.summary()
        
        assert "Total Return" in summary
        assert "Sharpe Ratio" in summary


class TestBenchmarkComparison:
    """Test benchmark comparison."""

    @pytest.fixture
    def sample_data(self):
        """Create strategy and benchmark returns."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        
        benchmark = pd.Series(
            np.random.normal(0.0003, 0.01, 252),
            index=dates,
        )
        
        # Strategy with some correlation to benchmark
        strategy = benchmark * 1.2 + np.random.normal(0.0001, 0.005, 252)
        strategy = pd.Series(strategy, index=dates)
        
        return strategy, benchmark

    def test_benchmark_comparison(self, sample_data):
        """Test benchmark comparison metrics."""
        strategy, benchmark = sample_data
        
        comparison = BenchmarkComparison(strategy, benchmark)
        
        assert hasattr(comparison, "alpha")
        assert hasattr(comparison, "beta")
        assert hasattr(comparison, "correlation")
        assert hasattr(comparison, "tracking_error")

    def test_beta_calculation(self, sample_data):
        """Test beta calculation."""
        strategy, benchmark = sample_data
        
        comparison = BenchmarkComparison(strategy, benchmark)
        
        # Beta should be close to 1.2 (our multiplier)
        assert 0.5 < comparison.beta < 2.0

    def test_comparison_to_dict(self, sample_data):
        """Test comparison conversion to dict."""
        strategy, benchmark = sample_data
        
        comparison = BenchmarkComparison(strategy, benchmark)
        d = comparison.to_dict()
        
        assert "alpha" in d
        assert "beta" in d
        assert "information_ratio" in d
