"""Tests for walk-forward analysis."""

import pytest
import numpy as np
import pandas as pd

from src.backtest.walk_forward import (
    WalkForwardAnalyzer,
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardWindow,
    AnchoredWalkForward,
    grid_search_optimizer,
)


class TestWalkForwardConfig:
    """Test WalkForwardConfig."""

    def test_parse_period_years(self):
        """Test parsing year periods."""
        config = WalkForwardConfig()
        offset = config.parse_period("2Y")
        
        assert offset == pd.DateOffset(years=2)

    def test_parse_period_months(self):
        """Test parsing month periods."""
        config = WalkForwardConfig()
        offset = config.parse_period("6M")
        
        assert offset == pd.DateOffset(months=6)

    def test_parse_period_weeks(self):
        """Test parsing week periods."""
        config = WalkForwardConfig()
        offset = config.parse_period("4W")
        
        assert offset == pd.DateOffset(weeks=4)


class TestWalkForwardAnalyzer:
    """Test WalkForwardAnalyzer."""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data (3 years)."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=756, freq="B")  # ~3 years
        
        returns = np.random.normal(0.0003, 0.015, (756, 3))
        prices = 100 * np.cumprod(1 + returns, axis=0)
        
        return pd.DataFrame(
            prices,
            index=dates,
            columns=["AAPL", "MSFT", "GOOGL"],
        )

    @pytest.fixture
    def simple_strategy_factory(self):
        """Create strategy factory."""
        def factory(**kwargs):
            class SimpleStrategy:
                def generate_weights(self, prices, **kw):
                    n = len(prices.columns)
                    return pd.Series(1.0/n, index=prices.columns)
            return SimpleStrategy()
        return factory

    def test_walk_forward_basic(self, sample_prices, simple_strategy_factory):
        """Test basic walk-forward analysis."""
        config = WalkForwardConfig(
            train_period="1Y",
            test_period="3M",
            step_size="3M",
        )
        
        analyzer = WalkForwardAnalyzer(config)
        result = analyzer.run(simple_strategy_factory, sample_prices)
        
        assert isinstance(result, WalkForwardResult)
        assert len(result.windows) > 0

    def test_walk_forward_windows(self, sample_prices, simple_strategy_factory):
        """Test window generation."""
        config = WalkForwardConfig(
            train_period="1Y",
            test_period="3M",
            step_size="3M",
        )
        
        analyzer = WalkForwardAnalyzer(config)
        result = analyzer.run(simple_strategy_factory, sample_prices)
        
        for window in result.windows:
            assert window.train_start < window.train_end
            assert window.train_end <= window.test_start
            assert window.test_start < window.test_end

    def test_walk_forward_metrics(self, sample_prices, simple_strategy_factory):
        """Test window metrics."""
        config = WalkForwardConfig(
            train_period="1Y",
            test_period="3M",
            step_size="3M",
        )
        
        analyzer = WalkForwardAnalyzer(config)
        result = analyzer.run(simple_strategy_factory, sample_prices)
        
        for window in result.windows:
            if window.test_metrics:
                assert window.test_metrics.total_return is not None

    def test_walk_forward_combined_metrics(self, sample_prices, simple_strategy_factory):
        """Test combined OOS metrics."""
        config = WalkForwardConfig(
            train_period="1Y",
            test_period="3M",
            step_size="3M",
        )
        
        analyzer = WalkForwardAnalyzer(config)
        result = analyzer.run(simple_strategy_factory, sample_prices)
        
        if result.combined_metrics:
            assert result.combined_metrics.total_return is not None

    def test_walk_forward_summary(self, sample_prices, simple_strategy_factory):
        """Test result summary."""
        config = WalkForwardConfig(
            train_period="1Y",
            test_period="3M",
            step_size="3M",
        )
        
        analyzer = WalkForwardAnalyzer(config)
        result = analyzer.run(simple_strategy_factory, sample_prices)
        
        summary = result.summary()
        assert "Walk-Forward" in summary


class TestAnchoredWalkForward:
    """Test AnchoredWalkForward."""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=756, freq="B")
        
        returns = np.random.normal(0.0003, 0.015, (756, 3))
        prices = 100 * np.cumprod(1 + returns, axis=0)
        
        return pd.DataFrame(
            prices,
            index=dates,
            columns=["AAPL", "MSFT", "GOOGL"],
        )

    @pytest.fixture
    def simple_strategy_factory(self):
        """Create strategy factory."""
        def factory(**kwargs):
            class SimpleStrategy:
                def generate_weights(self, prices, **kw):
                    n = len(prices.columns)
                    return pd.Series(1.0/n, index=prices.columns)
            return SimpleStrategy()
        return factory

    def test_anchored_basic(self, sample_prices, simple_strategy_factory):
        """Test anchored walk-forward."""
        analyzer = AnchoredWalkForward(
            test_period="3M",
            min_train_period="1Y",
            step_size="3M",
        )
        
        result = analyzer.run(simple_strategy_factory, sample_prices)
        
        assert isinstance(result, WalkForwardResult)
        assert len(result.windows) > 0

    def test_anchored_expanding_window(self, sample_prices, simple_strategy_factory):
        """Test that training window expands."""
        analyzer = AnchoredWalkForward(
            test_period="3M",
            min_train_period="1Y",
            step_size="3M",
        )
        
        result = analyzer.run(simple_strategy_factory, sample_prices)
        
        # All windows should start from same anchor
        if len(result.windows) > 1:
            first_start = result.windows[0].train_start
            for window in result.windows:
                assert window.train_start == first_start


class TestGridSearchOptimizer:
    """Test grid search optimizer."""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        
        returns = np.random.normal(0.0003, 0.015, (252, 3))
        prices = 100 * np.cumprod(1 + returns, axis=0)
        
        return pd.DataFrame(
            prices,
            index=dates,
            columns=["AAPL", "MSFT", "GOOGL"],
        )

    def test_grid_search_basic(self, sample_prices):
        """Test basic grid search."""
        def strategy_factory(weight_a=0.5):
            class ParamStrategy:
                def __init__(self, w):
                    self.weight_a = w
                    
                def generate_weights(self, prices, **kw):
                    n = len(prices.columns)
                    weights = pd.Series(1.0/n, index=prices.columns)
                    return weights
            return ParamStrategy(weight_a)
        
        param_grid = {"weight_a": [0.3, 0.5, 0.7]}
        
        best_params = grid_search_optimizer(
            strategy_factory,
            sample_prices,
            param_grid,
            metric="sharpe_ratio",
        )
        
        assert "weight_a" in best_params
        assert best_params["weight_a"] in [0.3, 0.5, 0.7]
