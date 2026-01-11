"""Tests for factor-based strategy."""

import pytest
import numpy as np
import pandas as pd

from src.strategy.factor_strategy import (
    FactorStrategy,
    FactorConfig,
    FactorWeightMethod,
    MultiFactorStrategy,
)


class TestFactorStrategy:
    """Test cases for FactorStrategy."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns DataFrame."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=60, freq="M")
        stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        
        return pd.DataFrame(
            np.random.normal(0.01, 0.05, (60, 5)),
            index=dates,
            columns=stocks,
        )

    @pytest.fixture
    def fundamental_data(self):
        """Create sample fundamental data."""
        stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        
        return pd.DataFrame({
            "pe_ratio": [25, 30, 22, 60, 15],
            "book_to_market": [0.3, 0.25, 0.2, 0.15, 0.4],
            "roe": [0.30, 0.35, 0.25, 0.20, 0.40],
            "debt_to_equity": [1.5, 0.8, 0.5, 1.2, 0.3],
            "market_cap": [2.5e12, 2.0e12, 1.5e12, 1.2e12, 0.8e12],
        }, index=stocks)

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        return FactorStrategy(
            factors=["value", "momentum"],
            weight_method=FactorWeightMethod.RANK,
            long_only=True,
        )

    def test_value_score(self, strategy, fundamental_data):
        """Test value score calculation."""
        scores = strategy.calculate_value_score(fundamental_data)
        
        assert len(scores) == len(fundamental_data)
        # All scores should be between 0 and 1 (percentile rank)
        assert scores.min() >= 0
        assert scores.max() <= 1

    def test_momentum_score(self, strategy, sample_returns):
        """Test momentum score calculation."""
        scores = strategy.calculate_momentum_score(sample_returns, lookback=12)
        
        assert len(scores) == len(sample_returns.columns)
        assert scores.min() >= 0
        assert scores.max() <= 1

    def test_quality_score(self, strategy, fundamental_data):
        """Test quality score calculation."""
        scores = strategy.calculate_quality_score(fundamental_data)
        
        assert len(scores) == len(fundamental_data)

    def test_low_volatility_score(self, strategy, sample_returns):
        """Test low volatility score calculation."""
        scores = strategy.calculate_low_volatility_score(sample_returns)
        
        # Lower vol should have higher score
        volatilities = sample_returns.std()
        assert len(scores) == len(volatilities)

    def test_size_score(self, strategy, fundamental_data):
        """Test size score calculation."""
        scores = strategy.calculate_size_score(fundamental_data)
        
        assert len(scores) == len(fundamental_data)
        # Smaller market cap should have higher score
        smallest = fundamental_data["market_cap"].idxmin()
        largest = fundamental_data["market_cap"].idxmax()
        assert scores[smallest] > scores[largest]

    def test_generate_weights_rank(self, strategy):
        """Test weight generation with rank method."""
        scores = pd.Series({
            "AAPL": 0.8,
            "MSFT": 0.6,
            "GOOGL": 0.4,
            "AMZN": 0.2,
        })
        
        weights = strategy.generate_weights(scores)
        
        assert abs(weights.sum() - 1.0) < 0.01
        # Higher score should have higher weight
        assert weights["AAPL"] > weights["AMZN"]

    def test_generate_weights_top_n(self):
        """Test weight generation with top N constraint."""
        strategy = FactorStrategy(
            factors=["value"],
            top_n=2,
        )
        
        scores = pd.Series({
            "AAPL": 0.8,
            "MSFT": 0.6,
            "GOOGL": 0.4,
            "AMZN": 0.2,
        })
        
        weights = strategy.generate_weights(scores)
        
        # Only top 2 should have weights
        assert (weights > 0).sum() == 2

    def test_rebalance(self, strategy):
        """Test rebalancing calculation."""
        current = pd.Series({"AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.25, "AMZN": 0.25})
        target = pd.Series({"AAPL": 0.40, "MSFT": 0.30, "GOOGL": 0.20, "AMZN": 0.10})
        
        trades = strategy.rebalance(current, target, threshold=0.05)
        
        assert trades["AAPL"] > 0  # Buy
        assert trades["AMZN"] < 0  # Sell


class TestMultiFactorStrategy:
    """Test cases for MultiFactorStrategy."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns DataFrame."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=60, freq="M")
        stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        
        return pd.DataFrame(
            np.random.normal(0.01, 0.05, (60, 5)),
            index=dates,
            columns=stocks,
        )

    @pytest.fixture
    def fundamental_data(self):
        """Create sample fundamental data."""
        stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        
        return pd.DataFrame({
            "pe_ratio": [25, 30, 22, 60, 15],
            "book_to_market": [0.3, 0.25, 0.2, 0.15, 0.4],
            "roe": [0.30, 0.35, 0.25, 0.20, 0.40],
            "market_cap": [2.5e12, 2.0e12, 1.5e12, 1.2e12, 0.8e12],
        }, index=stocks)

    def test_calculate_all_factor_scores(self, fundamental_data, sample_returns):
        """Test multi-factor score calculation."""
        factors = [
            FactorConfig(name="value", weight=1.0),
            FactorConfig(name="momentum", weight=1.0),
            FactorConfig(name="quality", weight=0.5),
        ]
        strategy = MultiFactorStrategy(factors=factors)
        
        scores = strategy.calculate_all_factor_scores(fundamental_data, sample_returns)
        
        assert "value" in scores
        assert "momentum" in scores
        assert "quality" in scores

    def test_composite_score(self, fundamental_data, sample_returns):
        """Test composite score calculation."""
        factors = [
            FactorConfig(name="value", weight=1.0),
            FactorConfig(name="momentum", weight=1.0),
        ]
        strategy = MultiFactorStrategy(factors=factors)
        
        factor_scores = strategy.calculate_all_factor_scores(fundamental_data, sample_returns)
        composite = strategy.calculate_composite_score(factor_scores)
        
        assert len(composite) > 0
        assert composite.min() >= 0

    def test_regime_weights(self):
        """Test regime-specific factor weights."""
        factors = [
            FactorConfig(name="value", weight=1.0),
            FactorConfig(name="momentum", weight=1.0),
        ]
        regime_weights = {
            "expansion": {"value": 0.3, "momentum": 0.7},
            "contraction": {"value": 0.7, "momentum": 0.3},
        }
        
        strategy = MultiFactorStrategy(
            factors=factors,
            regime_factor_weights=regime_weights,
        )
        
        exp_weights = strategy.get_regime_weights("expansion")
        
        assert exp_weights["momentum"] > exp_weights["value"]
