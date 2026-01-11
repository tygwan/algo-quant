"""Tests for portfolio optimizer."""

import pytest
import numpy as np
import pandas as pd

from src.strategy.optimizer import (
    PortfolioOptimizer,
    OptimizationConstraints,
    OptimizationResult,
)


class TestPortfolioOptimizer:
    """Test cases for PortfolioOptimizer."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns DataFrame."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        
        return pd.DataFrame({
            "AAPL": np.random.normal(0.001, 0.02, 252),
            "MSFT": np.random.normal(0.0008, 0.018, 252),
            "GOOGL": np.random.normal(0.0009, 0.022, 252),
            "AMZN": np.random.normal(0.0012, 0.025, 252),
        }, index=dates)

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance."""
        return PortfolioOptimizer()

    def test_mean_variance(self, optimizer, sample_returns):
        """Test mean-variance optimization."""
        result = optimizer.mean_variance(sample_returns, target_return=0.10)
        
        assert isinstance(result, OptimizationResult)
        assert abs(result.weights.sum() - 1.0) < 0.01
        assert result.weights.min() >= -0.01  # Long-only by default

    def test_minimum_variance(self, optimizer, sample_returns):
        """Test minimum variance optimization."""
        result = optimizer.minimum_variance(sample_returns)
        
        assert result.success
        assert abs(result.weights.sum() - 1.0) < 0.01
        # Volatility should be positive
        assert result.volatility > 0

    def test_maximum_sharpe(self, optimizer, sample_returns):
        """Test maximum Sharpe ratio optimization."""
        result = optimizer.maximum_sharpe(sample_returns, risk_free_rate=0.02)
        
        assert result.success
        assert abs(result.weights.sum() - 1.0) < 0.01

    def test_risk_parity(self, optimizer, sample_returns):
        """Test risk parity optimization."""
        result = optimizer.risk_parity(sample_returns)
        
        assert result.success
        assert abs(result.weights.sum() - 1.0) < 0.01

    def test_equal_weight(self, optimizer, sample_returns):
        """Test equal weight portfolio."""
        result = optimizer.equal_weight(sample_returns)
        
        n_assets = len(sample_returns.columns)
        expected_weight = 1.0 / n_assets
        
        for weight in result.weights:
            assert abs(weight - expected_weight) < 0.001

    def test_inverse_volatility(self, optimizer, sample_returns):
        """Test inverse volatility weighting."""
        result = optimizer.inverse_volatility(sample_returns)
        
        assert abs(result.weights.sum() - 1.0) < 0.01
        # Lower vol assets should have higher weight
        volatilities = sample_returns.std()
        lowest_vol = volatilities.idxmin()
        highest_vol = volatilities.idxmax()
        assert result.weights[lowest_vol] > result.weights[highest_vol]

    def test_efficient_frontier(self, optimizer, sample_returns):
        """Test efficient frontier generation."""
        frontier = optimizer.efficient_frontier(sample_returns, n_points=10)
        
        assert len(frontier) > 0
        assert "return" in frontier.columns
        assert "volatility" in frontier.columns
        assert "sharpe" in frontier.columns

    def test_constraints(self, sample_returns):
        """Test with custom constraints."""
        constraints = OptimizationConstraints(
            min_weight=0.1,
            max_weight=0.4,
            long_only=True,
        )
        optimizer = PortfolioOptimizer(constraints=constraints)
        
        result = optimizer.minimum_variance(sample_returns)
        
        assert result.weights.min() >= 0.1 - 0.01
        assert result.weights.max() <= 0.4 + 0.01

    def test_result_to_dict(self, optimizer, sample_returns):
        """Test result conversion to dict."""
        result = optimizer.equal_weight(sample_returns)
        
        d = result.to_dict()
        
        assert "weights" in d
        assert "expected_return" in d
        assert "volatility" in d
        assert "sharpe_ratio" in d
