"""Tests for CAPM implementation."""

import pytest
import numpy as np
import pandas as pd

from src.factors.capm import CAPM
from src.factors.base import FactorModelResult


class TestCAPM:
    """Test cases for CAPM model."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample stock returns."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        # Simulate returns with beta = 1.2
        market_returns = np.random.normal(0.0005, 0.01, 252)
        stock_returns = 0.0001 + 1.2 * market_returns + np.random.normal(0, 0.005, 252)
        
        return pd.Series(stock_returns, index=dates)

    @pytest.fixture
    def market_returns(self):
        """Create sample market returns."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        returns = np.random.normal(0.0005, 0.01, 252)
        return pd.Series(returns, index=dates)

    @pytest.fixture
    def risk_free_rate(self):
        """Create sample risk-free rate."""
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        return pd.Series(0.0001, index=dates)  # ~2.5% annualized

    def test_fit_basic(self, sample_returns, market_returns):
        """Test basic CAPM fitting."""
        capm = CAPM()
        capm.fit(sample_returns, market_returns)
        
        assert capm.is_fitted
        assert capm.get_beta() is not None
        assert capm.get_alpha() is not None

    def test_beta_calculation(self, sample_returns, market_returns):
        """Test beta calculation accuracy."""
        capm = CAPM()
        capm.fit(sample_returns, market_returns)
        
        # Beta should be close to 1.2 (the value used in sample generation)
        assert abs(capm.get_beta() - 1.2) < 0.2

    def test_r_squared(self, sample_returns, market_returns):
        """Test R-squared is reasonable."""
        capm = CAPM()
        capm.fit(sample_returns, market_returns)
        
        # R² should be between 0 and 1
        assert 0 <= capm.get_r_squared() <= 1
        # With beta = 1.2, R² should be relatively high
        assert capm.get_r_squared() > 0.3

    def test_result_attributes(self, sample_returns, market_returns):
        """Test result object attributes."""
        capm = CAPM()
        capm.fit(sample_returns, market_returns)
        
        result = capm.result
        assert isinstance(result, FactorModelResult)
        assert "Mkt-RF" in result.loadings
        assert "Mkt-RF" in result.t_stats
        assert "Mkt-RF" in result.p_values
        assert result.residuals is not None

    def test_summary(self, sample_returns, market_returns):
        """Test summary output."""
        capm = CAPM()
        capm.fit(sample_returns, market_returns)
        
        summary = capm.summary()
        assert "Factor Model Results" in summary
        assert "Mkt-RF" in summary
        assert "R-squared" in summary

    def test_predict(self, sample_returns, market_returns):
        """Test prediction method."""
        capm = CAPM()
        capm.fit(sample_returns, market_returns)
        
        factor_df = pd.DataFrame({"Mkt-RF": market_returns})
        predictions = capm.predict(factor_df)
        
        assert len(predictions) == len(market_returns)
        assert not predictions.isna().any()

    def test_rolling_beta(self, sample_returns, market_returns):
        """Test rolling beta calculation."""
        capm = CAPM()
        
        rolling_beta = capm.calculate_rolling_beta(
            sample_returns, market_returns, window=60
        )
        
        assert len(rolling_beta) == len(sample_returns)
        # First values should be NaN due to window
        assert rolling_beta.iloc[:30].isna().all()
        # Later values should be valid
        valid_betas = rolling_beta.dropna()
        assert len(valid_betas) > 0

    def test_expected_return(self, sample_returns, market_returns):
        """Test expected return calculation."""
        capm = CAPM()
        capm.fit(sample_returns, market_returns)
        
        expected = capm.expected_return(
            risk_free_rate=0.02,
            market_return=0.10
        )
        
        # Expected return = Rf + beta * (Rm - Rf)
        beta = capm.get_beta()
        manual_calc = 0.02 + beta * (0.10 - 0.02)
        
        assert abs(expected - manual_calc) < 1e-6

    def test_security_market_line(self, sample_returns, market_returns):
        """Test SML generation."""
        capm = CAPM()
        capm.fit(sample_returns, market_returns)
        
        sml = capm.security_market_line(
            risk_free_rate=0.02,
            market_return=0.10,
            num_points=50
        )
        
        assert len(sml) == 50
        assert "beta" in sml.columns
        assert "expected_return" in sml.columns
        # At beta = 0, expected return should equal risk-free rate
        assert abs(sml[sml["beta"] == 0]["expected_return"].values[0] - 0.02) < 0.01

    def test_not_fitted_error(self):
        """Test error when accessing results before fitting."""
        capm = CAPM()
        
        with pytest.raises(ValueError, match="not been fitted"):
            capm.get_beta()
        
        with pytest.raises(ValueError, match="not been fitted"):
            capm.get_alpha()

    def test_insufficient_data(self, market_returns):
        """Test error with insufficient data."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        short_returns = pd.Series(np.random.randn(10) * 0.01, index=dates)
        short_market = market_returns.iloc[:10]
        
        capm = CAPM()
        with pytest.raises(ValueError, match="Insufficient data"):
            capm.fit(short_returns, short_market)

    def test_with_risk_free_rate(self, sample_returns, market_returns, risk_free_rate):
        """Test fitting with explicit risk-free rate."""
        capm = CAPM()
        capm.fit(sample_returns, market_returns, risk_free_rate)
        
        assert capm.is_fitted
        # Results should be slightly different with risk-free adjustment
        assert capm.get_beta() is not None

    def test_factor_names(self):
        """Test factor names method."""
        capm = CAPM()
        factors = capm.get_factor_names()
        
        assert factors == ["Mkt-RF"]
