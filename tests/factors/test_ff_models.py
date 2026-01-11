"""Tests for Fama-French model implementations."""

import pytest
import numpy as np
import pandas as pd

from src.factors.ff3 import FamaFrench3
from src.factors.ff5 import FamaFrench5


class TestFamaFrench3:
    """Test cases for FF3 model."""

    @pytest.fixture
    def sample_factor_returns(self):
        """Create sample FF3 factor returns."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="M")
        
        return pd.DataFrame({
            "Mkt-RF": np.random.normal(0.005, 0.04, 100),
            "SMB": np.random.normal(0.002, 0.02, 100),
            "HML": np.random.normal(0.003, 0.02, 100),
            "RF": np.full(100, 0.002),
        }, index=dates)

    @pytest.fixture
    def sample_stock_returns(self, sample_factor_returns):
        """Create sample stock returns with known factor exposure."""
        np.random.seed(43)
        factors = sample_factor_returns
        
        # Generate returns with known loadings
        # beta = 1.1, smb = 0.5, hml = -0.3
        returns = (
            0.001  # alpha
            + 1.1 * factors["Mkt-RF"]
            + 0.5 * factors["SMB"]
            - 0.3 * factors["HML"]
            + np.random.normal(0, 0.01, len(factors))
        )
        
        return pd.Series(returns.values, index=factors.index)

    def test_fit_basic(self, sample_stock_returns, sample_factor_returns):
        """Test basic FF3 fitting."""
        model = FamaFrench3()
        model.fit(sample_stock_returns, sample_factor_returns)
        
        assert model.is_fitted
        assert len(model.get_loadings()) == 3

    def test_factor_loadings(self, sample_stock_returns, sample_factor_returns):
        """Test factor loading extraction."""
        model = FamaFrench3()
        model.fit(sample_stock_returns, sample_factor_returns)
        
        loadings = model.get_loadings()
        
        # Check loadings are close to true values
        assert abs(loadings["Mkt-RF"] - 1.1) < 0.3
        assert abs(loadings["SMB"] - 0.5) < 0.3
        assert abs(loadings["HML"] - (-0.3)) < 0.3

    def test_convenience_methods(self, sample_stock_returns, sample_factor_returns):
        """Test convenience accessor methods."""
        model = FamaFrench3()
        model.fit(sample_stock_returns, sample_factor_returns)
        
        assert model.get_market_beta() == model.get_loadings()["Mkt-RF"]
        assert model.get_size_loading() == model.get_loadings()["SMB"]
        assert model.get_value_loading() == model.get_loadings()["HML"]

    def test_risk_decomposition(self, sample_stock_returns, sample_factor_returns):
        """Test risk decomposition."""
        model = FamaFrench3()
        model.fit(sample_stock_returns, sample_factor_returns)
        
        decomp = model.decompose_risk()
        
        assert "total_variance" in decomp
        assert "factor_variance" in decomp
        assert "idiosyncratic_variance" in decomp
        assert decomp["systematic_risk_pct"] + decomp["idiosyncratic_risk_pct"] == pytest.approx(100)

    def test_missing_factors_error(self, sample_stock_returns):
        """Test error when factors are missing."""
        incomplete_factors = pd.DataFrame({
            "Mkt-RF": np.random.randn(100),
            "SMB": np.random.randn(100),
            # Missing HML
        })
        
        model = FamaFrench3()
        with pytest.raises(ValueError, match="Missing factors"):
            model.fit(sample_stock_returns, incomplete_factors)

    def test_factor_names(self):
        """Test factor names."""
        model = FamaFrench3()
        assert model.get_factor_names() == ["Mkt-RF", "SMB", "HML"]


class TestFamaFrench5:
    """Test cases for FF5 model."""

    @pytest.fixture
    def sample_factor_returns(self):
        """Create sample FF5 factor returns."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="M")
        
        return pd.DataFrame({
            "Mkt-RF": np.random.normal(0.005, 0.04, 100),
            "SMB": np.random.normal(0.002, 0.02, 100),
            "HML": np.random.normal(0.003, 0.02, 100),
            "RMW": np.random.normal(0.002, 0.015, 100),
            "CMA": np.random.normal(0.001, 0.015, 100),
            "RF": np.full(100, 0.002),
        }, index=dates)

    @pytest.fixture
    def sample_stock_returns(self, sample_factor_returns):
        """Create sample stock returns with known factor exposure."""
        np.random.seed(43)
        factors = sample_factor_returns
        
        # Generate returns with known loadings
        returns = (
            0.001  # alpha
            + 1.1 * factors["Mkt-RF"]
            + 0.5 * factors["SMB"]
            - 0.3 * factors["HML"]
            + 0.4 * factors["RMW"]
            - 0.2 * factors["CMA"]
            + np.random.normal(0, 0.01, len(factors))
        )
        
        return pd.Series(returns.values, index=factors.index)

    def test_fit_basic(self, sample_stock_returns, sample_factor_returns):
        """Test basic FF5 fitting."""
        model = FamaFrench5()
        model.fit(sample_stock_returns, sample_factor_returns)
        
        assert model.is_fitted
        assert len(model.get_loadings()) == 5

    def test_factor_loadings(self, sample_stock_returns, sample_factor_returns):
        """Test factor loading extraction."""
        model = FamaFrench5()
        model.fit(sample_stock_returns, sample_factor_returns)
        
        loadings = model.get_loadings()
        
        # Check loadings are close to true values
        assert abs(loadings["Mkt-RF"] - 1.1) < 0.3
        assert abs(loadings["RMW"] - 0.4) < 0.3
        assert abs(loadings["CMA"] - (-0.2)) < 0.3

    def test_convenience_methods(self, sample_stock_returns, sample_factor_returns):
        """Test convenience accessor methods."""
        model = FamaFrench5()
        model.fit(sample_stock_returns, sample_factor_returns)
        
        assert model.get_market_beta() == model.get_loadings()["Mkt-RF"]
        assert model.get_profitability_loading() == model.get_loadings()["RMW"]
        assert model.get_investment_loading() == model.get_loadings()["CMA"]

    def test_compare_with_ff3(self, sample_stock_returns, sample_factor_returns):
        """Test FF3 vs FF5 comparison."""
        model = FamaFrench5()
        comparison = model.compare_with_ff3(
            sample_stock_returns,
            sample_factor_returns
        )
        
        assert "ff3_r_squared" in comparison
        assert "ff5_r_squared" in comparison
        assert "r_squared_improvement" in comparison
        # FF5 should have higher or equal R²
        assert comparison["r_squared_improvement"] >= -0.01

    def test_factor_names(self):
        """Test factor names."""
        model = FamaFrench5()
        assert model.get_factor_names() == ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
