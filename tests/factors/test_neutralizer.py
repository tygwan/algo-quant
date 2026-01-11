"""Tests for factor neutralization."""

import pytest
import numpy as np
import pandas as pd

from src.factors.neutralizer import FactorNeutralizer


class TestFactorNeutralizer:
    """Test cases for FactorNeutralizer."""

    @pytest.fixture
    def sample_weights(self):
        """Create sample portfolio weights."""
        return pd.Series({
            "AAPL": 0.20,
            "MSFT": 0.15,
            "GOOGL": 0.15,
            "AMZN": 0.15,
            "META": 0.10,
            "NVDA": 0.10,
            "TSLA": 0.15,
        })

    @pytest.fixture
    def sample_betas(self):
        """Create sample stock betas."""
        return pd.Series({
            "AAPL": 1.2,
            "MSFT": 1.1,
            "GOOGL": 1.3,
            "AMZN": 1.4,
            "META": 1.5,
            "NVDA": 1.8,
            "TSLA": 2.0,
        })

    @pytest.fixture
    def sample_factor_loadings(self):
        """Create sample multi-factor loadings."""
        return pd.DataFrame({
            "Mkt-RF": [1.2, 1.1, 1.3, 1.4, 1.5, 1.8, 2.0],
            "SMB": [-0.2, -0.3, -0.1, -0.2, 0.1, 0.3, 0.4],
            "HML": [0.1, 0.2, -0.3, -0.4, -0.2, -0.5, -0.6],
        }, index=["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"])

    @pytest.fixture
    def sample_scores(self):
        """Create sample stock scores for long-short portfolio."""
        return pd.Series({
            "AAPL": 0.8,
            "MSFT": 0.7,
            "GOOGL": 0.6,
            "AMZN": 0.4,
            "META": 0.3,
            "NVDA": 0.2,
            "TSLA": 0.1,
        })

    def test_neutralize_single_factor(self, sample_weights, sample_betas):
        """Test single factor neutralization."""
        neutralizer = FactorNeutralizer()
        
        neutral_weights = neutralizer.neutralize_single_factor(
            sample_weights, sample_betas, target_loading=0.0
        )
        
        # Portfolio beta should be close to 0
        portfolio_beta = (neutral_weights * sample_betas.loc[neutral_weights.index]).sum()
        assert abs(portfolio_beta) < 0.01
        
        # Weights should still sum to 1
        assert abs(neutral_weights.sum() - 1.0) < 0.01

    def test_neutralize_to_target_beta(self, sample_weights, sample_betas):
        """Test neutralization to specific target beta."""
        neutralizer = FactorNeutralizer()
        target = 1.0
        
        neutral_weights = neutralizer.neutralize_single_factor(
            sample_weights, sample_betas, target_loading=target
        )
        
        portfolio_beta = (neutral_weights * sample_betas.loc[neutral_weights.index]).sum()
        assert abs(portfolio_beta - target) < 0.01

    def test_neutralize_multi_factor(self, sample_weights, sample_factor_loadings):
        """Test multi-factor neutralization."""
        neutralizer = FactorNeutralizer()
        
        neutral_weights = neutralizer.neutralize_multi_factor(
            sample_weights,
            sample_factor_loadings,
            target_loadings={"Mkt-RF": 1.0, "SMB": 0.0, "HML": 0.0}
        )
        
        # Verify loadings
        loadings = neutralizer.calculate_portfolio_loadings(
            neutral_weights, sample_factor_loadings
        )
        
        assert abs(loadings["Mkt-RF"] - 1.0) < 0.02
        assert abs(loadings["SMB"]) < 0.02
        assert abs(loadings["HML"]) < 0.02

    def test_long_short_portfolio_equal_weight(self, sample_scores, sample_betas):
        """Test long-short portfolio with equal weights."""
        neutralizer = FactorNeutralizer()
        
        weights = neutralizer.create_long_short_portfolio(
            sample_scores,
            long_pct=0.3,  # Top 30%
            short_pct=0.3,  # Bottom 30%
            method="equal"
        )
        
        # Should have long and short positions
        assert (weights > 0).any()
        assert (weights < 0).any()
        
        # Should be dollar-neutral
        assert abs(weights.sum()) < 0.01

    def test_long_short_with_neutralization(self, sample_scores, sample_betas):
        """Test long-short portfolio with beta neutralization."""
        neutralizer = FactorNeutralizer()
        
        weights = neutralizer.create_long_short_portfolio(
            sample_scores,
            factor_loadings=sample_betas,
            target_loadings=None,  # Will default to 0
            long_pct=0.3,
            short_pct=0.3,
        )
        
        # Should be beta-neutral
        portfolio_beta = (weights * sample_betas.loc[weights.index]).sum()
        assert abs(portfolio_beta) < 0.1

    def test_calculate_portfolio_loadings(self, sample_weights, sample_factor_loadings):
        """Test portfolio loading calculation."""
        neutralizer = FactorNeutralizer()
        
        loadings = neutralizer.calculate_portfolio_loadings(
            sample_weights, sample_factor_loadings
        )
        
        # Manual calculation for Mkt-RF
        expected_mkt = sum(
            sample_weights[asset] * sample_factor_loadings.loc[asset, "Mkt-RF"]
            for asset in sample_weights.index
        )
        
        assert abs(loadings["Mkt-RF"] - expected_mkt) < 0.01

    def test_verify_neutralization(self, sample_weights, sample_factor_loadings):
        """Test neutralization verification."""
        neutralizer = FactorNeutralizer()
        
        # First neutralize
        neutral_weights = neutralizer.neutralize_multi_factor(
            sample_weights,
            sample_factor_loadings,
            target_loadings={"Mkt-RF": 1.0, "SMB": 0.0}
        )
        
        # Then verify
        verification = neutralizer.verify_neutralization(
            neutral_weights,
            sample_factor_loadings,
            tolerance=0.05
        )
        
        assert verification["SMB_neutral"] is True
        assert abs(verification["SMB_loading"]) < 0.05

    def test_position_bounds(self, sample_weights, sample_betas):
        """Test position bounds are respected."""
        neutralizer = FactorNeutralizer(
            max_position=0.25,
            min_position=-0.1,
            allow_short=True
        )
        
        neutral_weights = neutralizer.neutralize_single_factor(
            sample_weights, sample_betas, target_loading=0.0
        )
        
        assert neutral_weights.max() <= 0.25 + 0.01
        assert neutral_weights.min() >= -0.1 - 0.01

    def test_no_short_constraint(self, sample_weights, sample_betas):
        """Test no-short constraint."""
        neutralizer = FactorNeutralizer(allow_short=False)
        
        neutral_weights = neutralizer.neutralize_single_factor(
            sample_weights, sample_betas, target_loading=1.0  # Target = 1.0 achievable without shorts
        )
        
        assert (neutral_weights >= -0.001).all()  # Small tolerance
