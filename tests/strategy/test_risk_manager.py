"""Tests for risk manager."""

import pytest
import numpy as np
import pandas as pd

from src.strategy.risk_manager import (
    RiskManager,
    PositionSizer,
    PositionSizingMethod,
    StopLoss,
)


class TestPositionSizer:
    """Test cases for PositionSizer."""

    @pytest.fixture
    def sizer(self):
        """Create position sizer instance."""
        return PositionSizer(
            method=PositionSizingMethod.FIXED_FRACTIONAL,
            risk_per_trade=0.02,
        )

    def test_fixed_fractional(self, sizer):
        """Test fixed fractional sizing."""
        size = sizer.calculate_size(
            capital=100000,
            price=100,
            volatility=0.20,
            stop_distance=0.05,
        )

        # Size = risk_per_trade / stop_distance = 0.02 / 0.05 = 0.4
        # But max_position = 0.20 (default), so size is capped at 0.20
        expected = min(0.02 / 0.05, sizer.max_position)
        assert abs(size.size - expected) < 0.01

    def test_volatility_target(self):
        """Test volatility targeting sizing."""
        sizer = PositionSizer(
            method=PositionSizingMethod.VOLATILITY_TARGET,
            target_volatility=0.15,
        )
        
        size = sizer.calculate_size(
            capital=100000,
            price=100,
            volatility=0.30,  # Higher than target
        )
        
        # Should scale down since vol > target
        assert size.size < 1.0

    def test_kelly(self):
        """Test Kelly criterion sizing."""
        sizer = PositionSizer(
            method=PositionSizingMethod.KELLY,
            kelly_fraction=0.5,
        )
        
        size = sizer.calculate_size(
            capital=100000,
            price=100,
            volatility=0.20,
            win_rate=0.6,
            avg_win=0.10,
            avg_loss=0.05,
        )
        
        assert size.size > 0
        assert size.size <= sizer.max_position

    def test_max_position_constraint(self, sizer):
        """Test max position constraint."""
        size = sizer.calculate_size(
            capital=100000,
            price=100,
            volatility=0.05,  # Very low vol would give large size
            stop_distance=0.01,
        )
        
        assert size.size <= sizer.max_position


class TestRiskManager:
    """Test cases for RiskManager."""

    @pytest.fixture
    def risk_mgr(self):
        """Create risk manager instance."""
        return RiskManager(
            max_drawdown=0.20,
            max_position_size=0.20,
            max_sector_exposure=0.40,
        )

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        
        return pd.DataFrame({
            "AAPL": np.random.normal(0.001, 0.02, 252),
            "MSFT": np.random.normal(0.0008, 0.018, 252),
            "GOOGL": np.random.normal(0.0009, 0.022, 252),
        }, index=dates)

    def test_check_position_limits(self, risk_mgr):
        """Test position limit checking."""
        # All positions exceed max_position_size (0.20)
        weights = {"AAPL": 0.25, "MSFT": 0.50, "GOOGL": 0.25}

        violations = risk_mgr.check_position_limits(weights)

        # All 3 positions should be in exceeded list
        assert len(violations["exceeded"]) == 3
        # MSFT has the largest violation (50%)
        assert any("MSFT" in v for v in violations["exceeded"])

    def test_check_sector_limits(self, risk_mgr):
        """Test sector limit checking."""
        weights = {"AAPL": 0.30, "MSFT": 0.30, "NVDA": 0.40}
        sectors = {
            "AAPL": "tech",
            "MSFT": "tech",
            "NVDA": "tech",
        }  # All tech = 100%
        
        violations = risk_mgr.check_sector_limits(weights, sectors)
        
        assert len(violations["exceeded"]) > 0

    def test_calculate_var(self, risk_mgr, sample_returns):
        """Test VaR calculation."""
        weights = pd.Series({"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2})
        
        var = risk_mgr.calculate_var(sample_returns, weights)
        
        assert var > 0
        assert var < 0.5  # Reasonable VaR

    def test_update_drawdown(self, risk_mgr):
        """Test drawdown tracking."""
        # Simulate portfolio value changes
        risk_mgr.update_drawdown(100000)
        risk_mgr.update_drawdown(105000)  # New high
        dd = risk_mgr.update_drawdown(95000)  # Drawdown
        
        expected_dd = (105000 - 95000) / 105000
        assert abs(dd - expected_dd) < 0.001

    def test_check_drawdown_limit(self, risk_mgr):
        """Test drawdown limit check."""
        risk_mgr.update_drawdown(100000)
        risk_mgr.update_drawdown(105000)
        
        # 10% drawdown - within limit
        assert risk_mgr.check_drawdown_limit(94500) is True
        
        # 25% drawdown - exceeds limit
        assert risk_mgr.check_drawdown_limit(78750) is False

    def test_calculate_stop_levels(self, risk_mgr):
        """Test stop loss calculation."""
        positions = {
            "AAPL": {
                "entry_price": 150,
                "current_price": 160,
                "high_since_entry": 165,
                "atr": 3.0,
            },
        }
        
        stops = risk_mgr.calculate_stop_levels(positions, method="trailing")
        
        assert "AAPL" in stops
        assert isinstance(stops["AAPL"], StopLoss)
        # Trailing stop should be below high
        assert stops["AAPL"].current_stop < 165

    def test_volatility_adjustment(self, risk_mgr):
        """Test volatility-based weight adjustment."""
        weights = {"AAPL": 0.5, "MSFT": 0.5}
        volatilities = {"AAPL": 0.25, "MSFT": 0.25}  # Combined > target
        
        adjusted = risk_mgr.adjust_for_volatility(weights, volatilities)
        
        # Should scale down to meet target
        assert adjusted.sum() <= 1.01

    def test_generate_risk_report(self, risk_mgr, sample_returns):
        """Test risk report generation."""
        weights = pd.Series({"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.25})
        
        report = risk_mgr.generate_risk_report(
            portfolio_value=100000,
            weights=weights,
            returns=sample_returns,
        )
        
        assert "drawdown" in report
        assert "var" in report
        assert "volatility" in report
        assert "position_violations" in report


class TestStopLoss:
    """Test cases for StopLoss."""

    def test_stop_triggered(self):
        """Test stop loss triggering."""
        stop = StopLoss(
            type="trailing",
            level=0.10,
            initial_price=100,
            current_stop=90,
        )
        
        assert stop.is_triggered(85) is True
        assert stop.is_triggered(95) is False
