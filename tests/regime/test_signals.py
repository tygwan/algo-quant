"""Tests for regime signal generator."""

import pytest
import numpy as np
import pandas as pd

from src.regime.rule_based import MarketRegime, RegimeClassification
from src.regime.signals import (
    RegimeSignalGenerator,
    AllocationSignal,
)


class TestRegimeSignalGenerator:
    """Test cases for RegimeSignalGenerator."""

    @pytest.fixture
    def generator(self):
        """Create signal generator instance."""
        return RegimeSignalGenerator()

    @pytest.fixture
    def expansion_classification(self):
        """Create expansion regime classification."""
        return RegimeClassification(
            regime=MarketRegime.EXPANSION,
            confidence=0.8,
            indicators={"gdp_growth": 3.0, "unemployment": 4.0},
            timestamp=pd.Timestamp("2020-01-01"),
        )

    @pytest.fixture
    def contraction_classification(self):
        """Create contraction regime classification."""
        return RegimeClassification(
            regime=MarketRegime.CONTRACTION,
            confidence=0.7,
            indicators={"gdp_growth": -1.0, "unemployment": 8.0},
            timestamp=pd.Timestamp("2020-01-01"),
        )

    @pytest.fixture
    def low_confidence_classification(self):
        """Create low confidence classification."""
        return RegimeClassification(
            regime=MarketRegime.EXPANSION,
            confidence=0.3,  # Below default min_confidence
            indicators={},
            timestamp=pd.Timestamp("2020-01-01"),
        )

    def test_generate_allocation_expansion(self, generator, expansion_classification):
        """Test allocation signal for expansion."""
        signal = generator.generate_allocation_signal(expansion_classification)
        
        assert isinstance(signal, AllocationSignal)
        assert signal.regime == MarketRegime.EXPANSION
        # Expansion should have high equity allocation
        assert signal.allocations["equity"] >= 0.5
        assert signal.allocations["cash"] < 0.2

    def test_generate_allocation_contraction(self, generator, contraction_classification):
        """Test allocation signal for contraction."""
        signal = generator.generate_allocation_signal(contraction_classification)
        
        # Contraction should have defensive allocation
        assert signal.allocations["equity"] < 0.4
        assert signal.allocations["fixed_income"] > 0.3
        assert signal.allocations["cash"] > 0.1

    def test_allocation_sums_to_one(self, generator, expansion_classification):
        """Test that allocation sums to 1."""
        signal = generator.generate_allocation_signal(expansion_classification)
        
        total = signal.total_allocation()
        assert abs(total - 1.0) < 0.01

    def test_low_confidence_blending(self, generator, low_confidence_classification):
        """Test blending with neutral allocation for low confidence."""
        signal = generator.generate_allocation_signal(
            low_confidence_classification,
            blend_with_neutral=True
        )
        
        # Should be blended towards neutral allocation
        assert signal.allocations["equity"] < 0.7  # Less aggressive than pure expansion

    def test_generate_risk_signal(self, generator, expansion_classification):
        """Test risk signal generation."""
        signal = generator.generate_risk_signal(
            expansion_classification,
            current_risk=0.5
        )
        
        assert "current_risk" in signal
        assert "target_risk" in signal
        assert "action" in signal
        assert signal["action"] in ["reduce", "increase", "maintain"]

    def test_generate_transition_signals(self, generator):
        """Test transition signal generation."""
        dates = pd.date_range("2020-01-01", periods=12, freq="ME")
        classifications = pd.DataFrame({
            "regime": ["expansion"] * 4 + ["peak"] * 4 + ["contraction"] * 4,
            "confidence": [0.8] * 12,
        }, index=dates)

        signals = generator.generate_transition_signals(classifications)

        assert "regime_change" in signals.columns
        assert "transition_signal" in signals.columns
        assert "regime_duration" in signals.columns

        # First row is change from NaN, plus 2 actual transitions = 3 total
        # Or skip first row: expansion->peak and peak->contraction = 2
        assert signals["regime_change"].iloc[1:].sum() == 2

    def test_backtest_signals(self, generator):
        """Test signal backtesting."""
        dates = pd.date_range("2020-01-01", periods=12, freq="ME")
        
        classifications = pd.DataFrame({
            "regime": ["expansion"] * 6 + ["contraction"] * 6,
            "confidence": [0.8] * 12,
        }, index=dates)
        
        returns = pd.DataFrame({
            "equity": np.random.normal(0.01, 0.05, 12),
            "fixed_income": np.random.normal(0.003, 0.01, 12),
            "commodities": np.random.normal(0.005, 0.03, 12),
            "cash": np.full(12, 0.001),
        }, index=dates)
        
        results = generator.backtest_signals(
            classifications,
            returns,
            initial_capital=100000
        )
        
        assert "total_return" in results
        assert "sharpe_ratio" in results
        assert "max_drawdown" in results
        assert "final_value" in results

    def test_custom_allocations(self):
        """Test generator with custom allocations."""
        custom_allocations = {
            MarketRegime.EXPANSION: {
                "equity": 0.90,
                "fixed_income": 0.05,
                "cash": 0.05,
            },
        }
        
        generator = RegimeSignalGenerator(allocations=custom_allocations)
        
        classification = RegimeClassification(
            regime=MarketRegime.EXPANSION,
            confidence=0.8,
            indicators={},
            timestamp=pd.Timestamp("2020-01-01"),
        )
        
        signal = generator.generate_allocation_signal(classification)
        
        assert signal.allocations["equity"] == 0.90


class TestAllocationSignal:
    """Test AllocationSignal dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        signal = AllocationSignal(
            regime=MarketRegime.EXPANSION,
            allocations={"equity": 0.7, "fixed_income": 0.2, "cash": 0.1},
            risk_level=0.7,
            confidence=0.8,
            timestamp=pd.Timestamp("2020-01-01"),
            metadata={"test": "value"},
        )
        
        d = signal.to_dict()
        
        assert d["regime"] == "expansion"
        assert d["allocations"]["equity"] == 0.7
        assert d["risk_level"] == 0.7
        assert d["metadata"]["test"] == "value"

    def test_total_allocation(self):
        """Test total allocation calculation."""
        signal = AllocationSignal(
            regime=MarketRegime.EXPANSION,
            allocations={"equity": 0.7, "fixed_income": 0.2, "cash": 0.1},
            risk_level=0.7,
            confidence=0.8,
            timestamp=pd.Timestamp("2020-01-01"),
        )
        
        assert abs(signal.total_allocation() - 1.0) < 0.01
