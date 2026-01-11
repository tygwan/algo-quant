"""Tests for regime classifiers."""

import pytest
import numpy as np
import pandas as pd

from src.regime.rule_based import (
    RuleBasedClassifier,
    YieldCurveClassifier,
    MarketRegime,
    RegimeClassification,
)


class TestRuleBasedClassifier:
    """Test cases for RuleBasedClassifier."""

    @pytest.fixture
    def classifier(self):
        """Create classifier instance."""
        return RuleBasedClassifier()

    @pytest.fixture
    def expansion_indicators(self):
        """Create indicators for expansion regime."""
        dates = pd.date_range("2020-01-01", periods=24, freq="M")
        return pd.DataFrame({
            "gdp_growth": [3.0] * 24,  # Strong growth
            "unemployment": [4.0] * 24,  # Low unemployment
            "unemployment_mom": [-0.1] * 24,  # Falling unemployment
            "yield_spread": [1.5] * 24,  # Normal spread
            "inflation_yoy": [2.0] * 24,  # Moderate inflation
            "diffusion_index": [65] * 24,  # Expanding
        }, index=dates)

    @pytest.fixture
    def contraction_indicators(self):
        """Create indicators for contraction regime."""
        dates = pd.date_range("2020-01-01", periods=24, freq="M")
        return pd.DataFrame({
            "gdp_growth": [-1.5] * 24,  # Negative growth
            "unemployment": [8.0] * 24,  # High unemployment
            "unemployment_mom": [0.3] * 24,  # Rising unemployment
            "yield_spread": [-0.5] * 24,  # Inverted curve
            "inflation_yoy": [0.5] * 24,  # Low inflation
            "diffusion_index": [25] * 24,  # Contracting
        }, index=dates)

    def test_classify_expansion(self, classifier, expansion_indicators):
        """Test classification of expansion regime."""
        result = classifier.classify(expansion_indicators)
        
        assert isinstance(result, RegimeClassification)
        assert result.regime == MarketRegime.EXPANSION
        assert result.confidence > 0.5

    def test_classify_contraction(self, classifier, contraction_indicators):
        """Test classification of contraction regime."""
        result = classifier.classify(contraction_indicators)
        
        assert result.regime == MarketRegime.CONTRACTION
        assert result.confidence > 0.5

    def test_classify_with_date(self, classifier, expansion_indicators):
        """Test classification at specific date."""
        target_date = expansion_indicators.index[12]
        result = classifier.classify(expansion_indicators, date=target_date)
        
        assert result.timestamp == target_date

    def test_classify_history(self, classifier, expansion_indicators):
        """Test historical classification."""
        history = classifier.classify_history(expansion_indicators)
        
        assert len(history) == len(expansion_indicators)
        assert "regime" in history.columns
        assert "confidence" in history.columns

    def test_classify_missing_indicators(self, classifier):
        """Test classification with missing indicators."""
        dates = pd.date_range("2020-01-01", periods=12, freq="M")
        sparse_indicators = pd.DataFrame({
            "gdp_growth": [2.0] * 12,  # Only one indicator
        }, index=dates)
        
        result = classifier.classify(sparse_indicators)
        
        # Should still produce a result
        assert result.regime is not None

    def test_custom_thresholds(self):
        """Test classifier with custom thresholds."""
        custom_thresholds = {
            "gdp_growth_positive": 1.0,  # Higher threshold
            "unemployment_low": 4.0,
        }
        classifier = RuleBasedClassifier(thresholds=custom_thresholds)
        
        assert classifier.thresholds["gdp_growth_positive"] == 1.0
        assert classifier.thresholds["unemployment_low"] == 4.0

    def test_disable_yield_curve(self):
        """Test classifier without yield curve."""
        classifier = RuleBasedClassifier(use_yield_curve=False)
        dates = pd.date_range("2020-01-01", periods=12, freq="M")
        indicators = pd.DataFrame({
            "gdp_growth": [3.0] * 12,
            "unemployment": [4.0] * 12,
            "yield_spread": [-1.0] * 12,  # Would indicate peak/contraction
        }, index=dates)
        
        result = classifier.classify(indicators)
        
        # Without yield curve, should lean towards expansion
        assert result.regime in [MarketRegime.EXPANSION, MarketRegime.TROUGH]


class TestYieldCurveClassifier:
    """Test cases for YieldCurveClassifier."""

    @pytest.fixture
    def classifier(self):
        """Create classifier instance."""
        return YieldCurveClassifier()

    @pytest.fixture
    def sample_indicators(self):
        """Create sample yield curve data."""
        dates = pd.date_range("2020-01-01", periods=36, freq="M")
        # Simulate cycle: steep -> flat -> inverted -> steep
        spreads = list(np.linspace(2.0, 0.0, 12)) + \
                  list(np.linspace(0.0, -0.5, 12)) + \
                  list(np.linspace(-0.5, 2.0, 12))
        
        return pd.DataFrame({
            "yield_spread": spreads,
        }, index=dates)

    def test_classify_steep_curve(self, classifier):
        """Test classification with steep yield curve."""
        dates = pd.date_range("2020-01-01", periods=12, freq="M")
        indicators = pd.DataFrame({
            "yield_spread": [2.5] * 12,
        }, index=dates)
        
        result = classifier.classify(indicators)
        
        # Steep curve should indicate trough or expansion
        assert result.regime in [MarketRegime.TROUGH, MarketRegime.EXPANSION]

    def test_classify_inverted_curve(self, classifier):
        """Test classification with inverted yield curve."""
        dates = pd.date_range("2020-01-01", periods=12, freq="M")
        indicators = pd.DataFrame({
            "yield_spread": [-0.5] * 12,
        }, index=dates)
        
        result = classifier.classify(indicators)
        
        # Inverted curve should indicate peak or contraction
        assert result.regime in [MarketRegime.PEAK, MarketRegime.CONTRACTION]

    def test_classify_history(self, classifier, sample_indicators):
        """Test historical classification."""
        history = classifier.classify_history(sample_indicators)
        
        assert len(history) == len(sample_indicators)
        assert "regime" in history.columns

    def test_calculate_recession_signal(self, classifier, sample_indicators):
        """Test recession signal calculation."""
        signals = classifier.calculate_recession_signal(sample_indicators["yield_spread"])
        
        assert "inverted" in signals.columns
        assert "recession_prob" in signals.columns
        assert "inversion_duration" in signals.columns

    def test_missing_yield_spread(self, classifier):
        """Test classification without yield spread data."""
        dates = pd.date_range("2020-01-01", periods=12, freq="M")
        indicators = pd.DataFrame({
            "gdp_growth": [2.0] * 12,  # No yield_spread
        }, index=dates)
        
        result = classifier.classify(indicators)
        
        assert result.regime == MarketRegime.UNKNOWN


class TestMarketRegime:
    """Test MarketRegime enum."""

    def test_regime_values(self):
        """Test regime enum values."""
        assert MarketRegime.EXPANSION.value == "expansion"
        assert MarketRegime.CONTRACTION.value == "contraction"
        assert MarketRegime.PEAK.value == "peak"
        assert MarketRegime.TROUGH.value == "trough"

    def test_regime_string(self):
        """Test regime string representation."""
        assert str(MarketRegime.EXPANSION) == "expansion"


class TestRegimeClassification:
    """Test RegimeClassification dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        classification = RegimeClassification(
            regime=MarketRegime.EXPANSION,
            confidence=0.8,
            indicators={"gdp_growth": 3.0},
            timestamp=pd.Timestamp("2020-01-01"),
            details="Test",
        )
        
        d = classification.to_dict()
        
        assert d["regime"] == "expansion"
        assert d["confidence"] == 0.8
        assert d["indicators"]["gdp_growth"] == 3.0
