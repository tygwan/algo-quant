"""Tests for macro indicator processor."""

import pytest
import numpy as np
import pandas as pd

from src.regime.indicators import MacroIndicatorProcessor, EconomicIndicators


class TestMacroIndicatorProcessor:
    """Test cases for MacroIndicatorProcessor."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return MacroIndicatorProcessor(lookback_window=12)

    @pytest.fixture
    def sample_series(self):
        """Create sample time series."""
        dates = pd.date_range("2020-01-01", periods=60, freq="M")
        np.random.seed(42)
        values = np.cumsum(np.random.randn(60)) + 100
        return pd.Series(values, index=dates)

    @pytest.fixture
    def sample_indicators(self):
        """Create sample indicator DataFrame."""
        dates = pd.date_range("2020-01-01", periods=60, freq="M")
        np.random.seed(42)
        
        return pd.DataFrame({
            "gdp_growth": np.random.normal(2.0, 1.0, 60),
            "unemployment": np.random.normal(5.0, 1.0, 60).clip(3, 10),
            "inflation_yoy": np.random.normal(2.0, 0.5, 60),
            "yield_spread": np.random.normal(1.0, 0.5, 60),
        }, index=dates)

    def test_normalize_zscore(self, processor, sample_series):
        """Test z-score normalization."""
        normalized = processor.normalize(sample_series, method="zscore")
        
        # After normalization, should have ~0 mean and ~1 std (within window)
        assert len(normalized) == len(sample_series)
        # Check that values are reasonable z-scores
        valid_values = normalized.dropna()
        assert valid_values.abs().mean() < 3  # Most z-scores within 3

    def test_normalize_minmax(self, processor, sample_series):
        """Test min-max normalization."""
        normalized = processor.normalize(sample_series, method="minmax")
        
        valid_values = normalized.dropna()
        assert valid_values.min() >= 0
        assert valid_values.max() <= 1

    def test_normalize_percentile(self, processor, sample_series):
        """Test percentile normalization."""
        normalized = processor.normalize(sample_series, method="percentile")
        
        valid_values = normalized.dropna()
        assert valid_values.min() >= 0
        assert valid_values.max() <= 1

    def test_calculate_momentum_diff(self, processor, sample_series):
        """Test momentum calculation with diff."""
        momentum = processor.calculate_momentum(sample_series, periods=3, method="diff")
        
        assert len(momentum) == len(sample_series)
        # First 3 values should be NaN
        assert momentum.iloc[:3].isna().all()

    def test_calculate_momentum_pct(self, processor, sample_series):
        """Test momentum calculation with percentage."""
        momentum = processor.calculate_momentum(sample_series, periods=3, method="pct")
        
        valid = momentum.dropna()
        # Percentage changes should be reasonable
        assert abs(valid.mean()) < 1  # Not 100% changes typically

    def test_calculate_trend(self, processor, sample_series):
        """Test trend calculation."""
        trend = processor.calculate_trend(sample_series, short_window=3, long_window=12)
        
        assert len(trend) == len(sample_series)
        # Trend should be -1, 0, or 1
        assert set(trend.unique()).issubset({-1, 0, 1})

    def test_calculate_acceleration(self, processor, sample_series):
        """Test acceleration calculation."""
        acceleration = processor.calculate_acceleration(sample_series, periods=3)
        
        assert len(acceleration) == len(sample_series)

    def test_process_indicator(self, processor, sample_series):
        """Test full indicator processing."""
        processed = processor.process_indicator(sample_series, name="test")
        
        assert processed.raw is not None
        assert processed.normalized is not None
        assert processed.momentum is not None
        assert processed.trend is not None
        
        df = processed.to_dataframe()
        assert "raw" in df.columns
        assert "normalized" in df.columns

    def test_process_indicator_inverted(self, processor, sample_series):
        """Test inverted indicator processing."""
        processed = processor.process_indicator(sample_series, invert=True)
        
        # Inverted should be opposite sign
        assert (processed.raw.dropna() == -sample_series.dropna()).all()

    def test_create_composite_index(self, processor, sample_indicators):
        """Test composite index creation."""
        indicators = {
            col: sample_indicators[col]
            for col in ["gdp_growth", "unemployment"]
        }
        
        composite = processor.create_composite_index(
            indicators,
            invert_list=["unemployment"]
        )
        
        assert len(composite) == len(sample_indicators)

    def test_detect_yield_curve_inversion(self, processor):
        """Test yield curve inversion detection."""
        dates = pd.date_range("2020-01-01", periods=20, freq="M")
        # Create spread that goes from positive to negative
        spread = pd.Series([1.0, 0.8, 0.5, 0.2, 0.0, -0.1, -0.2, -0.3, -0.2, -0.1,
                           0.0, 0.2, 0.5, 0.7, 1.0, 1.2, 1.3, 1.2, 1.0, 0.8], index=dates)
        
        inverted = processor.detect_yield_curve_inversion(spread, threshold=0.0)
        
        # Should detect inversions where spread < 0
        assert inverted.iloc[5:10].any()  # Some inversions in this range
        assert not inverted.iloc[0]  # First value not inverted

    def test_calculate_recession_probability(self, processor):
        """Test recession probability calculation."""
        dates = pd.date_range("2020-01-01", periods=10, freq="M")
        spread = pd.Series([2.0, 1.0, 0.5, 0.0, -0.5, -1.0, 0.0, 0.5, 1.0, 1.5], index=dates)
        
        prob = processor.calculate_recession_probability(spread, model="simple")
        
        # Probability should be 0-1
        assert (prob >= 0).all()
        assert (prob <= 1).all()
        # Negative spread should have higher probability
        assert prob.iloc[5] > prob.iloc[0]

    def test_calculate_diffusion_index(self, processor, sample_indicators):
        """Test diffusion index calculation."""
        indicators = {
            col: sample_indicators[col]
            for col in sample_indicators.columns
        }
        
        diffusion = processor.calculate_diffusion_index(indicators)
        
        # Diffusion should be 0-100
        valid = diffusion.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()


class TestEconomicIndicators:
    """Test economic indicator constants."""

    def test_indicator_constants_exist(self):
        """Test that key indicator constants are defined."""
        assert hasattr(EconomicIndicators, "GDP")
        assert hasattr(EconomicIndicators, "UNRATE")
        assert hasattr(EconomicIndicators, "FEDFUNDS")
        assert hasattr(EconomicIndicators, "T10Y2Y")

    def test_indicator_values_are_strings(self):
        """Test that indicator values are valid FRED series IDs."""
        assert isinstance(EconomicIndicators.GDP, str)
        assert isinstance(EconomicIndicators.UNRATE, str)
        assert len(EconomicIndicators.GDP) > 0
