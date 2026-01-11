"""Tests for data preprocessor."""

import pytest
import numpy as np
import pandas as pd

from src.data.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Test cases for DataPreprocessor."""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return DataPreprocessor()

    @pytest.fixture
    def sample_df(self):
        """Create sample price DataFrame."""
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "open": [100, 101, np.nan, 103, 104, 105, 106, 107, 108, 109],
            "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            "low": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            "close": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "volume": [1000, 1100, np.nan, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        })

    @pytest.fixture
    def sample_prices(self):
        """Create sample price series."""
        return pd.Series([100, 102, 101, 105, 103, 107, 110, 108, 112, 115])

    # Missing Values Tests

    def test_handle_missing_ffill(self, preprocessor, sample_df):
        """Test forward fill for missing values."""
        result = preprocessor.handle_missing(sample_df, method="ffill")
        assert result["open"].isna().sum() == 0
        assert result.loc[2, "open"] == 101  # Forward filled

    def test_handle_missing_interpolate(self, preprocessor, sample_df):
        """Test interpolation for missing values."""
        result = preprocessor.handle_missing(sample_df, method="interpolate")
        assert result["open"].isna().sum() == 0

    def test_handle_missing_drop(self, preprocessor, sample_df):
        """Test dropping rows with missing values."""
        result = preprocessor.handle_missing(sample_df, method="drop")
        assert len(result) == 8  # 2 rows with NaN dropped

    def test_fill_missing_prices(self, preprocessor, sample_df):
        """Test specialized price filling."""
        result = preprocessor.fill_missing_prices(sample_df)
        assert result["open"].isna().sum() == 0
        assert result["volume"].loc[2] == 0  # Volume filled with 0, not ffill

    # Outlier Tests

    def test_detect_outliers_zscore(self, preprocessor):
        """Test Z-score outlier detection."""
        series = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is outlier
        outliers = preprocessor.detect_outliers_zscore(series, threshold=2.0)
        assert outliers.iloc[-1] == True
        assert outliers.iloc[0] == False

    def test_detect_outliers_iqr(self, preprocessor):
        """Test IQR outlier detection."""
        series = pd.Series([1, 2, 3, 4, 5, 50])  # 50 is outlier
        outliers = preprocessor.detect_outliers_iqr(series, k=1.5)
        assert outliers.iloc[-1] == True

    def test_handle_outliers_clip(self, preprocessor):
        """Test outlier clipping."""
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 100]})
        result = preprocessor.handle_outliers(df, columns=["value"], action="clip")
        assert result["value"].max() < 100

    # Returns Calculation Tests

    def test_calculate_returns_simple(self, preprocessor, sample_prices):
        """Test simple returns calculation."""
        returns = preprocessor.calculate_returns(sample_prices, method="simple")
        expected_first = (102 - 100) / 100
        assert abs(returns.iloc[1] - expected_first) < 0.001

    def test_calculate_returns_log(self, preprocessor, sample_prices):
        """Test log returns calculation."""
        returns = preprocessor.calculate_returns(sample_prices, method="log")
        expected_first = np.log(102 / 100)
        assert abs(returns.iloc[1] - expected_first) < 0.001

    def test_calculate_cumulative_returns(self, preprocessor):
        """Test cumulative returns calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        cum_returns = preprocessor.calculate_cumulative_returns(returns)
        # After 4 periods with these returns
        assert cum_returns.iloc[-1] > 0

    def test_calculate_rolling_returns(self, preprocessor, sample_prices):
        """Test rolling returns calculation."""
        result = preprocessor.calculate_rolling_returns(sample_prices, windows=[3, 5])
        assert "return_3d" in result.columns
        assert "return_5d" in result.columns

    # Normalization Tests

    def test_normalize_minmax(self, preprocessor):
        """Test min-max normalization."""
        df = pd.DataFrame({"value": [10, 20, 30, 40, 50]})
        result = preprocessor.normalize(df, method="minmax")
        assert result["value"].min() == 0.0
        assert result["value"].max() == 1.0

    def test_normalize_zscore(self, preprocessor):
        """Test z-score standardization."""
        df = pd.DataFrame({"value": [10, 20, 30, 40, 50]})
        result = preprocessor.normalize(df, method="zscore")
        assert abs(result["value"].mean()) < 0.001
        assert abs(result["value"].std() - 1.0) < 0.001

    def test_winsorize(self, preprocessor):
        """Test winsorization."""
        df = pd.DataFrame({"value": list(range(100))})
        result = preprocessor.winsorize(df, limits=(0.05, 0.95))
        assert result["value"].min() >= 4  # 5th percentile
        assert result["value"].max() <= 95  # 95th percentile

    # Resampling Tests

    def test_resample_ohlcv(self, preprocessor, sample_df):
        """Test OHLCV resampling."""
        sample_df = sample_df.dropna()  # Remove NaN for clean resample
        result = preprocessor.resample_ohlcv(sample_df, freq="W")
        assert len(result) < len(sample_df)
        assert "open" in result.columns
        assert "close" in result.columns

    # Pipeline Tests

    def test_process_price_data(self, preprocessor, sample_df):
        """Test complete preprocessing pipeline."""
        result = preprocessor.process_price_data(
            sample_df,
            handle_missing=True,
            calculate_returns=True,
        )
        assert "returns" in result.columns
        assert "log_returns" in result.columns
        assert result["open"].isna().sum() == 0
