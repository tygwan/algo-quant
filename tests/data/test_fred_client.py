"""Tests for FRED API client."""

import pytest
from unittest.mock import patch

import pandas as pd

from src.data.fred_client import FREDClient, FREDIndicators


class TestFREDClient:
    """Test cases for FREDClient."""

    @pytest.fixture
    def client(self):
        """Create FRED client instance."""
        return FREDClient(api_key="test_api_key")

    @pytest.fixture
    def mock_observations_response(self):
        """Mock observations response."""
        return {
            "observations": [
                {"date": "2024-01-01", "value": "3.7"},
                {"date": "2024-02-01", "value": "3.9"},
                {"date": "2024-03-01", "value": "3.8"},
            ]
        }

    def test_init(self, client):
        """Test client initialization."""
        assert client.api_key == "test_api_key"
        assert "stlouisfed.org" in client.base_url

    def test_get_auth_params(self, client):
        """Test authentication parameters."""
        params = client._get_auth_params()
        assert params["api_key"] == "test_api_key"
        assert params["file_type"] == "json"

    @patch.object(FREDClient, "get")
    def test_get_series(self, mock_get, client, mock_observations_response):
        """Test fetching series data."""
        mock_get.return_value = mock_observations_response

        series = client.get_series("UNRATE", start="2024-01-01", end="2024-03-01")

        assert isinstance(series, pd.Series)
        assert len(series) == 3
        assert series.name == "UNRATE"
        mock_get.assert_called_once()

    @patch.object(FREDClient, "get")
    def test_get_series_empty(self, mock_get, client):
        """Test handling empty series data."""
        mock_get.return_value = {}

        series = client.get_series("INVALID")

        assert isinstance(series, pd.Series)
        assert len(series) == 0

    @patch.object(FREDClient, "get_series")
    def test_get_unemployment(self, mock_get_series, client):
        """Test get_unemployment convenience method."""
        client.get_unemployment(start="2024-01-01")
        mock_get_series.assert_called_once_with(FREDIndicators.UNRATE, "2024-01-01", None)

    @patch.object(FREDClient, "get_series")
    def test_get_gdp(self, mock_get_series, client):
        """Test get_gdp convenience method."""
        client.get_gdp(real=True)
        mock_get_series.assert_called_once_with(FREDIndicators.GDPC1, None, None)

    def test_fred_indicators(self):
        """Test FRED indicators constants."""
        assert FREDIndicators.UNRATE == "UNRATE"
        assert FREDIndicators.GDP == "GDP"
        assert FREDIndicators.FEDFUNDS == "FEDFUNDS"
        assert FREDIndicators.T10Y2Y == "T10Y2Y"
