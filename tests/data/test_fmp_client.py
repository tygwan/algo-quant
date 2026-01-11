"""Tests for FMP API client."""

import pytest
from datetime import date
from unittest.mock import Mock, patch

import pandas as pd

from src.data.fmp_client import FMPClient
from src.data.base_client import APIError, AuthenticationError


class TestFMPClient:
    """Test cases for FMPClient."""

    @pytest.fixture
    def client(self):
        """Create FMP client instance."""
        return FMPClient(api_key="test_api_key")

    @pytest.fixture
    def mock_historical_response(self):
        """Mock historical price data response."""
        return {
            "symbol": "AAPL",
            "historical": [
                {
                    "date": "2024-01-02",
                    "open": 185.0,
                    "high": 186.5,
                    "low": 184.0,
                    "close": 186.0,
                    "adjClose": 185.5,
                    "volume": 50000000,
                },
                {
                    "date": "2024-01-03",
                    "open": 186.0,
                    "high": 187.0,
                    "low": 185.0,
                    "close": 185.5,
                    "adjClose": 185.0,
                    "volume": 45000000,
                },
            ],
        }

    @pytest.fixture
    def mock_profile_response(self):
        """Mock company profile response."""
        return [
            {
                "symbol": "AAPL",
                "companyName": "Apple Inc.",
                "industry": "Consumer Electronics",
                "sector": "Technology",
                "mktCap": 3000000000000,
            }
        ]

    def test_init(self, client):
        """Test client initialization."""
        assert client.api_key == "test_api_key"
        assert client.base_url == "https://financialmodelingprep.com/api/v3"

    def test_get_auth_params(self, client):
        """Test authentication parameters."""
        params = client._get_auth_params()
        assert params == {"apikey": "test_api_key"}

    @patch.object(FMPClient, "get")
    def test_get_historical_prices(
        self, mock_get, client, mock_historical_response
    ):
        """Test fetching historical prices."""
        mock_get.return_value = mock_historical_response

        df = client.get_historical_prices("AAPL", start="2024-01-01", end="2024-01-03")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "date" in df.columns
        assert "close" in df.columns
        assert "adj_close" in df.columns
        mock_get.assert_called_once()

    @patch.object(FMPClient, "get")
    def test_get_historical_prices_empty(self, mock_get, client):
        """Test handling empty historical data."""
        mock_get.return_value = {}

        df = client.get_historical_prices("INVALID")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    @patch.object(FMPClient, "get")
    def test_get_company_profile(self, mock_get, client, mock_profile_response):
        """Test fetching company profile."""
        mock_get.return_value = mock_profile_response

        profile = client.get_company_profile("AAPL")

        assert profile["symbol"] == "AAPL"
        assert profile["companyName"] == "Apple Inc."
        mock_get.assert_called_once()

    @patch.object(FMPClient, "get")
    def test_get_financial_statements(self, mock_get, client):
        """Test fetching financial statements."""
        mock_response = [
            {
                "date": "2023-12-31",
                "revenue": 383285000000,
                "netIncome": 96995000000,
            }
        ]
        mock_get.return_value = mock_response

        df = client.get_financial_statements("AAPL", statement_type="income")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        mock_get.assert_called_once()

    def test_get_financial_statements_invalid_type(self, client):
        """Test invalid statement type raises error."""
        with pytest.raises(ValueError, match="Invalid statement_type"):
            client.get_financial_statements("AAPL", statement_type="invalid")

    def test_format_date_string(self):
        """Test date formatting with string input."""
        result = FMPClient._format_date("2024-01-15")
        assert result == "2024-01-15"

    def test_format_date_object(self):
        """Test date formatting with date object."""
        result = FMPClient._format_date(date(2024, 1, 15))
        assert result == "2024-01-15"
