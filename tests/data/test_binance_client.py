"""Tests for Binance API client."""

import pytest
from unittest.mock import patch

import pandas as pd

from src.data.binance_client import BinanceClient


class TestBinanceClient:
    """Test cases for BinanceClient."""

    @pytest.fixture
    def client(self):
        """Create Binance client instance."""
        return BinanceClient(
            api_key="test_api_key",
            api_secret="test_api_secret",
            testnet=True,
        )

    @pytest.fixture
    def mock_klines_response(self):
        """Mock klines response."""
        return [
            [1704067200000, "42000.00", "42500.00", "41800.00", "42300.00", 
             "1000.5", 1704153599999, "42150000.00", 5000, "500.25", "21075000.00", "0"],
            [1704153600000, "42300.00", "43000.00", "42100.00", "42800.00",
             "1200.3", 1704239999999, "51360000.00", 6000, "600.15", "25680000.00", "0"],
        ]

    def test_init(self, client):
        """Test client initialization."""
        assert client.api_key == "test_api_key"
        assert client.testnet is True
        assert "testnet" in client.base_url

    def test_init_live(self):
        """Test live trading URL."""
        client = BinanceClient(
            api_key="key",
            api_secret="secret",
            testnet=False,
        )
        assert "api.binance.com" in client.base_url

    def test_get_signature(self, client):
        """Test HMAC signature generation."""
        params = {"symbol": "BTCUSDT", "timestamp": 1234567890}
        signature = client._get_signature(params)
        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA256 hex

    @patch.object(BinanceClient, "_request")
    def test_get_klines(self, mock_request, client, mock_klines_response):
        """Test fetching kline data."""
        mock_request.return_value = mock_klines_response

        df = client.get_klines("BTCUSDT", interval="1d", limit=2)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "open" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

    @patch.object(BinanceClient, "_request")
    def test_get_ticker(self, mock_request, client):
        """Test fetching ticker data."""
        mock_request.return_value = {
            "symbol": "BTCUSDT",
            "lastPrice": "42500.00",
            "priceChange": "500.00",
            "priceChangePercent": "1.19",
            "highPrice": "43000.00",
            "lowPrice": "42000.00",
            "volume": "10000.00",
            "quoteVolume": "425000000.00",
        }

        ticker = client.get_ticker("BTCUSDT")

        assert ticker["symbol"] == "BTCUSDT"
        assert ticker["price"] == 42500.00
        assert ticker["change_percent"] == 1.19

    @patch.object(BinanceClient, "_request")
    def test_get_balance(self, mock_request, client):
        """Test fetching account balance."""
        mock_request.return_value = {
            "balances": [
                {"asset": "BTC", "free": "1.5", "locked": "0.5"},
                {"asset": "USDT", "free": "10000.0", "locked": "0.0"},
                {"asset": "ETH", "free": "0.0", "locked": "0.0"},
            ]
        }

        balance = client.get_balance()

        assert isinstance(balance, pd.DataFrame)
        assert len(balance) == 2  # ETH should be excluded (0 balance)
        assert "BTC" in balance["asset"].values
