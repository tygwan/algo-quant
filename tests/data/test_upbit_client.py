"""Tests for Upbit API client."""

import pytest
from unittest.mock import patch

import pandas as pd

from src.data.upbit_client import UpbitClient


class TestUpbitClient:
    """Test cases for UpbitClient."""

    @pytest.fixture
    def client(self):
        """Create Upbit client instance."""
        return UpbitClient(
            access_key="test_access_key",
            secret_key="test_secret_key",
        )

    @pytest.fixture
    def mock_candles_response(self):
        """Mock candles response."""
        return [
            {
                "market": "KRW-BTC",
                "candle_date_time_kst": "2024-01-02T09:00:00",
                "opening_price": 60000000.0,
                "high_price": 61000000.0,
                "low_price": 59000000.0,
                "trade_price": 60500000.0,
                "candle_acc_trade_volume": 100.5,
                "candle_acc_trade_price": 6050000000.0,
            },
            {
                "market": "KRW-BTC",
                "candle_date_time_kst": "2024-01-01T09:00:00",
                "opening_price": 59000000.0,
                "high_price": 60500000.0,
                "low_price": 58500000.0,
                "trade_price": 60000000.0,
                "candle_acc_trade_volume": 150.3,
                "candle_acc_trade_price": 9000000000.0,
            },
        ]

    def test_init(self, client):
        """Test client initialization."""
        assert client.access_key == "test_access_key"
        assert "api.upbit.com" in client.base_url

    def test_get_jwt_token(self, client):
        """Test JWT token generation."""
        token = client._get_jwt_token()
        assert isinstance(token, str)
        assert len(token) > 0

    @patch.object(UpbitClient, "_request")
    def test_get_candles(self, mock_request, client, mock_candles_response):
        """Test fetching candle data."""
        mock_request.return_value = mock_candles_response

        df = client.get_candles("KRW-BTC", interval="days", count=2)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "open" in df.columns
        assert "close" in df.columns
        # Should be sorted by date ascending
        assert df.iloc[0]["date"] < df.iloc[1]["date"]

    @patch.object(UpbitClient, "_request")
    def test_get_ticker(self, mock_request, client):
        """Test fetching ticker data."""
        mock_request.return_value = [
            {
                "market": "KRW-BTC",
                "trade_price": 60000000.0,
                "change": "RISE",
                "signed_change_rate": 0.025,
                "signed_change_price": 1500000.0,
                "high_price": 61000000.0,
                "low_price": 59000000.0,
                "acc_trade_volume_24h": 1000.5,
                "acc_trade_price_24h": 60000000000.0,
            }
        ]

        tickers = client.get_ticker("KRW-BTC")

        assert len(tickers) == 1
        assert tickers[0]["market"] == "KRW-BTC"
        assert tickers[0]["price"] == 60000000.0

    @patch.object(UpbitClient, "_request")
    def test_get_balance(self, mock_request, client):
        """Test fetching account balance."""
        mock_request.return_value = [
            {
                "currency": "KRW",
                "balance": "1000000.0",
                "locked": "0.0",
                "avg_buy_price": "0",
                "unit_currency": "KRW",
            },
            {
                "currency": "BTC",
                "balance": "0.5",
                "locked": "0.1",
                "avg_buy_price": "55000000",
                "unit_currency": "KRW",
            },
        ]

        balance = client.get_balance()

        assert isinstance(balance, pd.DataFrame)
        assert len(balance) == 2
        assert "BTC" in balance["currency"].values

    @patch.object(UpbitClient, "_request")
    def test_get_markets(self, mock_request, client):
        """Test fetching market list."""
        mock_request.return_value = [
            {
                "market": "KRW-BTC",
                "korean_name": "비트코인",
                "english_name": "Bitcoin",
                "market_warning": "NONE",
            },
            {
                "market": "KRW-ETH",
                "korean_name": "이더리움",
                "english_name": "Ethereum",
                "market_warning": "NONE",
            },
        ]

        markets = client.get_markets()

        assert isinstance(markets, pd.DataFrame)
        assert len(markets) == 2
        assert "KRW-BTC" in markets["market"].values
