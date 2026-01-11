"""Tests for Korea Investment Securities API client."""

import pytest
from unittest.mock import patch, MagicMock

import pandas as pd

from src.data.kis_client import KISClient


class TestKISClient:
    """Test cases for KISClient."""

    @pytest.fixture
    def client(self):
        """Create KIS client instance."""
        with patch.object(KISClient, "_refresh_token"):
            return KISClient(
                app_key="test_app_key",
                app_secret="test_app_secret",
                account_no="12345678",
                account_code="01",
                is_paper=True,
            )

    def test_init(self, client):
        """Test client initialization."""
        assert client.app_key == "test_app_key"
        assert client.account_no == "12345678"
        assert client.is_paper is True
        assert "openapivts" in client.base_url

    def test_init_live(self):
        """Test live trading URL."""
        with patch.object(KISClient, "_refresh_token"):
            client = KISClient(
                app_key="key",
                app_secret="secret",
                account_no="12345678",
                is_paper=False,
            )
        assert "openapi.koreainvestment.com" in client.base_url

    @patch.object(KISClient, "_request")
    def test_get_price(self, mock_request, client):
        """Test fetching current price."""
        mock_request.return_value = {
            "rt_cd": "0",
            "output": {
                "stck_prpr": "70000",
                "prdy_vrss": "1000",
                "prdy_ctrt": "1.45",
                "acml_vol": "10000000",
                "stck_hgpr": "71000",
                "stck_lwpr": "69000",
                "stck_oprc": "69500",
            }
        }

        price = client.get_price("005930")

        assert price["symbol"] == "005930"
        assert price["price"] == 70000
        assert price["change"] == 1000

    @patch.object(KISClient, "_request")
    def test_get_balance(self, mock_request, client):
        """Test fetching account balance."""
        mock_request.return_value = {
            "rt_cd": "0",
            "output1": [
                {
                    "pdno": "005930",
                    "prdt_name": "삼성전자",
                    "hldg_qty": "100",
                    "pchs_avg_pric": "65000.00",
                    "prpr": "70000",
                    "evlu_pfls_amt": "500000",
                    "evlu_pfls_rt": "7.69",
                }
            ]
        }

        balance = client.get_balance()

        assert isinstance(balance, pd.DataFrame)
        assert len(balance) == 1
        assert balance.iloc[0]["symbol"] == "005930"
