"""Kiwoom Securities API client."""

import hashlib
import json
import logging
import time
from datetime import date, datetime
from typing import Any

import pandas as pd
import requests

from .base_client import BaseClient, APIError, AuthenticationError

logger = logging.getLogger(__name__)


class KiwoomClient(BaseClient):
    """Client for Kiwoom Securities Open API.
    
    Supports both paper trading (모의투자) and live trading (실전투자).
    
    API Documentation: https://openapi.kiwoom.com
    """

    PAPER_BASE_URL = "https://openapivts.kiwoom.com:21443"
    LIVE_BASE_URL = "https://openapi.kiwoom.com:8443"

    def __init__(
        self,
        app_key: str,
        app_secret: str,
        account_no: str,
        is_paper: bool = True,
        requests_per_minute: int = 60,
    ):
        """Initialize Kiwoom client.
        
        Args:
            app_key: Kiwoom App Key
            app_secret: Kiwoom App Secret
            account_no: Account number
            is_paper: True for paper trading, False for live trading
            requests_per_minute: Rate limit (Kiwoom allows ~5/sec)
        """
        base_url = self.PAPER_BASE_URL if is_paper else self.LIVE_BASE_URL
        
        super().__init__(
            base_url=base_url,
            api_key=app_key,
            requests_per_minute=requests_per_minute,
        )
        
        self.app_key = app_key
        self.app_secret = app_secret
        self.account_no = account_no
        self.is_paper = is_paper
        
        self._access_token: str | None = None
        self._token_expires_at: float = 0

    def _get_auth_params(self) -> dict[str, str]:
        """Return empty dict - Kiwoom uses headers for auth."""
        return {}

    def _get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers with access token."""
        self._ensure_token()
        return {
            "authorization": f"Bearer {self._access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "content-type": "application/json; charset=utf-8",
        }

    def _ensure_token(self) -> None:
        """Ensure we have a valid access token."""
        if self._access_token and time.time() < self._token_expires_at - 60:
            return
        self._refresh_token()

    def _refresh_token(self) -> None:
        """Get a new access token."""
        endpoint = "/oauth2/token"
        url = f"{self.base_url}{endpoint}"
        
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "secretkey": self.app_secret,
        }
        
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code != 200:
            raise AuthenticationError(f"Token refresh failed: {response.text}")
        
        data = response.json()
        self._access_token = data["access_token"]
        expires_in = data.get("expires_in", 86400)
        self._token_expires_at = time.time() + expires_in - 3600
        
        logger.info("Kiwoom access token refreshed")

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make an HTTP request with Kiwoom-specific headers."""
        self.rate_limiter.wait()

        url = f"{self.base_url}{endpoint}"
        headers = self._get_auth_headers()
        
        if "tr_id" in kwargs:
            headers["tr_id"] = kwargs.pop("tr_id")
        if "custtype" in kwargs:
            headers["custtype"] = kwargs.pop("custtype")

        logger.debug(f"{method.upper()} {url}")

        try:
            if method.upper() == "GET":
                response = self.session.get(
                    url, params=params, headers=headers, timeout=self.timeout
                )
            else:
                response = self.session.post(
                    url, json=params, headers=headers, timeout=self.timeout
                )

            if response.status_code == 401:
                raise AuthenticationError("Invalid credentials", status_code=401)
            
            data = response.json()
            
            # Check for error response
            if data.get("rt_cd") != "0":
                raise APIError(
                    f"Kiwoom API error: {data.get('msg1', 'Unknown error')}",
                    status_code=response.status_code,
                )

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def get_price(self, symbol: str) -> dict[str, Any]:
        """Get current price for a stock.
        
        Args:
            symbol: Stock code (e.g., '005930' for Samsung Electronics)
            
        Returns:
            Dictionary with current price data
        """
        endpoint = "/uapi/domestic-stock/v1/quotations/inquire-price"
        
        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": symbol,
        }
        
        data = self._request(
            "GET", endpoint, params=params, 
            tr_id="FHKST01010100", custtype="P"
        )
        
        output = data.get("output", {})
        return {
            "symbol": symbol,
            "price": int(output.get("stck_prpr", 0)),
            "change": int(output.get("prdy_vrss", 0)),
            "change_rate": float(output.get("prdy_ctrt", 0)),
            "volume": int(output.get("acml_vol", 0)),
            "high": int(output.get("stck_hgpr", 0)),
            "low": int(output.get("stck_lwpr", 0)),
            "open": int(output.get("stck_oprc", 0)),
        }

    def get_daily_prices(
        self,
        symbol: str,
        start: date | str | None = None,
        end: date | str | None = None,
        period: str = "D",
    ) -> pd.DataFrame:
        """Get daily OHLCV data for a stock.
        
        Args:
            symbol: Stock code
            start: Start date
            end: End date
            period: 'D' for daily, 'W' for weekly, 'M' for monthly
            
        Returns:
            DataFrame with OHLCV data
        """
        endpoint = "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        
        end_str = self._format_date(end or datetime.now()).replace("-", "")
        start_str = self._format_date(start or "2020-01-01").replace("-", "")
        
        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": symbol,
            "fid_input_date_1": start_str,
            "fid_input_date_2": end_str,
            "fid_period_div_code": period,
            "fid_org_adj_prc": "0",
        }
        
        data = self._request(
            "GET", endpoint, params=params,
            tr_id="FHKST03010100", custtype="P"
        )
        
        output = data.get("output2", [])
        
        if not output:
            return pd.DataFrame()
        
        records = []
        for item in output:
            if item.get("stck_bsop_date"):
                records.append({
                    "date": pd.to_datetime(item["stck_bsop_date"]),
                    "open": int(item.get("stck_oprc", 0)),
                    "high": int(item.get("stck_hgpr", 0)),
                    "low": int(item.get("stck_lwpr", 0)),
                    "close": int(item.get("stck_clpr", 0)),
                    "volume": int(item.get("acml_vol", 0)),
                })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("date").reset_index(drop=True)
        
        return df

    def get_balance(self) -> pd.DataFrame:
        """Get account balance and positions.
        
        Returns:
            DataFrame with position data
        """
        endpoint = "/uapi/domestic-stock/v1/trading/inquire-balance"
        
        params = {
            "CANO": self.account_no[:8],
            "ACNT_PRDT_CD": self.account_no[8:] if len(self.account_no) > 8 else "01",
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }
        
        tr_id = "VTTC8434R" if self.is_paper else "TTTC8434R"
        
        data = self._request("GET", endpoint, params=params, tr_id=tr_id, custtype="P")
        
        output1 = data.get("output1", [])
        
        if not output1:
            return pd.DataFrame()
        
        records = []
        for item in output1:
            qty = int(item.get("hldg_qty", 0))
            if qty > 0:
                records.append({
                    "symbol": item.get("pdno", ""),
                    "name": item.get("prdt_name", ""),
                    "quantity": qty,
                    "avg_price": float(item.get("pchs_avg_pric", 0)),
                    "current_price": int(item.get("prpr", 0)),
                    "pnl": int(item.get("evlu_pfls_amt", 0)),
                    "pnl_rate": float(item.get("evlu_pfls_rt", 0)),
                })
        
        return pd.DataFrame(records)

    def create_order(
        self,
        symbol: str,
        quantity: int,
        price: int,
        side: str,
        order_type: str = "limit",
    ) -> dict[str, Any]:
        """Create a stock order.
        
        Args:
            symbol: Stock code
            quantity: Number of shares
            price: Order price (0 for market order)
            side: 'buy' or 'sell'
            order_type: 'limit' or 'market'
            
        Returns:
            Order result dictionary
        """
        endpoint = "/uapi/domestic-stock/v1/trading/order-cash"
        
        if order_type == "market":
            ord_dvsn = "01"
            price = 0
        else:
            ord_dvsn = "00"
        
        params = {
            "CANO": self.account_no[:8],
            "ACNT_PRDT_CD": self.account_no[8:] if len(self.account_no) > 8 else "01",
            "PDNO": symbol,
            "ORD_DVSN": ord_dvsn,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": str(price),
        }
        
        if side == "buy":
            tr_id = "VTTC0802U" if self.is_paper else "TTTC0802U"
        else:
            tr_id = "VTTC0801U" if self.is_paper else "TTTC0801U"
        
        data = self._request("POST", endpoint, params=params, tr_id=tr_id, custtype="P")
        
        output = data.get("output", {})
        return {
            "order_no": output.get("ODNO"),
            "order_time": output.get("ORD_TMD"),
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
        }

    def search_condition(
        self,
        condition_name: str,
    ) -> pd.DataFrame:
        """Search stocks by saved condition.
        
        Note: Conditions must be created in HTS first.
        
        Args:
            condition_name: Name of the saved condition
            
        Returns:
            DataFrame with matching stocks
        """
        # Note: Condition search requires additional setup in HTS
        # This is a placeholder for the API structure
        logger.warning("Condition search requires HTS setup. Not implemented in REST API.")
        return pd.DataFrame()

    @staticmethod
    def _format_date(d: date | str) -> str:
        """Format date to YYYY-MM-DD string."""
        if isinstance(d, str):
            return d
        return d.strftime("%Y-%m-%d")
