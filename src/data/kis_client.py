"""Korea Investment & Securities (KIS) API client."""

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


class KISClient(BaseClient):
    """Client for Korea Investment & Securities Open API.
    
    Supports both paper trading (모의투자) and live trading (실전투자).
    
    API Documentation: https://apiportal.koreainvestment.com
    """

    PAPER_BASE_URL = "https://openapivts.koreainvestment.com:29443"
    LIVE_BASE_URL = "https://openapi.koreainvestment.com:9443"

    def __init__(
        self,
        app_key: str,
        app_secret: str,
        account_no: str,
        account_code: str = "01",
        is_paper: bool = True,
        requests_per_minute: int = 60,
    ):
        """Initialize KIS client.
        
        Args:
            app_key: KIS App Key
            app_secret: KIS App Secret
            account_no: Account number (8 digits)
            account_code: Account code ("01" for regular, "02" for pension)
            is_paper: True for paper trading, False for live trading
            requests_per_minute: Rate limit
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
        self.account_code = account_code
        self.is_paper = is_paper
        
        self._access_token: str | None = None
        self._token_expires_at: float = 0

    def _get_auth_params(self) -> dict[str, str]:
        """Return empty dict - KIS uses headers for auth."""
        return {}

    def _get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers with access token."""
        self._ensure_token()
        return {
            "authorization": f"Bearer {self._access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }

    def _ensure_token(self) -> None:
        """Ensure we have a valid access token."""
        if self._access_token and time.time() < self._token_expires_at - 60:
            return
        self._refresh_token()

    def _refresh_token(self) -> None:
        """Get a new access token."""
        endpoint = "/oauth2/tokenP"
        url = f"{self.base_url}{endpoint}"
        
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }
        
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code != 200:
            raise AuthenticationError(f"Token refresh failed: {response.text}")
        
        data = response.json()
        self._access_token = data["access_token"]
        # Token expires in 24 hours, but we'll refresh earlier
        self._token_expires_at = time.time() + 86400 - 3600
        
        logger.info("KIS access token refreshed")

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make an HTTP request with KIS-specific headers."""
        self.rate_limiter.wait()

        url = f"{self.base_url}{endpoint}"
        headers = self._get_auth_headers()
        
        if "tr_id" in kwargs:
            headers["tr_id"] = kwargs.pop("tr_id")
        
        headers["content-type"] = "application/json; charset=utf-8"

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
            
            # KIS returns rt_cd for result code
            if data.get("rt_cd") != "0":
                raise APIError(
                    f"KIS API error: {data.get('msg1', 'Unknown error')}",
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
            "FID_COND_MRKT_DIV_CODE": "J",  # KOSPI/KOSDAQ
            "FID_INPUT_ISCD": symbol,
        }
        
        tr_id = "FHKST01010100"
        
        data = self._request("GET", endpoint, params=params, tr_id=tr_id)
        
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
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """Get daily OHLCV data for a stock.
        
        Args:
            symbol: Stock code
            start: Start date
            end: End date
            adjusted: If True, use adjusted prices
            
        Returns:
            DataFrame with OHLCV data
        """
        endpoint = "/uapi/domestic-stock/v1/quotations/inquire-daily-price"
        
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": symbol,
            "FID_PERIOD_DIV_CODE": "D",  # Daily
            "FID_ORG_ADJ_PRC": "0" if adjusted else "1",
        }
        
        tr_id = "FHKST01010400"
        
        data = self._request("GET", endpoint, params=params, tr_id=tr_id)
        
        output = data.get("output", [])
        
        if not output:
            return pd.DataFrame()
        
        records = []
        for item in output:
            records.append({
                "date": pd.to_datetime(item["stck_bsop_date"]),
                "open": int(item["stck_oprc"]),
                "high": int(item["stck_hgpr"]),
                "low": int(item["stck_lwpr"]),
                "close": int(item["stck_clpr"]),
                "volume": int(item["acml_vol"]),
            })
        
        df = pd.DataFrame(records)
        df = df.sort_values("date").reset_index(drop=True)
        
        # Filter by date range
        if start:
            start_dt = pd.to_datetime(start)
            df = df[df["date"] >= start_dt]
        if end:
            end_dt = pd.to_datetime(end)
            df = df[df["date"] <= end_dt]
        
        return df

    def get_balance(self) -> pd.DataFrame:
        """Get account balance and positions.
        
        Returns:
            DataFrame with position data
        """
        endpoint = "/uapi/domestic-stock/v1/trading/inquire-balance"
        
        params = {
            "CANO": self.account_no,
            "ACNT_PRDT_CD": self.account_code,
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
        
        data = self._request("GET", endpoint, params=params, tr_id=tr_id)
        
        output1 = data.get("output1", [])
        
        if not output1:
            return pd.DataFrame()
        
        records = []
        for item in output1:
            if int(item.get("hldg_qty", 0)) > 0:
                records.append({
                    "symbol": item["pdno"],
                    "name": item["prdt_name"],
                    "quantity": int(item["hldg_qty"]),
                    "avg_price": float(item["pchs_avg_pric"]),
                    "current_price": int(item["prpr"]),
                    "pnl": int(item["evlu_pfls_amt"]),
                    "pnl_rate": float(item["evlu_pfls_rt"]),
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
        
        # Order division codes
        if order_type == "market":
            ord_dvsn = "01"  # Market order
            price = 0
        else:
            ord_dvsn = "00"  # Limit order
        
        params = {
            "CANO": self.account_no,
            "ACNT_PRDT_CD": self.account_code,
            "PDNO": symbol,
            "ORD_DVSN": ord_dvsn,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": str(price),
        }
        
        if side == "buy":
            tr_id = "VTTC0802U" if self.is_paper else "TTTC0802U"
        else:
            tr_id = "VTTC0801U" if self.is_paper else "TTTC0801U"
        
        data = self._request("POST", endpoint, params=params, tr_id=tr_id)
        
        output = data.get("output", {})
        return {
            "order_no": output.get("ODNO"),
            "order_time": output.get("ORD_TMD"),
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
        }

    def get_order_history(
        self,
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pd.DataFrame:
        """Get order history.
        
        Args:
            start: Start date
            end: End date
            
        Returns:
            DataFrame with order history
        """
        endpoint = "/uapi/domestic-stock/v1/trading/inquire-daily-ccld"
        
        today = datetime.now().strftime("%Y%m%d")
        start_str = self._format_date(start).replace("-", "") if start else today
        end_str = self._format_date(end).replace("-", "") if end else today
        
        params = {
            "CANO": self.account_no,
            "ACNT_PRDT_CD": self.account_code,
            "INQR_STRT_DT": start_str,
            "INQR_END_DT": end_str,
            "SLL_BUY_DVSN_CD": "00",  # All
            "INQR_DVSN": "00",
            "PDNO": "",
            "CCLD_DVSN": "00",
            "ORD_GNO_BRNO": "",
            "ODNO": "",
            "INQR_DVSN_3": "00",
            "INQR_DVSN_1": "",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }
        
        tr_id = "VTTC8001R" if self.is_paper else "TTTC8001R"
        
        data = self._request("GET", endpoint, params=params, tr_id=tr_id)
        
        output = data.get("output1", [])
        
        if not output:
            return pd.DataFrame()
        
        records = []
        for item in output:
            records.append({
                "order_date": item.get("ord_dt"),
                "order_no": item.get("odno"),
                "symbol": item.get("pdno"),
                "name": item.get("prdt_name"),
                "side": "buy" if item.get("sll_buy_dvsn_cd") == "02" else "sell",
                "quantity": int(item.get("ord_qty", 0)),
                "price": int(item.get("ord_unpr", 0)),
                "filled_qty": int(item.get("tot_ccld_qty", 0)),
                "filled_price": int(item.get("avg_prvs", 0)),
            })
        
        return pd.DataFrame(records)

    @staticmethod
    def _format_date(d: date | str) -> str:
        """Format date to YYYY-MM-DD string."""
        if isinstance(d, str):
            return d
        return d.strftime("%Y-%m-%d")
