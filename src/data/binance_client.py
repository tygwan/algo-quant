"""Binance API client."""

import hashlib
import hmac
import logging
import time
from datetime import date, datetime
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import requests

from .base_client import BaseClient, APIError, AuthenticationError

logger = logging.getLogger(__name__)


class BinanceClient(BaseClient):
    """Client for Binance API.
    
    Supports both testnet and live trading.
    
    API Documentation: https://binance-docs.github.io/apidocs
    """

    LIVE_BASE_URL = "https://api.binance.com"
    TESTNET_BASE_URL = "https://testnet.binance.vision"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        requests_per_minute: int = 1200,
    ):
        """Initialize Binance client.
        
        Args:
            api_key: Binance API Key
            api_secret: Binance API Secret
            testnet: True for testnet, False for live trading
            requests_per_minute: Rate limit
        """
        base_url = self.TESTNET_BASE_URL if testnet else self.LIVE_BASE_URL
        
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            requests_per_minute=requests_per_minute,
        )
        
        self.api_secret = api_secret
        self.testnet = testnet

    def _get_auth_params(self) -> dict[str, str]:
        """Return empty dict - Binance uses headers/signature for auth."""
        return {}

    def _get_signature(self, params: dict[str, Any]) -> str:
        """Generate HMAC SHA256 signature."""
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | list:
        """Make an HTTP request with Binance-specific auth."""
        self.rate_limiter.wait()

        url = f"{self.base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key}
        
        params = params or {}
        
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["signature"] = self._get_signature(params)

        logger.debug(f"{method.upper()} {url}")

        try:
            if method.upper() == "GET":
                response = self.session.get(
                    url, params=params, headers=headers, timeout=self.timeout
                )
            else:
                response = self.session.post(
                    url, params=params, headers=headers, timeout=self.timeout
                )

            if response.status_code == 401:
                raise AuthenticationError("Invalid API key", status_code=401)
            
            data = response.json()
            
            if isinstance(data, dict) and "code" in data:
                raise APIError(
                    f"Binance API error: {data.get('msg', 'Unknown error')}",
                    status_code=response.status_code,
                )

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def get_klines(
        self,
        symbol: str,
        interval: str = "1d",
        start: date | str | None = None,
        end: date | str | None = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Get candlestick/kline data.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval ('1m', '5m', '1h', '1d', etc.)
            start: Start date
            end: End date
            limit: Number of klines to return (max 1000)
            
        Returns:
            DataFrame with OHLCV data
        """
        endpoint = "/api/v3/klines"
        
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, 1000),
        }
        
        if start:
            params["startTime"] = self._date_to_ms(start)
        if end:
            params["endTime"] = self._date_to_ms(end)
        
        data = self._request("GET", endpoint, params=params)
        
        if not data:
            return pd.DataFrame()
        
        records = []
        for kline in data:
            records.append({
                "date": pd.to_datetime(kline[0], unit="ms"),
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5]),
                "close_time": pd.to_datetime(kline[6], unit="ms"),
                "quote_volume": float(kline[7]),
                "trades": int(kline[8]),
            })
        
        df = pd.DataFrame(records)
        return df

    def get_ticker(self, symbol: str) -> dict[str, Any]:
        """Get 24hr ticker price change statistics.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            
        Returns:
            Dictionary with ticker data
        """
        endpoint = "/api/v3/ticker/24hr"
        
        params = {"symbol": symbol.upper()}
        
        data = self._request("GET", endpoint, params=params)
        
        return {
            "symbol": data["symbol"],
            "price": float(data["lastPrice"]),
            "change": float(data["priceChange"]),
            "change_percent": float(data["priceChangePercent"]),
            "high": float(data["highPrice"]),
            "low": float(data["lowPrice"]),
            "volume": float(data["volume"]),
            "quote_volume": float(data["quoteVolume"]),
        }

    def get_ticker_price(self, symbol: str | None = None) -> dict[str, Any] | list:
        """Get latest price for a symbol or all symbols.
        
        Args:
            symbol: Trading pair (optional, returns all if None)
            
        Returns:
            Price data
        """
        endpoint = "/api/v3/ticker/price"
        
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
        
        return self._request("GET", endpoint, params=params)

    def get_balance(self) -> pd.DataFrame:
        """Get account balance.
        
        Returns:
            DataFrame with balance data
        """
        endpoint = "/api/v3/account"
        
        data = self._request("GET", endpoint, signed=True)
        
        balances = data.get("balances", [])
        
        records = []
        for balance in balances:
            free = float(balance["free"])
            locked = float(balance["locked"])
            if free > 0 or locked > 0:
                records.append({
                    "asset": balance["asset"],
                    "free": free,
                    "locked": locked,
                    "total": free + locked,
                })
        
        return pd.DataFrame(records)

    def create_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None = None,
        order_type: str = "LIMIT",
    ) -> dict[str, Any]:
        """Create a new order.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            price: Limit price (required for LIMIT orders)
            order_type: 'LIMIT' or 'MARKET'
            
        Returns:
            Order response
        """
        endpoint = "/api/v3/order"
        
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(quantity),
        }
        
        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("Price is required for LIMIT orders")
            params["price"] = str(price)
            params["timeInForce"] = "GTC"
        
        data = self._request("POST", endpoint, params=params, signed=True)
        
        return {
            "order_id": data.get("orderId"),
            "symbol": data.get("symbol"),
            "side": data.get("side"),
            "type": data.get("type"),
            "quantity": float(data.get("origQty", 0)),
            "price": float(data.get("price", 0)),
            "status": data.get("status"),
        }

    def get_open_orders(self, symbol: str | None = None) -> pd.DataFrame:
        """Get all open orders.
        
        Args:
            symbol: Trading pair (optional)
            
        Returns:
            DataFrame with open orders
        """
        endpoint = "/api/v3/openOrders"
        
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
        
        data = self._request("GET", endpoint, params=params, signed=True)
        
        if not data:
            return pd.DataFrame()
        
        records = []
        for order in data:
            records.append({
                "order_id": order["orderId"],
                "symbol": order["symbol"],
                "side": order["side"],
                "type": order["type"],
                "quantity": float(order["origQty"]),
                "price": float(order["price"]),
                "filled": float(order["executedQty"]),
                "status": order["status"],
                "time": pd.to_datetime(order["time"], unit="ms"),
            })
        
        return pd.DataFrame(records)

    def cancel_order(self, symbol: str, order_id: int) -> dict[str, Any]:
        """Cancel an order.
        
        Args:
            symbol: Trading pair
            order_id: Order ID to cancel
            
        Returns:
            Cancellation response
        """
        endpoint = "/api/v3/order"
        
        params = {
            "symbol": symbol.upper(),
            "orderId": order_id,
        }
        
        return self._request("DELETE", endpoint, params=params, signed=True)

    def get_exchange_info(self, symbol: str | None = None) -> dict[str, Any]:
        """Get exchange trading rules and symbol info.
        
        Args:
            symbol: Trading pair (optional)
            
        Returns:
            Exchange info
        """
        endpoint = "/api/v3/exchangeInfo"
        
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
        
        return self._request("GET", endpoint, params=params)

    @staticmethod
    def _date_to_ms(d: date | str) -> int:
        """Convert date to milliseconds timestamp."""
        if isinstance(d, str):
            d = datetime.strptime(d, "%Y-%m-%d").date()
        dt = datetime.combine(d, datetime.min.time())
        return int(dt.timestamp() * 1000)
