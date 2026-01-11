"""Upbit API client."""

import hashlib
import jwt
import logging
import time
import uuid
from datetime import date, datetime
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import requests

from .base_client import BaseClient, APIError, AuthenticationError

logger = logging.getLogger(__name__)


class UpbitClient(BaseClient):
    """Client for Upbit API.
    
    Upbit is the largest Korean cryptocurrency exchange.
    
    API Documentation: https://docs.upbit.com
    """

    BASE_URL = "https://api.upbit.com"

    def __init__(
        self,
        access_key: str,
        secret_key: str,
        requests_per_minute: int = 600,
    ):
        """Initialize Upbit client.
        
        Args:
            access_key: Upbit Access Key
            secret_key: Upbit Secret Key
            requests_per_minute: Rate limit
        """
        super().__init__(
            base_url=self.BASE_URL,
            api_key=access_key,
            requests_per_minute=requests_per_minute,
        )
        
        self.access_key = access_key
        self.secret_key = secret_key

    def _get_auth_params(self) -> dict[str, str]:
        """Return empty dict - Upbit uses JWT for auth."""
        return {}

    def _get_jwt_token(self, query: dict[str, Any] | None = None) -> str:
        """Generate JWT token for authentication."""
        payload = {
            "access_key": self.access_key,
            "nonce": str(uuid.uuid4()),
            "timestamp": int(time.time() * 1000),
        }
        
        if query:
            query_string = urlencode(query).encode()
            m = hashlib.sha512()
            m.update(query_string)
            payload["query_hash"] = m.hexdigest()
            payload["query_hash_alg"] = "SHA512"
        
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        return token

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        auth_required: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | list:
        """Make an HTTP request with Upbit-specific auth."""
        self.rate_limiter.wait()

        url = f"{self.base_url}{endpoint}"
        headers = {"Accept": "application/json"}
        
        if auth_required:
            token = self._get_jwt_token(params)
            headers["Authorization"] = f"Bearer {token}"

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
                raise AuthenticationError("Invalid API key", status_code=401)
            
            data = response.json()
            
            if isinstance(data, dict) and "error" in data:
                error = data["error"]
                raise APIError(
                    f"Upbit API error: {error.get('message', 'Unknown error')}",
                    status_code=response.status_code,
                )

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def get_candles(
        self,
        market: str,
        interval: str = "days",
        count: int = 200,
        to: datetime | str | None = None,
    ) -> pd.DataFrame:
        """Get candlestick data.
        
        Args:
            market: Market code (e.g., 'KRW-BTC')
            interval: 'minutes/1', 'minutes/5', 'minutes/15', 'minutes/30',
                     'minutes/60', 'minutes/240', 'days', 'weeks', 'months'
            count: Number of candles (max 200)
            to: Last candle datetime
            
        Returns:
            DataFrame with OHLCV data
        """
        if interval.startswith("minutes"):
            unit = interval.split("/")[1] if "/" in interval else "1"
            endpoint = f"/v1/candles/minutes/{unit}"
        else:
            endpoint = f"/v1/candles/{interval}"
        
        params: dict[str, Any] = {
            "market": market.upper(),
            "count": min(count, 200),
        }
        
        if to:
            if isinstance(to, str):
                to = datetime.fromisoformat(to)
            params["to"] = to.strftime("%Y-%m-%dT%H:%M:%S")
        
        data = self._request("GET", endpoint, params=params)
        
        if not data:
            return pd.DataFrame()
        
        records = []
        for candle in data:
            records.append({
                "date": pd.to_datetime(candle["candle_date_time_kst"]),
                "open": float(candle["opening_price"]),
                "high": float(candle["high_price"]),
                "low": float(candle["low_price"]),
                "close": float(candle["trade_price"]),
                "volume": float(candle["candle_acc_trade_volume"]),
                "quote_volume": float(candle["candle_acc_trade_price"]),
            })
        
        df = pd.DataFrame(records)
        df = df.sort_values("date").reset_index(drop=True)
        
        return df

    def get_ticker(self, markets: str | list[str]) -> list[dict[str, Any]]:
        """Get current ticker(s).
        
        Args:
            markets: Market code(s) (e.g., 'KRW-BTC' or ['KRW-BTC', 'KRW-ETH'])
            
        Returns:
            List of ticker data
        """
        endpoint = "/v1/ticker"
        
        if isinstance(markets, list):
            markets_str = ",".join(m.upper() for m in markets)
        else:
            markets_str = markets.upper()
        
        params = {"markets": markets_str}
        
        data = self._request("GET", endpoint, params=params)
        
        result = []
        for ticker in data:
            result.append({
                "market": ticker["market"],
                "price": float(ticker["trade_price"]),
                "change": ticker["change"],
                "change_rate": float(ticker["signed_change_rate"]),
                "change_price": float(ticker["signed_change_price"]),
                "high": float(ticker["high_price"]),
                "low": float(ticker["low_price"]),
                "volume": float(ticker["acc_trade_volume_24h"]),
                "quote_volume": float(ticker["acc_trade_price_24h"]),
            })
        
        return result

    def get_markets(self) -> pd.DataFrame:
        """Get list of all available markets.
        
        Returns:
            DataFrame with market info
        """
        endpoint = "/v1/market/all"
        
        params = {"isDetails": "true"}
        
        data = self._request("GET", endpoint, params=params)
        
        records = []
        for market in data:
            records.append({
                "market": market["market"],
                "korean_name": market["korean_name"],
                "english_name": market["english_name"],
                "market_warning": market.get("market_warning", "NONE"),
            })
        
        return pd.DataFrame(records)

    def get_balance(self) -> pd.DataFrame:
        """Get account balance.
        
        Returns:
            DataFrame with balance data
        """
        endpoint = "/v1/accounts"
        
        data = self._request("GET", endpoint, auth_required=True)
        
        records = []
        for account in data:
            balance = float(account["balance"])
            locked = float(account["locked"])
            if balance > 0 or locked > 0:
                records.append({
                    "currency": account["currency"],
                    "balance": balance,
                    "locked": locked,
                    "avg_buy_price": float(account["avg_buy_price"]),
                    "unit_currency": account["unit_currency"],
                })
        
        return pd.DataFrame(records)

    def create_order(
        self,
        market: str,
        side: str,
        volume: float | None = None,
        price: float | None = None,
        ord_type: str = "limit",
    ) -> dict[str, Any]:
        """Create a new order.
        
        Args:
            market: Market code (e.g., 'KRW-BTC')
            side: 'bid' (buy) or 'ask' (sell)
            volume: Order volume (required for limit/market sell)
            price: Order price (required for limit/market buy)
            ord_type: 'limit', 'price' (market buy), 'market' (market sell)
            
        Returns:
            Order response
        """
        endpoint = "/v1/orders"
        
        params: dict[str, Any] = {
            "market": market.upper(),
            "side": side.lower(),
            "ord_type": ord_type,
        }
        
        if volume is not None:
            params["volume"] = str(volume)
        if price is not None:
            params["price"] = str(price)
        
        data = self._request("POST", endpoint, params=params, auth_required=True)
        
        return {
            "uuid": data.get("uuid"),
            "market": data.get("market"),
            "side": data.get("side"),
            "ord_type": data.get("ord_type"),
            "volume": float(data.get("volume") or 0),
            "price": float(data.get("price") or 0),
            "state": data.get("state"),
        }

    def get_order(self, uuid: str) -> dict[str, Any]:
        """Get order detail.
        
        Args:
            uuid: Order UUID
            
        Returns:
            Order detail
        """
        endpoint = "/v1/order"
        
        params = {"uuid": uuid}
        
        return self._request("GET", endpoint, params=params, auth_required=True)

    def get_orders(
        self,
        market: str | None = None,
        state: str = "wait",
        page: int = 1,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Get order list.
        
        Args:
            market: Market code (optional)
            state: 'wait', 'done', 'cancel'
            page: Page number
            limit: Items per page
            
        Returns:
            DataFrame with orders
        """
        endpoint = "/v1/orders"
        
        params: dict[str, Any] = {
            "state": state,
            "page": page,
            "limit": limit,
            "order_by": "desc",
        }
        
        if market:
            params["market"] = market.upper()
        
        data = self._request("GET", endpoint, params=params, auth_required=True)
        
        if not data:
            return pd.DataFrame()
        
        records = []
        for order in data:
            records.append({
                "uuid": order["uuid"],
                "market": order["market"],
                "side": order["side"],
                "ord_type": order["ord_type"],
                "volume": float(order.get("volume") or 0),
                "price": float(order.get("price") or 0),
                "executed_volume": float(order.get("executed_volume") or 0),
                "state": order["state"],
                "created_at": pd.to_datetime(order["created_at"]),
            })
        
        return pd.DataFrame(records)

    def cancel_order(self, uuid: str) -> dict[str, Any]:
        """Cancel an order.
        
        Args:
            uuid: Order UUID
            
        Returns:
            Cancellation response
        """
        endpoint = "/v1/order"
        
        params = {"uuid": uuid}
        
        return self._request("DELETE", endpoint, params=params, auth_required=True)

    def get_orderbook(self, markets: str | list[str]) -> list[dict[str, Any]]:
        """Get orderbook.
        
        Args:
            markets: Market code(s)
            
        Returns:
            Orderbook data
        """
        endpoint = "/v1/orderbook"
        
        if isinstance(markets, list):
            markets_str = ",".join(m.upper() for m in markets)
        else:
            markets_str = markets.upper()
        
        params = {"markets": markets_str}
        
        return self._request("GET", endpoint, params=params)
