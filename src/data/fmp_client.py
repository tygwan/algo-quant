"""Financial Modeling Prep (FMP) API client."""

import logging
from datetime import date, datetime
from typing import Any

import pandas as pd

from .base_client import BaseClient

logger = logging.getLogger(__name__)


class FMPClient(BaseClient):
    """Client for Financial Modeling Prep API.
    
    Provides access to stock prices, financial statements, and company profiles.
    
    API Documentation: https://site.financialmodelingprep.com/developer/docs
    """

    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(self, api_key: str, requests_per_minute: int = 300):
        super().__init__(
            base_url=self.BASE_URL,
            api_key=api_key,
            requests_per_minute=requests_per_minute,
        )

    def _get_auth_params(self) -> dict[str, str]:
        """Return FMP authentication parameters."""
        return {"apikey": self.api_key}

    def get_historical_prices(
        self,
        symbol: str,
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pd.DataFrame:
        """Get historical daily prices for a symbol.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            start: Start date (inclusive)
            end: End date (inclusive)
            
        Returns:
            DataFrame with columns: date, open, high, low, close, adjClose, volume
        """
        endpoint = f"historical-price-full/{symbol.upper()}"
        params: dict[str, Any] = {}

        if start:
            params["from"] = self._format_date(start)
        if end:
            params["to"] = self._format_date(end)

        data = self.get(endpoint, params=params)

        if "historical" not in data:
            logger.warning(f"No historical data found for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(data["historical"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Standardize column names
        df = df.rename(columns={"adjClose": "adj_close"})
        
        columns = ["date", "open", "high", "low", "close", "adj_close", "volume"]
        available_cols = [c for c in columns if c in df.columns]
        
        return df[available_cols]

    def get_financial_statements(
        self,
        symbol: str,
        statement_type: str = "income",
        period: str = "annual",
        limit: int = 10,
    ) -> pd.DataFrame:
        """Get financial statements for a company.
        
        Args:
            symbol: Stock ticker symbol
            statement_type: 'income', 'balance', or 'cash'
            period: 'annual' or 'quarter'
            limit: Number of periods to return
            
        Returns:
            DataFrame with financial statement data
        """
        endpoint_map = {
            "income": "income-statement",
            "balance": "balance-sheet-statement",
            "cash": "cash-flow-statement",
        }

        if statement_type not in endpoint_map:
            raise ValueError(f"Invalid statement_type: {statement_type}")

        endpoint = f"{endpoint_map[statement_type]}/{symbol.upper()}"
        params = {"period": period, "limit": limit}

        data = self.get(endpoint, params=params)

        if not data:
            logger.warning(f"No financial statements found for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

        return df

    def get_company_profile(self, symbol: str) -> dict[str, Any]:
        """Get company profile information.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with company profile data
        """
        endpoint = f"profile/{symbol.upper()}"
        data = self.get(endpoint)

        if not data:
            logger.warning(f"No profile found for {symbol}")
            return {}

        return data[0] if isinstance(data, list) else data

    def get_key_metrics(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> pd.DataFrame:
        """Get key financial metrics for a company.
        
        Args:
            symbol: Stock ticker symbol
            period: 'annual' or 'quarter'
            limit: Number of periods to return
            
        Returns:
            DataFrame with key metrics (P/E, P/B, ROE, etc.)
        """
        endpoint = f"key-metrics/{symbol.upper()}"
        params = {"period": period, "limit": limit}

        data = self.get(endpoint, params=params)

        if not data:
            logger.warning(f"No key metrics found for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

        return df

    def get_stock_list(self) -> list[dict[str, Any]]:
        """Get list of all available stock symbols.
        
        Returns:
            List of dictionaries with symbol, name, and exchange
        """
        endpoint = "stock/list"
        return self.get(endpoint)

    @staticmethod
    def _format_date(d: date | str) -> str:
        """Format date to YYYY-MM-DD string."""
        if isinstance(d, str):
            return d
        return d.strftime("%Y-%m-%d")
