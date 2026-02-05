"""Finnhub API client for economic calendar, news, and market data."""

import logging
from datetime import date, datetime, timedelta
from typing import Any, Literal

import pandas as pd

from .base_client import BaseClient

logger = logging.getLogger(__name__)


class FinnhubClient(BaseClient):
    """Client for Finnhub API.

    Provides access to:
    - Economic Calendar (FOMC, employment, GDP releases)
    - Market News
    - Company Profiles
    - Market Sentiment

    API Documentation: https://finnhub.io/docs/api

    Example:
        >>> client = FinnhubClient(api_key="your_key")
        >>> calendar = client.get_economic_calendar()
        >>> news = client.get_market_news()
    """

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str, requests_per_minute: int = 60):
        """Initialize Finnhub client.

        Args:
            api_key: Finnhub API key (free tier available)
            requests_per_minute: Rate limit (free tier: 60/min)
        """
        super().__init__(
            base_url=self.BASE_URL,
            api_key=api_key,
            requests_per_minute=requests_per_minute,
        )

    def _get_auth_params(self) -> dict[str, str]:
        """Return Finnhub authentication parameters."""
        return {"token": self.api_key}

    # ========== Economic Calendar ==========

    def get_economic_calendar(
        self,
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pd.DataFrame:
        """Get economic calendar events.

        Includes GDP, employment, inflation releases, FOMC meetings, etc.

        Args:
            start: Start date (default: today)
            end: End date (default: 30 days from start)

        Returns:
            DataFrame with economic events
        """
        if start is None:
            start = date.today()
        if isinstance(start, str):
            start = date.fromisoformat(start)

        if end is None:
            end = start + timedelta(days=30)
        if isinstance(end, str):
            end = date.fromisoformat(end)

        endpoint = "calendar/economic"
        params = {
            "from": start.isoformat(),
            "to": end.isoformat(),
        }

        data = self.get(endpoint, params=params)

        if not data or "economicCalendar" not in data:
            logger.warning("No economic calendar data")
            return pd.DataFrame()

        df = pd.DataFrame(data["economicCalendar"])

        if df.empty:
            return df

        # Parse datetime
        if "time" in df.columns:
            df["datetime"] = pd.to_datetime(df["time"])
        elif "date" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"])

        # Sort by date
        if "datetime" in df.columns:
            df = df.sort_values("datetime")

        return df

    def get_upcoming_events(
        self,
        days_ahead: int = 7,
        impact: Literal["all", "high", "medium", "low"] = "all",
    ) -> pd.DataFrame:
        """Get upcoming high-impact economic events.

        Args:
            days_ahead: Number of days to look ahead
            impact: Filter by impact level

        Returns:
            DataFrame with upcoming events
        """
        start = date.today()
        end = start + timedelta(days=days_ahead)

        df = self.get_economic_calendar(start, end)

        if df.empty:
            return df

        # Filter by impact
        if impact != "all" and "impact" in df.columns:
            df = df[df["impact"].str.lower() == impact]

        return df

    def get_fomc_calendar(
        self,
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pd.DataFrame:
        """Get FOMC meeting dates and decisions.

        Args:
            start: Start date
            end: End date

        Returns:
            DataFrame with FOMC events
        """
        df = self.get_economic_calendar(start, end)

        if df.empty:
            return df

        # Filter for FOMC events
        fomc_keywords = ["FOMC", "Federal Reserve", "Fed Rate", "Interest Rate Decision"]

        if "event" in df.columns:
            mask = df["event"].str.contains("|".join(fomc_keywords), case=False, na=False)
            df = df[mask]

        return df

    # ========== Market News ==========

    def get_market_news(
        self,
        category: Literal["general", "forex", "crypto", "merger"] = "general",
        min_id: int = 0,
    ) -> pd.DataFrame:
        """Get latest market news.

        Args:
            category: News category
            min_id: Minimum news ID (for pagination)

        Returns:
            DataFrame with news articles
        """
        endpoint = "news"
        params = {
            "category": category,
        }

        if min_id > 0:
            params["minId"] = min_id

        data = self.get(endpoint, params=params)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], unit="s")

        return df

    def get_company_news(
        self,
        symbol: str,
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pd.DataFrame:
        """Get news for a specific company.

        Args:
            symbol: Stock ticker symbol
            start: Start date
            end: End date

        Returns:
            DataFrame with company news
        """
        if start is None:
            start = date.today() - timedelta(days=30)
        if isinstance(start, str):
            start = date.fromisoformat(start)

        if end is None:
            end = date.today()
        if isinstance(end, str):
            end = date.fromisoformat(end)

        endpoint = "company-news"
        params = {
            "symbol": symbol.upper(),
            "from": start.isoformat(),
            "to": end.isoformat(),
        }

        data = self.get(endpoint, params=params)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], unit="s")

        return df

    # ========== Company Information ==========

    def get_company_profile(self, symbol: str) -> dict[str, Any]:
        """Get company profile.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Company profile data
        """
        endpoint = "stock/profile2"
        params = {"symbol": symbol.upper()}

        data = self.get(endpoint, params=params)

        return data if data else {}

    def get_basic_financials(
        self,
        symbol: str,
        metric: Literal["all", "margin", "profitability", "growth", "valuation"] = "all",
    ) -> dict[str, Any]:
        """Get basic financial metrics.

        Args:
            symbol: Stock ticker symbol
            metric: Metric category

        Returns:
            Financial metrics
        """
        endpoint = "stock/metric"
        params = {
            "symbol": symbol.upper(),
            "metric": metric,
        }

        data = self.get(endpoint, params=params)

        return data if data else {}

    # ========== Market Sentiment ==========

    def get_social_sentiment(
        self,
        symbol: str,
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pd.DataFrame:
        """Get social media sentiment for a stock.

        Args:
            symbol: Stock ticker symbol
            start: Start date
            end: End date

        Returns:
            DataFrame with sentiment data
        """
        if start is None:
            start = date.today() - timedelta(days=7)
        if isinstance(start, str):
            start = date.fromisoformat(start)

        if end is None:
            end = date.today()
        if isinstance(end, str):
            end = date.fromisoformat(end)

        endpoint = "stock/social-sentiment"
        params = {
            "symbol": symbol.upper(),
            "from": start.isoformat(),
            "to": end.isoformat(),
        }

        data = self.get(endpoint, params=params)

        if not data or "data" not in data:
            return pd.DataFrame()

        return pd.DataFrame(data["data"])

    def get_insider_sentiment(
        self,
        symbol: str,
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pd.DataFrame:
        """Get insider trading sentiment.

        Args:
            symbol: Stock ticker symbol
            start: Start date
            end: End date

        Returns:
            DataFrame with insider sentiment
        """
        if start is None:
            start = date.today() - timedelta(days=90)
        if isinstance(start, str):
            start = date.fromisoformat(start)

        if end is None:
            end = date.today()
        if isinstance(end, str):
            end = date.fromisoformat(end)

        endpoint = "stock/insider-sentiment"
        params = {
            "symbol": symbol.upper(),
            "from": start.isoformat(),
            "to": end.isoformat(),
        }

        data = self.get(endpoint, params=params)

        if not data or "data" not in data:
            return pd.DataFrame()

        return pd.DataFrame(data["data"])

    # ========== Earnings ==========

    def get_earnings_calendar(
        self,
        start: date | str | None = None,
        end: date | str | None = None,
        symbol: str | None = None,
    ) -> pd.DataFrame:
        """Get earnings calendar.

        Args:
            start: Start date
            end: End date
            symbol: Optional symbol filter

        Returns:
            DataFrame with earnings dates
        """
        if start is None:
            start = date.today()
        if isinstance(start, str):
            start = date.fromisoformat(start)

        if end is None:
            end = start + timedelta(days=30)
        if isinstance(end, str):
            end = date.fromisoformat(end)

        endpoint = "calendar/earnings"
        params = {
            "from": start.isoformat(),
            "to": end.isoformat(),
        }

        if symbol:
            params["symbol"] = symbol.upper()

        data = self.get(endpoint, params=params)

        if not data or "earningsCalendar" not in data:
            return pd.DataFrame()

        return pd.DataFrame(data["earningsCalendar"])

    # ========== IPO Calendar ==========

    def get_ipo_calendar(
        self,
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pd.DataFrame:
        """Get IPO calendar.

        Args:
            start: Start date
            end: End date

        Returns:
            DataFrame with IPO dates
        """
        if start is None:
            start = date.today()
        if isinstance(start, str):
            start = date.fromisoformat(start)

        if end is None:
            end = start + timedelta(days=90)
        if isinstance(end, str):
            end = date.fromisoformat(end)

        endpoint = "calendar/ipo"
        params = {
            "from": start.isoformat(),
            "to": end.isoformat(),
        }

        data = self.get(endpoint, params=params)

        if not data or "ipoCalendar" not in data:
            return pd.DataFrame()

        return pd.DataFrame(data["ipoCalendar"])

    # ========== Market Status ==========

    def get_market_status(
        self,
        exchange: str = "US",
    ) -> dict[str, Any]:
        """Get market status (open/closed).

        Args:
            exchange: Exchange code

        Returns:
            Market status data
        """
        endpoint = "stock/market-status"
        params = {"exchange": exchange}

        return self.get(endpoint, params=params) or {}

    def get_market_holidays(
        self,
        exchange: str = "US",
    ) -> pd.DataFrame:
        """Get market holiday calendar.

        Args:
            exchange: Exchange code

        Returns:
            DataFrame with holiday dates
        """
        endpoint = "stock/market-holiday"
        params = {"exchange": exchange}

        data = self.get(endpoint, params=params)

        if not data or "data" not in data:
            return pd.DataFrame()

        return pd.DataFrame(data["data"])
