"""FRED (Federal Reserve Economic Data) API client."""

import logging
from datetime import date
from typing import Any

import pandas as pd

from .base_client import BaseClient

logger = logging.getLogger(__name__)


class FREDIndicators:
    """Common FRED series IDs for macroeconomic indicators."""

    # GDP and Output
    GDP = "GDP"                      # Gross Domestic Product
    GDPC1 = "GDPC1"                  # Real GDP
    
    # Employment
    UNRATE = "UNRATE"                # Unemployment Rate
    PAYEMS = "PAYEMS"                # Total Nonfarm Payrolls
    ICSA = "ICSA"                    # Initial Jobless Claims
    
    # Inflation
    CPIAUCSL = "CPIAUCSL"            # Consumer Price Index
    PCEPI = "PCEPI"                  # PCE Price Index
    
    # Interest Rates
    FEDFUNDS = "FEDFUNDS"            # Federal Funds Rate
    DFF = "DFF"                      # Daily Federal Funds Rate
    DGS10 = "DGS10"                  # 10-Year Treasury Rate
    DGS2 = "DGS2"                    # 2-Year Treasury Rate
    T10Y2Y = "T10Y2Y"                # 10Y-2Y Treasury Spread
    T10Y3M = "T10Y3M"                # 10Y-3M Treasury Spread
    
    # Money Supply
    M2SL = "M2SL"                    # M2 Money Stock
    
    # Housing
    HOUST = "HOUST"                  # Housing Starts
    
    # Consumer
    UMCSENT = "UMCSENT"              # Consumer Sentiment
    
    # Leading Indicators
    USSLIND = "USSLIND"              # Leading Index


class FREDClient(BaseClient):
    """Client for FRED (Federal Reserve Economic Data) API.
    
    Provides access to macroeconomic indicators for regime classification.
    
    API Documentation: https://fred.stlouisfed.org/docs/api/fred/
    """

    BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(self, api_key: str, requests_per_minute: int = 120):
        super().__init__(
            base_url=self.BASE_URL,
            api_key=api_key,
            requests_per_minute=requests_per_minute,
        )

    def _get_auth_params(self) -> dict[str, str]:
        """Return FRED authentication parameters."""
        return {"api_key": self.api_key, "file_type": "json"}

    def get_series(
        self,
        series_id: str,
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pd.Series:
        """Get time series data for a FRED series.
        
        Args:
            series_id: FRED series ID (e.g., 'GDP', 'UNRATE')
            start: Start date (inclusive)
            end: End date (inclusive)
            
        Returns:
            pandas Series with date index and values
        """
        endpoint = "series/observations"
        params: dict[str, Any] = {"series_id": series_id}

        if start:
            params["observation_start"] = self._format_date(start)
        if end:
            params["observation_end"] = self._format_date(end)

        data = self.get(endpoint, params=params)

        if "observations" not in data:
            logger.warning(f"No observations found for {series_id}")
            return pd.Series(dtype=float)

        observations = data["observations"]
        
        dates = []
        values = []
        for obs in observations:
            dates.append(pd.to_datetime(obs["date"]))
            val = obs["value"]
            values.append(float(val) if val != "." else None)

        series = pd.Series(values, index=dates, name=series_id)
        series = series.dropna()
        
        return series

    def get_series_info(self, series_id: str) -> dict[str, Any]:
        """Get metadata for a FRED series.
        
        Args:
            series_id: FRED series ID
            
        Returns:
            Dictionary with series metadata
        """
        endpoint = "series"
        params = {"series_id": series_id}

        data = self.get(endpoint, params=params)

        if "seriess" not in data or not data["seriess"]:
            logger.warning(f"No series info found for {series_id}")
            return {}

        return data["seriess"][0]

    def get_gdp(
        self,
        start: date | str | None = None,
        end: date | str | None = None,
        real: bool = True,
    ) -> pd.Series:
        """Get GDP data.
        
        Args:
            start: Start date
            end: End date
            real: If True, get real GDP (inflation-adjusted)
            
        Returns:
            GDP time series
        """
        series_id = FREDIndicators.GDPC1 if real else FREDIndicators.GDP
        return self.get_series(series_id, start, end)

    def get_unemployment(
        self,
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pd.Series:
        """Get unemployment rate data.
        
        Args:
            start: Start date
            end: End date
            
        Returns:
            Unemployment rate time series
        """
        return self.get_series(FREDIndicators.UNRATE, start, end)

    def get_federal_funds_rate(
        self,
        start: date | str | None = None,
        end: date | str | None = None,
        daily: bool = False,
    ) -> pd.Series:
        """Get Federal Funds Rate.
        
        Args:
            start: Start date
            end: End date
            daily: If True, get daily rate; else monthly
            
        Returns:
            Federal funds rate time series
        """
        series_id = FREDIndicators.DFF if daily else FREDIndicators.FEDFUNDS
        return self.get_series(series_id, start, end)

    def get_yield_curve(
        self,
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pd.DataFrame:
        """Get yield curve data (10Y-2Y and 10Y-3M spreads).
        
        Args:
            start: Start date
            end: End date
            
        Returns:
            DataFrame with yield curve spreads
        """
        spread_10y2y = self.get_series(FREDIndicators.T10Y2Y, start, end)
        spread_10y3m = self.get_series(FREDIndicators.T10Y3M, start, end)
        
        df = pd.DataFrame({
            "T10Y2Y": spread_10y2y,
            "T10Y3M": spread_10y3m,
        })
        
        return df.dropna()

    def get_treasury_rates(
        self,
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pd.DataFrame:
        """Get Treasury rates (2Y, 10Y).
        
        Args:
            start: Start date
            end: End date
            
        Returns:
            DataFrame with Treasury rates
        """
        dgs2 = self.get_series(FREDIndicators.DGS2, start, end)
        dgs10 = self.get_series(FREDIndicators.DGS10, start, end)
        
        df = pd.DataFrame({
            "DGS2": dgs2,
            "DGS10": dgs10,
        })
        
        return df.dropna()

    def get_inflation(
        self,
        start: date | str | None = None,
        end: date | str | None = None,
        measure: str = "cpi",
    ) -> pd.Series:
        """Get inflation data.
        
        Args:
            start: Start date
            end: End date
            measure: 'cpi' for CPI, 'pce' for PCE
            
        Returns:
            Inflation index time series
        """
        series_id = FREDIndicators.CPIAUCSL if measure == "cpi" else FREDIndicators.PCEPI
        return self.get_series(series_id, start, end)

    def get_consumer_sentiment(
        self,
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pd.Series:
        """Get University of Michigan Consumer Sentiment.
        
        Args:
            start: Start date
            end: End date
            
        Returns:
            Consumer sentiment time series
        """
        return self.get_series(FREDIndicators.UMCSENT, start, end)

    def get_leading_index(
        self,
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pd.Series:
        """Get Leading Economic Index.
        
        Args:
            start: Start date
            end: End date
            
        Returns:
            Leading index time series
        """
        return self.get_series(FREDIndicators.USSLIND, start, end)

    @staticmethod
    def _format_date(d: date | str) -> str:
        """Format date to YYYY-MM-DD string."""
        if isinstance(d, str):
            return d
        return d.strftime("%Y-%m-%d")
