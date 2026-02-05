"""Yahoo Finance client using yfinance package.

No API key required - completely free.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Literal

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

logger = logging.getLogger(__name__)


class YFinanceClient:
    """Client for fetching data from Yahoo Finance.

    Provides free access to:
    - Historical stock prices (OHLCV)
    - Stock info and fundamentals
    - Dividend and split history
    - Index data (^GSPC, ^DJI, ^IXIC, etc.)
    - ETF data

    Example:
        >>> client = YFinanceClient()
        >>> prices = client.get_historical_prices("AAPL", period="1y")
        >>> info = client.get_stock_info("MSFT")
    """

    def __init__(self):
        """Initialize Yahoo Finance client."""
        if yf is None:
            raise ImportError(
                "yfinance package not installed. "
                "Install with: pip install yfinance"
            )

    def get_historical_prices(
        self,
        symbol: str,
        start: date | str | None = None,
        end: date | str | None = None,
        period: str | None = None,
        interval: Literal["1d", "1wk", "1mo"] = "1d",
    ) -> pd.DataFrame:
        """Get historical price data for a symbol.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL", "MSFT")
            start: Start date (YYYY-MM-DD or date object)
            end: End date (YYYY-MM-DD or date object)
            period: Alternative to start/end. Valid periods:
                    1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
            interval: Data interval (1d=daily, 1wk=weekly, 1mo=monthly)

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
        """
        ticker = yf.Ticker(symbol)

        if period:
            df = ticker.history(period=period, interval=interval)
        else:
            start_str = str(start) if start else None
            end_str = str(end) if end else None
            df = ticker.history(start=start_str, end=end_str, interval=interval)

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

        # Standardize column names
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]

        # Remove timezone info from index
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        logger.info(f"Fetched {len(df)} rows for {symbol}")
        return df

    def get_multiple_prices(
        self,
        symbols: list[str],
        start: date | str | None = None,
        end: date | str | None = None,
        period: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Get historical prices for multiple symbols.

        Args:
            symbols: List of ticker symbols
            start: Start date
            end: End date
            period: Alternative to start/end

        Returns:
            Dict mapping symbol to DataFrame
        """
        results = {}

        for symbol in symbols:
            try:
                df = self.get_historical_prices(
                    symbol, start=start, end=end, period=period
                )
                if not df.empty:
                    results[symbol] = df
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")

        logger.info(f"Fetched prices for {len(results)}/{len(symbols)} symbols")
        return results

    def get_stock_info(self, symbol: str) -> dict:
        """Get stock information and fundamentals.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with company info, financials, etc.
        """
        ticker = yf.Ticker(symbol)
        return ticker.info

    def get_financials(self, symbol: str) -> dict[str, pd.DataFrame]:
        """Get financial statements.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with keys: income_stmt, balance_sheet, cash_flow
        """
        ticker = yf.Ticker(symbol)

        return {
            "income_stmt": ticker.income_stmt,
            "balance_sheet": ticker.balance_sheet,
            "cash_flow": ticker.cashflow,
            "quarterly_income_stmt": ticker.quarterly_income_stmt,
            "quarterly_balance_sheet": ticker.quarterly_balance_sheet,
            "quarterly_cash_flow": ticker.quarterly_cashflow,
        }

    def get_dividends(self, symbol: str) -> pd.Series:
        """Get dividend history.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Series with dividend amounts indexed by date
        """
        ticker = yf.Ticker(symbol)
        return ticker.dividends

    def get_splits(self, symbol: str) -> pd.Series:
        """Get stock split history.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Series with split ratios indexed by date
        """
        ticker = yf.Ticker(symbol)
        return ticker.splits

    def get_recommendations(self, symbol: str) -> pd.DataFrame:
        """Get analyst recommendations.

        Args:
            symbol: Stock ticker symbol

        Returns:
            DataFrame with analyst recommendations
        """
        ticker = yf.Ticker(symbol)
        return ticker.recommendations

    def get_index_data(
        self,
        index: Literal["sp500", "dow", "nasdaq", "russell2000", "vix"] = "sp500",
        start: date | str | None = None,
        end: date | str | None = None,
        period: str | None = None,
    ) -> pd.DataFrame:
        """Get index data.

        Args:
            index: Index name (sp500, dow, nasdaq, russell2000, vix)
            start: Start date
            end: End date
            period: Alternative to start/end

        Returns:
            DataFrame with index prices
        """
        index_map = {
            "sp500": "^GSPC",
            "dow": "^DJI",
            "nasdaq": "^IXIC",
            "russell2000": "^RUT",
            "vix": "^VIX",
        }

        symbol = index_map.get(index, index)
        return self.get_historical_prices(symbol, start=start, end=end, period=period)

    def get_sector_etfs(
        self,
        start: date | str | None = None,
        end: date | str | None = None,
        period: str = "1y",
    ) -> dict[str, pd.DataFrame]:
        """Get sector ETF data (SPDR sector ETFs).

        Args:
            start: Start date
            end: End date
            period: Alternative to start/end

        Returns:
            Dict mapping sector name to DataFrame
        """
        sector_etfs = {
            "Technology": "XLK",
            "Healthcare": "XLV",
            "Financials": "XLF",
            "Consumer Discretionary": "XLY",
            "Communication Services": "XLC",
            "Industrials": "XLI",
            "Consumer Staples": "XLP",
            "Energy": "XLE",
            "Utilities": "XLU",
            "Real Estate": "XLRE",
            "Materials": "XLB",
        }

        results = {}
        for sector, etf in sector_etfs.items():
            df = self.get_historical_prices(etf, start=start, end=end, period=period)
            if not df.empty:
                results[sector] = df

        return results

    def get_market_cap(self, symbol: str) -> float | None:
        """Get current market capitalization.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Market cap in dollars, or None if unavailable
        """
        info = self.get_stock_info(symbol)
        return info.get("marketCap")

    def download_bulk(
        self,
        symbols: list[str],
        start: date | str | None = None,
        end: date | str | None = None,
        period: str | None = None,
    ) -> pd.DataFrame:
        """Download data for multiple symbols efficiently.

        Uses yfinance's download function which is faster for bulk downloads.

        Args:
            symbols: List of ticker symbols
            start: Start date
            end: End date
            period: Alternative to start/end

        Returns:
            DataFrame with MultiIndex columns (symbol, OHLCV)
        """
        if period:
            df = yf.download(symbols, period=period, group_by="ticker", progress=False)
        else:
            df = yf.download(
                symbols,
                start=str(start) if start else None,
                end=str(end) if end else None,
                group_by="ticker",
                progress=False,
            )

        logger.info(f"Bulk downloaded {len(df)} rows for {len(symbols)} symbols")
        return df
