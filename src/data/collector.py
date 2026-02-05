"""Data collection orchestrator for market data."""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .fmp_client import FMPClient
from .fred_client import FREDClient, FREDIndicators
from .vix_client import VIXClient
from .finnhub_client import FinnhubClient
from .yfinance_client import YFinanceClient
from .cache import DataCache

# Import Fama-French loader from factors module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from factors.ff_data import FamaFrenchDataLoader

logger = logging.getLogger(__name__)


@dataclass
class CollectionConfig:
    """Configuration for data collection."""

    # Date range
    start_date: date = field(default_factory=lambda: date(2015, 1, 1))
    end_date: date = field(default_factory=date.today)

    # Data storage
    data_dir: Path = field(default_factory=lambda: Path("data"))

    # Universe settings
    sp500_symbols: list[str] = field(default_factory=list)

    # Collection flags
    collect_prices: bool = True
    collect_financials: bool = True
    collect_metrics: bool = True
    collect_macro: bool = True


# S&P 500 constituents (top 100 by weight for initial testing)
SP500_TOP_100 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "GOOG", "BRK.B", "TSLA", "UNH",
    "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
    "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO", "TMO", "ACN",
    "ABT", "CRM", "DHR", "NKE", "LIN", "NEE", "CMCSA", "TXN", "PM", "VZ",
    "ADBE", "BMY", "WFC", "INTC", "AMD", "UNP", "ORCL", "COP", "RTX", "HON",
    "QCOM", "LOW", "UPS", "CAT", "IBM", "SPGI", "BA", "GE", "ELV", "SBUX",
    "INTU", "DE", "MS", "BLK", "AMAT", "AXP", "GS", "PLD", "ISRG", "MDLZ",
    "GILD", "ADP", "TJX", "SYK", "ADI", "BKNG", "CVS", "MMC", "LMT", "VRTX",
    "C", "AMT", "TMUS", "MO", "CI", "CB", "ZTS", "REGN", "SCHW", "EOG",
    "SO", "DUK", "BDX", "LRCX", "PGR", "CME", "FISV", "NOC", "SLB", "ETN",
]


class DataCollector:
    """Orchestrates data collection from multiple sources.

    Example:
        >>> collector = DataCollector(fmp_key="...", fred_key="...")
        >>> collector.collect_all()
        >>>
        >>> # Or collect specific data
        >>> prices = collector.collect_stock_prices(["AAPL", "MSFT"])
        >>> macro = collector.collect_macro_indicators()
    """

    def __init__(
        self,
        fmp_key: str | None = None,
        fred_key: str | None = None,
        finnhub_key: str | None = None,
        config: CollectionConfig | None = None,
    ):
        """Initialize data collector.

        Args:
            fmp_key: FMP API key
            fred_key: FRED API key
            finnhub_key: Finnhub API key
            config: Collection configuration
        """
        self.config = config or CollectionConfig()

        # Initialize clients
        self.fmp = FMPClient(fmp_key) if fmp_key else None
        self.fred = FREDClient(fred_key) if fred_key else None
        self.finnhub = FinnhubClient(finnhub_key) if finnhub_key else None
        self.vix = VIXClient(fred_api_key=fred_key)
        self.ff_loader = FamaFrenchDataLoader(cache_dir=self.config.data_dir / ".cache" / "ff_data")

        # Initialize Yahoo Finance client (no API key required - free)
        try:
            self.yfinance = YFinanceClient()
        except ImportError:
            self.yfinance = None
            logger.warning("yfinance not installed. Install with: pip install yfinance")

        # Initialize cache
        self.cache = DataCache()

        # Set default universe
        if not self.config.sp500_symbols:
            self.config.sp500_symbols = SP500_TOP_100

        # Create data directories
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Create data storage directories."""
        dirs = [
            self.config.data_dir / "raw" / "prices" / "us",
            self.config.data_dir / "raw" / "fundamentals" / "financials",
            self.config.data_dir / "raw" / "fundamentals" / "metrics",
            self.config.data_dir / "raw" / "macro",
            self.config.data_dir / "raw" / "vix",
            self.config.data_dir / "raw" / "factors",
            self.config.data_dir / "raw" / "calendar",
            self.config.data_dir / "raw" / "news",
            self.config.data_dir / "processed" / "returns",
            self.config.data_dir / "processed" / "factors",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    # ========== Stock Price Collection ==========

    def collect_stock_prices(
        self,
        symbols: list[str] | None = None,
        start: date | None = None,
        end: date | None = None,
        source: str = "auto",
    ) -> dict[str, pd.DataFrame]:
        """Collect historical prices for multiple stocks.

        Args:
            symbols: List of stock symbols (default: S&P 500 top 100)
            start: Start date (default: config start_date)
            end: End date (default: config end_date)
            source: Data source - "fmp", "yfinance", or "auto" (uses yfinance if fmp unavailable)

        Returns:
            Dictionary of symbol -> price DataFrame
        """
        # Determine which client to use
        use_fmp = source == "fmp" or (source == "auto" and self.fmp)
        use_yfinance = source == "yfinance" or (source == "auto" and not self.fmp)

        if use_fmp and not self.fmp:
            raise ValueError("FMP client not initialized. Provide fmp_key.")
        if use_yfinance and not self.yfinance:
            raise ValueError("YFinance client not available. Install with: pip install yfinance")

        symbols = symbols or self.config.sp500_symbols
        start = start or self.config.start_date
        end = end or self.config.end_date

        results = {}
        failed = []

        client_name = "FMP" if use_fmp else "Yahoo Finance"
        logger.info(f"Collecting prices for {len(symbols)} stocks from {start} to {end} using {client_name}")

        for i, symbol in enumerate(symbols, 1):
            try:
                if use_fmp:
                    df = self.fmp.get_historical_prices(symbol, start=start, end=end)
                else:
                    df = self.yfinance.get_historical_prices(symbol, start=start, end=end)

                if df.empty:
                    logger.warning(f"[{i}/{len(symbols)}] {symbol}: No data")
                    failed.append(symbol)
                    continue

                # Save to file
                output_path = self.config.data_dir / "raw" / "prices" / "us" / f"{symbol}.parquet"
                df.to_parquet(output_path)

                results[symbol] = df
                logger.info(f"[{i}/{len(symbols)}] {symbol}: {len(df)} rows")

            except Exception as e:
                logger.error(f"[{i}/{len(symbols)}] {symbol}: {e}")
                failed.append(symbol)

        logger.info(f"Collected {len(results)} stocks, {len(failed)} failed")

        if failed:
            logger.warning(f"Failed symbols: {failed[:10]}{'...' if len(failed) > 10 else ''}")

        return results

    def collect_single_stock(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
        source: str = "auto",
    ) -> pd.DataFrame:
        """Collect prices for a single stock.

        Args:
            symbol: Stock ticker symbol
            start: Start date
            end: End date
            source: Data source - "fmp", "yfinance", or "auto"

        Returns:
            Price DataFrame
        """
        use_fmp = source == "fmp" or (source == "auto" and self.fmp)

        if use_fmp and not self.fmp:
            raise ValueError("FMP client not initialized. Provide fmp_key.")
        if not use_fmp and not self.yfinance:
            raise ValueError("YFinance client not available. Install with: pip install yfinance")

        start = start or self.config.start_date
        end = end or self.config.end_date

        if use_fmp:
            df = self.fmp.get_historical_prices(symbol, start=start, end=end)
        else:
            df = self.yfinance.get_historical_prices(symbol, start=start, end=end)

        if not df.empty:
            output_path = self.config.data_dir / "raw" / "prices" / "us" / f"{symbol}.parquet"
            df.to_parquet(output_path)

        return df

    def collect_index_data(
        self,
        indexes: list[str] | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Collect market index data using Yahoo Finance (free).

        Args:
            indexes: List of index names (sp500, dow, nasdaq, russell2000, vix)
            start: Start date
            end: End date

        Returns:
            Dictionary of index_name -> price DataFrame
        """
        if not self.yfinance:
            raise ValueError("YFinance client not available. Install with: pip install yfinance")

        indexes = indexes or ["sp500", "dow", "nasdaq", "russell2000", "vix"]
        start = start or self.config.start_date
        end = end or self.config.end_date

        results = {}

        logger.info(f"Collecting {len(indexes)} market indexes from {start} to {end}")

        for index_name in indexes:
            try:
                df = self.yfinance.get_index_data(index=index_name, start=start, end=end)

                if not df.empty:
                    output_path = self.config.data_dir / "raw" / "prices" / "us" / f"INDEX_{index_name}.parquet"
                    df.to_parquet(output_path)
                    results[index_name] = df
                    logger.info(f"{index_name}: {len(df)} rows")

            except Exception as e:
                logger.error(f"{index_name}: {e}")

        return results

    def collect_sector_etfs(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Collect sector ETF data using Yahoo Finance (free).

        Args:
            start: Start date
            end: End date

        Returns:
            Dictionary of sector_name -> price DataFrame
        """
        if not self.yfinance:
            raise ValueError("YFinance client not available. Install with: pip install yfinance")

        start = start or self.config.start_date
        end = end or self.config.end_date

        logger.info(f"Collecting sector ETF data from {start} to {end}")

        results = self.yfinance.get_sector_etfs(start=start, end=end)

        # Save each sector
        for sector_name, df in results.items():
            safe_name = sector_name.replace(" ", "_").lower()
            output_path = self.config.data_dir / "raw" / "prices" / "us" / f"SECTOR_{safe_name}.parquet"
            df.to_parquet(output_path)

        logger.info(f"Collected {len(results)} sector ETFs")

        return results

    # ========== Financial Data Collection ==========

    def collect_financials(
        self,
        symbols: list[str] | None = None,
        statement_types: list[str] | None = None,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """Collect financial statements for multiple stocks.

        Args:
            symbols: List of stock symbols
            statement_types: List of statement types ('income', 'balance', 'cash')

        Returns:
            Nested dict: symbol -> statement_type -> DataFrame
        """
        if not self.fmp:
            raise ValueError("FMP client not initialized. Provide fmp_key.")

        symbols = symbols or self.config.sp500_symbols
        statement_types = statement_types or ["income", "balance", "cash"]

        results = {}

        logger.info(f"Collecting financials for {len(symbols)} stocks")

        for i, symbol in enumerate(symbols, 1):
            results[symbol] = {}

            for stmt_type in statement_types:
                try:
                    df = self.fmp.get_financial_statements(
                        symbol,
                        statement_type=stmt_type,
                        period="quarter",
                        limit=40,  # ~10 years of quarterly data
                    )

                    if not df.empty:
                        output_path = (
                            self.config.data_dir / "raw" / "fundamentals" / "financials" /
                            f"{symbol}_{stmt_type}.parquet"
                        )
                        df.to_parquet(output_path)
                        results[symbol][stmt_type] = df

                except Exception as e:
                    logger.error(f"{symbol} {stmt_type}: {e}")

            if (i % 10) == 0:
                logger.info(f"Processed {i}/{len(symbols)} stocks")

        return results

    def collect_key_metrics(
        self,
        symbols: list[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Collect key financial metrics for multiple stocks.

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary of symbol -> metrics DataFrame
        """
        if not self.fmp:
            raise ValueError("FMP client not initialized. Provide fmp_key.")

        symbols = symbols or self.config.sp500_symbols
        results = {}

        logger.info(f"Collecting key metrics for {len(symbols)} stocks")

        for i, symbol in enumerate(symbols, 1):
            try:
                df = self.fmp.get_key_metrics(symbol, period="quarter", limit=40)

                if not df.empty:
                    output_path = (
                        self.config.data_dir / "raw" / "fundamentals" / "metrics" /
                        f"{symbol}_metrics.parquet"
                    )
                    df.to_parquet(output_path)
                    results[symbol] = df

            except Exception as e:
                logger.error(f"{symbol} metrics: {e}")

            if (i % 10) == 0:
                logger.info(f"Processed {i}/{len(symbols)} stocks")

        return results

    # ========== Macro Indicator Collection ==========

    def collect_macro_indicators(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> dict[str, pd.Series]:
        """Collect macroeconomic indicators from FRED.

        Args:
            start: Start date
            end: End date

        Returns:
            Dictionary of indicator_name -> time series
        """
        if not self.fred:
            raise ValueError("FRED client not initialized. Provide fred_key.")

        start = start or self.config.start_date
        end = end or self.config.end_date

        # Core indicators for regime classification
        indicators = {
            "gdp": FREDIndicators.GDPC1,
            "unemployment": FREDIndicators.UNRATE,
            "initial_claims": FREDIndicators.ICSA,
            "cpi": FREDIndicators.CPIAUCSL,
            "pce": FREDIndicators.PCEPI,
            "fed_funds": FREDIndicators.FEDFUNDS,
            "treasury_10y": FREDIndicators.DGS10,
            "treasury_2y": FREDIndicators.DGS2,
            "yield_spread_10y2y": FREDIndicators.T10Y2Y,
            "yield_spread_10y3m": FREDIndicators.T10Y3M,
            "consumer_sentiment": FREDIndicators.UMCSENT,
            "leading_index": FREDIndicators.USSLIND,
            "housing_starts": FREDIndicators.HOUST,
            "nonfarm_payrolls": FREDIndicators.PAYEMS,
            "m2_money": FREDIndicators.M2SL,
        }

        results = {}

        logger.info(f"Collecting {len(indicators)} macro indicators from {start} to {end}")

        for name, series_id in indicators.items():
            try:
                series = self.fred.get_series(series_id, start=start, end=end)

                if not series.empty:
                    output_path = self.config.data_dir / "raw" / "macro" / f"{name}.parquet"
                    series.to_frame(name).to_parquet(output_path)
                    results[name] = series
                    logger.info(f"{name}: {len(series)} observations")
                else:
                    logger.warning(f"{name}: No data")

            except Exception as e:
                logger.error(f"{name}: {e}")

        return results

    def collect_regime_indicators(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """Collect and align indicators needed for regime classification.

        Returns a DataFrame with all indicators aligned to a common date index.

        Args:
            start: Start date
            end: End date

        Returns:
            DataFrame with aligned macro indicators
        """
        indicators = self.collect_macro_indicators(start=start, end=end)

        # Combine into single DataFrame
        df = pd.DataFrame(indicators)

        # Forward fill to handle different frequencies
        df = df.ffill()

        # Save combined data
        output_path = self.config.data_dir / "processed" / "regime_indicators.parquet"
        df.to_parquet(output_path)

        logger.info(f"Combined regime indicators: {df.shape}")

        return df

    # ========== VIX Data Collection ==========

    def collect_vix_data(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """Collect VIX volatility index data.

        Args:
            start: Start date
            end: End date

        Returns:
            DataFrame with VIX OHLC data
        """
        start = start or self.config.start_date
        end = end or self.config.end_date

        logger.info(f"Collecting VIX data from {start} to {end}")

        df = self.vix.get_vix_history(start_date=start, end_date=end)

        if not df.empty:
            output_path = self.config.data_dir / "raw" / "vix" / "vix_daily.parquet"
            df.to_parquet(output_path)
            logger.info(f"VIX data: {len(df)} rows")

        return df

    # ========== Fama-French Factor Collection ==========

    def collect_ff_factors(
        self,
        frequency: str = "daily",
        include_momentum: bool = True,
    ) -> pd.DataFrame:
        """Collect Fama-French factor data.

        Args:
            frequency: "daily" or "monthly"
            include_momentum: Include momentum factor

        Returns:
            DataFrame with factor returns
        """
        start = str(self.config.start_date)
        end = str(self.config.end_date)

        logger.info(f"Collecting FF5 factors ({frequency}) from {start} to {end}")

        if include_momentum:
            df = self.ff_loader.load_factors_with_momentum(
                num_factors=5,
                frequency=frequency,
                start_date=start,
                end_date=end,
            )
        else:
            df = self.ff_loader.load_ff5_factors(
                frequency=frequency,
                start_date=start,
                end_date=end,
            )

        if not df.empty:
            output_path = self.config.data_dir / "raw" / "factors" / f"ff5_mom_{frequency}.parquet"
            df.to_parquet(output_path)
            logger.info(f"FF factors: {len(df)} rows, columns: {list(df.columns)}")

        return df

    # ========== Economic Calendar Collection ==========

    def collect_economic_calendar(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """Collect economic calendar from Finnhub.

        Args:
            start: Start date
            end: End date

        Returns:
            DataFrame with economic events
        """
        if not self.finnhub:
            logger.warning("Finnhub client not initialized. Skipping economic calendar.")
            return pd.DataFrame()

        start = start or date.today()
        end = end or (start + timedelta(days=90))

        logger.info(f"Collecting economic calendar from {start} to {end}")

        df = self.finnhub.get_economic_calendar(start=start, end=end)

        if not df.empty:
            output_path = self.config.data_dir / "raw" / "calendar" / "economic_calendar.parquet"
            df.to_parquet(output_path)
            logger.info(f"Economic calendar: {len(df)} events")

        return df

    def collect_earnings_calendar(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """Collect earnings calendar from Finnhub.

        Args:
            start: Start date
            end: End date

        Returns:
            DataFrame with earnings dates
        """
        if not self.finnhub:
            logger.warning("Finnhub client not initialized. Skipping earnings calendar.")
            return pd.DataFrame()

        start = start or date.today()
        end = end or (start + timedelta(days=30))

        logger.info(f"Collecting earnings calendar from {start} to {end}")

        df = self.finnhub.get_earnings_calendar(start=start, end=end)

        if not df.empty:
            output_path = self.config.data_dir / "raw" / "calendar" / "earnings_calendar.parquet"
            df.to_parquet(output_path)
            logger.info(f"Earnings calendar: {len(df)} events")

        return df

    # ========== Market News Collection ==========

    def collect_market_news(
        self,
        category: str = "general",
    ) -> pd.DataFrame:
        """Collect market news from Finnhub.

        Args:
            category: News category (general, forex, crypto, merger)

        Returns:
            DataFrame with news articles
        """
        if not self.finnhub:
            logger.warning("Finnhub client not initialized. Skipping market news.")
            return pd.DataFrame()

        logger.info(f"Collecting {category} market news")

        df = self.finnhub.get_market_news(category=category)

        if not df.empty:
            output_path = self.config.data_dir / "raw" / "news" / f"news_{category}.parquet"
            df.to_parquet(output_path)
            logger.info(f"Market news: {len(df)} articles")

        return df

    # ========== Unified Collection ==========

    def collect_all(self) -> dict[str, Any]:
        """Collect all configured data.

        Returns:
            Dictionary with collected data summary
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "start_date": str(self.config.start_date),
                "end_date": str(self.config.end_date),
                "symbols_count": len(self.config.sp500_symbols),
            },
            "prices": {},
            "financials": {},
            "metrics": {},
            "macro": {},
            "vix": {},
            "factors": {},
            "calendar": {},
        }

        # Stock prices (FMP)
        if self.config.collect_prices and self.fmp:
            logger.info("=== Collecting Stock Prices ===")
            prices = self.collect_stock_prices()
            results["prices"] = {
                "count": len(prices),
                "symbols": list(prices.keys())[:10],
            }

        # Macro indicators (FRED)
        if self.config.collect_macro and self.fred:
            logger.info("=== Collecting Macro Indicators ===")
            macro = self.collect_macro_indicators()
            results["macro"] = {
                "count": len(macro),
                "indicators": list(macro.keys()),
            }

        # VIX data (always collect - no API key needed for CBOE)
        logger.info("=== Collecting VIX Data ===")
        try:
            vix = self.collect_vix_data()
            results["vix"] = {"count": len(vix)}
        except Exception as e:
            logger.error(f"VIX collection failed: {e}")
            results["vix"] = {"error": str(e)}

        # Fama-French factors (always collect - free data)
        logger.info("=== Collecting Fama-French Factors ===")
        try:
            ff = self.collect_ff_factors()
            results["factors"] = {
                "count": len(ff),
                "columns": list(ff.columns),
            }
        except Exception as e:
            logger.error(f"FF factors collection failed: {e}")
            results["factors"] = {"error": str(e)}

        # Economic calendar (Finnhub)
        if self.finnhub:
            logger.info("=== Collecting Economic Calendar ===")
            try:
                calendar = self.collect_economic_calendar()
                results["calendar"] = {"count": len(calendar)}
            except Exception as e:
                logger.error(f"Calendar collection failed: {e}")
                results["calendar"] = {"error": str(e)}

        # Financial statements (FMP)
        if self.config.collect_financials and self.fmp:
            logger.info("=== Collecting Financial Statements ===")
            financials = self.collect_financials()
            results["financials"] = {"count": len(financials)}

        # Key metrics (FMP)
        if self.config.collect_metrics and self.fmp:
            logger.info("=== Collecting Key Metrics ===")
            metrics = self.collect_key_metrics()
            results["metrics"] = {"count": len(metrics)}

        logger.info("=== Collection Complete ===")

        return results

    # ========== Data Loading ==========

    def load_stock_prices(self, symbol: str) -> pd.DataFrame:
        """Load saved stock price data.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Price DataFrame
        """
        path = self.config.data_dir / "raw" / "prices" / "us" / f"{symbol}.parquet"

        if path.exists():
            return pd.read_parquet(path)

        return pd.DataFrame()

    def load_all_prices(self) -> dict[str, pd.DataFrame]:
        """Load all saved stock price data.

        Returns:
            Dictionary of symbol -> price DataFrame
        """
        price_dir = self.config.data_dir / "raw" / "prices" / "us"
        results = {}

        for path in price_dir.glob("*.parquet"):
            symbol = path.stem
            results[symbol] = pd.read_parquet(path)

        return results

    def load_macro_indicators(self) -> pd.DataFrame:
        """Load saved macro indicator data.

        Returns:
            DataFrame with macro indicators
        """
        path = self.config.data_dir / "processed" / "regime_indicators.parquet"

        if path.exists():
            return pd.read_parquet(path)

        # Try loading individual indicators
        macro_dir = self.config.data_dir / "raw" / "macro"
        indicators = {}

        for path in macro_dir.glob("*.parquet"):
            name = path.stem
            df = pd.read_parquet(path)
            indicators[name] = df[name] if name in df.columns else df.iloc[:, 0]

        if indicators:
            return pd.DataFrame(indicators).ffill()

        return pd.DataFrame()

    # ========== Data Status ==========

    def get_collection_status(self) -> dict[str, Any]:
        """Get status of collected data.

        Returns:
            Dictionary with data status
        """
        status = {
            "prices": {"count": 0, "symbols": []},
            "financials": {"count": 0},
            "metrics": {"count": 0},
            "macro": {"count": 0, "indicators": []},
        }

        # Check prices
        price_dir = self.config.data_dir / "raw" / "prices" / "us"
        if price_dir.exists():
            files = list(price_dir.glob("*.parquet"))
            status["prices"]["count"] = len(files)
            status["prices"]["symbols"] = [f.stem for f in files[:10]]

        # Check financials
        fin_dir = self.config.data_dir / "raw" / "fundamentals" / "financials"
        if fin_dir.exists():
            status["financials"]["count"] = len(list(fin_dir.glob("*.parquet")))

        # Check metrics
        metrics_dir = self.config.data_dir / "raw" / "fundamentals" / "metrics"
        if metrics_dir.exists():
            status["metrics"]["count"] = len(list(metrics_dir.glob("*.parquet")))

        # Check macro
        macro_dir = self.config.data_dir / "raw" / "macro"
        if macro_dir.exists():
            files = list(macro_dir.glob("*.parquet"))
            status["macro"]["count"] = len(files)
            status["macro"]["indicators"] = [f.stem for f in files]

        return status
