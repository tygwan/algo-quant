"""VIX (CBOE Volatility Index) data client."""

import io
import logging
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class VIXClient:
    """Client for VIX volatility index data.

    Fetches VIX data from multiple sources:
    - CBOE historical data (via GitHub mirror)
    - FRED API (alternative source)

    Example:
        >>> vix = VIXClient()
        >>> df = vix.get_vix_history()
        >>> print(df.tail())
    """

    # CBOE VIX data mirror on GitHub
    CBOE_GITHUB_URL = "https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv"

    # FRED series for VIX
    FRED_VIX_SERIES = "VIXCLS"  # CBOE VIX Close

    def __init__(
        self,
        fred_api_key: str | None = None,
        cache_dir: str | Path = ".cache/vix",
        use_cache: bool = True,
    ):
        """Initialize VIX client.

        Args:
            fred_api_key: Optional FRED API key for alternative data source
            cache_dir: Directory for caching data
            use_cache: Whether to use cached data
        """
        self.fred_api_key = fred_api_key
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache

        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_vix_history(
        self,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
        source: Literal["cboe", "fred"] = "cboe",
    ) -> pd.DataFrame:
        """Get historical VIX data.

        Args:
            start_date: Start date for data
            end_date: End date for data
            source: Data source ("cboe" or "fred")

        Returns:
            DataFrame with columns: date, open, high, low, close
        """
        if source == "cboe":
            df = self._fetch_cboe_data()
        elif source == "fred":
            df = self._fetch_fred_data()
        else:
            raise ValueError(f"Unknown source: {source}")

        # Filter by date
        if start_date:
            start_date = pd.to_datetime(start_date)
            df = df[df.index >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date)
            df = df[df.index <= end_date]

        return df

    def _fetch_cboe_data(self) -> pd.DataFrame:
        """Fetch VIX data from CBOE GitHub mirror.

        Returns:
            DataFrame with VIX OHLC data
        """
        cache_file = self.cache_dir / "vix_cboe.parquet"

        # Check cache (1 day TTL)
        if self.use_cache and cache_file.exists():
            import time
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age < 86400:  # 24 hours
                logger.info("Loading VIX from cache")
                return pd.read_parquet(cache_file)

        logger.info(f"Downloading VIX from CBOE GitHub: {self.CBOE_GITHUB_URL}")

        try:
            response = requests.get(self.CBOE_GITHUB_URL, timeout=30)
            response.raise_for_status()

            df = pd.read_csv(io.StringIO(response.text))

            # Standardize column names
            df.columns = df.columns.str.lower().str.strip()

            # Parse date
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

            # Rename columns if needed
            rename_map = {
                "vix open": "open",
                "vix high": "high",
                "vix low": "low",
                "vix close": "close",
            }
            df = df.rename(columns=rename_map)

            # Keep only OHLC
            cols = ["open", "high", "low", "close"]
            available = [c for c in cols if c in df.columns]
            df = df[available]

            # Sort by date
            df = df.sort_index()

            # Cache
            if self.use_cache:
                df.to_parquet(cache_file)

            logger.info(f"VIX data: {len(df)} rows, {df.index[0]} to {df.index[-1]}")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch CBOE VIX data: {e}")

            # Fall back to cache if available
            if cache_file.exists():
                logger.warning("Using stale cache")
                return pd.read_parquet(cache_file)

            raise

    def _fetch_fred_data(self) -> pd.DataFrame:
        """Fetch VIX data from FRED.

        Returns:
            DataFrame with VIX close data
        """
        if not self.fred_api_key:
            raise ValueError("FRED API key required for FRED source")

        cache_file = self.cache_dir / "vix_fred.parquet"

        # Check cache
        if self.use_cache and cache_file.exists():
            import time
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age < 86400:
                return pd.read_parquet(cache_file)

        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": self.FRED_VIX_SERIES,
            "api_key": self.fred_api_key,
            "file_type": "json",
        }

        logger.info(f"Fetching VIX from FRED: {self.FRED_VIX_SERIES}")

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        if "observations" not in data:
            raise ValueError("No VIX data from FRED")

        rows = []
        for obs in data["observations"]:
            val = obs["value"]
            if val != ".":
                rows.append({
                    "date": obs["date"],
                    "close": float(val),
                })

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        # Cache
        if self.use_cache:
            df.to_parquet(cache_file)

        return df

    def get_vix_term_structure(
        self,
        date: date | str | None = None,
    ) -> pd.DataFrame:
        """Get VIX term structure (futures curve).

        Note: This requires additional data sources (e.g., CBOE futures data).
        Currently returns VIX spot only.

        Args:
            date: Date for term structure

        Returns:
            DataFrame with VIX term structure
        """
        # TODO: Implement VIX futures term structure
        # This would require VIX futures data from CBOE or another source
        logger.warning("VIX term structure not yet implemented - returning spot only")

        vix = self.get_vix_history()

        if date:
            date = pd.to_datetime(date)
            if date in vix.index:
                return vix.loc[[date]]

        return vix.tail(1)

    def calculate_vix_metrics(
        self,
        lookback_days: int = 252,
    ) -> dict:
        """Calculate VIX-based market metrics.

        Args:
            lookback_days: Number of days to analyze

        Returns:
            Dictionary with VIX metrics
        """
        vix = self.get_vix_history()

        if len(vix) < lookback_days:
            lookback_days = len(vix)

        recent = vix.tail(lookback_days)["close"]

        current = recent.iloc[-1]
        mean = recent.mean()
        std = recent.std()
        percentile = (recent < current).sum() / len(recent) * 100

        # VIX regime classification
        if current < 15:
            regime = "low_volatility"
        elif current < 20:
            regime = "normal"
        elif current < 30:
            regime = "elevated"
        else:
            regime = "high_volatility"

        return {
            "current": current,
            "mean": mean,
            "std": std,
            "percentile": percentile,
            "min": recent.min(),
            "max": recent.max(),
            "regime": regime,
            "z_score": (current - mean) / std if std > 0 else 0,
        }

    def is_fear_elevated(
        self,
        threshold: float = 25.0,
    ) -> bool:
        """Check if VIX indicates elevated market fear.

        Args:
            threshold: VIX threshold for elevated fear

        Returns:
            True if VIX is above threshold
        """
        vix = self.get_vix_history()
        current = vix["close"].iloc[-1]

        return current > threshold
