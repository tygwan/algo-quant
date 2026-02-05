"""Data module for API clients and data processing."""

from .base_client import (
    BaseClient,
    RateLimiter,
    APIError,
    RateLimitError,
    AuthenticationError,
)
from .fmp_client import FMPClient
from .fred_client import FREDClient, FREDIndicators
from .kis_client import KISClient
from .kiwoom_client import KiwoomClient
from .binance_client import BinanceClient
from .upbit_client import UpbitClient
from .preprocessor import DataPreprocessor
from .cache import DataCache, CacheConfig, CachedClient, create_cache
from .collector import DataCollector, CollectionConfig, SP500_TOP_100
from .vix_client import VIXClient
from .finnhub_client import FinnhubClient
from .yfinance_client import YFinanceClient

__all__ = [
    # Base
    "BaseClient",
    "RateLimiter",
    "APIError",
    "RateLimitError",
    "AuthenticationError",
    # US Stocks
    "FMPClient",
    # Macro
    "FREDClient",
    "FREDIndicators",
    # Korean Stocks
    "KISClient",
    "KiwoomClient",
    # Crypto
    "BinanceClient",
    "UpbitClient",
    # Processing
    "DataPreprocessor",
    # Cache
    "DataCache",
    "CacheConfig",
    "CachedClient",
    "create_cache",
    # Collector
    "DataCollector",
    "CollectionConfig",
    "SP500_TOP_100",
    # Additional Sources
    "VIXClient",
    "FinnhubClient",
    "YFinanceClient",
]
