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
]
