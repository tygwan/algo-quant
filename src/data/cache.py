"""Local caching system for API data."""

import hashlib
import json
import logging
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

import pandas as pd

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheConfig:
    """Cache configuration with TTL settings."""
    
    # Default TTL values in seconds
    DAILY_PRICES = 86400       # 1 day
    INTRADAY_PRICES = 300      # 5 minutes
    FINANCIAL_STATEMENTS = 604800  # 1 week
    COMPANY_PROFILE = 604800   # 1 week
    MACRO_INDICATORS = 86400   # 1 day
    CRYPTO_PRICES = 60         # 1 minute
    
    def __init__(
        self,
        cache_dir: str = ".cache",
        default_ttl: int = 86400,
        enabled: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl
        self.enabled = enabled
        
        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)


class DataCache:
    """File-based cache for API responses.
    
    Supports caching of DataFrames, Series, dicts, and other picklable objects.
    Uses Parquet for DataFrames (efficient) and pickle for other objects.
    """

    def __init__(self, config: CacheConfig | None = None):
        """Initialize cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        self._stats = {"hits": 0, "misses": 0, "writes": 0}

    def _get_cache_path(self, key: str, ext: str = ".pkl") -> Path:
        """Get file path for cache key."""
        # Use hash for long keys
        if len(key) > 100:
            key = hashlib.md5(key.encode()).hexdigest()
        
        # Sanitize key for filesystem
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        return self.config.cache_dir / f"{safe_key}{ext}"

    def _is_expired(self, cache_path: Path, ttl: int) -> bool:
        """Check if cache file is expired."""
        if not cache_path.exists():
            return True
        
        mtime = cache_path.stat().st_mtime
        age = time.time() - mtime
        return age > ttl

    def get(
        self,
        key: str,
        ttl: int | None = None,
    ) -> Any | None:
        """Get value from cache.
        
        Args:
            key: Cache key
            ttl: Time-to-live in seconds
            
        Returns:
            Cached value or None if not found/expired
        """
        if not self.config.enabled:
            return None
        
        ttl = ttl or self.config.default_ttl
        
        # Try Parquet first (for DataFrames)
        parquet_path = self._get_cache_path(key, ".parquet")
        if parquet_path.exists() and not self._is_expired(parquet_path, ttl):
            try:
                self._stats["hits"] += 1
                logger.debug(f"Cache hit (parquet): {key}")
                return pd.read_parquet(parquet_path)
            except Exception as e:
                logger.warning(f"Failed to read parquet cache: {e}")
        
        # Try pickle
        pkl_path = self._get_cache_path(key, ".pkl")
        if pkl_path.exists() and not self._is_expired(pkl_path, ttl):
            try:
                with open(pkl_path, "rb") as f:
                    self._stats["hits"] += 1
                    logger.debug(f"Cache hit (pickle): {key}")
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to read pickle cache: {e}")
        
        self._stats["misses"] += 1
        logger.debug(f"Cache miss: {key}")
        return None

    def set(self, key: str, value: Any) -> None:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        if not self.config.enabled:
            return
        
        try:
            if isinstance(value, pd.DataFrame):
                path = self._get_cache_path(key, ".parquet")
                value.to_parquet(path)
            elif isinstance(value, pd.Series):
                # Convert Series to DataFrame for Parquet
                path = self._get_cache_path(key, ".parquet")
                value.to_frame().to_parquet(path)
            else:
                path = self._get_cache_path(key, ".pkl")
                with open(path, "wb") as f:
                    pickle.dump(value, f)
            
            self._stats["writes"] += 1
            logger.debug(f"Cache write: {key}")
            
        except Exception as e:
            logger.warning(f"Failed to write cache: {e}")

    def delete(self, key: str) -> bool:
        """Delete a cache entry.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        deleted = False
        
        for ext in [".parquet", ".pkl"]:
            path = self._get_cache_path(key, ext)
            if path.exists():
                path.unlink()
                deleted = True
        
        return deleted

    def clear(self, older_than: int | None = None) -> int:
        """Clear cache entries.
        
        Args:
            older_than: Only clear entries older than this (seconds)
            
        Returns:
            Number of entries cleared
        """
        count = 0
        now = time.time()
        
        for path in self.config.cache_dir.glob("*"):
            if path.is_file():
                if older_than is None or (now - path.stat().st_mtime) > older_than:
                    path.unlink()
                    count += 1
        
        logger.info(f"Cleared {count} cache entries")
        return count

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0
        
        return {
            **self._stats,
            "hit_rate": round(hit_rate, 3),
            "total_requests": total,
        }

    def cached(
        self,
        ttl: int | None = None,
        key_prefix: str = "",
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator for caching function results.
        
        Args:
            ttl: Time-to-live in seconds
            key_prefix: Prefix for cache key
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            def wrapper(*args: Any, **kwargs: Any) -> T:
                # Generate cache key from function name and arguments
                key_parts = [key_prefix, func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                key = "_".join(key_parts)
                
                # Try to get from cache
                cached_value = self.get(key, ttl=ttl)
                if cached_value is not None:
                    return cached_value
                
                # Call function and cache result
                result = func(*args, **kwargs)
                self.set(key, result)
                return result
            
            return wrapper
        return decorator

    def get_size(self) -> dict[str, Any]:
        """Get cache size information."""
        total_size = 0
        file_count = 0
        
        for path in self.config.cache_dir.glob("*"):
            if path.is_file():
                total_size += path.stat().st_size
                file_count += 1
        
        return {
            "total_bytes": total_size,
            "total_mb": round(total_size / (1024 * 1024), 2),
            "file_count": file_count,
        }


class CachedClient:
    """Mixin for adding caching to API clients."""

    def __init__(self, cache: DataCache | None = None):
        self._cache = cache or DataCache()

    def _cache_key(self, method: str, *args: Any, **kwargs: Any) -> str:
        """Generate cache key for API call."""
        parts = [self.__class__.__name__, method]
        parts.extend(str(arg) for arg in args)
        parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return "_".join(parts)

    def _get_cached(
        self,
        key: str,
        ttl: int = CacheConfig.DAILY_PRICES,
    ) -> Any | None:
        """Get cached value."""
        return self._cache.get(key, ttl=ttl)

    def _set_cached(self, key: str, value: Any) -> None:
        """Set cached value."""
        self._cache.set(key, value)


def create_cache(
    cache_dir: str = ".cache",
    enabled: bool = True,
    default_ttl: int = 86400,
) -> DataCache:
    """Create a configured cache instance.
    
    Args:
        cache_dir: Directory for cache files
        enabled: Whether caching is enabled
        default_ttl: Default time-to-live in seconds
        
    Returns:
        Configured DataCache instance
    """
    config = CacheConfig(
        cache_dir=cache_dir,
        enabled=enabled,
        default_ttl=default_ttl,
    )
    return DataCache(config)
