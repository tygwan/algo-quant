"""Tests for caching system."""

import pytest
import tempfile
import time
from pathlib import Path

import pandas as pd

from src.data.cache import DataCache, CacheConfig, create_cache


class TestDataCache:
    """Test cases for DataCache."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create cache instance."""
        config = CacheConfig(cache_dir=temp_cache_dir, enabled=True)
        return DataCache(config)

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "value": [1, 2, 3, 4, 5],
        })

    def test_cache_disabled(self, temp_cache_dir):
        """Test cache when disabled."""
        config = CacheConfig(cache_dir=temp_cache_dir, enabled=False)
        cache = DataCache(config)
        
        cache.set("test_key", {"data": 123})
        result = cache.get("test_key")
        
        assert result is None

    def test_set_get_dict(self, cache):
        """Test caching dictionary."""
        data = {"key": "value", "number": 42}
        cache.set("test_dict", data)
        
        result = cache.get("test_dict")
        
        assert result == data

    def test_set_get_dataframe(self, cache, sample_df):
        """Test caching DataFrame."""
        cache.set("test_df", sample_df)
        
        result = cache.get("test_df")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        pd.testing.assert_frame_equal(result, sample_df)

    def test_set_get_series(self, cache):
        """Test caching Series."""
        series = pd.Series([1, 2, 3, 4, 5], name="test")
        cache.set("test_series", series)
        
        result = cache.get("test_series")
        
        assert isinstance(result, pd.DataFrame)  # Series stored as DataFrame
        assert len(result) == 5

    def test_cache_expiry(self, cache):
        """Test cache expiration."""
        cache.set("expiry_test", {"data": "test"})
        
        # Should exist with long TTL
        result = cache.get("expiry_test", ttl=3600)
        assert result is not None
        
        # Should be "expired" with 0 TTL
        time.sleep(0.1)
        result = cache.get("expiry_test", ttl=0)
        assert result is None

    def test_cache_miss(self, cache):
        """Test cache miss."""
        result = cache.get("nonexistent_key")
        assert result is None

    def test_delete(self, cache):
        """Test cache deletion."""
        cache.set("delete_test", {"data": "test"})
        
        assert cache.get("delete_test") is not None
        
        deleted = cache.delete("delete_test")
        
        assert deleted is True
        assert cache.get("delete_test") is None

    def test_clear(self, cache):
        """Test clearing cache."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        count = cache.clear()
        
        assert count >= 2
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_get_stats(self, cache):
        """Test cache statistics."""
        cache.set("stats_test", "data")
        cache.get("stats_test")  # Hit
        cache.get("nonexistent")  # Miss
        
        stats = cache.get_stats()
        
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
        assert stats["writes"] >= 1
        assert "hit_rate" in stats

    def test_get_size(self, cache):
        """Test cache size calculation."""
        cache.set("size_test", {"large": "data" * 1000})
        
        size_info = cache.get_size()
        
        assert size_info["total_bytes"] > 0
        assert size_info["file_count"] >= 1

    def test_cached_decorator(self, cache):
        """Test caching decorator."""
        call_count = 0
        
        @cache.cached(ttl=3600)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        result1 = expensive_function(5)
        result2 = expensive_function(5)
        
        assert result1 == 10
        assert result2 == 10
        assert call_count == 1  # Only called once due to cache

    def test_create_cache(self, temp_cache_dir):
        """Test create_cache factory function."""
        cache = create_cache(
            cache_dir=temp_cache_dir,
            enabled=True,
            default_ttl=3600,
        )
        
        assert cache.config.enabled is True
        assert cache.config.default_ttl == 3600

    def test_long_key_handling(self, cache):
        """Test handling of very long cache keys."""
        long_key = "x" * 200  # Very long key
        cache.set(long_key, {"data": "test"})
        
        result = cache.get(long_key)
        
        assert result == {"data": "test"}
