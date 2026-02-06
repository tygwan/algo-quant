"""Tests for realtime symbol normalization."""

from __future__ import annotations

from src.ui.services.realtime_hub import _normalize_symbols


def test_normalize_symbols_compacts_crypto_pairs_and_preserves_stock_formats() -> None:
    normalized = _normalize_symbols([
        " btc/usdt ",
        "ETH-USDT",
        "sol_usdt",
        "AAPL",
        "BRK-B",
        "BTCUSDT",
    ])

    assert normalized == ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AAPL", "BRK-B"]
