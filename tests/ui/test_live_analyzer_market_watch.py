"""Tests for realtime market-watch ranking helpers."""

from __future__ import annotations

import pandas as pd

from src.ui.layouts.live_analyzer import _build_market_watch_rows


def test_build_market_watch_rows_ranks_by_score() -> None:
    idx = pd.date_range("2026-01-01", periods=6, freq="s", tz="UTC")
    prices = pd.DataFrame(
        {
            "AAA": [100.0, 100.5, 101.0, 101.5, 102.0, 102.5],
            "BBB": [100.0, 99.8, 99.6, 99.5, 99.2, 99.0],
            "CCC": [100.0, 100.0, 100.1, 100.0, 100.1, 100.2],
        },
        index=idx,
    )

    snapshot = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "price": [102.5, 99.0, 100.2],
            "volume": [12000, 8500, 4300],
            "source": ["finnhub", "finnhub", "finnhub"],
        }
    )

    rows = _build_market_watch_rows(prices, snapshot_df=snapshot, lookback_points=6)

    assert len(rows) == 3
    assert rows[0]["symbol"] == "AAA"
    assert rows[0]["rank"] == 1
    assert rows[-1]["symbol"] == "BBB"
    assert rows[0]["volume"] == 12000


def test_build_market_watch_rows_handles_empty_frame() -> None:
    rows = _build_market_watch_rows(pd.DataFrame())
    assert rows == []
