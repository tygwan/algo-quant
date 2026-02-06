"""Tests for realtime candle resampling helpers."""

from __future__ import annotations

import pandas as pd

from src.ui.layouts.live_analyzer import _resample_ticks_to_candles


def test_resample_ticks_to_candles_tick_mode() -> None:
    ticks = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00.000Z",
                    "2026-01-01T00:00:00.400Z",
                    "2026-01-01T00:00:00.900Z",
                ]
            ),
            "price": [100.0, 101.5, 99.7],
            "volume": [5, 3, 2],
        }
    )

    candles = _resample_ticks_to_candles(ticks, "tick")

    assert list(candles.columns) == ["open", "high", "low", "close", "volume"]
    assert len(candles) == 3
    assert float(candles.iloc[1]["open"]) == 101.5
    assert float(candles.iloc[2]["volume"]) == 2.0


def test_resample_ticks_to_candles_seconds_mode() -> None:
    ticks = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00.100Z",
                    "2026-01-01T00:00:00.700Z",
                    "2026-01-01T00:00:01.200Z",
                    "2026-01-01T00:00:01.900Z",
                ]
            ),
            "price": [100.0, 101.0, 99.0, 102.0],
            "volume": [1, 2, 3, 4],
        }
    )

    candles = _resample_ticks_to_candles(ticks, "1s")

    assert len(candles) == 2

    first = candles.iloc[0]
    second = candles.iloc[1]

    assert float(first["open"]) == 100.0
    assert float(first["high"]) == 101.0
    assert float(first["low"]) == 100.0
    assert float(first["close"]) == 101.0
    assert float(first["volume"]) == 3.0

    assert float(second["open"]) == 99.0
    assert float(second["high"]) == 102.0
    assert float(second["low"]) == 99.0
    assert float(second["close"]) == 102.0
    assert float(second["volume"]) == 7.0
