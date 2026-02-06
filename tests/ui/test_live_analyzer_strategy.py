"""Tests for live analyzer paper strategy helper functions."""

from __future__ import annotations

import pandas as pd

from src.ui.layouts.live_analyzer import (
    _latest_prices_for_symbols,
    _select_strategy_targets,
)


def test_select_strategy_targets_momentum_prefers_winners() -> None:
    idx = pd.date_range("2025-01-01", periods=6, freq="D")
    prices = pd.DataFrame(
        {
            "AAA": [100, 102, 104, 106, 108, 110],
            "BBB": [100, 99, 98, 97, 96, 95],
            "CCC": [100, 100, 100.5, 101, 101.5, 102],
        },
        index=idx,
    )

    targets = _select_strategy_targets(prices, strategy="momentum", top_n=2)

    assert targets == ["AAA", "CCC"]


def test_select_strategy_targets_mean_reversion_prefers_losers() -> None:
    idx = pd.date_range("2025-01-01", periods=6, freq="D")
    prices = pd.DataFrame(
        {
            "AAA": [100, 101, 102, 103, 104, 105],
            "BBB": [100, 99, 98, 97, 96, 95],
            "CCC": [100, 100, 101, 102, 103, 104],
        },
        index=idx,
    )

    targets = _select_strategy_targets(prices, strategy="mean_reversion", top_n=2)

    assert targets[0] == "BBB"
    assert len(targets) == 2


def test_latest_prices_for_symbols_filters_invalid_data() -> None:
    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    prices = pd.DataFrame(
        {
            "AAA": [100.0, 101.0, 102.0],
            "BBB": [0.0, 0.0, 0.0],
            "CCC": [None, None, None],
        },
        index=idx,
    )

    latest = _latest_prices_for_symbols(prices, ["AAA", "BBB", "CCC", "DDD"])

    assert latest == {"AAA": 102.0}
