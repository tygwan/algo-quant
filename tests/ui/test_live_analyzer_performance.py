"""Tests for live analyzer performance helpers and watchlist presets."""

from __future__ import annotations

import pandas as pd

from src.ui.layouts.live_analyzer import (
    _extract_close_series,
    _dynamic_refresh_interval_ms,
    _fetch_historical_prices,
    _historical_request_timeout_sec,
    _limit_realtime_tickers,
    _load_live_watchlists,
)


def test_load_live_watchlists_includes_builtin_presets() -> None:
    watchlists = _load_live_watchlists()

    assert "us_mega" in watchlists
    assert "crypto_major" in watchlists
    assert len(watchlists["us_mega"]) >= 5


def test_limit_realtime_tickers_dedupes_and_caps_size() -> None:
    raw = [f"SYM{i}" for i in range(60)] + ["SYM1", "SYM2"]

    limited = _limit_realtime_tickers(raw, max_tickers=24)

    assert len(limited) == 24
    assert limited[0] == "SYM0"
    assert limited[1] == "SYM1"


def test_dynamic_refresh_interval_ms_adapts_to_mode_and_symbol_count() -> None:
    fast = _dynamic_refresh_interval_ms(
        base_ms=3000,
        analysis_type="board",
        resolution="tick",
        ticker_count=8,
    )
    throttled = _dynamic_refresh_interval_ms(
        base_ms=3000,
        analysis_type="board",
        resolution="tick",
        ticker_count=48,
    )

    assert fast <= 1000
    assert throttled >= 2000
    assert throttled > fast


def test_historical_request_timeout_sec_clamps_env(monkeypatch) -> None:
    monkeypatch.setenv("AQ_YF_TIMEOUT_SEC", "0.2")
    assert _historical_request_timeout_sec() == 2.0

    monkeypatch.setenv("AQ_YF_TIMEOUT_SEC", "120")
    assert _historical_request_timeout_sec() == 30.0

    monkeypatch.setenv("AQ_YF_TIMEOUT_SEC", "7")
    assert _historical_request_timeout_sec() == 7.0


def test_fetch_historical_prices_uses_timeout_and_non_threaded_download(monkeypatch) -> None:
    import yfinance as yf

    calls: list[dict[str, object]] = []

    def fake_download(
        tickers,
        period=None,
        interval=None,
        group_by=None,
        progress=None,
        auto_adjust=None,
        threads=None,
        timeout=None,
    ):
        del period, interval, group_by, progress, auto_adjust
        calls.append({"threads": threads, "timeout": timeout})
        idx = pd.date_range("2025-01-01", periods=3, freq="D")
        if isinstance(tickers, list):
            cols = pd.MultiIndex.from_product([["Close"], tickers])
            data = [[100 + i + j for j in range(len(tickers))] for i in range(3)]
            return pd.DataFrame(data, index=idx, columns=cols)
        return pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=idx)

    monkeypatch.setattr(yf, "download", fake_download)
    monkeypatch.setenv("AQ_YF_TIMEOUT_SEC", "6")

    prices_df, failed = _fetch_historical_prices(["AAPL", "MSFT"], period="1mo", interval="1d")

    assert failed == []
    assert list(prices_df.columns) == ["AAPL", "MSFT"]
    assert calls
    assert calls[0]["threads"] is False
    assert calls[0]["timeout"] == 6.0


def test_extract_close_series_handles_multiindex_close_dataframe() -> None:
    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    cols = pd.MultiIndex.from_tuples(
        [
            ("Close", "AAPL"),
            ("Close", "MSFT"),
            ("Volume", "AAPL"),
        ]
    )
    df = pd.DataFrame(
        [
            [100.0, 200.0, 1000.0],
            [101.0, 201.0, 1100.0],
            [102.0, 202.0, 1200.0],
        ],
        index=idx,
        columns=cols,
    )

    series = _extract_close_series(df, ticker="AAPL")

    assert series is not None
    assert len(series) == 3
    assert float(series.iloc[-1]) == 102.0


def test_fetch_historical_prices_handles_single_ticker_multiindex_download(monkeypatch) -> None:
    import yfinance as yf

    def fake_download(
        tickers,
        period=None,
        interval=None,
        group_by=None,
        progress=None,
        auto_adjust=None,
        threads=None,
        timeout=None,
    ):
        del period, interval, group_by, progress, auto_adjust, threads, timeout
        idx = pd.date_range("2025-01-01", periods=3, freq="D")
        if isinstance(tickers, list):
            # Force fallback path
            return pd.DataFrame()
        cols = pd.MultiIndex.from_tuples(
            [
                ("Close", tickers),
                ("Volume", tickers),
            ]
        )
        return pd.DataFrame(
            [
                [100.0, 1000.0],
                [101.0, 1100.0],
                [102.0, 1200.0],
            ],
            index=idx,
            columns=cols,
        )

    monkeypatch.setattr(yf, "download", fake_download)

    prices_df, failed = _fetch_historical_prices(["AAPL"], period="1mo", interval="1d")

    assert failed == []
    assert list(prices_df.columns) == ["AAPL"]
    assert len(prices_df) == 3
