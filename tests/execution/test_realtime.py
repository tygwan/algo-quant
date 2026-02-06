"""Tests for realtime stream components."""

import asyncio

from src.execution.realtime import (
    BinanceStream,
    FinnhubStockStream,
    StreamConfig,
    StreamType,
)


def test_stream_name_mapping_for_stream_types():
    """Each stream type should map to expected Binance suffix."""
    price_stream = BinanceStream(
        StreamConfig(symbols=["BTCUSDT"], stream_type=StreamType.PRICE)
    )
    trade_stream = BinanceStream(
        StreamConfig(symbols=["BTCUSDT"], stream_type=StreamType.TRADE)
    )

    assert price_stream._build_stream_names() == ["btcusdt@ticker"]
    assert trade_stream._build_stream_names() == ["btcusdt@trade"]


def test_mock_data_generation_and_dataframe_conversion():
    """Mock generation should populate buffer and convert to DataFrame."""

    async def scenario():
        stream = BinanceStream(
            StreamConfig(symbols=["BTCUSDT"], stream_type=StreamType.TRADE)
        )
        await stream._generate_mock_data()

        latest = stream.get_latest("BTCUSDT")
        assert latest is not None
        assert latest.price > 0

        df = stream.to_dataframe("BTCUSDT")
        assert not df.empty
        assert {"symbol", "price", "volume", "timestamp"}.issubset(df.columns)

    asyncio.run(scenario())


def test_finnhub_trade_message_parsing():
    """Finnhub trade payload should update per-symbol latest value."""

    async def scenario():
        stream = FinnhubStockStream(
            StreamConfig(symbols=["AAPL"], stream_type=StreamType.TRADE),
            api_key="dummy",
        )
        message = {
            "type": "trade",
            "data": [
                {"s": "AAPL", "p": 191.25, "v": 100.0, "t": 1738700000000},
                {"s": "MSFT", "p": 413.10, "v": 200.0, "t": 1738700001000},
            ],
        }
        await stream._handle_message(message)

        aapl = stream.get_latest("AAPL")
        msft = stream.get_latest("MSFT")
        assert aapl is not None
        assert msft is not None
        assert aapl.price == 191.25
        assert msft.price == 413.10

    asyncio.run(scenario())
