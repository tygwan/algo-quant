"""Tests for execution broker integrations."""

import asyncio

import pandas as pd
import pytest

from src.execution.broker import (
    BinanceBroker,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    PaperBroker,
)


def test_order_request_validation():
    """Validate core order constraints."""
    bad_qty = OrderRequest(symbol="AAPL", side=OrderSide.BUY, quantity=0)
    assert bad_qty.validate()[0] is False

    missing_limit = OrderRequest(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=1,
        order_type=OrderType.LIMIT,
    )
    assert missing_limit.validate()[0] is False

    good_market = OrderRequest(symbol="AAPL", side=OrderSide.BUY, quantity=1)
    assert good_market.validate() == (True, "Valid")


def test_paper_broker_buy_sell_cycle():
    """Paper broker should fill basic buy/sell flow."""

    async def scenario():
        broker = PaperBroker(initial_cash=1000.0, commission_rate=0.0)
        await broker.connect()

        broker.update_price("AAPL", 10.0)

        buy = await broker.submit_order(
            OrderRequest(symbol="AAPL", side=OrderSide.BUY, quantity=50)
        )
        assert buy.status == OrderStatus.FILLED

        positions = await broker.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"
        assert positions[0].quantity == 50

        sell = await broker.submit_order(
            OrderRequest(symbol="AAPL", side=OrderSide.SELL, quantity=10)
        )
        assert sell.status == OrderStatus.FILLED

        positions = await broker.get_positions()
        assert positions[0].quantity == 40

        account = await broker.get_account()
        assert account.cash == pytest.approx(599.7, abs=1e-6)

        await broker.disconnect()

    asyncio.run(scenario())


def test_paper_broker_rejects_insufficient_funds_and_positions():
    """Paper broker should reject invalid account-level constraints."""

    async def scenario():
        broker = PaperBroker(initial_cash=100.0, commission_rate=0.0)
        await broker.connect()
        broker.update_price("AAPL", 10.0)

        too_large_buy = await broker.submit_order(
            OrderRequest(symbol="AAPL", side=OrderSide.BUY, quantity=20)
        )
        assert too_large_buy.status == OrderStatus.REJECTED

        invalid_sell = await broker.submit_order(
            OrderRequest(symbol="AAPL", side=OrderSide.SELL, quantity=1)
        )
        assert invalid_sell.status == OrderStatus.REJECTED

    asyncio.run(scenario())


def test_binance_broker_adapter_with_mock_client(monkeypatch):
    """BinanceBroker should adapt BinanceClient sync API into async interface."""

    class DummyBinanceClient:
        def __init__(self, api_key, api_secret, testnet=True):
            self.api_key = api_key
            self.api_secret = api_secret
            self.testnet = testnet

        def get_balance(self):
            return pd.DataFrame(
                [
                    {"asset": "USDT", "free": 1000.0, "locked": 50.0, "total": 1050.0},
                    {"asset": "BTC", "free": 0.01, "locked": 0.0, "total": 0.01},
                ]
            )

        def create_order(self, symbol, side, quantity, price=None, order_type="MARKET"):
            return {
                "order_id": 123,
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "quantity": quantity,
                "price": price or 0,
                "status": "FILLED",
            }

        def get_open_orders(self, symbol=None):
            return pd.DataFrame(
                [
                    {
                        "order_id": 456,
                        "symbol": "BTCUSDT",
                        "side": "BUY",
                        "type": "LIMIT",
                        "quantity": 0.1,
                        "price": 40000,
                        "filled": 0.0,
                        "status": "NEW",
                        "time": pd.Timestamp("2026-02-05"),
                    }
                ]
            )

        def cancel_order(self, symbol, order_id):
            return {"symbol": symbol, "orderId": order_id, "status": "CANCELED"}

    monkeypatch.setattr("src.data.binance_client.BinanceClient", DummyBinanceClient)

    async def scenario():
        broker = BinanceBroker(api_key="k", api_secret="s", testnet=True)
        assert await broker.connect() is True

        account = await broker.get_account()
        assert account.cash == 1000.0
        assert account.portfolio_value == 1050.0

        positions = await broker.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "BTC"

        result = await broker.submit_order(
            OrderRequest(symbol="BTCUSDT", side=OrderSide.BUY, quantity=0.01)
        )
        assert result.status == OrderStatus.FILLED

        open_orders = await broker.get_open_orders()
        assert len(open_orders) == 1
        assert open_orders[0].status == OrderStatus.SUBMITTED

        assert await broker.cancel_order("456") is True
        await broker.disconnect()

    asyncio.run(scenario())
