"""Tests for execution engine workflows."""

import asyncio

from src.execution.broker import OrderSide, OrderStatus, PaperBroker
from src.execution.executor import ExecutionConfig, ExecutionEngine, ExecutionMode


def test_execution_engine_paper_order_flow():
    """Engine should connect, execute a paper trade, and stop cleanly."""

    async def scenario():
        broker = PaperBroker(initial_cash=1000.0, commission_rate=0.0)
        config = ExecutionConfig(
            mode=ExecutionMode.PAPER,
            symbols=[],
            enable_auto_rebalance=False,
        )
        engine = ExecutionEngine(config=config, broker=broker)

        await engine.start()
        result = await engine.submit_order("AAPL", OrderSide.BUY, quantity=2)
        assert result.status == OrderStatus.FILLED

        trades = engine.get_trade_history()
        assert len(trades) == 1
        assert trades.iloc[0]["symbol"] == "AAPL"

        await engine.stop()
        assert engine.get_state().is_connected is False

    asyncio.run(scenario())


def test_execution_engine_weight_validation():
    """Target weights should reject sums above 100%."""
    engine = ExecutionEngine(
        config=ExecutionConfig(mode=ExecutionMode.PAPER, symbols=[]),
        broker=PaperBroker(),
    )

    try:
        engine.set_target_weights({"AAPL": 0.8, "MSFT": 0.3})
    except ValueError as exc:
        assert "must be <= 1.0" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid target weights")
