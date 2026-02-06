"""Tests for auto-rebalancing logic."""

from src.execution.rebalancer import (
    AutoRebalancer,
    Position,
    RebalanceConfig,
    RebalanceTrigger,
)


def test_threshold_trigger_detection():
    """Threshold trigger should detect drift above configured bound."""
    config = RebalanceConfig(trigger=RebalanceTrigger.THRESHOLD, drift_threshold=0.05)
    rebalancer = AutoRebalancer(config)

    needed, reason = rebalancer.check_rebalance_needed(
        current_weights={"AAPL": 0.70, "MSFT": 0.30},
        target_weights={"AAPL": 0.60, "MSFT": 0.40},
    )

    assert needed is True
    assert "threshold" in reason.lower()


def test_calculate_trades_sells_then_buys():
    """Trade optimization should place sells before buys."""
    rebalancer = AutoRebalancer(
        RebalanceConfig(min_trade_size=1.0, max_single_position=1.0)
    )

    positions = {
        "AAPL": Position("AAPL", quantity=70, avg_cost=100, current_price=100),
        "MSFT": Position("MSFT", quantity=30, avg_cost=100, current_price=100),
    }

    trades = rebalancer.calculate_trades(
        positions=positions,
        target_weights={"AAPL": 0.5, "MSFT": 0.5},
        portfolio_value=10000,
        prices={"AAPL": 100, "MSFT": 100},
    )

    assert len(trades) == 2
    assert trades[0].side == "sell"
    assert trades[1].side == "buy"


def test_execute_rebalance_dry_run_records_result():
    """Dry-run rebalance should return generated orders and success state."""
    rebalancer = AutoRebalancer(RebalanceConfig(min_trade_size=1.0))

    positions = {
        "AAPL": Position("AAPL", quantity=70, avg_cost=100, current_price=100),
        "MSFT": Position("MSFT", quantity=30, avg_cost=100, current_price=100),
    }

    result = rebalancer.execute_rebalance(
        positions=positions,
        target_weights={"AAPL": 0.5, "MSFT": 0.5},
        portfolio_value=10000,
        prices={"AAPL": 100, "MSFT": 100},
        execute_fn=None,
    )

    assert result.success is True
    assert len(result.orders) == 2
    assert "dry run" in result.message.lower()
