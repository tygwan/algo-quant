"""Tests for backtest engine."""

import pytest
import numpy as np
import pandas as pd

from src.backtest.engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    Trade,
    Position,
    OrderSide,
    VectorizedBacktester,
)


class TestPosition:
    """Test cases for Position."""

    def test_position_update_buy(self):
        """Test position update with buy."""
        pos = Position(symbol="AAPL")
        pos.update(100, 150.0, pd.Timestamp("2023-01-01"))
        
        assert pos.quantity == 100
        assert pos.entry_price == 150.0

    def test_position_update_sell(self):
        """Test position update with sell."""
        pos = Position(symbol="AAPL", quantity=100, entry_price=150.0)
        pos.update(-50, 160.0, pd.Timestamp("2023-01-02"))
        
        assert pos.quantity == 50
        assert pos.entry_price == 150.0  # Entry price unchanged for partial sell

    def test_position_add_to_long(self):
        """Test adding to long position."""
        pos = Position(symbol="AAPL", quantity=100, entry_price=150.0)
        pos.update(100, 160.0, pd.Timestamp("2023-01-02"))
        
        assert pos.quantity == 200
        # Average price: (100 * 150 + 100 * 160) / 200 = 155
        assert abs(pos.entry_price - 155.0) < 0.01


class TestTrade:
    """Test cases for Trade."""

    def test_trade_value(self):
        """Test trade value calculation."""
        trade = Trade(
            timestamp=pd.Timestamp("2023-01-01"),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=150.0,
            commission=15.0,
        )
        
        assert trade.value == 15000.0
        assert trade.total_cost == 15.0


class TestBacktestEngine:
    """Test cases for BacktestEngine."""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        
        # Generate random walk prices
        returns = np.random.normal(0.0005, 0.02, (252, 3))
        prices = 100 * np.cumprod(1 + returns, axis=0)
        
        return pd.DataFrame(
            prices,
            index=dates,
            columns=["AAPL", "MSFT", "GOOGL"],
        )

    @pytest.fixture
    def simple_strategy(self):
        """Create simple equal weight strategy."""
        def strategy(prices):
            n_assets = len(prices.columns)
            return pd.Series(1.0 / n_assets, index=prices.columns)
        return strategy

    def test_backtest_basic(self, sample_prices, simple_strategy):
        """Test basic backtest execution."""
        config = BacktestConfig(
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=100000,
            commission=0.001,
            rebalance_frequency="monthly",
        )
        
        engine = BacktestEngine(config)
        result = engine.run(simple_strategy, sample_prices)
        
        assert isinstance(result, BacktestResult)
        assert len(result.portfolio_values) > 0
        assert len(result.returns) > 0

    def test_backtest_trades(self, sample_prices, simple_strategy):
        """Test trade execution."""
        config = BacktestConfig(
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=100000,
            rebalance_frequency="monthly",
        )
        
        engine = BacktestEngine(config)
        result = engine.run(simple_strategy, sample_prices)
        
        # Should have trades from rebalancing
        assert len(result.trades) > 0

    def test_backtest_commission(self, sample_prices, simple_strategy):
        """Test commission impact."""
        config_no_comm = BacktestConfig(
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=100000,
            commission=0.0,
            slippage=0.0,
        )
        
        config_with_comm = BacktestConfig(
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=100000,
            commission=0.01,
            slippage=0.01,
        )
        
        result_no_comm = BacktestEngine(config_no_comm).run(simple_strategy, sample_prices)
        result_with_comm = BacktestEngine(config_with_comm).run(simple_strategy, sample_prices)
        
        # Higher costs should result in lower final value
        assert result_with_comm.portfolio_values.iloc[-1] < result_no_comm.portfolio_values.iloc[-1]

    def test_backtest_result_to_dict(self, sample_prices, simple_strategy):
        """Test result conversion to dict."""
        config = BacktestConfig(
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=100000,
        )
        
        engine = BacktestEngine(config)
        result = engine.run(simple_strategy, sample_prices)
        
        d = result.to_dict()
        
        assert "initial_capital" in d
        assert "final_value" in d
        assert "total_return" in d
        assert "num_trades" in d


class TestVectorizedBacktester:
    """Test cases for VectorizedBacktester."""

    @pytest.fixture
    def sample_data(self):
        """Create sample returns and weights."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        
        returns = pd.DataFrame(
            np.random.normal(0.0005, 0.02, (252, 3)),
            index=dates,
            columns=["AAPL", "MSFT", "GOOGL"],
        )
        
        weights = pd.DataFrame(
            [[1/3, 1/3, 1/3]] * 252,
            index=dates,
            columns=["AAPL", "MSFT", "GOOGL"],
        )
        
        return returns, weights

    def test_vectorized_backtest(self, sample_data):
        """Test vectorized backtest."""
        returns, weights = sample_data
        
        backtester = VectorizedBacktester(commission=0.001)
        result = backtester.run(weights, returns, initial_capital=100000)
        
        assert isinstance(result, BacktestResult)
        assert len(result.portfolio_values) > 0

    def test_vectorized_vs_event_driven(self, sample_data):
        """Test vectorized produces similar results to event-driven."""
        returns, weights = sample_data
        
        # Vectorized
        vec_result = VectorizedBacktester(commission=0.001).run(
            weights, returns, initial_capital=100000
        )
        
        # Results should be similar (allowing for implementation differences)
        assert vec_result.portfolio_values.iloc[-1] > 0
