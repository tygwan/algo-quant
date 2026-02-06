#!/usr/bin/env python3
"""Offline paper-trading demo for algo-quant.

Runs a simple moving-average crossover strategy against synthetic prices,
executes trades via ExecutionEngine in PAPER mode, and prints a summary.
"""

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from src.execution.broker import OrderSide, PaperBroker
from src.execution.executor import ExecutionConfig, ExecutionEngine, ExecutionMode


@dataclass
class DemoConfig:
    symbol: str = "AAPL"
    steps: int = 120
    initial_price: float = 100.0
    drift: float = 0.0005
    volatility: float = 0.01
    ma_window: int = 20
    quantity: float = 1.0
    seed: int = 42


class SmaPaperDemo:
    """SMA crossover demo runner using paper execution."""

    def __init__(self, config: DemoConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.broker = PaperBroker(initial_cash=100000.0, commission_rate=0.001)
        self.engine = ExecutionEngine(
            config=ExecutionConfig(
                mode=ExecutionMode.PAPER,
                symbols=[],
                enable_auto_rebalance=False,
            ),
            broker=self.broker,
        )
        self.prices: list[float] = []
        self.holding = False

    def _next_price(self, last_price: float) -> float:
        """Generate next synthetic price using GBM-like step."""
        shock = self.rng.normal(self.config.drift, self.config.volatility)
        return max(last_price * (1 + shock), 0.01)

    async def run(self) -> None:
        """Run the strategy simulation."""
        await self.engine.start()

        price = self.config.initial_price
        for _ in range(self.config.steps):
            price = self._next_price(price)
            self.prices.append(price)
            self.broker.update_price(self.config.symbol, price)

            if len(self.prices) < self.config.ma_window:
                continue

            window = self.prices[-self.config.ma_window :]
            ma = float(np.mean(window))

            if not self.holding and price > ma:
                await self.engine.submit_order(
                    self.config.symbol,
                    OrderSide.BUY,
                    quantity=self.config.quantity,
                )
                self.holding = True
            elif self.holding and price < ma:
                await self.engine.submit_order(
                    self.config.symbol,
                    OrderSide.SELL,
                    quantity=self.config.quantity,
                )
                self.holding = False

            await self.engine._update_account_state()  # keep engine state fresh for summary

        # Flat position at the end for cleaner demo summary
        if self.holding:
            await self.engine.submit_order(
                self.config.symbol,
                OrderSide.SELL,
                quantity=self.config.quantity,
            )
            await self.engine._update_account_state()
            self.holding = False

        summary = self.engine.get_performance_summary()
        trades = self.engine.get_trade_history()

        await self.engine.stop()

        self._print_report(summary, trades)

    def _print_report(self, summary: dict, trades: pd.DataFrame) -> None:
        """Print demo result report."""
        print("\n" + "=" * 70)
        print("algo-quant PAPER TRADING DEMO RESULT")
        print("=" * 70)
        print(f"Timestamp      : {datetime.now().isoformat(timespec='seconds')}")
        print(f"Symbol         : {self.config.symbol}")
        print(f"Simulation     : {self.config.steps} steps, MA({self.config.ma_window})")
        print(f"Price Range    : {min(self.prices):.2f} ~ {max(self.prices):.2f}")
        print(f"Trades         : {len(trades)}")

        if summary:
            total_return = float(summary.get("total_return", 0.0)) * 100
            print(f"Portfolio Value: {summary.get('portfolio_value', 0.0):.2f}")
            print(f"Cash           : {summary.get('cash', 0.0):.2f}")
            print(f"Total Return   : {total_return:.2f}%")
            print(f"Commission     : {summary.get('total_commission', 0.0):.2f}")
            print(f"Win Rate       : {float(summary.get('win_rate', 0.0)) * 100:.2f}%")

        if not trades.empty:
            print("\nRecent Trades")
            print("-" * 70)
            print(trades.tail(10).to_string(index=False))


def parse_args() -> DemoConfig:
    parser = argparse.ArgumentParser(description="Run offline paper-trading demo")
    parser.add_argument("--symbol", default="AAPL", help="Trading symbol")
    parser.add_argument("--steps", type=int, default=120, help="Number of simulation steps")
    parser.add_argument("--initial-price", type=float, default=100.0, help="Initial synthetic price")
    parser.add_argument("--ma-window", type=int, default=20, help="SMA window")
    parser.add_argument("--quantity", type=float, default=1.0, help="Order quantity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    return DemoConfig(
        symbol=args.symbol.upper(),
        steps=args.steps,
        initial_price=args.initial_price,
        ma_window=args.ma_window,
        quantity=args.quantity,
        seed=args.seed,
    )


def main() -> None:
    config = parse_args()
    asyncio.run(SmaPaperDemo(config).run())


if __name__ == "__main__":
    main()
