"""Broker integration layer for order execution."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import uuid

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(Enum):
    """Time in force options."""
    DAY = "day"           # Good for day
    GTC = "gtc"           # Good till cancelled
    IOC = "ioc"           # Immediate or cancel
    FOK = "fok"           # Fill or kill
    GTD = "gtd"           # Good till date


@dataclass
class OrderRequest:
    """Order request to submit to broker."""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def validate(self) -> tuple[bool, str]:
        """Validate order request."""
        if self.quantity <= 0:
            return False, "Quantity must be positive"

        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if self.limit_price is None or self.limit_price <= 0:
                return False, "Limit price required for limit orders"

        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if self.stop_price is None or self.stop_price <= 0:
                return False, "Stop price required for stop orders"

        return True, "Valid"


@dataclass
class OrderResult:
    """Result of order execution."""
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    status: OrderStatus
    quantity: float
    filled_quantity: float = 0.0
    average_price: float = 0.0
    commission: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""

    @property
    def is_complete(self) -> bool:
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        ]

    @property
    def fill_rate(self) -> float:
        if self.quantity == 0:
            return 0
        return self.filled_quantity / self.quantity

    @property
    def total_value(self) -> float:
        return self.filled_quantity * self.average_price

    @property
    def total_cost(self) -> float:
        return self.total_value + self.commission


@dataclass
class AccountInfo:
    """Broker account information."""
    account_id: str
    cash: float
    buying_power: float
    portfolio_value: float
    equity: float
    margin_used: float = 0.0
    day_trades_remaining: int = 3
    pattern_day_trader: bool = False


@dataclass
class PositionInfo:
    """Position information from broker."""
    symbol: str
    quantity: float
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    side: str = "long"


class BrokerInterface(ABC):
    """Abstract interface for broker integrations."""

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from broker."""
        pass

    @abstractmethod
    async def get_account(self) -> AccountInfo:
        """Get account information."""
        pass

    @abstractmethod
    async def get_positions(self) -> list[PositionInfo]:
        """Get current positions."""
        pass

    @abstractmethod
    async def submit_order(self, order: OrderRequest) -> OrderResult:
        """Submit an order."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass

    @abstractmethod
    async def get_order(self, order_id: str) -> Optional[OrderResult]:
        """Get order status."""
        pass

    @abstractmethod
    async def get_open_orders(self) -> list[OrderResult]:
        """Get all open orders."""
        pass


class PaperBroker(BrokerInterface):
    """Paper trading broker for simulation."""

    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission_rate: float = 0.001,
    ):
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate

        self._cash = initial_cash
        self._positions: dict[str, PositionInfo] = {}
        self._orders: dict[str, OrderResult] = {}
        self._prices: dict[str, float] = {}
        self._connected = False

    async def connect(self) -> bool:
        """Connect to paper broker."""
        self._connected = True
        logger.info("Connected to paper broker")
        return True

    async def disconnect(self) -> None:
        """Disconnect from paper broker."""
        self._connected = False
        logger.info("Disconnected from paper broker")

    async def get_account(self) -> AccountInfo:
        """Get paper account information."""
        portfolio_value = self._cash + sum(
            p.market_value for p in self._positions.values()
        )

        return AccountInfo(
            account_id="paper_account",
            cash=self._cash,
            buying_power=self._cash,
            portfolio_value=portfolio_value,
            equity=portfolio_value,
        )

    async def get_positions(self) -> list[PositionInfo]:
        """Get current positions."""
        # Update market values
        for symbol, pos in self._positions.items():
            if symbol in self._prices:
                price = self._prices[symbol]
                pos.market_value = pos.quantity * price
                pos.unrealized_pnl = (price - pos.avg_cost) * pos.quantity
                pos.unrealized_pnl_pct = (price - pos.avg_cost) / pos.avg_cost if pos.avg_cost > 0 else 0

        return list(self._positions.values())

    async def submit_order(self, order: OrderRequest) -> OrderResult:
        """Submit a paper order."""
        # Validate order
        valid, msg = order.validate()
        if not valid:
            return OrderResult(
                order_id=str(uuid.uuid4()),
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                side=order.side,
                status=OrderStatus.REJECTED,
                quantity=order.quantity,
                message=msg,
            )

        # Get price (use limit price or simulated market price)
        if order.order_type == OrderType.MARKET:
            price = self._prices.get(order.symbol, 100.0)
            # Add slippage for market orders
            slippage = 0.0005
            if order.side == OrderSide.BUY:
                price *= (1 + slippage)
            else:
                price *= (1 - slippage)
        else:
            price = order.limit_price or self._prices.get(order.symbol, 100.0)

        # Calculate values
        trade_value = order.quantity * price
        commission = trade_value * self.commission_rate

        # Check buying power for buys
        if order.side == OrderSide.BUY:
            if trade_value + commission > self._cash:
                return OrderResult(
                    order_id=str(uuid.uuid4()),
                    client_order_id=order.client_order_id,
                    symbol=order.symbol,
                    side=order.side,
                    status=OrderStatus.REJECTED,
                    quantity=order.quantity,
                    message="Insufficient buying power",
                )

        # Check position for sells
        if order.side == OrderSide.SELL:
            pos = self._positions.get(order.symbol)
            if not pos or pos.quantity < order.quantity:
                return OrderResult(
                    order_id=str(uuid.uuid4()),
                    client_order_id=order.client_order_id,
                    symbol=order.symbol,
                    side=order.side,
                    status=OrderStatus.REJECTED,
                    quantity=order.quantity,
                    message="Insufficient position",
                )

        # Execute order
        order_id = str(uuid.uuid4())

        if order.side == OrderSide.BUY:
            self._cash -= trade_value + commission
            self._update_position(order.symbol, order.quantity, price)
        else:
            self._cash += trade_value - commission
            self._update_position(order.symbol, -order.quantity, price)

        result = OrderResult(
            order_id=order_id,
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            status=OrderStatus.FILLED,
            quantity=order.quantity,
            filled_quantity=order.quantity,
            average_price=price,
            commission=commission,
            message="Order filled",
        )

        self._orders[order_id] = result
        logger.info(f"Paper order filled: {order.side.value} {order.quantity} {order.symbol} @ {price:.2f}")

        return result

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a paper order."""
        if order_id in self._orders:
            order = self._orders[order_id]
            if not order.is_complete:
                order.status = OrderStatus.CANCELLED
                return True
        return False

    async def get_order(self, order_id: str) -> Optional[OrderResult]:
        """Get order status."""
        return self._orders.get(order_id)

    async def get_open_orders(self) -> list[OrderResult]:
        """Get all open orders."""
        return [
            o for o in self._orders.values()
            if not o.is_complete
        ]

    def update_price(self, symbol: str, price: float) -> None:
        """Update price for a symbol."""
        self._prices[symbol] = price

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update multiple prices."""
        self._prices.update(prices)

    def _update_position(self, symbol: str, quantity_change: float, price: float) -> None:
        """Update position after trade."""
        if symbol in self._positions:
            pos = self._positions[symbol]
            if quantity_change > 0:
                # Buying - update average cost
                total_cost = pos.avg_cost * pos.quantity + price * quantity_change
                pos.quantity += quantity_change
                pos.avg_cost = total_cost / pos.quantity if pos.quantity > 0 else 0
            else:
                # Selling
                pos.quantity += quantity_change

            pos.market_value = pos.quantity * price

            # Remove position if quantity is 0
            if pos.quantity <= 0:
                del self._positions[symbol]
        else:
            # New position
            if quantity_change > 0:
                self._positions[symbol] = PositionInfo(
                    symbol=symbol,
                    quantity=quantity_change,
                    avg_cost=price,
                    market_value=quantity_change * price,
                    unrealized_pnl=0,
                    unrealized_pnl_pct=0,
                )


class BinanceBroker(BrokerInterface):
    """Binance broker integration."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self._connected = False
        self._client = None

    async def connect(self) -> bool:
        """Connect to Binance."""
        try:
            # Import Binance client if available
            from src.data.binance import BinanceClient

            self._client = BinanceClient(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet,
            )
            self._connected = True
            logger.info(f"Connected to Binance {'testnet' if self.testnet else 'mainnet'}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Binance."""
        self._connected = False
        self._client = None
        logger.info("Disconnected from Binance")

    async def get_account(self) -> AccountInfo:
        """Get Binance account information."""
        if not self._client:
            raise RuntimeError("Not connected to Binance")

        account = await self._client.get_account()

        # Calculate total values
        total_btc = float(account.get("totalWalletBalance", 0))

        return AccountInfo(
            account_id=str(account.get("accountId", "binance")),
            cash=total_btc,  # In BTC
            buying_power=float(account.get("availableBalance", 0)),
            portfolio_value=total_btc,
            equity=total_btc,
        )

    async def get_positions(self) -> list[PositionInfo]:
        """Get Binance positions."""
        if not self._client:
            raise RuntimeError("Not connected to Binance")

        balances = await self._client.get_balances()
        positions = []

        for balance in balances:
            quantity = float(balance.get("free", 0)) + float(balance.get("locked", 0))
            if quantity > 0:
                positions.append(PositionInfo(
                    symbol=balance["asset"],
                    quantity=quantity,
                    avg_cost=0,  # Not available from Binance
                    market_value=0,  # Would need price data
                    unrealized_pnl=0,
                    unrealized_pnl_pct=0,
                ))

        return positions

    async def submit_order(self, order: OrderRequest) -> OrderResult:
        """Submit order to Binance."""
        if not self._client:
            raise RuntimeError("Not connected to Binance")

        try:
            result = await self._client.place_order(
                symbol=order.symbol,
                side=order.side.value.upper(),
                order_type=order.order_type.value.upper(),
                quantity=order.quantity,
                price=order.limit_price,
            )

            return OrderResult(
                order_id=str(result["orderId"]),
                client_order_id=result.get("clientOrderId", order.client_order_id),
                symbol=order.symbol,
                side=order.side,
                status=self._map_status(result["status"]),
                quantity=float(result["origQty"]),
                filled_quantity=float(result["executedQty"]),
                average_price=float(result.get("avgPrice", 0)),
                commission=0,  # Would need to query fills
            )
        except Exception as e:
            return OrderResult(
                order_id="",
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                side=order.side,
                status=OrderStatus.REJECTED,
                quantity=order.quantity,
                message=str(e),
            )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel Binance order."""
        if not self._client:
            return False

        try:
            await self._client.cancel_order(order_id)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False

    async def get_order(self, order_id: str) -> Optional[OrderResult]:
        """Get Binance order status."""
        if not self._client:
            return None

        try:
            result = await self._client.get_order(order_id)
            return OrderResult(
                order_id=str(result["orderId"]),
                client_order_id=result.get("clientOrderId", ""),
                symbol=result["symbol"],
                side=OrderSide(result["side"].lower()),
                status=self._map_status(result["status"]),
                quantity=float(result["origQty"]),
                filled_quantity=float(result["executedQty"]),
                average_price=float(result.get("avgPrice", 0)),
            )
        except Exception:
            return None

    async def get_open_orders(self) -> list[OrderResult]:
        """Get open Binance orders."""
        if not self._client:
            return []

        try:
            orders = await self._client.get_open_orders()
            return [
                OrderResult(
                    order_id=str(o["orderId"]),
                    client_order_id=o.get("clientOrderId", ""),
                    symbol=o["symbol"],
                    side=OrderSide(o["side"].lower()),
                    status=self._map_status(o["status"]),
                    quantity=float(o["origQty"]),
                    filled_quantity=float(o["executedQty"]),
                )
                for o in orders
            ]
        except Exception:
            return []

    def _map_status(self, binance_status: str) -> OrderStatus:
        """Map Binance status to OrderStatus."""
        mapping = {
            "NEW": OrderStatus.SUBMITTED,
            "PARTIALLY_FILLED": OrderStatus.PARTIAL,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }
        return mapping.get(binance_status, OrderStatus.PENDING)
