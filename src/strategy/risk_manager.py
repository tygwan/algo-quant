"""Risk management for portfolio strategies."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd


class PositionSizingMethod(Enum):
    """Position sizing methods."""
    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY = "kelly"
    VOLATILITY_TARGET = "volatility_target"
    EQUAL_RISK = "equal_risk"


@dataclass
class PositionSize:
    """Position sizing result.
    
    Attributes:
        size: Position size (fraction of capital)
        notional: Notional value
        risk: Estimated risk
        method: Method used
    """
    size: float
    notional: float
    risk: float
    method: str


@dataclass
class StopLoss:
    """Stop loss configuration and levels.
    
    Attributes:
        type: Stop type (fixed, trailing, atr)
        level: Stop level
        initial_price: Entry price
        current_stop: Current stop price
    """
    type: str
    level: float
    initial_price: float
    current_stop: float
    
    def is_triggered(self, current_price: float) -> bool:
        """Check if stop is triggered."""
        return current_price <= self.current_stop


class PositionSizer:
    """Position sizing calculator.
    
    Supports various position sizing methods:
    - Fixed Fractional: Risk fixed % of capital per trade
    - Kelly Criterion: Optimal fraction based on edge and odds
    - Volatility Targeting: Size inversely to volatility
    - Equal Risk: Equal risk contribution
    
    Example:
        >>> sizer = PositionSizer(method="volatility_target", target_vol=0.15)
        >>> size = sizer.calculate_size(signal_strength=0.8, volatility=0.20)
    """
    
    def __init__(
        self,
        method: PositionSizingMethod | str = PositionSizingMethod.FIXED_FRACTIONAL,
        risk_per_trade: float = 0.02,
        target_volatility: float = 0.15,
        max_position: float = 0.20,
        kelly_fraction: float = 0.5,
    ):
        """Initialize position sizer.
        
        Args:
            method: Position sizing method
            risk_per_trade: Risk per trade (for fixed fractional)
            target_volatility: Target portfolio volatility
            max_position: Maximum position size
            kelly_fraction: Fraction of Kelly criterion to use
        """
        if isinstance(method, str):
            method = PositionSizingMethod(method)
        
        self.method = method
        self.risk_per_trade = risk_per_trade
        self.target_volatility = target_volatility
        self.max_position = max_position
        self.kelly_fraction = kelly_fraction
    
    def calculate_size(
        self,
        capital: float,
        price: float,
        volatility: float,
        win_rate: float | None = None,
        avg_win: float | None = None,
        avg_loss: float | None = None,
        stop_distance: float | None = None,
    ) -> PositionSize:
        """Calculate position size.
        
        Args:
            capital: Available capital
            price: Current asset price
            volatility: Asset volatility (annualized)
            win_rate: Historical win rate (for Kelly)
            avg_win: Average win size (for Kelly)
            avg_loss: Average loss size (for Kelly)
            stop_distance: Distance to stop loss (for fixed fractional)
            
        Returns:
            PositionSize result
        """
        if self.method == PositionSizingMethod.FIXED_FRACTIONAL:
            size = self._fixed_fractional(
                capital, volatility, stop_distance
            )
        elif self.method == PositionSizingMethod.KELLY:
            size = self._kelly(win_rate, avg_win, avg_loss)
        elif self.method == PositionSizingMethod.VOLATILITY_TARGET:
            size = self._volatility_target(volatility)
        elif self.method == PositionSizingMethod.EQUAL_RISK:
            size = self._equal_risk(volatility)
        else:
            size = self.risk_per_trade
        
        # Apply max position constraint
        size = min(size, self.max_position)
        
        notional = capital * size
        risk = size * volatility
        
        return PositionSize(
            size=size,
            notional=notional,
            risk=risk,
            method=self.method.value,
        )
    
    def _fixed_fractional(
        self,
        capital: float,
        volatility: float,
        stop_distance: float | None,
    ) -> float:
        """Fixed fractional position sizing."""
        if stop_distance is not None and stop_distance > 0:
            # Risk-based sizing
            return self.risk_per_trade / stop_distance
        else:
            # Volatility-based sizing
            return self.risk_per_trade / (volatility * np.sqrt(252))
    
    def _kelly(
        self,
        win_rate: float | None,
        avg_win: float | None,
        avg_loss: float | None,
    ) -> float:
        """Kelly criterion position sizing."""
        if win_rate is None or avg_win is None or avg_loss is None:
            return self.risk_per_trade
        
        if avg_loss == 0:
            return self.max_position
        
        # Kelly formula: f = (p * b - q) / b
        # where p = win rate, q = 1-p, b = avg_win / avg_loss
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        kelly = (p * b - q) / b
        
        # Apply fraction of Kelly
        return max(0, kelly * self.kelly_fraction)
    
    def _volatility_target(
        self,
        volatility: float,
    ) -> float:
        """Volatility targeting position sizing."""
        if volatility <= 0:
            return self.max_position
        
        return self.target_volatility / volatility
    
    def _equal_risk(
        self,
        volatility: float,
    ) -> float:
        """Equal risk contribution sizing."""
        # Assumes we want equal risk from all positions
        # Size inversely proportional to volatility
        if volatility <= 0:
            return self.risk_per_trade
        
        return self.risk_per_trade / volatility


class RiskManager:
    """Portfolio risk management.
    
    Manages position sizing, stop losses, risk limits,
    and portfolio-level risk controls.
    
    Example:
        >>> risk_mgr = RiskManager(max_drawdown=0.15, var_limit=0.10)
        >>> if risk_mgr.check_risk_limits(portfolio):
        ...     execute_trades()
    """
    
    def __init__(
        self,
        max_drawdown: float = 0.20,
        max_position_size: float = 0.20,
        max_sector_exposure: float = 0.40,
        max_single_loss: float = 0.05,
        var_limit: float = 0.10,
        var_confidence: float = 0.95,
        volatility_target: float = 0.15,
    ):
        """Initialize risk manager.
        
        Args:
            max_drawdown: Maximum allowed drawdown
            max_position_size: Maximum single position
            max_sector_exposure: Maximum sector exposure
            max_single_loss: Maximum loss on single position
            var_limit: Value at Risk limit
            var_confidence: VaR confidence level
            volatility_target: Target portfolio volatility
        """
        self.max_drawdown = max_drawdown
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.max_single_loss = max_single_loss
        self.var_limit = var_limit
        self.var_confidence = var_confidence
        self.volatility_target = volatility_target
        
        self._high_water_mark: float = 0
        self._current_drawdown: float = 0
    
    def check_position_limits(
        self,
        weights: pd.Series | dict[str, float],
    ) -> dict[str, list[str]]:
        """Check position size limits.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Dictionary of violations
        """
        if isinstance(weights, dict):
            weights = pd.Series(weights)
        
        violations = {"exceeded": [], "warnings": []}
        
        for asset, weight in weights.items():
            if abs(weight) > self.max_position_size:
                violations["exceeded"].append(
                    f"{asset}: {weight:.1%} > {self.max_position_size:.1%}"
                )
            elif abs(weight) > self.max_position_size * 0.8:
                violations["warnings"].append(
                    f"{asset}: {weight:.1%} approaching limit"
                )
        
        return violations
    
    def check_sector_limits(
        self,
        weights: pd.Series | dict[str, float],
        sectors: dict[str, str],
    ) -> dict[str, list[str]]:
        """Check sector exposure limits.
        
        Args:
            weights: Portfolio weights
            sectors: Asset to sector mapping
            
        Returns:
            Dictionary of violations
        """
        if isinstance(weights, dict):
            weights = pd.Series(weights)
        
        # Calculate sector exposures
        sector_exposure = {}
        for asset, weight in weights.items():
            sector = sectors.get(asset, "other")
            sector_exposure[sector] = sector_exposure.get(sector, 0) + weight
        
        violations = {"exceeded": [], "warnings": []}
        
        for sector, exposure in sector_exposure.items():
            if abs(exposure) > self.max_sector_exposure:
                violations["exceeded"].append(
                    f"{sector}: {exposure:.1%} > {self.max_sector_exposure:.1%}"
                )
            elif abs(exposure) > self.max_sector_exposure * 0.8:
                violations["warnings"].append(
                    f"{sector}: {exposure:.1%} approaching limit"
                )
        
        return violations
    
    def calculate_var(
        self,
        returns: pd.Series | pd.DataFrame,
        weights: pd.Series | None = None,
        method: Literal["historical", "parametric"] = "historical",
    ) -> float:
        """Calculate Value at Risk.
        
        Args:
            returns: Historical returns
            weights: Portfolio weights (for multi-asset)
            method: VaR calculation method
            
        Returns:
            VaR at confidence level
        """
        if isinstance(returns, pd.DataFrame):
            if weights is None:
                weights = pd.Series(1/len(returns.columns), index=returns.columns)
            portfolio_returns = (returns * weights).sum(axis=1)
        else:
            portfolio_returns = returns
        
        if method == "historical":
            var = np.percentile(portfolio_returns, (1 - self.var_confidence) * 100)
        else:
            # Parametric (assumes normal distribution)
            from scipy import stats
            mean = portfolio_returns.mean()
            std = portfolio_returns.std()
            var = mean + std * stats.norm.ppf(1 - self.var_confidence)
        
        return abs(var)
    
    def update_drawdown(
        self,
        portfolio_value: float,
    ) -> float:
        """Update and return current drawdown.
        
        Args:
            portfolio_value: Current portfolio value
            
        Returns:
            Current drawdown
        """
        self._high_water_mark = max(self._high_water_mark, portfolio_value)
        
        if self._high_water_mark > 0:
            self._current_drawdown = (
                self._high_water_mark - portfolio_value
            ) / self._high_water_mark
        
        return self._current_drawdown
    
    def check_drawdown_limit(
        self,
        portfolio_value: float,
    ) -> bool:
        """Check if drawdown limit is breached.
        
        Args:
            portfolio_value: Current portfolio value
            
        Returns:
            True if within limits, False if breached
        """
        current_dd = self.update_drawdown(portfolio_value)
        return current_dd <= self.max_drawdown
    
    def calculate_stop_levels(
        self,
        positions: dict[str, dict],
        method: Literal["fixed", "atr", "trailing"] = "trailing",
        atr_multiplier: float = 2.0,
        trailing_pct: float = 0.10,
    ) -> dict[str, StopLoss]:
        """Calculate stop loss levels for positions.
        
        Args:
            positions: Position data (entry_price, current_price, atr)
            method: Stop loss method
            atr_multiplier: ATR multiplier for ATR stops
            trailing_pct: Trailing stop percentage
            
        Returns:
            Dictionary of stop loss configurations
        """
        stops = {}
        
        for asset, data in positions.items():
            entry_price = data.get("entry_price", data.get("current_price"))
            current_price = data.get("current_price", entry_price)
            atr = data.get("atr", current_price * 0.02)
            high_since_entry = data.get("high_since_entry", current_price)
            
            if method == "fixed":
                stop_price = entry_price * (1 - trailing_pct)
            elif method == "atr":
                stop_price = current_price - atr_multiplier * atr
            elif method == "trailing":
                stop_price = high_since_entry * (1 - trailing_pct)
            else:
                stop_price = entry_price * 0.9
            
            stops[asset] = StopLoss(
                type=method,
                level=trailing_pct if method != "atr" else atr_multiplier,
                initial_price=entry_price,
                current_stop=stop_price,
            )
        
        return stops
    
    def adjust_for_volatility(
        self,
        target_weights: pd.Series | dict[str, float],
        volatilities: pd.Series | dict[str, float],
    ) -> pd.Series:
        """Adjust weights for volatility targeting.
        
        Args:
            target_weights: Target portfolio weights
            volatilities: Asset volatilities
            
        Returns:
            Volatility-adjusted weights
        """
        if isinstance(target_weights, dict):
            target_weights = pd.Series(target_weights)
        if isinstance(volatilities, dict):
            volatilities = pd.Series(volatilities)
        
        # Calculate implied portfolio volatility
        # (simplified - assumes no correlation)
        port_vol = np.sqrt((target_weights ** 2 * volatilities ** 2).sum())
        
        if port_vol > self.volatility_target:
            # Scale down weights
            scale = self.volatility_target / port_vol
            adjusted = target_weights * scale
            
            # Allocate excess to cash
            cash_weight = 1 - adjusted.sum()
            if "cash" in adjusted.index:
                adjusted["cash"] += cash_weight
            else:
                adjusted["cash"] = cash_weight
            
            return adjusted
        
        return target_weights
    
    def generate_risk_report(
        self,
        portfolio_value: float,
        weights: pd.Series,
        returns: pd.DataFrame,
        positions: dict[str, dict] | None = None,
    ) -> dict:
        """Generate comprehensive risk report.
        
        Args:
            portfolio_value: Current portfolio value
            weights: Portfolio weights
            returns: Historical returns
            positions: Position details
            
        Returns:
            Risk report dictionary
        """
        # Calculate metrics
        var = self.calculate_var(returns, weights)
        drawdown = self.update_drawdown(portfolio_value)
        position_violations = self.check_position_limits(weights)
        
        # Portfolio stats
        port_returns = (returns * weights).sum(axis=1)
        volatility = port_returns.std() * np.sqrt(252)
        
        report = {
            "portfolio_value": portfolio_value,
            "drawdown": drawdown,
            "drawdown_limit": self.max_drawdown,
            "drawdown_breached": drawdown > self.max_drawdown,
            "var": var,
            "var_limit": self.var_limit,
            "var_breached": var > self.var_limit,
            "volatility": volatility,
            "volatility_target": self.volatility_target,
            "position_violations": position_violations,
            "high_water_mark": self._high_water_mark,
        }
        
        if positions:
            stops = self.calculate_stop_levels(positions)
            triggered = [
                asset for asset, stop in stops.items()
                if stop.is_triggered(positions[asset].get("current_price", 0))
            ]
            report["stops_triggered"] = triggered
        
        return report
