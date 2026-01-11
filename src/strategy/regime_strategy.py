"""Regime-adaptive portfolio strategy."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.regime.rule_based import MarketRegime


@dataclass
class RegimeAllocation:
    """Asset allocation for a specific regime.
    
    Attributes:
        regime: Market regime
        allocations: Asset class allocations
        risk_budget: Risk budget allocation
        rebalance_threshold: Threshold for rebalancing
    """
    regime: MarketRegime
    allocations: dict[str, float]
    risk_budget: dict[str, float] | None = None
    rebalance_threshold: float = 0.05


class RegimeAdaptiveStrategy:
    """Regime-adaptive portfolio strategy.
    
    Adjusts portfolio allocation based on detected market regime.
    Supports gradual transitions between regimes.
    
    Example:
        >>> strategy = RegimeAdaptiveStrategy()
        >>> current_regime = RegimeClassifier.classify(indicators)
        >>> target = strategy.get_target_allocation(current_regime.regime)
    """
    
    # Default allocations by regime
    DEFAULT_ALLOCATIONS = {
        MarketRegime.EXPANSION: {
            "us_equity": 0.40,
            "intl_equity": 0.20,
            "emerging_equity": 0.10,
            "investment_grade_bonds": 0.10,
            "high_yield_bonds": 0.05,
            "commodities": 0.10,
            "cash": 0.05,
        },
        MarketRegime.PEAK: {
            "us_equity": 0.25,
            "intl_equity": 0.10,
            "emerging_equity": 0.05,
            "investment_grade_bonds": 0.25,
            "high_yield_bonds": 0.05,
            "commodities": 0.15,
            "cash": 0.15,
        },
        MarketRegime.CONTRACTION: {
            "us_equity": 0.15,
            "intl_equity": 0.05,
            "emerging_equity": 0.00,
            "investment_grade_bonds": 0.40,
            "high_yield_bonds": 0.00,
            "commodities": 0.05,
            "cash": 0.35,
        },
        MarketRegime.TROUGH: {
            "us_equity": 0.30,
            "intl_equity": 0.15,
            "emerging_equity": 0.05,
            "investment_grade_bonds": 0.25,
            "high_yield_bonds": 0.05,
            "commodities": 0.10,
            "cash": 0.10,
        },
        MarketRegime.UNKNOWN: {
            "us_equity": 0.25,
            "intl_equity": 0.10,
            "emerging_equity": 0.05,
            "investment_grade_bonds": 0.30,
            "high_yield_bonds": 0.00,
            "commodities": 0.10,
            "cash": 0.20,
        },
    }
    
    def __init__(
        self,
        allocations: dict[MarketRegime, dict[str, float]] | None = None,
        transition_periods: int = 3,
        min_allocation: float = 0.0,
        max_allocation: float = 1.0,
    ):
        """Initialize strategy.
        
        Args:
            allocations: Custom allocations by regime
            transition_periods: Periods for gradual transition
            min_allocation: Minimum allocation per asset
            max_allocation: Maximum allocation per asset
        """
        self.allocations = allocations or self.DEFAULT_ALLOCATIONS
        self.transition_periods = transition_periods
        self.min_allocation = min_allocation
        self.max_allocation = max_allocation
        
        self._current_regime: MarketRegime | None = None
        self._transition_progress: float = 1.0
        self._transition_from: dict[str, float] | None = None
    
    def get_target_allocation(
        self,
        regime: MarketRegime,
        confidence: float = 1.0,
    ) -> dict[str, float]:
        """Get target allocation for regime.
        
        Args:
            regime: Current market regime
            confidence: Classification confidence (0-1)
            
        Returns:
            Target allocation dictionary
        """
        target = self.allocations.get(
            regime,
            self.allocations[MarketRegime.UNKNOWN]
        ).copy()
        
        # Blend with neutral allocation if confidence is low
        if confidence < 0.5:
            neutral = self.allocations[MarketRegime.UNKNOWN]
            blend_factor = confidence / 0.5
            
            for asset in target:
                target[asset] = (
                    blend_factor * target[asset] + 
                    (1 - blend_factor) * neutral.get(asset, 0)
                )
        
        # Apply constraints
        for asset in target:
            target[asset] = max(self.min_allocation, 
                                min(self.max_allocation, target[asset]))
        
        # Normalize to sum to 1
        total = sum(target.values())
        if total > 0:
            target = {k: v / total for k, v in target.items()}
        
        return target
    
    def calculate_transition_allocation(
        self,
        from_allocation: dict[str, float],
        to_allocation: dict[str, float],
        progress: float,
    ) -> dict[str, float]:
        """Calculate blended allocation during transition.
        
        Args:
            from_allocation: Starting allocation
            to_allocation: Target allocation
            progress: Transition progress (0-1)
            
        Returns:
            Blended allocation
        """
        all_assets = set(from_allocation.keys()) | set(to_allocation.keys())
        
        blended = {}
        for asset in all_assets:
            from_weight = from_allocation.get(asset, 0)
            to_weight = to_allocation.get(asset, 0)
            blended[asset] = from_weight + progress * (to_weight - from_weight)
        
        return blended
    
    def update_regime(
        self,
        new_regime: MarketRegime,
        current_allocation: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Update regime and get new target allocation.
        
        Args:
            new_regime: New detected regime
            current_allocation: Current portfolio allocation
            
        Returns:
            Target allocation (may be transitioning)
        """
        if self._current_regime is None:
            # First regime assignment
            self._current_regime = new_regime
            self._transition_progress = 1.0
            return self.get_target_allocation(new_regime)
        
        if new_regime != self._current_regime:
            # Regime change - start transition
            self._transition_from = current_allocation or self.get_target_allocation(
                self._current_regime
            )
            self._current_regime = new_regime
            self._transition_progress = 0.0
        
        # Progress transition
        if self._transition_progress < 1.0:
            self._transition_progress += 1.0 / self.transition_periods
            self._transition_progress = min(1.0, self._transition_progress)
            
            target = self.get_target_allocation(new_regime)
            return self.calculate_transition_allocation(
                self._transition_from,
                target,
                self._transition_progress,
            )
        
        return self.get_target_allocation(new_regime)
    
    def get_regime_risk_profile(
        self,
        regime: MarketRegime,
    ) -> dict[str, Any]:
        """Get risk profile for regime.
        
        Args:
            regime: Market regime
            
        Returns:
            Risk parameters for regime
        """
        profiles = {
            MarketRegime.EXPANSION: {
                "max_equity": 0.80,
                "min_cash": 0.05,
                "volatility_target": 0.15,
                "max_drawdown": 0.20,
                "rebalance_frequency": "monthly",
            },
            MarketRegime.PEAK: {
                "max_equity": 0.50,
                "min_cash": 0.10,
                "volatility_target": 0.12,
                "max_drawdown": 0.15,
                "rebalance_frequency": "monthly",
            },
            MarketRegime.CONTRACTION: {
                "max_equity": 0.30,
                "min_cash": 0.25,
                "volatility_target": 0.08,
                "max_drawdown": 0.10,
                "rebalance_frequency": "weekly",
            },
            MarketRegime.TROUGH: {
                "max_equity": 0.60,
                "min_cash": 0.10,
                "volatility_target": 0.12,
                "max_drawdown": 0.15,
                "rebalance_frequency": "monthly",
            },
        }
        
        return profiles.get(regime, profiles[MarketRegime.TROUGH])
    
    def calculate_trades(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        portfolio_value: float,
        min_trade_value: float = 100,
    ) -> dict[str, float]:
        """Calculate trade amounts for rebalancing.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target weights
            portfolio_value: Total portfolio value
            min_trade_value: Minimum trade value
            
        Returns:
            Trade amounts by asset (positive = buy)
        """
        all_assets = set(current_weights.keys()) | set(target_weights.keys())
        
        trades = {}
        for asset in all_assets:
            current = current_weights.get(asset, 0) * portfolio_value
            target = target_weights.get(asset, 0) * portfolio_value
            diff = target - current
            
            if abs(diff) >= min_trade_value:
                trades[asset] = diff
        
        return trades


class TacticalOverlay:
    """Tactical overlay for regime strategy.
    
    Adds short-term tactical adjustments to strategic allocation.
    """
    
    def __init__(
        self,
        max_deviation: float = 0.10,
        signals: dict[str, float] | None = None,
    ):
        """Initialize overlay.
        
        Args:
            max_deviation: Maximum deviation from strategic allocation
            signals: Initial tactical signals
        """
        self.max_deviation = max_deviation
        self.signals = signals or {}
    
    def update_signal(
        self,
        asset: str,
        signal: float,
    ) -> None:
        """Update tactical signal for asset.
        
        Args:
            asset: Asset name
            signal: Signal strength (-1 to 1)
        """
        self.signals[asset] = np.clip(signal, -1, 1)
    
    def apply_overlay(
        self,
        strategic_allocation: dict[str, float],
    ) -> dict[str, float]:
        """Apply tactical overlay to strategic allocation.
        
        Args:
            strategic_allocation: Base strategic allocation
            
        Returns:
            Adjusted allocation
        """
        adjusted = strategic_allocation.copy()
        
        for asset, signal in self.signals.items():
            if asset in adjusted:
                deviation = signal * self.max_deviation
                adjusted[asset] = np.clip(
                    adjusted[asset] + deviation,
                    0,
                    1
                )
        
        # Normalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        
        return adjusted
