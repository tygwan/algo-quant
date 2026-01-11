"""Regime-based trading signal generation."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .rule_based import MarketRegime, RegimeClassification


@dataclass
class AllocationSignal:
    """Asset allocation signal based on regime.
    
    Attributes:
        regime: Current market regime
        allocations: Target allocation weights by asset class
        risk_level: Risk level (0-1)
        confidence: Signal confidence (0-1)
        timestamp: Signal generation time
        metadata: Additional signal metadata
    """
    regime: MarketRegime
    allocations: dict[str, float]
    risk_level: float
    confidence: float
    timestamp: pd.Timestamp
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "regime": self.regime.value,
            "allocations": self.allocations,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    def total_allocation(self) -> float:
        """Get total allocation (should sum to 1)."""
        return sum(self.allocations.values())


class RegimeSignalGenerator:
    """Generate trading signals based on regime classification.
    
    Provides asset allocation recommendations and risk signals
    based on the current market regime and transition probabilities.
    
    Example:
        >>> generator = RegimeSignalGenerator()
        >>> signal = generator.generate_allocation_signal(regime_classification)
        >>> print(f"Equity: {signal.allocations['equity']:.1%}")
    """
    
    # Default regime-based allocations
    DEFAULT_ALLOCATIONS = {
        MarketRegime.EXPANSION: {
            "equity": 0.70,
            "fixed_income": 0.15,
            "commodities": 0.10,
            "cash": 0.05,
        },
        MarketRegime.PEAK: {
            "equity": 0.40,
            "fixed_income": 0.30,
            "commodities": 0.15,
            "cash": 0.15,
        },
        MarketRegime.CONTRACTION: {
            "equity": 0.20,
            "fixed_income": 0.45,
            "commodities": 0.05,
            "cash": 0.30,
        },
        MarketRegime.TROUGH: {
            "equity": 0.50,
            "fixed_income": 0.25,
            "commodities": 0.10,
            "cash": 0.15,
        },
        MarketRegime.UNKNOWN: {
            "equity": 0.40,
            "fixed_income": 0.30,
            "commodities": 0.10,
            "cash": 0.20,
        },
    }
    
    # Risk levels by regime
    DEFAULT_RISK_LEVELS = {
        MarketRegime.EXPANSION: 0.7,
        MarketRegime.PEAK: 0.4,
        MarketRegime.CONTRACTION: 0.2,
        MarketRegime.TROUGH: 0.5,
        MarketRegime.UNKNOWN: 0.4,
    }
    
    def __init__(
        self,
        allocations: dict[MarketRegime, dict[str, float]] | None = None,
        risk_levels: dict[MarketRegime, float] | None = None,
        min_confidence: float = 0.5,
    ):
        """Initialize signal generator.
        
        Args:
            allocations: Custom allocation rules by regime
            risk_levels: Custom risk levels by regime
            min_confidence: Minimum confidence for signals
        """
        self.allocations = allocations or self.DEFAULT_ALLOCATIONS
        self.risk_levels = risk_levels or self.DEFAULT_RISK_LEVELS
        self.min_confidence = min_confidence
    
    def generate_allocation_signal(
        self,
        classification: RegimeClassification,
        blend_with_neutral: bool = True,
    ) -> AllocationSignal:
        """Generate allocation signal from regime classification.
        
        Args:
            classification: Regime classification result
            blend_with_neutral: Blend with neutral allocation if confidence is low
            
        Returns:
            AllocationSignal with recommended allocations
        """
        regime = classification.regime
        confidence = classification.confidence
        
        # Get target allocation
        target_alloc = self.allocations.get(
            regime,
            self.allocations[MarketRegime.UNKNOWN]
        )
        
        # Get risk level
        risk_level = self.risk_levels.get(
            regime,
            self.risk_levels[MarketRegime.UNKNOWN]
        )
        
        # Blend with neutral if confidence is low
        if blend_with_neutral and confidence < self.min_confidence:
            neutral_alloc = self.allocations[MarketRegime.UNKNOWN]
            blend_factor = confidence / self.min_confidence
            
            target_alloc = {
                asset: blend_factor * target + (1 - blend_factor) * neutral_alloc[asset]
                for asset, target in target_alloc.items()
            }
            
            # Also blend risk level
            risk_level = blend_factor * risk_level + (1 - blend_factor) * 0.4
        
        return AllocationSignal(
            regime=regime,
            allocations=target_alloc,
            risk_level=risk_level,
            confidence=confidence,
            timestamp=classification.timestamp,
            metadata={
                "classification_details": classification.details,
                "indicators": classification.indicators,
            },
        )
    
    def generate_risk_signal(
        self,
        classification: RegimeClassification,
        current_risk: float = 0.5,
    ) -> dict[str, Any]:
        """Generate risk adjustment signal.
        
        Args:
            classification: Regime classification result
            current_risk: Current portfolio risk level
            
        Returns:
            Risk signal dictionary
        """
        regime = classification.regime
        target_risk = self.risk_levels.get(regime, 0.4)
        
        # Calculate adjustment
        risk_adjustment = target_risk - current_risk
        
        return {
            "regime": regime.value,
            "current_risk": current_risk,
            "target_risk": target_risk,
            "risk_adjustment": risk_adjustment,
            "action": "reduce" if risk_adjustment < -0.1 else "increase" if risk_adjustment > 0.1 else "maintain",
            "urgency": abs(risk_adjustment),
            "confidence": classification.confidence,
        }
    
    def generate_transition_signals(
        self,
        classifications: pd.DataFrame,
        lookback: int = 3,
    ) -> pd.DataFrame:
        """Generate signals based on regime transitions.
        
        Args:
            classifications: Historical regime classifications
            lookback: Periods to look back for transition detection
            
        Returns:
            DataFrame with transition signals
        """
        signals = pd.DataFrame(index=classifications.index)
        
        # Current regime
        signals["regime"] = classifications["regime"]
        
        # Previous regime
        signals["prev_regime"] = classifications["regime"].shift(1)
        
        # Regime change flag
        signals["regime_change"] = signals["regime"] != signals["prev_regime"]
        
        # Regime stability (consecutive periods in same regime)
        regime_groups = (signals["regime"] != signals["regime"].shift(1)).cumsum()
        signals["regime_duration"] = regime_groups.groupby(regime_groups).cumcount() + 1
        
        # Transition signals
        transitions = []
        for i in range(len(signals)):
            if signals["regime_change"].iloc[i]:
                prev = signals["prev_regime"].iloc[i]
                curr = signals["regime"].iloc[i]
                
                # Classify transition type
                if curr == MarketRegime.CONTRACTION.value:
                    signal = "defensive"
                elif curr == MarketRegime.EXPANSION.value:
                    signal = "aggressive"
                elif curr == MarketRegime.TROUGH.value:
                    signal = "recovery"
                elif curr == MarketRegime.PEAK.value:
                    signal = "cautious"
                else:
                    signal = "neutral"
                
                transitions.append(signal)
            else:
                transitions.append("hold")
        
        signals["transition_signal"] = transitions
        
        return signals
    
    def backtest_signals(
        self,
        classifications: pd.DataFrame,
        returns: pd.DataFrame,
        initial_capital: float = 100000,
    ) -> dict[str, Any]:
        """Backtest allocation signals against historical returns.
        
        Args:
            classifications: Historical regime classifications
            returns: Asset class returns DataFrame
            initial_capital: Starting capital
            
        Returns:
            Backtest results dictionary
        """
        # Align dates
        common_dates = classifications.index.intersection(returns.index)
        classifications = classifications.loc[common_dates]
        returns = returns.loc[common_dates]
        
        # Generate signals for each period
        portfolio_values = [initial_capital]
        allocations_history = []
        
        for i in range(len(common_dates)):
            date = common_dates[i]
            regime_str = classifications.loc[date, "regime"]
            confidence = classifications.loc[date, "confidence"]
            
            try:
                regime = MarketRegime(regime_str)
            except ValueError:
                regime = MarketRegime.UNKNOWN
            
            # Get allocation
            alloc = self.allocations.get(regime, self.allocations[MarketRegime.UNKNOWN])
            allocations_history.append(alloc)
            
            if i > 0:
                # Calculate return
                period_return = sum(
                    alloc.get(asset, 0) * returns.loc[date, asset]
                    for asset in returns.columns
                    if asset in alloc
                )
                portfolio_values.append(portfolio_values[-1] * (1 + period_return))
        
        # Calculate metrics
        portfolio_series = pd.Series(portfolio_values[1:], index=common_dates)
        portfolio_returns = portfolio_series.pct_change().dropna()
        
        total_return = (portfolio_values[-1] / initial_capital) - 1
        
        if len(portfolio_returns) > 0:
            sharpe = (
                portfolio_returns.mean() * np.sqrt(252) / 
                portfolio_returns.std() if portfolio_returns.std() > 0 else 0
            )
            max_dd = (portfolio_series / portfolio_series.cummax() - 1).min()
        else:
            sharpe = 0
            max_dd = 0
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "final_value": portfolio_values[-1],
            "portfolio_values": portfolio_series,
            "n_periods": len(common_dates),
        }


class CompositeSignalGenerator:
    """Generate signals from multiple regime classifiers.
    
    Combines signals from different classifiers using weighted averaging
    or voting to produce more robust recommendations.
    """
    
    def __init__(
        self,
        classifiers: dict[str, tuple[Any, float]],
        base_generator: RegimeSignalGenerator | None = None,
    ):
        """Initialize composite generator.
        
        Args:
            classifiers: Dict of name -> (classifier, weight)
            base_generator: Base signal generator for allocations
        """
        self.classifiers = classifiers
        self.generator = base_generator or RegimeSignalGenerator()
    
    def generate_consensus_signal(
        self,
        indicators: pd.DataFrame,
        date: pd.Timestamp | None = None,
    ) -> AllocationSignal:
        """Generate consensus signal from all classifiers.
        
        Args:
            indicators: Economic indicators
            date: Target date
            
        Returns:
            Consensus allocation signal
        """
        regime_votes = {}
        total_weight = 0
        
        for name, (classifier, weight) in self.classifiers.items():
            classification = classifier.classify(indicators, date)
            regime = classification.regime
            
            if regime not in regime_votes:
                regime_votes[regime] = 0
            regime_votes[regime] += weight * classification.confidence
            total_weight += weight
        
        # Normalize votes
        if total_weight > 0:
            regime_votes = {r: v / total_weight for r, v in regime_votes.items()}
        
        # Get winning regime
        if regime_votes:
            winning_regime = max(regime_votes, key=regime_votes.get)
            confidence = regime_votes[winning_regime]
        else:
            winning_regime = MarketRegime.UNKNOWN
            confidence = 0.0
        
        # Generate signal for winning regime
        dummy_classification = RegimeClassification(
            regime=winning_regime,
            confidence=confidence,
            indicators={},
            timestamp=date or indicators.index[-1],
            details=f"Consensus from {len(self.classifiers)} classifiers",
        )
        
        signal = self.generator.generate_allocation_signal(dummy_classification)
        signal.metadata["regime_votes"] = {r.value: v for r, v in regime_votes.items()}
        
        return signal
