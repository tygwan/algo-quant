"""Rule-based regime classification using economic indicators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class MarketRegime(Enum):
    """Market regime states based on business cycle."""
    
    EXPANSION = "expansion"      # GDP↑, Unemployment↓, Early/Mid cycle
    PEAK = "peak"               # GDP high, Inflation↑, Late cycle
    CONTRACTION = "contraction"  # GDP↓, Unemployment↑, Recession
    TROUGH = "trough"           # GDP low, Recovery starting
    UNKNOWN = "unknown"         # Insufficient data
    
    def __str__(self) -> str:
        return self.value


@dataclass
class RegimeClassification:
    """Result of regime classification."""
    regime: MarketRegime
    confidence: float
    indicators: dict[str, float]
    timestamp: pd.Timestamp
    details: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "indicators": self.indicators,
            "timestamp": self.timestamp,
            "details": self.details,
        }


class RegimeClassifier(ABC):
    """Abstract base class for regime classifiers."""
    
    @abstractmethod
    def classify(
        self,
        indicators: pd.DataFrame,
        date: pd.Timestamp | None = None,
    ) -> RegimeClassification:
        """Classify the current regime.
        
        Args:
            indicators: DataFrame with economic indicators
            date: Date to classify (default: latest)
            
        Returns:
            RegimeClassification result
        """
        pass
    
    @abstractmethod
    def classify_history(
        self,
        indicators: pd.DataFrame,
    ) -> pd.DataFrame:
        """Classify regime for entire history.
        
        Args:
            indicators: DataFrame with economic indicators
            
        Returns:
            DataFrame with regime classifications
        """
        pass


class RuleBasedClassifier(RegimeClassifier):
    """Rule-based regime classifier using economic rules.
    
    Uses a combination of economic indicators to classify the current
    market regime based on established macroeconomic relationships.
    
    Rules based on:
    - GDP growth direction and level
    - Unemployment rate trends
    - Yield curve slope
    - Inflation trends
    - Leading indicators
    
    Example:
        >>> classifier = RuleBasedClassifier()
        >>> result = classifier.classify(indicators_df)
        >>> print(f"Current regime: {result.regime}")
    """
    
    # Default thresholds
    DEFAULT_THRESHOLDS = {
        "gdp_growth_positive": 0.0,
        "gdp_growth_strong": 2.5,
        "unemployment_low": 5.0,
        "unemployment_high": 7.0,
        "yield_curve_inverted": 0.0,
        "inflation_high": 3.0,
        "diffusion_expanding": 50,
    }
    
    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
        use_yield_curve: bool = True,
        use_diffusion: bool = True,
    ):
        """Initialize classifier.
        
        Args:
            thresholds: Custom thresholds for rules
            use_yield_curve: Include yield curve in classification
            use_diffusion: Include diffusion index in classification
        """
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.use_yield_curve = use_yield_curve
        self.use_diffusion = use_diffusion
    
    def _get_indicator_value(
        self,
        indicators: pd.DataFrame,
        column: str,
        date: pd.Timestamp | None = None,
    ) -> float | None:
        """Get indicator value at date."""
        if column not in indicators.columns:
            return None
        
        series = indicators[column]
        
        if date is not None:
            # Find nearest date
            if date in series.index:
                return series.loc[date]
            else:
                idx = series.index.get_indexer([date], method="ffill")[0]
                if idx >= 0:
                    return series.iloc[idx]
                return None
        else:
            # Latest value
            return series.dropna().iloc[-1] if len(series.dropna()) > 0 else None
    
    def _calculate_regime_scores(
        self,
        indicators: pd.DataFrame,
        date: pd.Timestamp | None = None,
    ) -> dict[MarketRegime, float]:
        """Calculate score for each regime.
        
        Returns dict mapping regime to confidence score.
        """
        scores = {
            MarketRegime.EXPANSION: 0.0,
            MarketRegime.PEAK: 0.0,
            MarketRegime.CONTRACTION: 0.0,
            MarketRegime.TROUGH: 0.0,
        }
        
        # GDP Growth
        gdp_growth = self._get_indicator_value(indicators, "gdp_growth", date)
        if gdp_growth is not None:
            if gdp_growth > self.thresholds["gdp_growth_strong"]:
                scores[MarketRegime.EXPANSION] += 2.0
            elif gdp_growth > self.thresholds["gdp_growth_positive"]:
                scores[MarketRegime.EXPANSION] += 1.0
                scores[MarketRegime.TROUGH] += 0.5
            elif gdp_growth < -1.0:
                scores[MarketRegime.CONTRACTION] += 2.0
            else:
                scores[MarketRegime.PEAK] += 0.5
                scores[MarketRegime.TROUGH] += 0.5
        
        # Unemployment
        unemployment = self._get_indicator_value(indicators, "unemployment", date)
        unemployment_mom = self._get_indicator_value(indicators, "unemployment_mom", date)
        
        if unemployment is not None:
            if unemployment < self.thresholds["unemployment_low"]:
                scores[MarketRegime.EXPANSION] += 1.0
                scores[MarketRegime.PEAK] += 0.5
            elif unemployment > self.thresholds["unemployment_high"]:
                scores[MarketRegime.CONTRACTION] += 1.0
                scores[MarketRegime.TROUGH] += 0.5
        
        if unemployment_mom is not None:
            if unemployment_mom < -0.1:
                scores[MarketRegime.EXPANSION] += 1.0
                scores[MarketRegime.TROUGH] += 0.5
            elif unemployment_mom > 0.2:
                scores[MarketRegime.CONTRACTION] += 1.5
        
        # Yield Curve
        if self.use_yield_curve:
            yield_spread = self._get_indicator_value(indicators, "yield_spread", date)
            if yield_spread is not None:
                if yield_spread < self.thresholds["yield_curve_inverted"]:
                    scores[MarketRegime.PEAK] += 1.5
                    scores[MarketRegime.CONTRACTION] += 1.0
                elif yield_spread > 1.5:
                    scores[MarketRegime.EXPANSION] += 1.0
                    scores[MarketRegime.TROUGH] += 0.5
        
        # Inflation
        inflation = self._get_indicator_value(indicators, "inflation_yoy", date)
        if inflation is not None:
            if inflation > self.thresholds["inflation_high"]:
                scores[MarketRegime.PEAK] += 1.0
            elif inflation < 1.0:
                scores[MarketRegime.CONTRACTION] += 0.5
                scores[MarketRegime.TROUGH] += 0.5
        
        # Diffusion Index
        if self.use_diffusion:
            diffusion = self._get_indicator_value(indicators, "diffusion_index", date)
            if diffusion is not None:
                if diffusion > 60:
                    scores[MarketRegime.EXPANSION] += 1.5
                elif diffusion > self.thresholds["diffusion_expanding"]:
                    scores[MarketRegime.EXPANSION] += 0.5
                    scores[MarketRegime.TROUGH] += 0.5
                elif diffusion < 30:
                    scores[MarketRegime.CONTRACTION] += 1.5
                else:
                    scores[MarketRegime.PEAK] += 0.5
        
        return scores
    
    def classify(
        self,
        indicators: pd.DataFrame,
        date: pd.Timestamp | None = None,
    ) -> RegimeClassification:
        """Classify the regime at a specific date.
        
        Args:
            indicators: DataFrame with economic indicators
            date: Date to classify (default: latest)
            
        Returns:
            RegimeClassification result
        """
        if date is None:
            date = indicators.index[-1]
        
        # Calculate scores
        scores = self._calculate_regime_scores(indicators, date)
        
        # Determine winning regime
        total_score = sum(scores.values())
        if total_score == 0:
            return RegimeClassification(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                indicators={},
                timestamp=date,
                details="Insufficient data for classification",
            )
        
        best_regime = max(scores, key=scores.get)
        confidence = scores[best_regime] / total_score
        
        # Collect indicator values
        indicator_values = {}
        for col in indicators.columns:
            val = self._get_indicator_value(indicators, col, date)
            if val is not None:
                indicator_values[col] = val
        
        return RegimeClassification(
            regime=best_regime,
            confidence=confidence,
            indicators=indicator_values,
            timestamp=date,
            details=f"Scores: {scores}",
        )
    
    def classify_history(
        self,
        indicators: pd.DataFrame,
    ) -> pd.DataFrame:
        """Classify regime for entire history.
        
        Args:
            indicators: DataFrame with economic indicators
            
        Returns:
            DataFrame with columns: regime, confidence
        """
        results = []
        
        for date in indicators.index:
            classification = self.classify(indicators, date)
            results.append({
                "date": date,
                "regime": classification.regime.value,
                "confidence": classification.confidence,
            })
        
        df = pd.DataFrame(results)
        df = df.set_index("date")
        
        return df


class YieldCurveClassifier(RegimeClassifier):
    """Classifier focused on yield curve signals.
    
    Uses yield curve shape and dynamics to predict recessions
    and classify market regimes.
    """
    
    def __init__(
        self,
        inversion_threshold: float = 0.0,
        steepening_threshold: float = 0.5,
        lead_periods: int = 12,
    ):
        """Initialize classifier.
        
        Args:
            inversion_threshold: Spread below this = inverted
            steepening_threshold: Spread change for steepening signal
            lead_periods: Months yield curve leads recession
        """
        self.inversion_threshold = inversion_threshold
        self.steepening_threshold = steepening_threshold
        self.lead_periods = lead_periods
    
    def classify(
        self,
        indicators: pd.DataFrame,
        date: pd.Timestamp | None = None,
    ) -> RegimeClassification:
        """Classify based on yield curve."""
        if date is None:
            date = indicators.index[-1]
        
        if "yield_spread" not in indicators.columns:
            return RegimeClassification(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                indicators={},
                timestamp=date,
                details="Yield spread data not available",
            )
        
        spread = indicators["yield_spread"]
        current_spread = spread.loc[:date].iloc[-1]
        
        # Calculate spread changes
        spread_3m = spread.diff(3).loc[:date].iloc[-1] if len(spread) > 3 else 0
        spread_12m = spread.diff(12).loc[:date].iloc[-1] if len(spread) > 12 else 0
        
        # Classification logic
        if current_spread < self.inversion_threshold:
            # Inverted curve - late cycle or recession warning
            if spread_3m < 0:
                regime = MarketRegime.PEAK
                confidence = 0.8
            else:
                regime = MarketRegime.CONTRACTION
                confidence = 0.7
        elif current_spread > 2.0 and spread_3m > 0:
            # Steep and steepening - early cycle
            regime = MarketRegime.TROUGH
            confidence = 0.7
        elif current_spread > 1.0:
            # Normal steep curve - expansion
            regime = MarketRegime.EXPANSION
            confidence = 0.6
        else:
            # Flat curve - could be transitioning
            regime = MarketRegime.PEAK
            confidence = 0.5
        
        return RegimeClassification(
            regime=regime,
            confidence=confidence,
            indicators={
                "yield_spread": current_spread,
                "spread_3m_change": spread_3m,
                "spread_12m_change": spread_12m,
            },
            timestamp=date,
            details=f"Spread: {current_spread:.2f}%",
        )
    
    def classify_history(
        self,
        indicators: pd.DataFrame,
    ) -> pd.DataFrame:
        """Classify regime for entire history."""
        results = []
        
        for date in indicators.index:
            classification = self.classify(indicators, date)
            results.append({
                "date": date,
                "regime": classification.regime.value,
                "confidence": classification.confidence,
            })
        
        df = pd.DataFrame(results)
        df = df.set_index("date")
        
        return df
    
    def calculate_recession_signal(
        self,
        spread: pd.Series,
    ) -> pd.DataFrame:
        """Calculate recession warning signals.
        
        Args:
            spread: Yield spread time series
            
        Returns:
            DataFrame with recession signals
        """
        df = pd.DataFrame(index=spread.index)
        
        # Inversion signal
        df["inverted"] = spread < self.inversion_threshold
        
        # Duration of inversion
        df["inversion_duration"] = df["inverted"].groupby(
            (~df["inverted"]).cumsum()
        ).cumsum()
        
        # Recession probability (simple model)
        # Based on historical relationship
        df["recession_prob"] = (0.5 - 0.25 * spread).clip(0, 1)
        
        # Signal strength
        df["signal_strength"] = np.where(
            df["inverted"],
            np.minimum(df["inversion_duration"] / 3, 1.0),
            0.0
        )
        
        return df
