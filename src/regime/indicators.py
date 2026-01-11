"""Macro economic indicator processing for regime classification."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd


class EconomicIndicators:
    """FRED economic indicator series IDs.
    
    Reference: https://fred.stlouisfed.org/
    """
    
    # GDP and Output
    GDP = "GDPC1"              # Real GDP (Quarterly)
    GDP_GROWTH = "A191RL1Q225SBEA"  # Real GDP Growth Rate
    INDPRO = "INDPRO"          # Industrial Production Index
    
    # Employment
    UNRATE = "UNRATE"          # Unemployment Rate
    PAYEMS = "PAYEMS"          # Total Nonfarm Payrolls
    ICSA = "ICSA"              # Initial Claims (Weekly)
    
    # Inflation
    CPIAUCSL = "CPIAUCSL"      # Consumer Price Index
    PCEPI = "PCEPI"            # PCE Price Index
    CPILFESL = "CPILFESL"      # Core CPI (ex Food & Energy)
    
    # Interest Rates
    FEDFUNDS = "FEDFUNDS"      # Federal Funds Rate
    DFF = "DFF"                # Fed Funds Effective Rate (Daily)
    DGS10 = "DGS10"            # 10-Year Treasury
    DGS2 = "DGS2"              # 2-Year Treasury
    
    # Yield Curve
    T10Y2Y = "T10Y2Y"          # 10Y-2Y Spread
    T10Y3M = "T10Y3M"          # 10Y-3M Spread
    
    # Sentiment and Surveys
    UMCSENT = "UMCSENT"        # Consumer Sentiment
    
    # Credit
    BAMLH0A0HYM2 = "BAMLH0A0HYM2"  # High Yield Spread
    
    # NBER Recession Indicator
    USREC = "USREC"            # NBER Recession Indicator
    USRECM = "USRECM"          # NBER Recession Indicator (Monthly)


@dataclass
class ProcessedIndicator:
    """Processed economic indicator with derived metrics."""
    raw: pd.Series
    normalized: pd.Series
    momentum: pd.Series
    trend: pd.Series
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame({
            "raw": self.raw,
            "normalized": self.normalized,
            "momentum": self.momentum,
            "trend": self.trend,
        })


class MacroIndicatorProcessor:
    """Process macroeconomic indicators for regime classification.
    
    Provides methods to normalize, transform, and derive signals from
    raw FRED economic data.
    
    Example:
        >>> processor = MacroIndicatorProcessor()
        >>> processed = processor.process_indicator(gdp_series, name="GDP")
        >>> momentum = processor.calculate_momentum(unrate, periods=3)
    """
    
    def __init__(
        self,
        lookback_window: int = 60,
        normalize_method: Literal["zscore", "minmax", "percentile"] = "zscore",
    ):
        """Initialize processor.
        
        Args:
            lookback_window: Window for rolling calculations (months)
            normalize_method: Normalization method
        """
        self.lookback_window = lookback_window
        self.normalize_method = normalize_method
    
    def normalize(
        self,
        series: pd.Series,
        method: str | None = None,
        window: int | None = None,
    ) -> pd.Series:
        """Normalize a time series.
        
        Args:
            series: Raw time series
            method: Normalization method (zscore, minmax, percentile)
            window: Rolling window for calculation
            
        Returns:
            Normalized series
        """
        method = method or self.normalize_method
        window = window or self.lookback_window
        
        if method == "zscore":
            rolling_mean = series.rolling(window=window, min_periods=window//2).mean()
            rolling_std = series.rolling(window=window, min_periods=window//2).std()
            return (series - rolling_mean) / rolling_std
        
        elif method == "minmax":
            rolling_min = series.rolling(window=window, min_periods=window//2).min()
            rolling_max = series.rolling(window=window, min_periods=window//2).max()
            return (series - rolling_min) / (rolling_max - rolling_min)
        
        elif method == "percentile":
            def percentile_rank(x):
                return (x.rank() / len(x)).iloc[-1]
            return series.rolling(window=window, min_periods=window//2).apply(percentile_rank)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def calculate_momentum(
        self,
        series: pd.Series,
        periods: int = 12,
        method: Literal["diff", "pct", "log"] = "diff",
    ) -> pd.Series:
        """Calculate momentum (rate of change).
        
        Args:
            series: Input time series
            periods: Number of periods for momentum calculation
            method: Momentum calculation method
                - diff: Simple difference
                - pct: Percentage change
                - log: Log difference
                
        Returns:
            Momentum series
        """
        if method == "diff":
            return series.diff(periods)
        elif method == "pct":
            return series.pct_change(periods)
        elif method == "log":
            return np.log(series / series.shift(periods))
        else:
            raise ValueError(f"Unknown momentum method: {method}")
    
    def calculate_trend(
        self,
        series: pd.Series,
        short_window: int = 3,
        long_window: int = 12,
    ) -> pd.Series:
        """Calculate trend using moving average crossover.
        
        Args:
            series: Input time series
            short_window: Short-term MA window
            long_window: Long-term MA window
            
        Returns:
            Trend signal (-1, 0, 1)
        """
        short_ma = series.rolling(window=short_window, min_periods=1).mean()
        long_ma = series.rolling(window=long_window, min_periods=1).mean()
        
        # Trend signal: 1 if short > long, -1 if short < long
        trend = pd.Series(0, index=series.index)
        trend[short_ma > long_ma] = 1
        trend[short_ma < long_ma] = -1
        
        return trend
    
    def calculate_acceleration(
        self,
        series: pd.Series,
        periods: int = 3,
    ) -> pd.Series:
        """Calculate acceleration (second derivative).
        
        Args:
            series: Input time series
            periods: Periods for calculation
            
        Returns:
            Acceleration series
        """
        momentum = self.calculate_momentum(series, periods, "diff")
        return self.calculate_momentum(momentum, periods, "diff")
    
    def process_indicator(
        self,
        series: pd.Series,
        name: str = "indicator",
        invert: bool = False,
    ) -> ProcessedIndicator:
        """Process a single indicator with all transformations.
        
        Args:
            series: Raw indicator series
            name: Indicator name
            invert: Whether to invert the indicator (e.g., unemployment)
            
        Returns:
            ProcessedIndicator with all derived metrics
        """
        # Clean data
        clean_series = series.dropna()
        
        # Invert if needed (e.g., unemployment rate - lower is better)
        if invert:
            clean_series = -clean_series
        
        return ProcessedIndicator(
            raw=series,
            normalized=self.normalize(clean_series),
            momentum=self.calculate_momentum(clean_series, periods=3, method="diff"),
            trend=self.calculate_trend(clean_series),
        )
    
    def create_composite_index(
        self,
        indicators: dict[str, pd.Series],
        weights: dict[str, float] | None = None,
        invert_list: list[str] | None = None,
    ) -> pd.Series:
        """Create a composite index from multiple indicators.
        
        Args:
            indicators: Dictionary of indicator name -> series
            weights: Optional weights for each indicator
            invert_list: List of indicators to invert
            
        Returns:
            Composite index series
        """
        invert_list = invert_list or []
        
        # Default equal weights
        if weights is None:
            weights = {name: 1.0 / len(indicators) for name in indicators}
        
        # Normalize all indicators
        normalized = {}
        for name, series in indicators.items():
            norm = self.normalize(series)
            if name in invert_list:
                norm = -norm
            normalized[name] = norm
        
        # Create DataFrame and align dates
        df = pd.DataFrame(normalized)
        
        # Weighted sum
        composite = pd.Series(0, index=df.index, dtype=float)
        for name, weight in weights.items():
            if name in df.columns:
                composite += weight * df[name].fillna(0)
        
        return composite
    
    def detect_yield_curve_inversion(
        self,
        spread: pd.Series,
        threshold: float = 0.0,
        consecutive_periods: int = 1,
    ) -> pd.Series:
        """Detect yield curve inversion.
        
        Args:
            spread: Yield spread series (e.g., 10Y-2Y)
            threshold: Inversion threshold
            consecutive_periods: Consecutive periods below threshold
            
        Returns:
            Boolean series indicating inversion
        """
        inverted = spread < threshold
        
        if consecutive_periods > 1:
            # Require consecutive periods
            inverted = inverted.rolling(window=consecutive_periods).sum() >= consecutive_periods
        
        return inverted.fillna(False)
    
    def calculate_recession_probability(
        self,
        spread: pd.Series,
        model: Literal["probit", "simple"] = "simple",
    ) -> pd.Series:
        """Calculate recession probability from yield curve.
        
        Based on research showing yield curve inversion precedes recessions.
        
        Args:
            spread: Yield spread (e.g., T10Y3M or T10Y2Y)
            model: Probability model type
            
        Returns:
            Recession probability (0-1)
        """
        if model == "simple":
            # Simple linear mapping
            # Spread of +2% -> ~5% recession prob
            # Spread of -1% -> ~80% recession prob
            prob = 0.5 - 0.25 * spread
            return prob.clip(0, 1)
        
        elif model == "probit":
            # Probit-style transformation
            from scipy import stats
            z_score = -spread / spread.std()
            return pd.Series(stats.norm.cdf(z_score), index=spread.index)
        
        else:
            raise ValueError(f"Unknown model: {model}")
    
    def calculate_diffusion_index(
        self,
        indicators: dict[str, pd.Series],
        threshold: float = 0,
    ) -> pd.Series:
        """Calculate diffusion index (% of indicators improving).
        
        Args:
            indicators: Dictionary of indicator series
            threshold: Threshold for considering improvement
            
        Returns:
            Diffusion index (0-100)
        """
        # Calculate momentum for each indicator
        improving = pd.DataFrame()
        
        for name, series in indicators.items():
            momentum = self.calculate_momentum(series, periods=3, method="diff")
            improving[name] = momentum > threshold
        
        # Diffusion = % improving
        diffusion = improving.mean(axis=1) * 100
        
        return diffusion
