"""Factor-based portfolio strategy."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd


class FactorWeightMethod(Enum):
    """Methods for converting factor scores to weights."""
    RANK = "rank"              # Rank-based weights
    ZSCORE = "zscore"          # Z-score based weights
    QUINTILE = "quintile"      # Top/bottom quintile
    LINEAR = "linear"          # Linear mapping


@dataclass
class FactorConfig:
    """Configuration for a factor.
    
    Attributes:
        name: Factor name
        weight: Factor weight in composite score
        direction: 1 for higher is better, -1 for lower is better
        lookback: Lookback period for calculation
    """
    name: str
    weight: float = 1.0
    direction: int = 1
    lookback: int = 12


class FactorStrategy:
    """Factor-based portfolio construction strategy.
    
    Constructs portfolios based on factor scores (value, momentum,
    quality, size, low volatility).
    
    Example:
        >>> strategy = FactorStrategy(factors=["value", "momentum"])
        >>> scores = strategy.calculate_factor_scores(data)
        >>> weights = strategy.generate_weights(scores)
    """
    
    def __init__(
        self,
        factors: list[str] | list[FactorConfig] | None = None,
        weight_method: FactorWeightMethod = FactorWeightMethod.RANK,
        long_only: bool = True,
        top_n: int | None = None,
        top_pct: float | None = None,
    ):
        """Initialize strategy.
        
        Args:
            factors: List of factor names or FactorConfig objects
            weight_method: Method for converting scores to weights
            long_only: Whether to only go long
            top_n: Number of top stocks to include
            top_pct: Top percentage of stocks to include
        """
        self.factor_configs = self._parse_factors(factors)
        self.weight_method = weight_method
        self.long_only = long_only
        self.top_n = top_n
        self.top_pct = top_pct
    
    def _parse_factors(
        self,
        factors: list[str] | list[FactorConfig] | None,
    ) -> list[FactorConfig]:
        """Parse factor inputs into FactorConfig objects."""
        if factors is None:
            return []
        
        configs = []
        for f in factors:
            if isinstance(f, FactorConfig):
                configs.append(f)
            else:
                configs.append(FactorConfig(name=f))
        
        return configs
    
    def calculate_value_score(
        self,
        data: pd.DataFrame,
    ) -> pd.Series:
        """Calculate value factor score (low P/E, high B/M).
        
        Args:
            data: DataFrame with pe_ratio, book_to_market columns
            
        Returns:
            Value scores by asset
        """
        scores = pd.Series(0, index=data.index, dtype=float)
        
        if "pe_ratio" in data.columns:
            # Lower P/E is better
            pe = data["pe_ratio"].replace([np.inf, -np.inf], np.nan)
            pe_rank = pe.rank(ascending=True, pct=True)
            scores += pe_rank
        
        if "book_to_market" in data.columns:
            # Higher B/M is better
            bm_rank = data["book_to_market"].rank(ascending=False, pct=True)
            scores += bm_rank
        
        return scores / 2 if scores.any() else scores
    
    def calculate_momentum_score(
        self,
        returns: pd.DataFrame,
        lookback: int = 12,
        skip_recent: int = 1,
    ) -> pd.Series:
        """Calculate momentum factor score.
        
        Args:
            returns: Historical returns DataFrame
            lookback: Lookback period (months)
            skip_recent: Periods to skip (avoid reversal)
            
        Returns:
            Momentum scores by asset
        """
        if len(returns) < lookback + skip_recent:
            return pd.Series(0, index=returns.columns)
        
        # Cumulative return excluding recent periods
        start_idx = -lookback - skip_recent
        end_idx = -skip_recent if skip_recent > 0 else None
        
        period_returns = returns.iloc[start_idx:end_idx]
        cumulative = (1 + period_returns).prod() - 1
        
        return cumulative.rank(ascending=False, pct=True)
    
    def calculate_quality_score(
        self,
        data: pd.DataFrame,
    ) -> pd.Series:
        """Calculate quality factor score (profitability, stability).
        
        Args:
            data: DataFrame with roe, debt_to_equity, earnings_variability
            
        Returns:
            Quality scores by asset
        """
        scores = pd.Series(0, index=data.index, dtype=float)
        count = 0
        
        if "roe" in data.columns:
            # Higher ROE is better
            roe_rank = data["roe"].rank(ascending=False, pct=True)
            scores += roe_rank
            count += 1
        
        if "debt_to_equity" in data.columns:
            # Lower D/E is better
            de_rank = data["debt_to_equity"].rank(ascending=True, pct=True)
            scores += de_rank
            count += 1
        
        if "earnings_variability" in data.columns:
            # Lower variability is better
            var_rank = data["earnings_variability"].rank(ascending=True, pct=True)
            scores += var_rank
            count += 1
        
        return scores / count if count > 0 else scores
    
    def calculate_low_volatility_score(
        self,
        returns: pd.DataFrame,
        lookback: int = 252,
    ) -> pd.Series:
        """Calculate low volatility factor score.
        
        Args:
            returns: Historical returns DataFrame
            lookback: Lookback period
            
        Returns:
            Low volatility scores by asset
        """
        if len(returns) < lookback:
            lookback = len(returns)
        
        volatility = returns.iloc[-lookback:].std()
        
        # Lower volatility is better
        return volatility.rank(ascending=True, pct=True)
    
    def calculate_size_score(
        self,
        data: pd.DataFrame,
    ) -> pd.Series:
        """Calculate size factor score (small cap premium).

        Args:
            data: DataFrame with market_cap column

        Returns:
            Size scores by asset
        """
        if "market_cap" not in data.columns:
            return pd.Series(0, index=data.index)

        # Smaller is better (small cap premium) - rank descending so smaller gets higher rank
        return data["market_cap"].rank(ascending=False, pct=True)
    
    def calculate_composite_score(
        self,
        factor_scores: dict[str, pd.Series],
        factor_weights: dict[str, float] | None = None,
    ) -> pd.Series:
        """Calculate composite factor score.
        
        Args:
            factor_scores: Dictionary of factor name -> scores
            factor_weights: Optional custom weights
            
        Returns:
            Composite score
        """
        if not factor_scores:
            raise ValueError("No factor scores provided")
        
        # Get weights
        if factor_weights is None:
            weights = {config.name: config.weight 
                       for config in self.factor_configs}
        else:
            weights = factor_weights
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            total_weight = 1
        
        # Calculate weighted composite
        composite = None
        for name, scores in factor_scores.items():
            weight = weights.get(name, 1.0) / total_weight
            
            if composite is None:
                composite = weight * scores
            else:
                composite += weight * scores.reindex(composite.index, fill_value=0)
        
        return composite
    
    def generate_weights(
        self,
        scores: pd.Series,
        current_weights: pd.Series | None = None,
    ) -> pd.Series:
        """Generate portfolio weights from factor scores.
        
        Args:
            scores: Factor scores by asset
            current_weights: Current portfolio weights (for turnover control)
            
        Returns:
            Target weights
        """
        # Filter top assets if specified
        if self.top_n is not None:
            top_assets = scores.nlargest(self.top_n).index
            scores = scores.loc[top_assets]
        elif self.top_pct is not None:
            n = max(1, int(len(scores) * self.top_pct))
            top_assets = scores.nlargest(n).index
            scores = scores.loc[top_assets]
        
        # Convert scores to weights based on method
        if self.weight_method == FactorWeightMethod.RANK:
            # Higher score should get higher weight, so use ascending rank
            weights = scores.rank(ascending=True)
            weights = weights / weights.sum()
        
        elif self.weight_method == FactorWeightMethod.ZSCORE:
            z_scores = (scores - scores.mean()) / scores.std()
            if self.long_only:
                z_scores = z_scores - z_scores.min()
            weights = z_scores / z_scores.sum()
        
        elif self.weight_method == FactorWeightMethod.LINEAR:
            if self.long_only:
                weights = scores - scores.min()
            else:
                weights = scores - scores.mean()
            weights = weights / weights.abs().sum()
        
        else:  # QUINTILE
            # Equal weight top quintile
            top_quintile = scores.nlargest(max(1, len(scores) // 5))
            weights = pd.Series(0, index=scores.index)
            weights.loc[top_quintile.index] = 1 / len(top_quintile)
        
        return weights
    
    def rebalance(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        threshold: float = 0.05,
    ) -> pd.Series:
        """Calculate rebalancing trades.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target weights
            threshold: Minimum difference to trigger rebalance
            
        Returns:
            Trade weights (positive = buy, negative = sell)
        """
        # Align indices
        all_assets = current_weights.index.union(target_weights.index)
        current = current_weights.reindex(all_assets, fill_value=0)
        target = target_weights.reindex(all_assets, fill_value=0)
        
        # Calculate differences
        diff = target - current
        
        # Only rebalance if above threshold
        diff[diff.abs() < threshold] = 0
        
        return diff


class MultiFactorStrategy(FactorStrategy):
    """Multi-factor strategy combining multiple factors.
    
    Supports factor timing and regime-based factor selection.
    """
    
    def __init__(
        self,
        factors: list[FactorConfig],
        weight_method: FactorWeightMethod = FactorWeightMethod.RANK,
        regime_factor_weights: dict[str, dict[str, float]] | None = None,
        **kwargs,
    ):
        """Initialize multi-factor strategy.
        
        Args:
            factors: Factor configurations
            weight_method: Weighting method
            regime_factor_weights: Factor weights by regime
            **kwargs: Additional arguments for base class
        """
        super().__init__(factors=factors, weight_method=weight_method, **kwargs)
        self.regime_factor_weights = regime_factor_weights or {}
    
    def calculate_all_factor_scores(
        self,
        fundamental_data: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> dict[str, pd.Series]:
        """Calculate all factor scores.
        
        Args:
            fundamental_data: Fundamental data DataFrame
            returns: Historical returns
            
        Returns:
            Dictionary of factor name -> scores
        """
        scores = {}
        
        for config in self.factor_configs:
            name = config.name.lower()
            
            if name == "value":
                scores[config.name] = self.calculate_value_score(fundamental_data)
            elif name == "momentum":
                scores[config.name] = self.calculate_momentum_score(
                    returns, lookback=config.lookback
                )
            elif name == "quality":
                scores[config.name] = self.calculate_quality_score(fundamental_data)
            elif name == "low_volatility" or name == "lowvol":
                scores[config.name] = self.calculate_low_volatility_score(returns)
            elif name == "size":
                scores[config.name] = self.calculate_size_score(fundamental_data)
            
            # Apply direction
            if config.direction == -1:
                scores[config.name] = 1 - scores[config.name]
        
        return scores
    
    def get_regime_weights(
        self,
        regime: str,
    ) -> dict[str, float]:
        """Get factor weights for a specific regime.
        
        Args:
            regime: Market regime name
            
        Returns:
            Factor weights dictionary
        """
        if regime in self.regime_factor_weights:
            return self.regime_factor_weights[regime]
        
        # Default: equal weights
        return {config.name: config.weight for config in self.factor_configs}
