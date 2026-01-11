"""Base class for factor models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FactorModelResult:
    """Factor model regression results.
    
    Attributes:
        alpha: Intercept (abnormal return)
        loadings: Factor loadings (betas)
        r_squared: R-squared (goodness of fit)
        adj_r_squared: Adjusted R-squared
        t_stats: t-statistics for coefficients
        p_values: p-values for coefficients
        residuals: Model residuals
        std_errors: Standard errors of coefficients
    """
    alpha: float
    loadings: dict[str, float]
    r_squared: float
    adj_r_squared: float
    t_stats: dict[str, float]
    p_values: dict[str, float]
    residuals: pd.Series | None = None
    std_errors: dict[str, float] | None = None
    
    def summary(self) -> str:
        """Return a summary string of the results."""
        lines = [
            "=" * 50,
            "Factor Model Results",
            "=" * 50,
            f"Alpha (Intercept): {self.alpha:.6f}",
            "",
            "Factor Loadings:",
        ]
        
        for factor, loading in self.loadings.items():
            t_stat = self.t_stats.get(factor, 0)
            p_val = self.p_values.get(factor, 1)
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            lines.append(f"  {factor}: {loading:.4f} (t={t_stat:.2f}, p={p_val:.4f}) {sig}")
        
        lines.extend([
            "",
            f"R-squared: {self.r_squared:.4f}",
            f"Adj. R-squared: {self.adj_r_squared:.4f}",
            "=" * 50,
            "Significance: *** p<0.01, ** p<0.05, * p<0.1",
        ])
        
        return "\n".join(lines)


class FactorModel(ABC):
    """Abstract base class for factor models.
    
    Provides a common interface for CAPM and Fama-French models.
    """
    
    def __init__(self):
        self._result: FactorModelResult | None = None
        self._is_fitted: bool = False
    
    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._is_fitted
    
    @property
    def result(self) -> FactorModelResult:
        """Get model results."""
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        return self._result
    
    @abstractmethod
    def fit(
        self,
        returns: pd.Series,
        factor_returns: pd.DataFrame,
        risk_free_rate: pd.Series | None = None,
    ) -> "FactorModel":
        """Fit the factor model.
        
        Args:
            returns: Asset returns (Series with DatetimeIndex)
            factor_returns: Factor returns (DataFrame with DatetimeIndex)
            risk_free_rate: Risk-free rate (optional)
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def get_factor_names(self) -> list[str]:
        """Get list of factor names used by this model."""
        pass
    
    def get_loadings(self) -> dict[str, float]:
        """Get factor loadings."""
        return self.result.loadings
    
    def get_alpha(self) -> float:
        """Get alpha (abnormal return)."""
        return self.result.alpha
    
    def get_r_squared(self) -> float:
        """Get R-squared."""
        return self.result.r_squared
    
    def predict(self, factor_returns: pd.DataFrame) -> pd.Series:
        """Predict returns given factor returns.
        
        Args:
            factor_returns: Factor returns for prediction
            
        Returns:
            Predicted returns
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        predicted = pd.Series(self.result.alpha, index=factor_returns.index)
        
        for factor, loading in self.result.loadings.items():
            if factor in factor_returns.columns:
                predicted += loading * factor_returns[factor]
        
        return predicted
    
    def summary(self) -> str:
        """Get model summary."""
        return self.result.summary()


def calculate_excess_returns(
    returns: pd.Series,
    risk_free_rate: pd.Series,
) -> pd.Series:
    """Calculate excess returns over risk-free rate.
    
    Args:
        returns: Asset returns
        risk_free_rate: Risk-free rate
        
    Returns:
        Excess returns
    """
    # Align indices
    common_idx = returns.index.intersection(risk_free_rate.index)
    
    if len(common_idx) == 0:
        raise ValueError("No overlapping dates between returns and risk-free rate")
    
    return returns.loc[common_idx] - risk_free_rate.loc[common_idx]


def annualize_returns(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Annualize returns.
    
    Args:
        returns: Periodic returns
        periods_per_year: Number of periods in a year (252 for daily)
        
    Returns:
        Annualized return
    """
    total_return = (1 + returns).prod()
    n_periods = len(returns)
    
    if n_periods == 0:
        return 0.0
    
    return total_return ** (periods_per_year / n_periods) - 1


def annualize_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Annualize volatility.
    
    Args:
        returns: Periodic returns
        periods_per_year: Number of periods in a year
        
    Returns:
        Annualized volatility
    """
    return returns.std() * np.sqrt(periods_per_year)
