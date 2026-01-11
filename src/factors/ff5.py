"""Fama-French 5 Factor Model implementation."""

import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from .base import FactorModel, FactorModelResult


class FamaFrench5(FactorModel):
    """Fama-French 5 Factor Model.
    
    The FF5 model extends FF3 by adding profitability and investment factors:
    
    Ri - Rf = αi + βi(Rm-Rf) + si(SMB) + hi(HML) + ri(RMW) + ci(CMA) + εi
    
    Where:
        - Ri - Rf: Excess return of asset i
        - Rm - Rf: Market excess return (Mkt-RF)
        - SMB: Small Minus Big (size factor)
        - HML: High Minus Low (value factor)
        - RMW: Robust Minus Weak (profitability factor)
        - CMA: Conservative Minus Aggressive (investment factor)
        - αi: Alpha (abnormal return)
        - βi, si, hi, ri, ci: Factor loadings
    
    Example:
        >>> from src.factors import FamaFrench5, FamaFrenchDataLoader
        >>> loader = FamaFrenchDataLoader()
        >>> ff5_factors = loader.load_ff5_factors(frequency="monthly")
        >>> 
        >>> model = FamaFrench5()
        >>> model.fit(stock_returns, ff5_factors)
        >>> print(model.summary())
    """
    
    FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    
    def get_factor_names(self) -> list[str]:
        """Get factor names for FF5 model."""
        return self.FACTORS.copy()
    
    def fit(
        self,
        returns: pd.Series,
        factor_returns: pd.DataFrame,
        risk_free_rate: pd.Series | None = None,
    ) -> "FamaFrench5":
        """Fit the Fama-French 5 Factor model.
        
        Args:
            returns: Asset returns (with DatetimeIndex)
            factor_returns: FF5 factor returns DataFrame
                Should contain columns: Mkt-RF, SMB, HML, RMW, CMA, (RF optional)
            risk_free_rate: Risk-free rate (optional if RF in factor_returns)
            
        Returns:
            Self for method chaining
        """
        # Validate factor columns
        missing_factors = set(self.FACTORS) - set(factor_returns.columns)
        if missing_factors:
            raise ValueError(f"Missing factors in factor_returns: {missing_factors}")
        
        # Get risk-free rate
        if risk_free_rate is None and "RF" in factor_returns.columns:
            risk_free_rate = factor_returns["RF"]
        
        # Align indices
        common_idx = returns.index.intersection(factor_returns.index)
        
        if risk_free_rate is not None:
            common_idx = common_idx.intersection(risk_free_rate.index)
            rf = risk_free_rate.loc[common_idx]
            excess_returns = returns.loc[common_idx] - rf
        else:
            # Assume returns are already excess returns
            excess_returns = returns.loc[common_idx]
        
        factors = factor_returns.loc[common_idx, self.FACTORS]
        
        # Remove NaN values
        valid_mask = ~(excess_returns.isna() | factors.isna().any(axis=1))
        y = excess_returns[valid_mask]
        X = factors[valid_mask]
        
        if len(y) < 30:
            raise ValueError(f"Insufficient data points: {len(y)} (minimum 30 required)")
        
        # OLS Regression
        X_with_const = add_constant(X)
        model = OLS(y, X_with_const).fit()
        
        # Extract results
        loadings = {}
        t_stats = {"const": model.tvalues["const"]}
        p_values = {"const": model.pvalues["const"]}
        std_errors = {"const": model.bse["const"]}
        
        for factor in self.FACTORS:
            loadings[factor] = model.params[factor]
            t_stats[factor] = model.tvalues[factor]
            p_values[factor] = model.pvalues[factor]
            std_errors[factor] = model.bse[factor]
        
        self._result = FactorModelResult(
            alpha=model.params["const"],
            loadings=loadings,
            r_squared=model.rsquared,
            adj_r_squared=model.rsquared_adj,
            t_stats=t_stats,
            p_values=p_values,
            residuals=pd.Series(model.resid, index=y.index),
            std_errors=std_errors,
        )
        self._is_fitted = True
        
        return self
    
    def get_market_beta(self) -> float:
        """Get market beta (loading on Mkt-RF)."""
        return self.result.loadings["Mkt-RF"]
    
    def get_size_loading(self) -> float:
        """Get size loading (loading on SMB)."""
        return self.result.loadings["SMB"]
    
    def get_value_loading(self) -> float:
        """Get value loading (loading on HML)."""
        return self.result.loadings["HML"]
    
    def get_profitability_loading(self) -> float:
        """Get profitability loading (loading on RMW)."""
        return self.result.loadings["RMW"]
    
    def get_investment_loading(self) -> float:
        """Get investment loading (loading on CMA)."""
        return self.result.loadings["CMA"]
    
    def decompose_risk(self) -> dict[str, float]:
        """Decompose total risk into factor contributions.
        
        Returns:
            Dictionary with risk decomposition percentages
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Total variance = factor variance + idiosyncratic variance
        residual_var = self.result.residuals.var()
        total_var = residual_var / (1 - self.result.r_squared)
        
        # Factor contribution = R² (explained variance)
        factor_var = total_var * self.result.r_squared
        
        return {
            "total_variance": total_var,
            "factor_variance": factor_var,
            "idiosyncratic_variance": residual_var,
            "systematic_risk_pct": self.result.r_squared * 100,
            "idiosyncratic_risk_pct": (1 - self.result.r_squared) * 100,
        }
    
    def compare_with_ff3(
        self,
        returns: pd.Series,
        factor_returns: pd.DataFrame,
        risk_free_rate: pd.Series | None = None,
    ) -> dict:
        """Compare FF5 results with FF3 model.
        
        Args:
            returns: Asset returns
            factor_returns: Factor returns (must contain all FF5 factors)
            risk_free_rate: Risk-free rate (optional)
            
        Returns:
            Comparison dictionary
        """
        from .ff3 import FamaFrench3
        
        # Fit FF3 model
        ff3 = FamaFrench3()
        ff3.fit(returns, factor_returns, risk_free_rate)
        
        # Fit FF5 model if not already fitted
        if not self._is_fitted:
            self.fit(returns, factor_returns, risk_free_rate)
        
        return {
            "ff3_r_squared": ff3.result.r_squared,
            "ff5_r_squared": self.result.r_squared,
            "r_squared_improvement": self.result.r_squared - ff3.result.r_squared,
            "ff3_alpha": ff3.result.alpha,
            "ff5_alpha": self.result.alpha,
            "ff3_alpha_pvalue": ff3.result.p_values["const"],
            "ff5_alpha_pvalue": self.result.p_values["const"],
            "rmw_loading": self.result.loadings["RMW"],
            "rmw_pvalue": self.result.p_values["RMW"],
            "cma_loading": self.result.loadings["CMA"],
            "cma_pvalue": self.result.p_values["CMA"],
        }
