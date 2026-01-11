"""Capital Asset Pricing Model (CAPM) implementation."""

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from .base import FactorModel, FactorModelResult, calculate_excess_returns


class CAPM(FactorModel):
    """Capital Asset Pricing Model.
    
    The CAPM describes the relationship between systematic risk and expected
    return for assets. The model formula is:
    
    E(Ri) - Rf = αi + βi × (E(Rm) - Rf) + εi
    
    Where:
        - E(Ri): Expected return of asset i
        - Rf: Risk-free rate
        - αi: Alpha (abnormal return)
        - βi: Beta (systematic risk)
        - E(Rm): Expected market return
        - εi: Error term
    
    Example:
        >>> capm = CAPM()
        >>> capm.fit(stock_returns, market_returns, risk_free_rate)
        >>> print(f"Beta: {capm.get_beta()}")
        >>> print(f"Alpha: {capm.get_alpha()}")
    """
    
    FACTOR_NAME = "Mkt-RF"
    
    def __init__(self):
        super().__init__()
        self._beta: float | None = None
    
    def get_factor_names(self) -> list[str]:
        """Get factor names for CAPM."""
        return [self.FACTOR_NAME]
    
    def fit(
        self,
        returns: pd.Series,
        market_returns: pd.Series | pd.DataFrame,
        risk_free_rate: pd.Series | None = None,
    ) -> "CAPM":
        """Fit the CAPM model.
        
        Args:
            returns: Asset returns (daily/monthly)
            market_returns: Market returns (e.g., S&P 500)
            risk_free_rate: Risk-free rate (e.g., T-Bill rate)
            
        Returns:
            Self for method chaining
        """
        # Handle DataFrame input (extract market column)
        if isinstance(market_returns, pd.DataFrame):
            if self.FACTOR_NAME in market_returns.columns:
                mkt_rf = market_returns[self.FACTOR_NAME]
                # If Mkt-RF is already excess return, we may have RF column
                if "RF" in market_returns.columns and risk_free_rate is None:
                    risk_free_rate = market_returns["RF"]
            else:
                # Assume first column is market returns
                mkt_rf = market_returns.iloc[:, 0]
        else:
            mkt_rf = market_returns
        
        # Align indices
        common_idx = returns.index.intersection(mkt_rf.index)
        
        if risk_free_rate is not None:
            common_idx = common_idx.intersection(risk_free_rate.index)
            rf = risk_free_rate.loc[common_idx]
            excess_returns = returns.loc[common_idx] - rf
            # If market returns are already excess returns (Mkt-RF), use directly
            if isinstance(market_returns, pd.DataFrame) and self.FACTOR_NAME in market_returns.columns:
                market_excess = mkt_rf.loc[common_idx]
            else:
                market_excess = mkt_rf.loc[common_idx] - rf
        else:
            # Assume returns are already excess returns
            excess_returns = returns.loc[common_idx]
            market_excess = mkt_rf.loc[common_idx]
        
        # Remove NaN values
        valid_mask = ~(excess_returns.isna() | market_excess.isna())
        y = excess_returns[valid_mask]
        X = market_excess[valid_mask]
        
        if len(y) < 30:
            raise ValueError(f"Insufficient data points: {len(y)} (minimum 30 required)")
        
        # OLS Regression
        X_with_const = add_constant(X)
        model = OLS(y, X_with_const).fit()
        
        # Extract results
        alpha = model.params.iloc[0]  # const
        beta = model.params.iloc[1]   # market
        
        self._beta = beta
        self._result = FactorModelResult(
            alpha=alpha,
            loadings={self.FACTOR_NAME: beta},
            r_squared=model.rsquared,
            adj_r_squared=model.rsquared_adj,
            t_stats={
                "const": model.tvalues.iloc[0],
                self.FACTOR_NAME: model.tvalues.iloc[1],
            },
            p_values={
                "const": model.pvalues.iloc[0],
                self.FACTOR_NAME: model.pvalues.iloc[1],
            },
            residuals=pd.Series(model.resid, index=y.index),
            std_errors={
                "const": model.bse.iloc[0],
                self.FACTOR_NAME: model.bse.iloc[1],
            },
        )
        self._is_fitted = True
        
        return self
    
    def get_beta(self) -> float:
        """Get market beta."""
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        return self._beta
    
    def calculate_rolling_beta(
        self,
        returns: pd.Series,
        market_returns: pd.Series,
        window: int = 252,
        min_periods: int | None = None,
    ) -> pd.Series:
        """Calculate rolling beta.
        
        Args:
            returns: Asset returns
            market_returns: Market returns
            window: Rolling window size (default: 252 trading days)
            min_periods: Minimum periods for calculation
            
        Returns:
            Rolling beta series
        """
        if min_periods is None:
            min_periods = window // 2
        
        # Align indices
        common_idx = returns.index.intersection(market_returns.index)
        r = returns.loc[common_idx]
        m = market_returns.loc[common_idx]
        
        # Calculate rolling covariance and variance
        rolling_cov = r.rolling(window=window, min_periods=min_periods).cov(m)
        rolling_var = m.rolling(window=window, min_periods=min_periods).var()
        
        return rolling_cov / rolling_var
    
    def expected_return(
        self,
        risk_free_rate: float,
        market_return: float,
    ) -> float:
        """Calculate expected return using CAPM.
        
        Args:
            risk_free_rate: Current risk-free rate
            market_return: Expected market return
            
        Returns:
            Expected asset return
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        market_premium = market_return - risk_free_rate
        return risk_free_rate + self._beta * market_premium
    
    def security_market_line(
        self,
        risk_free_rate: float,
        market_return: float,
        beta_range: tuple[float, float] = (0, 2),
        num_points: int = 100,
    ) -> pd.DataFrame:
        """Generate Security Market Line (SML) data.
        
        Args:
            risk_free_rate: Risk-free rate
            market_return: Expected market return
            beta_range: Range of betas to plot
            num_points: Number of points to generate
            
        Returns:
            DataFrame with beta and expected_return columns
        """
        betas = np.linspace(beta_range[0], beta_range[1], num_points)
        market_premium = market_return - risk_free_rate
        expected_returns = risk_free_rate + betas * market_premium
        
        return pd.DataFrame({
            "beta": betas,
            "expected_return": expected_returns,
        })
