"""Portfolio optimization algorithms."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class OptimizationConstraints:
    """Constraints for portfolio optimization.
    
    Attributes:
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset
        long_only: Whether to allow only long positions
        target_return: Target portfolio return (for mean-variance)
        max_turnover: Maximum turnover allowed
    """
    min_weight: float = 0.0
    max_weight: float = 1.0
    long_only: bool = True
    target_return: float | None = None
    max_turnover: float | None = None


@dataclass
class OptimizationResult:
    """Result of portfolio optimization.
    
    Attributes:
        weights: Optimal portfolio weights
        expected_return: Expected portfolio return
        volatility: Portfolio volatility (std dev)
        sharpe_ratio: Sharpe ratio
        success: Whether optimization succeeded
        message: Status message
    """
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float
    success: bool
    message: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "weights": self.weights.to_dict(),
            "expected_return": self.expected_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "success": self.success,
            "message": self.message,
        }


class PortfolioOptimizer:
    """Portfolio optimization using various methods.
    
    Supports:
    - Mean-Variance Optimization (Markowitz)
    - Minimum Variance Portfolio
    - Maximum Sharpe Ratio Portfolio
    - Risk Parity Portfolio
    - Equal Weight Portfolio
    
    Example:
        >>> optimizer = PortfolioOptimizer()
        >>> result = optimizer.maximum_sharpe(returns, rf=0.02)
        >>> print(f"Optimal weights: {result.weights}")
        >>> print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    """
    
    def __init__(
        self,
        constraints: OptimizationConstraints | None = None,
        risk_free_rate: float = 0.0,
    ):
        """Initialize optimizer.
        
        Args:
            constraints: Optimization constraints
            risk_free_rate: Annual risk-free rate
        """
        self.constraints = constraints or OptimizationConstraints()
        self.risk_free_rate = risk_free_rate
    
    def _calculate_portfolio_stats(
        self,
        weights: np.ndarray,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        periods_per_year: int = 252,
    ) -> tuple[float, float, float]:
        """Calculate portfolio return, volatility, and Sharpe ratio.
        
        Args:
            weights: Portfolio weights
            mean_returns: Mean returns vector
            cov_matrix: Covariance matrix
            periods_per_year: Number of periods per year
            
        Returns:
            Tuple of (return, volatility, sharpe_ratio)
        """
        portfolio_return = np.sum(mean_returns * weights) * periods_per_year
        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(cov_matrix, weights)) * periods_per_year
        )
        
        if portfolio_volatility > 0:
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        else:
            sharpe = 0.0
        
        return portfolio_return, portfolio_volatility, sharpe
    
    def _get_bounds(self, n_assets: int) -> list[tuple[float, float]]:
        """Get weight bounds for optimization."""
        if self.constraints.long_only:
            min_w = max(0.0, self.constraints.min_weight)
        else:
            min_w = self.constraints.min_weight
        
        return [(min_w, self.constraints.max_weight)] * n_assets
    
    def mean_variance(
        self,
        returns: pd.DataFrame,
        target_return: float | None = None,
    ) -> OptimizationResult:
        """Mean-Variance Optimization (Markowitz).
        
        Minimize portfolio variance subject to target return constraint.
        
        Args:
            returns: Historical returns DataFrame
            target_return: Target annualized return
            
        Returns:
            OptimizationResult with optimal weights
        """
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        n_assets = len(returns.columns)
        
        target = target_return or self.constraints.target_return
        if target is None:
            # Default to mean of mean returns
            target = np.mean(mean_returns) * 252
        
        # Objective: minimize variance
        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # Weights sum to 1
        ]
        
        # Return constraint (annualized)
        constraints.append({
            "type": "eq",
            "fun": lambda x: np.sum(mean_returns * x) * 252 - target
        })
        
        bounds = self._get_bounds(n_assets)
        
        # Initial guess: equal weight
        x0 = np.ones(n_assets) / n_assets
        
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        
        weights = pd.Series(result.x, index=returns.columns)
        ret, vol, sharpe = self._calculate_portfolio_stats(
            result.x, mean_returns, cov_matrix
        )
        
        return OptimizationResult(
            weights=weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            success=result.success,
            message=result.message,
        )
    
    def minimum_variance(
        self,
        returns: pd.DataFrame,
    ) -> OptimizationResult:
        """Minimum Variance Portfolio.
        
        Find portfolio with lowest possible variance.
        
        Args:
            returns: Historical returns DataFrame
            
        Returns:
            OptimizationResult with optimal weights
        """
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        n_assets = len(returns.columns)
        
        # Objective: minimize variance
        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        ]
        
        bounds = self._get_bounds(n_assets)
        x0 = np.ones(n_assets) / n_assets
        
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        
        weights = pd.Series(result.x, index=returns.columns)
        ret, vol, sharpe = self._calculate_portfolio_stats(
            result.x, mean_returns, cov_matrix
        )
        
        return OptimizationResult(
            weights=weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            success=result.success,
            message=result.message,
        )
    
    def maximum_sharpe(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float | None = None,
    ) -> OptimizationResult:
        """Maximum Sharpe Ratio Portfolio.
        
        Find portfolio with highest Sharpe ratio.
        
        Args:
            returns: Historical returns DataFrame
            risk_free_rate: Risk-free rate (overrides default)
            
        Returns:
            OptimizationResult with optimal weights
        """
        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        n_assets = len(returns.columns)
        
        # Objective: minimize negative Sharpe ratio
        def objective(weights):
            ret = np.sum(mean_returns * weights) * 252
            vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)) * 252)
            if vol == 0:
                return 0
            return -(ret - rf) / vol
        
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        ]
        
        bounds = self._get_bounds(n_assets)
        x0 = np.ones(n_assets) / n_assets
        
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        
        weights = pd.Series(result.x, index=returns.columns)
        ret, vol, sharpe = self._calculate_portfolio_stats(
            result.x, mean_returns, cov_matrix
        )
        
        return OptimizationResult(
            weights=weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            success=result.success,
            message=result.message,
        )
    
    def risk_parity(
        self,
        returns: pd.DataFrame,
    ) -> OptimizationResult:
        """Risk Parity Portfolio.
        
        Equal risk contribution from each asset.
        
        Args:
            returns: Historical returns DataFrame
            
        Returns:
            OptimizationResult with optimal weights
        """
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        n_assets = len(returns.columns)
        
        # Objective: minimize squared differences in risk contributions
        def objective(weights):
            weights = np.array(weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            if portfolio_vol == 0:
                return 1e10
            
            # Marginal risk contribution
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            
            # Risk contribution
            risk_contrib = weights * marginal_contrib
            
            # Target: equal risk contribution
            target_contrib = portfolio_vol / n_assets
            
            # Sum of squared differences
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        ]
        
        bounds = self._get_bounds(n_assets)
        x0 = np.ones(n_assets) / n_assets
        
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        
        weights = pd.Series(result.x, index=returns.columns)
        ret, vol, sharpe = self._calculate_portfolio_stats(
            result.x, mean_returns, cov_matrix
        )
        
        return OptimizationResult(
            weights=weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            success=result.success,
            message=result.message,
        )
    
    def equal_weight(
        self,
        returns: pd.DataFrame,
    ) -> OptimizationResult:
        """Equal Weight Portfolio (1/N).
        
        Args:
            returns: Historical returns DataFrame
            
        Returns:
            OptimizationResult with equal weights
        """
        n_assets = len(returns.columns)
        weights_arr = np.ones(n_assets) / n_assets
        
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        
        weights = pd.Series(weights_arr, index=returns.columns)
        ret, vol, sharpe = self._calculate_portfolio_stats(
            weights_arr, mean_returns, cov_matrix
        )
        
        return OptimizationResult(
            weights=weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            success=True,
            message="Equal weight portfolio",
        )
    
    def inverse_volatility(
        self,
        returns: pd.DataFrame,
    ) -> OptimizationResult:
        """Inverse Volatility Weighting.
        
        Weight inversely proportional to volatility.
        
        Args:
            returns: Historical returns DataFrame
            
        Returns:
            OptimizationResult with inverse volatility weights
        """
        volatilities = returns.std()
        inv_vol = 1 / volatilities
        weights_arr = (inv_vol / inv_vol.sum()).values
        
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        
        weights = pd.Series(weights_arr, index=returns.columns)
        ret, vol, sharpe = self._calculate_portfolio_stats(
            weights_arr, mean_returns, cov_matrix
        )
        
        return OptimizationResult(
            weights=weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            success=True,
            message="Inverse volatility portfolio",
        )
    
    def efficient_frontier(
        self,
        returns: pd.DataFrame,
        n_points: int = 50,
    ) -> pd.DataFrame:
        """Generate efficient frontier.
        
        Args:
            returns: Historical returns DataFrame
            n_points: Number of points on frontier
            
        Returns:
            DataFrame with return, volatility, sharpe for each point
        """
        mean_returns = returns.mean() * 252
        
        # Range of target returns
        min_ret = mean_returns.min()
        max_ret = mean_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        results = []
        for target in target_returns:
            try:
                result = self.mean_variance(returns, target_return=target)
                if result.success:
                    results.append({
                        "return": result.expected_return,
                        "volatility": result.volatility,
                        "sharpe": result.sharpe_ratio,
                    })
            except Exception:
                continue
        
        return pd.DataFrame(results)
