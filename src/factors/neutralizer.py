"""Factor neutralization for portfolio construction."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds
from typing import Literal


class FactorNeutralizer:
    """Factor neutralization for portfolio construction.
    
    Provides methods to construct portfolios with zero (or target)
    exposure to specified factors. Useful for:
    - Market-neutral strategies (beta = 0)
    - Factor-neutral strategies (all factor loadings = 0)
    - Targeted factor tilts
    
    Example:
        >>> neutralizer = FactorNeutralizer()
        >>> 
        >>> # Make portfolio market-neutral
        >>> neutral_weights = neutralizer.neutralize_single_factor(
        ...     weights=initial_weights,
        ...     factor_loadings=stock_betas,
        ...     target_loading=0.0
        ... )
        >>> 
        >>> # Multi-factor neutralization
        >>> neutral_weights = neutralizer.neutralize_multi_factor(
        ...     weights=initial_weights,
        ...     factor_loadings_df=factor_loadings,
        ...     target_loadings={"Mkt-RF": 0, "SMB": 0}
        ... )
    """
    
    def __init__(
        self,
        allow_short: bool = True,
        max_position: float = 1.0,
        min_position: float = -1.0,
    ):
        """Initialize neutralizer.
        
        Args:
            allow_short: Whether to allow short positions
            max_position: Maximum position size (as fraction)
            min_position: Minimum position size (use negative for shorts)
        """
        self.allow_short = allow_short
        self.max_position = max_position
        self.min_position = min_position if allow_short else 0.0
    
    def neutralize_single_factor(
        self,
        weights: pd.Series,
        factor_loadings: pd.Series,
        target_loading: float = 0.0,
    ) -> pd.Series:
        """Neutralize portfolio to a single factor.
        
        Adjusts weights to achieve target factor loading while
        minimizing deviation from original weights.
        
        Args:
            weights: Original portfolio weights (asset -> weight)
            factor_loadings: Factor loadings for each asset (e.g., betas)
            target_loading: Target factor loading (0 for neutral)
            
        Returns:
            Adjusted portfolio weights
        """
        # Align assets
        common_assets = weights.index.intersection(factor_loadings.index)
        w = weights.loc[common_assets].values
        b = factor_loadings.loc[common_assets].values
        
        n = len(common_assets)
        
        # Current portfolio loading
        current_loading = np.sum(w * b)
        
        # If already at target, return original weights
        if abs(current_loading - target_loading) < 1e-8:
            return weights.loc[common_assets]
        
        # Optimization: minimize ||w_new - w||^2
        # subject to: sum(w_new * b) = target_loading
        #            sum(w_new) = 1
        #            min_position <= w_new <= max_position

        def objective(w_new):
            return np.sum((w_new - w) ** 2)

        def objective_grad(w_new):
            return 2 * (w_new - w)

        # Build constraint matrix: [ones; b] @ w = [1; target]
        A_eq = np.vstack([np.ones(n), b])
        b_eq = np.array([1.0, target_loading])

        linear_constraint = LinearConstraint(A_eq, b_eq, b_eq)
        bounds = Bounds(
            lb=np.full(n, self.min_position),
            ub=np.full(n, self.max_position),
        )

        result = minimize(
            objective,
            w,
            method="trust-constr",
            jac=objective_grad,
            constraints=linear_constraint,
            bounds=bounds,
            options={"maxiter": 1000, "gtol": 1e-8},
        )

        # Status 1 = iteration limit, 2 = constraint violation (may be close enough)
        if not result.success and result.status not in (1, 2):
            raise ValueError(f"Optimization failed: {result.message}")

        # For status 2 (constraint violation), check if solution is acceptable
        if result.status == 2:
            # Verify constraints are approximately satisfied
            actual_sum = np.sum(result.x)
            actual_loading = np.sum(result.x * b)
            if abs(actual_sum - 1.0) > 0.1 or abs(actual_loading - target_loading) > 0.1:
                raise ValueError(f"Optimization failed: {result.message}")

        return pd.Series(result.x, index=common_assets)

    def neutralize_multi_factor(
        self,
        weights: pd.Series,
        factor_loadings_df: pd.DataFrame,
        target_loadings: dict[str, float] | None = None,
    ) -> pd.Series:
        """Neutralize portfolio to multiple factors.
        
        Args:
            weights: Original portfolio weights
            factor_loadings_df: DataFrame with factor loadings
                (assets as index, factors as columns)
            target_loadings: Target loading for each factor
                (default: 0 for all factors)
                
        Returns:
            Adjusted portfolio weights
        """
        if target_loadings is None:
            target_loadings = {col: 0.0 for col in factor_loadings_df.columns}
        
        # Align assets
        common_assets = weights.index.intersection(factor_loadings_df.index)
        w = weights.loc[common_assets].values
        loadings = factor_loadings_df.loc[common_assets]
        
        n = len(common_assets)

        def objective(w_new):
            return np.sum((w_new - w) ** 2)

        def objective_grad(w_new):
            return 2 * (w_new - w)

        # Build constraint matrix: first row is sum=1, then factor constraints
        A_rows = [np.ones(n)]
        b_targets = [1.0]

        for factor, target in target_loadings.items():
            if factor in loadings.columns:
                A_rows.append(loadings[factor].values)
                b_targets.append(target)

        A_eq = np.vstack(A_rows)
        b_eq = np.array(b_targets)

        linear_constraint = LinearConstraint(A_eq, b_eq, b_eq)
        bounds = Bounds(
            lb=np.full(n, self.min_position),
            ub=np.full(n, self.max_position),
        )

        result = minimize(
            objective,
            w,
            method="trust-constr",
            jac=objective_grad,
            constraints=linear_constraint,
            bounds=bounds,
            options={"maxiter": 1000, "gtol": 1e-8},
        )

        # Status 1 = iteration limit, 2 = constraint violation (may be close enough)
        if not result.success and result.status not in (1, 2):
            raise ValueError(f"Optimization failed: {result.message}")

        # For status 2 (constraint violation), check if solution is acceptable
        if result.status == 2:
            actual_sum = np.sum(result.x)
            if abs(actual_sum - 1.0) > 0.1:
                raise ValueError(f"Optimization failed: {result.message}")
            # Check factor constraints
            for factor, target in target_loadings.items():
                if factor in loadings.columns:
                    actual = np.sum(result.x * loadings[factor].values)
                    if abs(actual - target) > 0.1:
                        raise ValueError(f"Optimization failed: {result.message}")

        return pd.Series(result.x, index=common_assets)

    def create_long_short_portfolio(
        self,
        scores: pd.Series,
        factor_loadings: pd.Series | pd.DataFrame | None = None,
        target_loadings: dict[str, float] | None = None,
        long_pct: float = 0.2,
        short_pct: float = 0.2,
        method: Literal["equal", "score"] = "equal",
    ) -> pd.Series:
        """Create a long-short portfolio from scores.
        
        Args:
            scores: Asset scores (higher = more attractive)
            factor_loadings: Factor loadings for neutralization (optional)
            target_loadings: Target factor loadings (optional)
            long_pct: Percentage of assets to go long
            short_pct: Percentage of assets to go short
            method: Weighting method
                - "equal": Equal weights within long/short groups
                - "score": Score-proportional weights
                
        Returns:
            Long-short portfolio weights (sum to 0)
        """
        n = len(scores)
        n_long = max(1, int(n * long_pct))
        n_short = max(1, int(n * short_pct))
        
        # Rank assets
        ranked = scores.sort_values(ascending=False)
        
        # Long top, short bottom
        long_assets = ranked.head(n_long).index
        short_assets = ranked.tail(n_short).index
        
        weights = pd.Series(0.0, index=scores.index)
        
        if method == "equal":
            weights.loc[long_assets] = 1.0 / n_long
            weights.loc[short_assets] = -1.0 / n_short
        elif method == "score":
            # Score-proportional weights
            long_scores = scores.loc[long_assets]
            short_scores = scores.loc[short_assets]
            
            # Normalize to sum to 1 (long) and -1 (short)
            weights.loc[long_assets] = long_scores / long_scores.sum()
            weights.loc[short_assets] = -short_scores / short_scores.sum()
        
        # Normalize to dollar-neutral (sum = 0)
        long_sum = weights[weights > 0].sum()
        short_sum = abs(weights[weights < 0].sum())
        target_sum = (long_sum + short_sum) / 2
        
        if long_sum > 0:
            weights[weights > 0] *= target_sum / long_sum
        if short_sum > 0:
            weights[weights < 0] *= target_sum / short_sum
        
        # Apply factor neutralization if provided
        if factor_loadings is not None:
            if isinstance(factor_loadings, pd.Series):
                weights = self.neutralize_single_factor(
                    weights, factor_loadings, target_loading=0.0
                )
            else:
                weights = self.neutralize_multi_factor(
                    weights, factor_loadings, target_loadings or {}
                )
        
        return weights
    
    def calculate_portfolio_loadings(
        self,
        weights: pd.Series,
        factor_loadings_df: pd.DataFrame,
    ) -> dict[str, float]:
        """Calculate portfolio factor loadings.
        
        Args:
            weights: Portfolio weights
            factor_loadings_df: Factor loadings for each asset
            
        Returns:
            Dictionary of portfolio factor loadings
        """
        common_assets = weights.index.intersection(factor_loadings_df.index)
        w = weights.loc[common_assets]
        loadings = factor_loadings_df.loc[common_assets]
        
        portfolio_loadings = {}
        for factor in loadings.columns:
            portfolio_loadings[factor] = np.sum(w * loadings[factor])
        
        return portfolio_loadings
    
    def verify_neutralization(
        self,
        weights: pd.Series,
        factor_loadings: pd.Series | pd.DataFrame,
        tolerance: float = 1e-4,
    ) -> dict[str, bool]:
        """Verify that portfolio is factor-neutral.
        
        Args:
            weights: Portfolio weights
            factor_loadings: Factor loadings
            tolerance: Tolerance for considering neutral
            
        Returns:
            Dictionary with verification results
        """
        if isinstance(factor_loadings, pd.Series):
            factor_loadings = factor_loadings.to_frame("factor")
        
        loadings = self.calculate_portfolio_loadings(weights, factor_loadings)
        
        results = {}
        for factor, loading in loadings.items():
            results[f"{factor}_loading"] = loading
            results[f"{factor}_neutral"] = abs(loading) < tolerance
        
        results["all_neutral"] = all(
            v for k, v in results.items() if k.endswith("_neutral")
        )
        
        return results
