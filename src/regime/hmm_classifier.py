"""Hidden Markov Model based regime classification."""

import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .rule_based import RegimeClassifier, RegimeClassification, MarketRegime

logger = logging.getLogger(__name__)


class HMMClassifier(RegimeClassifier):
    """Hidden Markov Model based regime classifier.
    
    Uses Gaussian HMM to identify latent market regimes from
    economic indicators. The model learns regime-specific
    distributions and transition probabilities.
    
    Features:
    - Automatic state labeling based on regime characteristics
    - Regime transition probability matrix
    - Smoothed state probabilities
    
    Example:
        >>> classifier = HMMClassifier(n_regimes=4)
        >>> classifier.fit(indicators_df)
        >>> current_regime = classifier.classify(indicators_df)
        >>> probabilities = classifier.get_state_probabilities(indicators_df)
    """
    
    def __init__(
        self,
        n_regimes: int = 4,
        covariance_type: Literal["spherical", "diag", "full", "tied"] = "diag",
        n_iter: int = 100,
        random_state: int = 42,
    ):
        """Initialize HMM classifier.
        
        Args:
            n_regimes: Number of hidden states (regimes)
            covariance_type: Type of covariance matrix
            n_iter: Maximum iterations for EM algorithm
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.state_to_regime = {}
        self._is_fitted = False
    
    def _prepare_features(
        self,
        indicators: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> np.ndarray:
        """Prepare features for HMM.
        
        Args:
            indicators: Raw indicators DataFrame
            columns: Columns to use (default: all numeric)
            
        Returns:
            Scaled feature array
        """
        if columns is None:
            columns = indicators.select_dtypes(include=[np.number]).columns.tolist()
        
        # Select and forward-fill missing values
        X = indicators[columns].ffill().bfill()
        
        if self._is_fitted:
            return self.scaler.transform(X)
        else:
            return self.scaler.fit_transform(X)
    
    def _label_states(
        self,
        indicators: pd.DataFrame,
        states: np.ndarray,
    ) -> dict[int, MarketRegime]:
        """Label HMM states with regime names.
        
        Uses indicator values to interpret each hidden state.
        
        Args:
            indicators: Original indicators
            states: Predicted state sequence
            
        Returns:
            Mapping from state index to MarketRegime
        """
        state_characteristics = {}
        
        for state in range(self.n_regimes):
            mask = states == state
            if mask.sum() == 0:
                continue
            
            state_data = indicators[mask]
            
            # Calculate characteristics
            chars = {}
            
            if "gdp_growth" in state_data.columns:
                chars["gdp_growth"] = state_data["gdp_growth"].mean()
            
            if "unemployment" in state_data.columns:
                chars["unemployment"] = state_data["unemployment"].mean()
            
            if "yield_spread" in state_data.columns:
                chars["yield_spread"] = state_data["yield_spread"].mean()
            
            state_characteristics[state] = chars
        
        # Label states based on characteristics
        state_to_regime = {}
        
        for state, chars in state_characteristics.items():
            gdp = chars.get("gdp_growth", 0)
            unemp = chars.get("unemployment", 5)
            spread = chars.get("yield_spread", 1)
            
            # Simple rule-based labeling
            if gdp > 2 and unemp < 5:
                regime = MarketRegime.EXPANSION
            elif gdp < 0 or unemp > 7:
                regime = MarketRegime.CONTRACTION
            elif spread < 0:
                regime = MarketRegime.PEAK
            else:
                regime = MarketRegime.TROUGH
            
            state_to_regime[state] = regime
        
        # Ensure all states have a label
        for state in range(self.n_regimes):
            if state not in state_to_regime:
                state_to_regime[state] = MarketRegime.UNKNOWN
        
        return state_to_regime
    
    def fit(
        self,
        indicators: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> "HMMClassifier":
        """Fit the HMM to historical data.
        
        Args:
            indicators: Historical indicator DataFrame
            columns: Columns to use for fitting
            
        Returns:
            Self for method chaining
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.warning("hmmlearn not installed. Using fallback.")
            self._is_fitted = False
            return self
        
        # Prepare features
        X = self._prepare_features(indicators, columns)
        
        # Remove rows with NaN (after scaling)
        valid_mask = ~np.isnan(X).any(axis=1)
        X_valid = X[valid_mask]
        
        if len(X_valid) < self.n_regimes * 10:
            raise ValueError(f"Insufficient data: {len(X_valid)} samples")
        
        # Fit HMM
        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.model.fit(X_valid)
        
        # Predict states for labeling
        states = self.model.predict(X_valid)
        
        # Label states
        indicators_valid = indicators.iloc[valid_mask]
        self.state_to_regime = self._label_states(indicators_valid, states)
        
        self._is_fitted = True
        logger.info(f"HMM fitted with {self.n_regimes} regimes")
        
        return self
    
    def predict_states(
        self,
        indicators: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> np.ndarray:
        """Predict hidden states.
        
        Args:
            indicators: Indicator DataFrame
            columns: Columns to use
            
        Returns:
            Array of predicted states
        """
        if not self._is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = self._prepare_features(indicators, columns)
        return self.model.predict(X)
    
    def get_state_probabilities(
        self,
        indicators: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Get probability of each state over time.
        
        Args:
            indicators: Indicator DataFrame
            columns: Columns to use
            
        Returns:
            DataFrame with state probabilities
        """
        if not self._is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = self._prepare_features(indicators, columns)
        probs = self.model.predict_proba(X)
        
        # Create DataFrame with regime names
        columns = [self.state_to_regime.get(i, MarketRegime.UNKNOWN).value 
                   for i in range(self.n_regimes)]
        
        return pd.DataFrame(probs, index=indicators.index, columns=columns)
    
    def get_transition_matrix(self) -> pd.DataFrame:
        """Get regime transition probability matrix.
        
        Returns:
            DataFrame with transition probabilities
        """
        if not self._is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        regime_names = [self.state_to_regime.get(i, MarketRegime.UNKNOWN).value 
                        for i in range(self.n_regimes)]
        
        return pd.DataFrame(
            self.model.transmat_,
            index=regime_names,
            columns=regime_names,
        )
    
    def classify(
        self,
        indicators: pd.DataFrame,
        date: pd.Timestamp | None = None,
    ) -> RegimeClassification:
        """Classify regime at specific date.
        
        Args:
            indicators: Indicator DataFrame
            date: Date to classify
            
        Returns:
            RegimeClassification result
        """
        if not self._is_fitted:
            return RegimeClassification(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                indicators={},
                timestamp=date or indicators.index[-1],
                details="Model not fitted",
            )
        
        if date is None:
            date = indicators.index[-1]
        
        # Get probabilities at date
        probs = self.get_state_probabilities(indicators)
        
        if date not in probs.index:
            # Find nearest date
            idx = probs.index.get_indexer([date], method="ffill")[0]
            if idx < 0:
                return RegimeClassification(
                    regime=MarketRegime.UNKNOWN,
                    confidence=0.0,
                    indicators={},
                    timestamp=date,
                    details="Date not in range",
                )
            date = probs.index[idx]
        
        # Get regime with highest probability
        date_probs = probs.loc[date]
        best_regime_name = date_probs.idxmax()
        confidence = date_probs.max()
        
        # Convert to MarketRegime enum
        regime = MarketRegime(best_regime_name)
        
        # Get indicator values at date
        indicator_values = {}
        for col in indicators.columns:
            if date in indicators.index:
                val = indicators.loc[date, col]
                if pd.notna(val):
                    indicator_values[col] = float(val)
        
        return RegimeClassification(
            regime=regime,
            confidence=confidence,
            indicators=indicator_values,
            timestamp=date,
            details=f"State probabilities: {date_probs.to_dict()}",
        )
    
    def classify_history(
        self,
        indicators: pd.DataFrame,
    ) -> pd.DataFrame:
        """Classify regime for entire history.
        
        Args:
            indicators: Indicator DataFrame
            
        Returns:
            DataFrame with regime and confidence
        """
        if not self._is_fitted:
            return pd.DataFrame(
                {"regime": MarketRegime.UNKNOWN.value, "confidence": 0.0},
                index=indicators.index,
            )
        
        probs = self.get_state_probabilities(indicators)
        
        results = pd.DataFrame(index=indicators.index)
        results["regime"] = probs.idxmax(axis=1)
        results["confidence"] = probs.max(axis=1)
        
        return results
    
    def get_expected_duration(self) -> dict[str, float]:
        """Get expected duration in each regime.
        
        Based on transition matrix diagonal elements.
        
        Returns:
            Dict mapping regime to expected duration (periods)
        """
        if not self._is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        durations = {}
        
        for i in range(self.n_regimes):
            regime = self.state_to_regime.get(i, MarketRegime.UNKNOWN)
            # Expected duration = 1 / (1 - p_ii)
            p_stay = self.model.transmat_[i, i]
            if p_stay < 1:
                duration = 1 / (1 - p_stay)
            else:
                duration = float("inf")
            
            durations[regime.value] = duration
        
        return durations
