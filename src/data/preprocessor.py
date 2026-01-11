"""Data preprocessing pipeline for quantitative analysis."""

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Data preprocessing utilities for financial time series.
    
    Handles common data quality issues in quantitative analysis:
    - Missing values
    - Outliers
    - Returns calculation
    - Normalization/Standardization
    - Corporate actions adjustment
    """

    def __init__(self):
        pass

    # ==================== Missing Values ====================

    def handle_missing(
        self,
        df: pd.DataFrame,
        method: Literal["ffill", "bfill", "interpolate", "drop", "mean", "median"] = "ffill",
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Handle missing values in DataFrame.
        
        Args:
            df: Input DataFrame
            method: Method to handle missing values
                - 'ffill': Forward fill
                - 'bfill': Backward fill
                - 'interpolate': Linear interpolation
                - 'drop': Drop rows with any missing values
                - 'mean': Fill with column mean
                - 'median': Fill with column median
            limit: Maximum number of consecutive NaNs to fill
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        if method == "ffill":
            df = df.ffill(limit=limit)
        elif method == "bfill":
            df = df.bfill(limit=limit)
        elif method == "interpolate":
            df = df.interpolate(method="linear", limit=limit)
        elif method == "drop":
            df = df.dropna()
        elif method == "mean":
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].fillna(df[col].mean())
        elif method == "median":
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].fillna(df[col].median())
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return df

    def fill_missing_prices(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        price_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fill missing prices with appropriate methods.
        
        Uses forward fill for OHLC prices (assumes no trading on missing days).
        
        Args:
            df: DataFrame with price data
            date_col: Name of date column
            price_cols: List of price columns to fill
            
        Returns:
            DataFrame with filled prices
        """
        df = df.copy()
        
        if price_cols is None:
            price_cols = ["open", "high", "low", "close", "adj_close"]
        
        existing_cols = [c for c in price_cols if c in df.columns]
        
        for col in existing_cols:
            df[col] = df[col].ffill()
        
        # Volume should be 0 on non-trading days, not forward filled
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0)
        
        return df

    # ==================== Outlier Detection ====================

    def detect_outliers_zscore(
        self,
        series: pd.Series,
        threshold: float = 3.0,
    ) -> pd.Series:
        """Detect outliers using Z-score method.
        
        Args:
            series: Input series
            threshold: Z-score threshold (default 3.0)
            
        Returns:
            Boolean series indicating outliers
        """
        z_scores = np.abs(stats.zscore(series.dropna()))
        outliers = pd.Series(False, index=series.index)
        outliers.loc[series.dropna().index] = z_scores > threshold
        return outliers

    def detect_outliers_iqr(
        self,
        series: pd.Series,
        k: float = 1.5,
    ) -> pd.Series:
        """Detect outliers using IQR method.
        
        Args:
            series: Input series
            k: IQR multiplier (default 1.5)
            
        Returns:
            Boolean series indicating outliers
        """
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        return (series < lower_bound) | (series > upper_bound)

    def detect_outliers_mad(
        self,
        series: pd.Series,
        threshold: float = 3.5,
    ) -> pd.Series:
        """Detect outliers using Median Absolute Deviation (MAD).
        
        More robust than Z-score for non-normal distributions.
        
        Args:
            series: Input series
            threshold: MAD threshold
            
        Returns:
            Boolean series indicating outliers
        """
        median = series.median()
        mad = np.median(np.abs(series - median))
        if mad == 0:
            return pd.Series(False, index=series.index)
        modified_z_scores = 0.6745 * (series - median) / mad
        return np.abs(modified_z_scores) > threshold

    def handle_outliers(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        method: Literal["zscore", "iqr", "mad"] = "zscore",
        action: Literal["remove", "clip", "nan"] = "clip",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Handle outliers in DataFrame.
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            method: Detection method ('zscore', 'iqr', 'mad')
            action: How to handle outliers
                - 'remove': Remove rows with outliers
                - 'clip': Clip values to bounds
                - 'nan': Replace outliers with NaN
            **kwargs: Additional arguments for detection method
            
        Returns:
            DataFrame with outliers handled
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        detect_func = {
            "zscore": self.detect_outliers_zscore,
            "iqr": self.detect_outliers_iqr,
            "mad": self.detect_outliers_mad,
        }[method]
        
        for col in columns:
            if col not in df.columns:
                continue
                
            outliers = detect_func(df[col], **kwargs)
            
            if action == "remove":
                df = df[~outliers]
            elif action == "clip":
                if method == "iqr":
                    k = kwargs.get("k", 1.5)
                    q1, q3 = df[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower = q1 - k * iqr
                    upper = q3 + k * iqr
                else:
                    lower = df[col][~outliers].min()
                    upper = df[col][~outliers].max()
                df[col] = df[col].clip(lower=lower, upper=upper)
            elif action == "nan":
                df.loc[outliers, col] = np.nan
        
        return df

    # ==================== Returns Calculation ====================

    def calculate_returns(
        self,
        prices: pd.Series | pd.DataFrame,
        method: Literal["simple", "log"] = "simple",
        periods: int = 1,
    ) -> pd.Series | pd.DataFrame:
        """Calculate returns from prices.
        
        Args:
            prices: Price series or DataFrame
            method: 'simple' for arithmetic returns, 'log' for logarithmic
            periods: Number of periods for return calculation
            
        Returns:
            Returns series or DataFrame
        """
        if method == "simple":
            returns = prices.pct_change(periods=periods)
        elif method == "log":
            returns = np.log(prices / prices.shift(periods))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return returns

    def calculate_cumulative_returns(
        self,
        returns: pd.Series | pd.DataFrame,
        method: Literal["simple", "log"] = "simple",
    ) -> pd.Series | pd.DataFrame:
        """Calculate cumulative returns.
        
        Args:
            returns: Returns series or DataFrame
            method: 'simple' or 'log' (must match returns calculation method)
            
        Returns:
            Cumulative returns
        """
        if method == "simple":
            return (1 + returns).cumprod() - 1
        elif method == "log":
            return returns.cumsum()
        else:
            raise ValueError(f"Unknown method: {method}")

    def calculate_rolling_returns(
        self,
        prices: pd.Series,
        windows: list[int] | None = None,
    ) -> pd.DataFrame:
        """Calculate rolling returns for multiple windows.
        
        Args:
            prices: Price series
            windows: List of window sizes (e.g., [5, 20, 60, 120, 252])
            
        Returns:
            DataFrame with rolling returns for each window
        """
        if windows is None:
            windows = [5, 20, 60, 120, 252]  # 1W, 1M, 3M, 6M, 1Y
        
        result = pd.DataFrame(index=prices.index)
        
        for window in windows:
            col_name = f"return_{window}d"
            result[col_name] = prices.pct_change(periods=window)
        
        return result

    # ==================== Normalization/Standardization ====================

    def normalize(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        method: Literal["minmax", "zscore", "robust"] = "minmax",
    ) -> pd.DataFrame:
        """Normalize/standardize numeric columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to normalize
            method: Normalization method
                - 'minmax': Scale to [0, 1]
                - 'zscore': Standardize to mean=0, std=1
                - 'robust': Use median and IQR
                
        Returns:
            Normalized DataFrame
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == "minmax":
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val != min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
            elif method == "zscore":
                mean = df[col].mean()
                std = df[col].std()
                if std != 0:
                    df[col] = (df[col] - mean) / std
            elif method == "robust":
                median = df[col].median()
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr != 0:
                    df[col] = (df[col] - median) / iqr
        
        return df

    def winsorize(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        limits: tuple[float, float] = (0.01, 0.99),
    ) -> pd.DataFrame:
        """Winsorize extreme values.
        
        Args:
            df: Input DataFrame
            columns: Columns to winsorize
            limits: Lower and upper percentile limits
            
        Returns:
            Winsorized DataFrame
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in df.columns:
                continue
            lower = df[col].quantile(limits[0])
            upper = df[col].quantile(limits[1])
            df[col] = df[col].clip(lower=lower, upper=upper)
        
        return df

    # ==================== Corporate Actions ====================

    def adjust_for_splits(
        self,
        df: pd.DataFrame,
        split_ratio: float,
        split_date: str | pd.Timestamp,
        price_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Adjust historical prices for stock splits.
        
        Args:
            df: DataFrame with date index and price columns
            split_ratio: Split ratio (e.g., 2.0 for 2-for-1 split)
            split_date: Date of the split
            price_cols: Price columns to adjust
            
        Returns:
            Split-adjusted DataFrame
        """
        df = df.copy()
        
        if price_cols is None:
            price_cols = ["open", "high", "low", "close", "adj_close"]
        
        split_date = pd.to_datetime(split_date)
        
        mask = df.index < split_date if isinstance(df.index, pd.DatetimeIndex) else df["date"] < split_date
        
        for col in price_cols:
            if col in df.columns:
                df.loc[mask, col] = df.loc[mask, col] / split_ratio
        
        if "volume" in df.columns:
            df.loc[mask, "volume"] = df.loc[mask, "volume"] * split_ratio
        
        return df

    # ==================== Point-in-Time ====================

    def ensure_point_in_time(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        report_date_col: str = "report_date",
    ) -> pd.DataFrame:
        """Ensure data reflects point-in-time availability.
        
        Financial data is often restated or delayed. This ensures
        we only use data that was actually available at each point.
        
        Args:
            df: DataFrame with fundamental data
            date_col: Column for the period date
            report_date_col: Column for when data was reported
            
        Returns:
            Point-in-time adjusted DataFrame
        """
        if report_date_col not in df.columns:
            logger.warning(f"No {report_date_col} column found. Returning original DataFrame.")
            return df
        
        df = df.copy()
        df = df.sort_values([date_col, report_date_col])
        
        # Keep only the first report for each period
        df = df.drop_duplicates(subset=[date_col], keep="first")
        
        return df

    # ==================== Resampling ====================

    def resample_ohlcv(
        self,
        df: pd.DataFrame,
        freq: str = "W",
        date_col: str = "date",
    ) -> pd.DataFrame:
        """Resample OHLCV data to different frequency.
        
        Args:
            df: DataFrame with OHLCV data
            freq: Target frequency ('W', 'M', 'Q', 'Y')
            date_col: Date column name
            
        Returns:
            Resampled DataFrame
        """
        df = df.copy()
        
        if date_col in df.columns:
            df = df.set_index(date_col)
        
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        
        # Only include columns that exist
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
        
        if "adj_close" in df.columns:
            agg_dict["adj_close"] = "last"
        
        resampled = df.resample(freq).agg(agg_dict)
        resampled = resampled.dropna()
        
        return resampled.reset_index()

    # ==================== Pipeline ====================

    def process_price_data(
        self,
        df: pd.DataFrame,
        handle_missing: bool = True,
        handle_outliers: bool = False,
        calculate_returns: bool = True,
    ) -> pd.DataFrame:
        """Complete preprocessing pipeline for price data.
        
        Args:
            df: Raw price DataFrame
            handle_missing: Whether to fill missing values
            handle_outliers: Whether to handle outliers
            calculate_returns: Whether to add returns column
            
        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()
        
        if handle_missing:
            df = self.fill_missing_prices(df)
        
        if handle_outliers:
            price_cols = ["open", "high", "low", "close"]
            existing_cols = [c for c in price_cols if c in df.columns]
            df = self.handle_outliers(df, columns=existing_cols, action="clip")
        
        if calculate_returns and "close" in df.columns:
            df["returns"] = self.calculate_returns(df["close"])
            df["log_returns"] = self.calculate_returns(df["close"], method="log")
        
        return df
