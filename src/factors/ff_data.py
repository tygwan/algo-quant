"""Fama-French factor data loader."""

import io
import logging
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class FamaFrenchDataLoader:
    """Load Fama-French factor data from Kenneth French Data Library.
    
    Supports:
    - 3 Factor Model: Mkt-RF, SMB, HML
    - 5 Factor Model: Mkt-RF, SMB, HML, RMW, CMA
    - Momentum Factor: Mom
    
    Example:
        >>> loader = FamaFrenchDataLoader()
        >>> ff3_monthly = loader.load_ff3_factors(frequency="monthly")
        >>> ff5_daily = loader.load_ff5_factors(frequency="daily")
    """
    
    BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
    
    # Dataset names
    FF3_MONTHLY = "F-F_Research_Data_Factors_CSV.zip"
    FF3_DAILY = "F-F_Research_Data_Factors_daily_CSV.zip"
    FF5_MONTHLY = "F-F_Research_Data_5_Factors_2x3_CSV.zip"
    FF5_DAILY = "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    MOM_MONTHLY = "F-F_Momentum_Factor_CSV.zip"
    MOM_DAILY = "F-F_Momentum_Factor_daily_CSV.zip"
    
    def __init__(
        self,
        cache_dir: str | Path = ".cache/ff_data",
        use_cache: bool = True,
    ):
        """Initialize the data loader.
        
        Args:
            cache_dir: Directory for caching downloaded data
            use_cache: Whether to use cached data
        """
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _download_and_parse(
        self,
        dataset_name: str,
        skip_footer: int = 0,
    ) -> pd.DataFrame:
        """Download and parse a Fama-French dataset.
        
        Args:
            dataset_name: Name of the dataset file
            skip_footer: Number of footer rows to skip
            
        Returns:
            Parsed DataFrame
        """
        cache_file = self.cache_dir / dataset_name.replace(".zip", ".parquet")
        
        # Check cache
        if self.use_cache and cache_file.exists():
            logger.info(f"Loading from cache: {cache_file}")
            return pd.read_parquet(cache_file)
        
        # Download
        url = f"{self.BASE_URL}/{dataset_name}"
        logger.info(f"Downloading: {url}")
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Parse zip file
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            # Get the CSV file (should be the only file)
            csv_name = [n for n in zf.namelist() if n.endswith(".CSV")][0]
            
            with zf.open(csv_name) as f:
                content = f.read().decode("utf-8")
        
        # Parse CSV content
        df = self._parse_ff_csv(content, skip_footer)
        
        # Cache
        if self.use_cache:
            df.to_parquet(cache_file)
            logger.info(f"Cached to: {cache_file}")
        
        return df
    
    def _parse_ff_csv(
        self,
        content: str,
        skip_footer: int = 0,
    ) -> pd.DataFrame:
        """Parse Fama-French CSV content.
        
        The FF data files have a specific format with multiple tables.
        We parse only the first table (main factor data).
        
        Args:
            content: CSV content as string
            skip_footer: Number of footer rows to skip
            
        Returns:
            Parsed DataFrame
        """
        lines = content.strip().split("\n")
        
        # Find the start of data (first line with numbers)
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip() and line.strip()[0].isdigit():
                data_start = i
                break
        
        # Find the end of first table (blank line or non-numeric line after data)
        data_end = len(lines)
        for i in range(data_start + 1, len(lines)):
            line = lines[i].strip()
            # Check for blank line or new section (non-numeric first character)
            if not line or (line and not line[0].isdigit() and line[0] != "-"):
                data_end = i
                break
        
        # Get header (line before data)
        header_line = lines[data_start - 1] if data_start > 0 else ""
        
        # Parse data
        data_lines = lines[data_start:data_end - skip_footer]
        
        # Create DataFrame
        df = pd.read_csv(
            io.StringIO("\n".join([header_line] + data_lines)),
            skipinitialspace=True,
        )
        
        # Rename first column to 'date' if unnamed
        if df.columns[0] == "Unnamed: 0" or df.columns[0] == "":
            df = df.rename(columns={df.columns[0]: "date"})
        
        # Parse date - handle both YYYYMM and YYYYMMDD formats
        date_col = df.columns[0]
        df[date_col] = df[date_col].astype(str)
        
        if df[date_col].str.len().iloc[0] == 6:  # Monthly: YYYYMM
            df["date"] = pd.to_datetime(df[date_col], format="%Y%m")
            # Set to end of month
            df["date"] = df["date"] + pd.offsets.MonthEnd(0)
        else:  # Daily: YYYYMMDD
            df["date"] = pd.to_datetime(df[date_col], format="%Y%m%d")
        
        # Drop original date column if different from 'date'
        if date_col != "date":
            df = df.drop(columns=[date_col])
        
        df = df.set_index("date")
        
        # Convert to decimal (data is in percentage)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100
        
        return df
    
    def load_ff3_factors(
        self,
        frequency: Literal["daily", "monthly"] = "monthly",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Load Fama-French 3 Factor data.
        
        Factors:
        - Mkt-RF: Market excess return
        - SMB: Small Minus Big (size factor)
        - HML: High Minus Low (value factor)
        - RF: Risk-free rate
        
        Args:
            frequency: Data frequency ("daily" or "monthly")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with factor returns
        """
        dataset = self.FF3_DAILY if frequency == "daily" else self.FF3_MONTHLY
        df = self._download_and_parse(dataset)
        
        # Filter by date
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        return df
    
    def load_ff5_factors(
        self,
        frequency: Literal["daily", "monthly"] = "monthly",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Load Fama-French 5 Factor data.
        
        Factors:
        - Mkt-RF: Market excess return
        - SMB: Small Minus Big (size factor)
        - HML: High Minus Low (value factor)
        - RMW: Robust Minus Weak (profitability factor)
        - CMA: Conservative Minus Aggressive (investment factor)
        - RF: Risk-free rate
        
        Args:
            frequency: Data frequency ("daily" or "monthly")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with factor returns
        """
        dataset = self.FF5_DAILY if frequency == "daily" else self.FF5_MONTHLY
        df = self._download_and_parse(dataset)
        
        # Filter by date
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        return df
    
    def load_momentum_factor(
        self,
        frequency: Literal["daily", "monthly"] = "monthly",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Load Momentum factor data.
        
        Args:
            frequency: Data frequency ("daily" or "monthly")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with momentum factor returns
        """
        dataset = self.MOM_DAILY if frequency == "daily" else self.MOM_MONTHLY
        df = self._download_and_parse(dataset)
        
        # Rename column
        df = df.rename(columns={df.columns[0]: "Mom"})
        
        # Filter by date
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        return df
    
    def load_factors_with_momentum(
        self,
        num_factors: Literal[3, 5] = 5,
        frequency: Literal["daily", "monthly"] = "monthly",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Load Fama-French factors with Momentum.
        
        Args:
            num_factors: 3 (Mkt-RF, SMB, HML) or 5 (adds RMW, CMA)
            frequency: Data frequency
            start_date: Start date
            end_date: End date
            
        Returns:
            Combined DataFrame with FF factors and Momentum
        """
        if num_factors == 3:
            ff = self.load_ff3_factors(frequency, start_date, end_date)
        else:
            ff = self.load_ff5_factors(frequency, start_date, end_date)
        
        mom = self.load_momentum_factor(frequency, start_date, end_date)
        
        # Merge
        df = ff.join(mom, how="inner")
        
        return df
    
    def clear_cache(self) -> int:
        """Clear cached data files.
        
        Returns:
            Number of files cleared
        """
        count = 0
        for file in self.cache_dir.glob("*.parquet"):
            file.unlink()
            count += 1
        
        logger.info(f"Cleared {count} cached files")
        return count
