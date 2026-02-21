from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

T_PLUS_2_SHIFT = 2
TRADING_DAYS_4_QUARTERS = 252


class TripleLagEngine:
    """
    Enforces Layer 1 Point-in-Time integrity with strict T+2 fundamental lag.
    Eliminates look-ahead bias and data leakage.
    """

    T0_COLS: List[str] = ["date", "PX_LAST", "PX_TURN_OVER"]

    def __init__(
        self,
        raw_path: str = "data/raw",
        processed_path: str = "data/processed",
    ) -> None:
        self.raw_path = Path(raw_path)
        self.processed_path = Path(processed_path)

    def _validate_monotonic_dates(self, df: pd.DataFrame, ticker: str) -> None:
        date_col = pd.to_datetime(df["date"])
        if not date_col.is_monotonic_increasing:
            raise ValueError(
                f"[{ticker}] Dates are not monotonically increasing. Look-ahead bias risk."
            )

    def _get_fundamental_columns(self, df: pd.DataFrame) -> List[str]:
        exclude = self.T0_COLS + ["Return", "Amihud_Illiq"]
        return [c for c in df.columns if c not in exclude]

    def process_asset(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        if "date" not in df.columns:
            raise ValueError(f"[{ticker}] Missing 'date' column.")
        if "PX_LAST" not in df.columns or "PX_TURN_OVER" not in df.columns:
            raise ValueError(f"[{ticker}] Missing required columns: PX_LAST, PX_TURN_OVER.")

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        self._validate_monotonic_dates(df, ticker)
        df = df.sort_values("date").reset_index(drop=True)

        df["Return"] = df["PX_LAST"].pct_change()
        safe_volume = df["PX_TURN_OVER"].replace(0, np.nan)
        df["Amihud_Illiq"] = np.abs(df["Return"]) / safe_volume

        if "ACTUAL_EPS" in df.columns and "CONSENSUS_EPS" in df.columns:
            forecast_error = df["ACTUAL_EPS"] - df["CONSENSUS_EPS"]
            rolling_std = (
                forecast_error.rolling(window=TRADING_DAYS_4_QUARTERS, min_periods=63)
                .std()
                .shift(1)
            )
            df["SUE"] = forecast_error / rolling_std

        fundamental_cols = self._get_fundamental_columns(df)
        for col in fundamental_cols:
            df[f"{col}_lag2"] = df[col].shift(T_PLUS_2_SHIFT)

        df = df.drop(columns=fundamental_cols)
        df = df.dropna().reset_index(drop=True)

        return df

    def execute_pipeline(self) -> None:
        self.processed_path.mkdir(parents=True, exist_ok=True)

        parquet_files = list(self.raw_path.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No Parquet files found in {self.raw_path}")

        for file_path in parquet_files:
            ticker = file_path.stem
            raw_df = pd.read_parquet(file_path)
            clean_df = self.process_asset(raw_df, ticker)
            output_path = self.processed_path / f"{ticker}_clean.parquet"
            clean_df.to_parquet(output_path, index=False)
            logging.info(f"Processed {ticker}. Shape: {clean_df.shape}")
