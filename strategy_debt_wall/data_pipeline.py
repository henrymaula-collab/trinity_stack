"""
Debt Wall Data Pipeline.

Consumes Capital IQ / Bloomberg debt maturity data.
Schema: date, ticker, market_cap, cash_equivalents, ttm_fcf,
        next_12m_debt_maturity, short_interest.

Point-in-time only; no look-ahead bias.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

REQUIRED_SCHEMA = [
    "date",
    "ticker",
    "market_cap",
    "cash_equivalents",
    "ttm_fcf",
    "next_12m_debt_maturity",
    "short_interest",
]


def _validate_future_timestamps(df: pd.DataFrame, reference_date: pd.Timestamp) -> None:
    """Raise if any date exceeds reference (look-ahead contamination)."""
    future_mask = pd.to_datetime(df["date"]) > reference_date
    if future_mask.any():
        n = future_mask.sum()
        raise ValueError(
            f"Future timestamp contamination: {n} rows with date > {reference_date}. "
            "Look-ahead bias detected."
        )


def load_debt_wall_data(
    path: str | Path,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Load corporate finance / debt maturity dataset.

    Required columns: date, ticker, market_cap, cash_equivalents, ttm_fcf,
    next_12m_debt_maturity, short_interest.

    Args:
        path: Path to CSV or Parquet file.
        reference_date: If provided, raises on any row with date > reference_date.

    Returns:
        DataFrame with validated schema.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Debt wall data file not found: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    missing = [c for c in REQUIRED_SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=REQUIRED_SCHEMA).reset_index(drop=True)

    if (df["market_cap"] <= 0).any():
        raise ValueError("market_cap must be positive")
    if (df["short_interest"] < 0).any():
        raise ValueError("short_interest must be non-negative")

    if reference_date is not None:
        _validate_future_timestamps(df, reference_date)

    return df[REQUIRED_SCHEMA]


def load_risk_free_rates(
    path: str | Path,
    rate_col: str = "rf_rate",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Load risk-free rate series for cost-of-borrowing overlay.

    Expected columns: date, rate (e.g. rf_rate or PX_LAST for government bond).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Risk-free rates file not found: {path}")

    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    df = df.rename(columns={date_col: "date", rate_col: "rf_rate"})
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "rf_rate"]].dropna()


def load_debt_maturity_dates(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load debt maturity dates per ticker for exit logic.

    Expected columns: ticker, maturity_date (or next_12m_maturity_date).
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "maturity_date"])

    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    if "maturity_date" not in df.columns and "next_12m_maturity_date" in df.columns:
        df = df.rename(columns={"next_12m_maturity_date": "maturity_date"})
    df["maturity_date"] = pd.to_datetime(df["maturity_date"])
    return df[["ticker", "maturity_date"]].drop_duplicates()
