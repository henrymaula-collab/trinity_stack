"""
Cross-Asset Signal Engine for Supply-Chain Lead-Lag.

Trigger conditions (causal, no look-ahead):
  1. Customer daily return > 2σ of its rolling 60-day volatility.
  2. Customer Earnings Surprise > threshold (if available).

Conviction scaled by revenue_dependency_pct.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

VOL_WINDOW = 60
SIGMA_THRESHOLD = 2.0
DEFAULT_EPS_SURPRISE_THRESHOLD = 0.5  # Standardized surprise (e.g. SUE)


def _validate_no_future_data(df: pd.DataFrame, signal_date: pd.Timestamp) -> None:
    """Ensure no signal uses future information."""
    if (pd.to_datetime(df["date"]) > signal_date).any():
        raise ValueError("Signal generation uses future dates. Look-ahead bias.")


def compute_customer_vol_60d(
    prices: pd.DataFrame,
    ticker_col: str = "customer_ticker",
    return_col: str = "customer_return",
) -> pd.Series:
    """
    Rolling 60-day volatility (std of returns), strictly lagged.

    Uses .shift(1) so vol at date t uses returns up to t-1 only.
    """
    vol = (
        prices.groupby(ticker_col)[return_col]
        .transform(lambda x: x.rolling(VOL_WINDOW, min_periods=20).std().shift(1))
    )
    return vol


def trigger_from_return_volatility(
    df: pd.DataFrame,
    sigma: float = SIGMA_THRESHOLD,
) -> pd.Series:
    """
    Boolean trigger: customer_return > sigma * customer_vol_60d.

    Both customer_return and customer_vol_60d must be strictly past (shifted).
    """
    if "customer_return" not in df.columns or "customer_vol_60d" not in df.columns:
        raise ValueError("Requires customer_return and customer_vol_60d")
    safe_vol = df["customer_vol_60d"].replace(0, np.nan)
    threshold = sigma * safe_vol
    return (df["customer_return"] > threshold) & threshold.notna()


def trigger_from_earnings_surprise(
    df: pd.DataFrame,
    surprise_col: str = "customer_eps_surprise",
    threshold: float = DEFAULT_EPS_SURPRISE_THRESHOLD,
) -> pd.Series:
    """
    Boolean trigger: customer EPS surprise (e.g. SUE) > threshold.

    Only applied where surprise_col exists and is non-NaN.
    """
    if surprise_col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[surprise_col] > threshold


def generate_signals(
    unified: pd.DataFrame,
    sigma: float = SIGMA_THRESHOLD,
    eps_surprise_threshold: Optional[float] = None,
    eps_surprise_col: Optional[str] = "customer_eps_surprise",
) -> pd.DataFrame:
    """
    Generate BUY signals for suppliers based on customer triggers.

    Trigger if:
      - customer_return > sigma * customer_vol_60d, OR
      - customer_eps_surprise > eps_surprise_threshold (if column exists)

    Conviction = revenue_dependency_pct / 100 (scaled 0–1).
    """
    df = unified.copy()

    ret_trigger = trigger_from_return_volatility(df, sigma=sigma)

    if eps_surprise_threshold is not None and eps_surprise_col and eps_surprise_col in df.columns:
        earn_trigger = trigger_from_earnings_surprise(
            df, surprise_col=eps_surprise_col, threshold=eps_surprise_threshold
        )
        df["trigger"] = ret_trigger | earn_trigger
    else:
        df["trigger"] = ret_trigger

    df["signal"] = np.where(df["trigger"], "BUY", None)
    df["conviction"] = np.where(df["trigger"], df["revenue_dependency_pct"] / 100.0, 0.0)

    return df
