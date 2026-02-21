"""
Cost-of-Borrowing Overlay.

Amplifies SHORT conviction when current risk-free rate >> fixed rate of
maturing debt (interest expense will spike post-refinancing).
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

RATE_GAP_THRESHOLD = 0.02  # 2% spread: rf - debt_rate
MAX_CONVICTION_BOOST = 0.30  # Up to 30% conviction increase


def compute_conviction_boost(
    rf_rate: float,
    debt_rate: Optional[float] = None,
    rate_gap_threshold: float = RATE_GAP_THRESHOLD,
    max_boost: float = MAX_CONVICTION_BOOST,
) -> float:
    """
    Boost conviction when rf >> debt rate (refinancing will be painful).

    If debt_rate is None, assume zero (worst-case: full rf exposure).
    Boost scales with (rf_rate - debt_rate) up to max_boost.
    """
    debt = debt_rate if debt_rate is not None else 0.0
    gap = max(0.0, rf_rate - debt)
    if gap < rate_gap_threshold:
        return 0.0
    # Linear ramp from threshold to 2*threshold, then cap
    ramp = min(1.0, (gap - rate_gap_threshold) / rate_gap_threshold)
    return ramp * max_boost


def apply_overlay(
    signals: pd.DataFrame,
    rf_rates: pd.DataFrame,
    debt_rate_col: Optional[str] = "debt_fixed_rate",
    rate_gap_threshold: float = RATE_GAP_THRESHOLD,
    max_boost: float = MAX_CONVICTION_BOOST,
) -> pd.DataFrame:
    """
    Merge risk-free rates by date and apply conviction boost to SHORT signals.
    """
    df = signals.copy()
    df["date"] = pd.to_datetime(df["date"])
    rf = rf_rates.copy()
    rf["date"] = pd.to_datetime(rf["date"])

    df = df.merge(rf, on="date", how="left")
    df["rf_rate"] = df["rf_rate"].fillna(0.0)

    debt = df[debt_rate_col].fillna(0.0) if debt_rate_col and debt_rate_col in df.columns else 0.0
    gap = (df["rf_rate"] - debt).clip(lower=0)
    ramp = ((gap - rate_gap_threshold) / rate_gap_threshold).clip(lower=0, upper=1)
    df["conviction_boost"] = ramp * max_boost
    df["base_conviction"] = df["trigger"].astype(float)
    df["conviction"] = (df["base_conviction"] * (1.0 + df["conviction_boost"])).clip(upper=1.0)
    return df
