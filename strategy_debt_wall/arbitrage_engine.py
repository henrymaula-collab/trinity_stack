"""
Debt Wall Arbitrage Engine.

Cash Runway Deficit = next_12m_debt_maturity - (cash_equivalents + (ttm_fcf * 0.5)).

SHORT / EXCLUDE trigger: Deficit > (market_cap * 0.10) AND maturity < 6 months.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFICIT_DILUTION_THRESHOLD = 0.10  # 10% of market cap
FCF_COVERAGE_FACTOR = 0.5  # Assume 50% of TTM FCF achievable in runway period
MATURITY_WINDOW_MONTHS = 6


def compute_cash_runway_deficit(
    cash_equivalents: pd.Series,
    ttm_fcf: pd.Series,
    next_12m_debt_maturity: pd.Series,
) -> pd.Series:
    """
    Deficit = next_12m_debt_maturity - (cash_equivalents + ttm_fcf * 0.5).

    Assumes 50% of TTM FCF can be generated in the runway period.
    """
    liquidity = cash_equivalents + (ttm_fcf * FCF_COVERAGE_FACTOR)
    deficit = next_12m_debt_maturity - liquidity
    return deficit


def months_to_maturity(date: pd.Series, maturity_date: pd.Series) -> pd.Series:
    """Months until maturity from observation date."""
    return ((pd.to_datetime(maturity_date) - pd.to_datetime(date)).dt.days / 30.44).clip(lower=0)


def generate_signals(
    df: pd.DataFrame,
    dilution_threshold: float = DEFICIT_DILUTION_THRESHOLD,
    maturity_months_max: float = MATURITY_WINDOW_MONTHS,
    maturity_date_col: Optional[str] = "maturity_date",
) -> pd.DataFrame:
    """
    Trigger SHORT / EXCLUDE when:
      - Deficit > market_cap * dilution_threshold (10% dilution implied)
      - Maturity < maturity_months_max (6 months) if maturity_date available

    If maturity_date_col is absent, treats as within window (12m debt implies near-term).
    """
    out = df.copy()
    out["deficit"] = compute_cash_runway_deficit(
        out["cash_equivalents"], out["ttm_fcf"], out["next_12m_debt_maturity"]
    )
    dilution_breach = out["deficit"] > (out["market_cap"] * dilution_threshold)

    if maturity_date_col and maturity_date_col in out.columns:
        out["months_to_maturity"] = months_to_maturity(out["date"], out[maturity_date_col])
        maturity_breach = out["months_to_maturity"] < maturity_months_max
    else:
        maturity_breach = pd.Series(True, index=out.index)

    out["trigger"] = dilution_breach & maturity_breach
    out["signal"] = np.where(out["trigger"], "SHORT", None)

    return out
