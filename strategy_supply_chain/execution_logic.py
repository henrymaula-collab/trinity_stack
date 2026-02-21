"""
Execution Logic for Supply-Chain Lead-Lag Strategy.

Entry: Buy supplier at open of T+1 following customer's signal.
Exit:  3 days before supplier's next earnings, or after 20 trading days max.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

HOLDING_DAYS_MAX = 20
DAYS_BEFORE_EARNINGS_EXIT = 3


def compute_entry_dates(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Entry = T+1 open after signal. Requires trading calendar.

    signals: must have columns [date, supplier_ticker, conviction] where signal='BUY'.
    """
    df = signals[signals["signal"] == "BUY"].copy()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["entry_date"] = df["date"]
    return df


def get_supplier_earnings_schedule(
    earnings: pd.DataFrame,
    supplier_col: str = "supplier_ticker",
) -> pd.DataFrame:
    """Map supplier -> next earnings date (sorted ascending per supplier)."""
    e = earnings.copy()
    e = e.rename(columns={e.columns[0]: supplier_col, e.columns[1]: "earnings_date"})
    e["earnings_date"] = pd.to_datetime(e["earnings_date"])
    e = e.sort_values([supplier_col, "earnings_date"]).drop_duplicates()
    return e


def compute_exit_date(
    entry_date: pd.Timestamp,
    supplier_ticker: str,
    supplier_earnings: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
    max_hold_days: int = HOLDING_DAYS_MAX,
    days_before_earnings: int = DAYS_BEFORE_EARNINGS_EXIT,
) -> pd.Timestamp:
    """
    Exit = min(
        entry + max_hold_days,
        supplier_next_earnings - days_before_earnings
    ).
    """
    mask = trading_calendar >= entry_date
    cal_from_entry = trading_calendar[mask]
    if len(cal_from_entry) == 0:
        return entry_date

    # Max holding period
    max_exit_idx = min(max_hold_days, len(cal_from_entry) - 1)
    max_exit_date = cal_from_entry[max_exit_idx]

    # Earnings-based exit: 3 trading days before supplier's next earnings
    if supplier_earnings.empty or "supplier_ticker" not in supplier_earnings.columns:
        return max_exit_date
    supp_earn = supplier_earnings[supplier_earnings["supplier_ticker"] == supplier_ticker]
    future_earnings = supp_earn[pd.to_datetime(supp_earn["earnings_date"]) > entry_date]
    if future_earnings.empty:
        return max_exit_date
    next_earn = pd.to_datetime(future_earnings["earnings_date"].min())
    cal_before_earn = cal_from_entry[cal_from_entry < next_earn]
    if len(cal_before_earn) > days_before_earnings:
        earn_exit_date = cal_before_earn[-(1 + days_before_earnings)]
    else:
        earn_exit_date = max_exit_date
    return min(max_exit_date, earn_exit_date)


def build_trades(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    earnings: Optional[pd.DataFrame] = None,
    max_hold_days: int = HOLDING_DAYS_MAX,
    days_before_earnings: int = DAYS_BEFORE_EARNINGS_EXIT,
) -> pd.DataFrame:
    """
    Build trade table: entry_date, exit_date, supplier_ticker, conviction.

    Uses prices['date'] as trading calendar. Exit = min(max_hold, 3d before earnings).
    """
    buys = signals[signals["signal"] == "BUY"].copy()
    if buys.empty:
        return pd.DataFrame(columns=["entry_date", "exit_date", "supplier_ticker", "conviction"])

    cal = pd.DatetimeIndex(pd.to_datetime(prices["date"]).unique()).sort_values()
    supplier_earnings = (
        get_supplier_earnings_schedule(earnings) if earnings is not None and not earnings.empty
        else pd.DataFrame(columns=["supplier_ticker", "earnings_date"])
    )

    trades = []
    for _, row in buys.iterrows():
        entry = pd.Timestamp(row["date"])
        supp = row["supplier_ticker"]
        conv = row["conviction"]

        # T+1 entry: first trading day after signal
        entry_mask = cal > entry
        if not entry_mask.any():
            continue
        entry_date = cal[entry_mask][0]

        # Exit
        if not supplier_earnings.empty and "supplier_ticker" in supplier_earnings.columns:
            exit_date = compute_exit_date(
                entry_date, supp, supplier_earnings, cal, max_hold_days, days_before_earnings
            )
        else:
            entry_idx = cal.get_indexer([entry_date], method="ffill")[0]
            exit_idx = min(entry_idx + max_hold_days, len(cal) - 1)
            exit_date = cal[exit_idx]

        trades.append({"entry_date": entry_date, "exit_date": exit_date, "supplier_ticker": supp, "conviction": conv})

    return pd.DataFrame(trades)
