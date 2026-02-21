"""
Supply-Chain Data Pipeline.

Handles Bloomberg BQL/SPLC (Supply Chain) or Capital IQ data.
Schema: date, supplier_ticker, customer_ticker, revenue_dependency_pct.
Also fetches daily prices and earnings dates for both customer and supplier.

No look-ahead bias: all data is strictly point-in-time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

REQUIRED_SCHEMA = ["date", "supplier_ticker", "customer_ticker", "revenue_dependency_pct"]


def _validate_future_timestamps(df: pd.DataFrame, reference_date: pd.Timestamp) -> None:
    """Raise if any date exceeds reference (look-ahead contamination)."""
    future_mask = pd.to_datetime(df["date"]) > reference_date
    if future_mask.any():
        n = future_mask.sum()
        raise ValueError(
            f"Future timestamp contamination: {n} rows with date > {reference_date}. "
            "Look-ahead bias detected."
        )


def load_supply_chain(
    path: str | Path,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Load supply-chain relationship dataset.

    Required columns: date, supplier_ticker, customer_ticker, revenue_dependency_pct.

    Args:
        path: Path to CSV or Parquet file.
        reference_date: If provided, raises on any row with date > reference_date.

    Returns:
        DataFrame with validated schema.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Supply-chain file not found: {path}")

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

    if (df["revenue_dependency_pct"] < 0).any() or (df["revenue_dependency_pct"] > 100).any():
        raise ValueError("revenue_dependency_pct must be in [0, 100]")

    if reference_date is not None:
        _validate_future_timestamps(df, reference_date)

    return df[REQUIRED_SCHEMA]


def fetch_bloomberg_prices_and_earnings(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch daily prices and earnings dates from Bloomberg BLPAPI.

    Requires pdblp and Bloomberg Terminal/B-PIPE running.

    Returns:
        prices: columns [date, ticker, PX_LAST, Return] (Return = pct_change, shifted)
        earnings: columns [ticker, earnings_date] (next-earnings per ticker)
    """
    try:
        import pdblp
    except ImportError:
        raise ImportError(
            "pdblp required for Bloomberg fetch. Install: pip install pdblp"
        )

    overrides = [
        ("CshAdjNormal", True),
        ("CshAdjAbnormal", True),
        ("CapChgExec", True),
    ]

    con = pdblp.BCon(debug=False, port=8194, timeout=60000)
    con.start()

    try:
        prices_df = con.bdh(
            tickers,
            "PX_LAST",
            start_date,
            end_date,
            longdata=True,
            ovrds=overrides,
        )
    finally:
        con.stop()

    if prices_df is None or prices_df.empty:
        raise ValueError("No price data returned from Bloomberg")

    prices_df = prices_df.rename(
        columns={"date": "date", "security": "ticker", "PX_LAST": "PX_LAST"}
    )
    prices_df["date"] = pd.to_datetime(prices_df["date"])
    prices_df = prices_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Returns: strictly past (shift(1))
    prices_df["Return"] = (
        prices_df.groupby("ticker")["PX_LAST"]
        .pct_change()
        .groupby(prices_df["ticker"])
        .shift(1)
    )

    # Earnings: fetch EARN_ANN_DT_TIME_HIST or equivalent
    try:
        con.start()
        earn_df = con.ref(tickers, "EARN_ANN_DT_TIME_HIST_WITH_EPS")
    except Exception:
        earn_df = None
    finally:
        try:
            con.stop()
        except Exception:
            pass

    earnings = pd.DataFrame(columns=["ticker", "earnings_date"])
    if earn_df is not None and not earn_df.empty:
        # Normalize structure depending on pdblp output
        if "value" in earn_df.columns:
            earnings = earn_df.rename(columns={"value": "earnings_date"})
        elif "EARN_ANN_DT_TIME_HIST_WITH_EPS" in earn_df.columns:
            earnings = earn_df.rename(
                columns={"EARN_ANN_DT_TIME_HIST_WITH_EPS": "earnings_date"}
            )
        if "ticker" not in earnings.columns and "security" in earnings.columns:
            earnings = earnings.rename(columns={"security": "ticker"})
        earnings["earnings_date"] = pd.to_datetime(earnings["earnings_date"])

    return prices_df, earnings


def load_prices_from_parquet(path: str | Path) -> pd.DataFrame:
    """
    Load daily prices from pre-saved Parquet (e.g. Trinity Layer 1 output).

    Expected columns: date, ticker, PX_LAST. Adds Return with .shift(1).
    """
    df = pd.read_parquet(path)
    required = ["date", "ticker", "PX_LAST"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Prices file missing columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])
    df["Return"] = df.groupby("ticker")["PX_LAST"].pct_change().groupby(df["ticker"]).shift(1)
    return df


def load_earnings_from_parquet(path: str | Path) -> pd.DataFrame:
    """
    Load earnings dates from pre-saved Parquet.

    Expected columns: ticker, earnings_date (or report_date).
    """
    df = pd.read_parquet(path)
    if "report_date" in df.columns and "earnings_date" not in df.columns:
        df = df.rename(columns={"report_date": "earnings_date"})
    if "ticker" not in df.columns:
        raise ValueError("Earnings file must have 'ticker' column")
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    return df[["ticker", "earnings_date"]].drop_duplicates()


def build_unified_dataset(
    supply_chain: pd.DataFrame,
    prices: pd.DataFrame,
    earnings: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Join supply-chain relationships with prices and earnings.

    Output: one row per (date, supplier_ticker, customer_ticker) with:
    - revenue_dependency_pct
    - customer_return, customer_vol_60d (rolling 60d vol, shifted)
    - supplier_return, supplier_earnings_next (next earnings date)
    """
    df = supply_chain.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Customer prices -> returns and 60d vol (strictly past)
    cust = prices.rename(columns={"ticker": "customer_ticker", "Return": "customer_return"})
    cust = cust[["date", "customer_ticker", "customer_return", "PX_LAST"]]
    cust = cust.rename(columns={"PX_LAST": "customer_px"})

    cust["customer_vol_60d"] = (
        cust.groupby("customer_ticker")["customer_return"]
        .transform(lambda x: x.rolling(60, min_periods=20).std().shift(1))
    )
    df = df.merge(cust, on=["date", "customer_ticker"], how="left")

    # Supplier prices
    supp = prices.rename(columns={"ticker": "supplier_ticker", "Return": "supplier_return"})
    supp = supp[["date", "supplier_ticker", "supplier_return"]]
    df = df.merge(supp, on=["date", "supplier_ticker"], how="left")

    if earnings is not None and not earnings.empty:
        supp_earn = earnings.rename(columns={"ticker": "supplier_ticker"})
        df = df.merge(supp_earn, on="supplier_ticker", how="left")
        df = df.rename(columns={"earnings_date": "supplier_earnings_next"})
    else:
        df["supplier_earnings_next"] = pd.NaT

    return df
