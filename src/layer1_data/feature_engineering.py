"""
Layer 1: Feature Engineering.
Processes raw price and fundamental data into PiT-safe features for Layer 3.
T+2 lag enforced; no look-ahead bias.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

T_PLUS_2_DAYS = 2

PRICE_REQUIRED: List[str] = ["date", "ticker", "PX_LAST", "PX_TURN_OVER", "BID_ASK_SPREAD_PCT"]
FUND_REQUIRED: List[str] = [
    "report_date",
    "ticker",
    "ACTUAL_EPS",
    "CONSENSUS_EPS",
    "EPS_STD",
    "ROIC",
    "Dilution",
    "Accruals",
]
LAYER3_OUTPUT_COLS: List[str] = [
    "date",
    "ticker",
    "Quality_Score",
    "BID_ASK_SPREAD_PCT",
    "PX_TURN_OVER",
    "SUE",
    "Amihud",
    "Vol_Compression",
    "is_january_prep",
    "dist_to_sma200",
    "Local_Rate",
    "Dividend_Yield",
    "forward_return",
]


def apply_triple_lag(
    df: pd.DataFrame, date_col: str = "report_date"
) -> pd.Series:
    """Shift fundamental dates forward by 2 business days (T+2)."""
    if date_col not in df.columns:
        raise ValueError(f"Missing column: {date_col}")
    dates = pd.to_datetime(df[date_col])
    return dates + pd.tseries.offsets.BusinessDay(n=T_PLUS_2_DAYS)


def calculate_sue(
    actual_eps: pd.Series,
    consensus_eps: pd.Series,
    eps_std: pd.Series,
) -> pd.Series:
    """SUE (PEAD): (actual - consensus) / eps_std."""
    num = actual_eps - consensus_eps
    denom = eps_std.replace(0, np.nan)
    return num / denom


def calculate_quality(
    roic: pd.Series,
    dilution: pd.Series,
    grouper: pd.Series | None = None,
) -> pd.Series:
    """
    Quality = Z(ROIC) - Z(Dilution).
    Cross-sectional z-scores. High dilution (share issuance) is bad.
    grouper: series for grouping (e.g. date). NaN -> 0.
    """
    grp = grouper if grouper is not None else pd.Series(roic.index, index=roic.index)
    def _z(s: pd.Series) -> pd.Series:
        t = s.groupby(grp).transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 1e-10 else 0.0
        )
        return t.fillna(0)
    return _z(roic) - _z(dilution)


def calculate_amihud(
    daily_return: pd.Series,
    volume_fiat: pd.Series,
) -> pd.Series:
    """Amihud Illiquidity: abs(return) / volume. inf -> NaN."""
    safe_vol = volume_fiat.replace(0, np.nan)
    out = np.abs(daily_return) / safe_vol
    return out.replace([np.inf, -np.inf], np.nan)


def calculate_vol_compression(returns: pd.Series) -> pd.Series:
    """Ratio of 20d rolling std to 100d rolling std. shift(1) for no look-ahead."""
    std_20 = returns.rolling(20, min_periods=2).std().shift(1)
    std_100 = returns.rolling(100, min_periods=20).std().shift(1)
    ratio = std_20 / std_100.replace(0, np.nan)
    return ratio


class FeatureEngineer:
    """
    Orchestrates Layer 1 feature construction.
    T+2 lag, merge_asof for PiT-safe fundamental alignment.
    """

    def build_features(
        self,
        raw_price_df: pd.DataFrame,
        raw_fundamentals_df: pd.DataFrame,
        macro_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Merge price + lagged fundamentals, compute features.
        If macro_df provided: maps Rates_FI/Rates_SE to Local_Rate per ticker (FH→FI, SS→SE).
        Output: Layer 3 required columns.
        """
        for col in PRICE_REQUIRED:
            if col not in raw_price_df.columns:
                raise ValueError(f"raw_price_df missing: {col}")
        for col in FUND_REQUIRED:
            if col not in raw_fundamentals_df.columns:
                raise ValueError(f"raw_fundamentals_df missing: {col}")

        if raw_price_df[["PX_LAST", "PX_TURN_OVER"]].isna().any().any():
            raise ValueError("NaNs in PX_LAST or PX_TURN_OVER. Fail-fast.")

        price = raw_price_df.copy()
        price["date"] = pd.to_datetime(price["date"])
        # merge_asof requires left keys sorted by 'on' column first (date)
        price = price.sort_values(["date", "ticker"]).reset_index(drop=True)

        fund = raw_fundamentals_df.copy()
        fund["report_date"] = pd.to_datetime(fund["report_date"])
        fund["effective_date"] = apply_triple_lag(fund, "report_date")
        # merge_asof requires right keys sorted by 'on' column first (effective_date)
        fund = fund.sort_values(["effective_date", "ticker"]).reset_index(drop=True)

        merged = pd.merge_asof(
            price,
            fund.drop(columns=["report_date"]),
            left_on="date",
            right_on="effective_date",
            by="ticker",
            direction="backward",
        )

        merged["Return"] = merged.groupby("ticker")["PX_LAST"].pct_change()

        # Seasonality (January Effect Prep): 1 if December rebalance, else 0
        merged["is_january_prep"] = (merged["date"].dt.month == 12).astype(int)

        # Trend Shield: distance to 200-day SMA, shift(1) to avoid look-ahead
        sma200 = merged.groupby("ticker")["PX_LAST"].transform(
            lambda x: x.rolling(200, min_periods=100).mean().shift(1)
        )
        merged["dist_to_sma200"] = ((merged["PX_LAST"] / sma200) - 1).replace(
            [np.inf, -np.inf], np.nan
        )
        merged["dist_to_sma200"] = merged.groupby("ticker")["dist_to_sma200"].shift(1)

        merged["forward_return"] = merged.groupby("ticker")["Return"].shift(-1)

        merged["SUE"] = calculate_sue(
            merged["ACTUAL_EPS"],
            merged["CONSENSUS_EPS"],
            merged["EPS_STD"],
        )

        merged["Quality_Score"] = calculate_quality(
            merged["ROIC"],
            merged["Dilution"],
            grouper=merged["date"],
        ).fillna(0)

        merged["Amihud"] = calculate_amihud(merged["Return"], merged["PX_TURN_OVER"])
        merged["Vol_Compression"] = merged.groupby("ticker")["Return"].transform(
            calculate_vol_compression
        )

        # Local_Rate: explicit mappning per hemvist — SS→Rates_SE, FH→Rates_FI; övriga→NaN (fail-fast)
        if macro_df is not None and "Rates_FI" in macro_df.columns and "Rates_SE" in macro_df.columns:
            macro = macro_df.copy()
            if "date" not in macro.columns:
                macro = macro.reset_index()
                if "index" in macro.columns:
                    macro = macro.rename(columns={"index": "date"})
            macro["date"] = pd.to_datetime(macro["date"])
            macro_sorted = macro[["date", "Rates_FI", "Rates_SE"]].sort_values("date").reset_index(drop=True)
            merged_sorted = merged.sort_values("date").reset_index(drop=True)
            merged = pd.merge_asof(
                merged_sorted,
                macro_sorted,
                on="date",
                direction="backward",
            )
            ticker_str = merged["ticker"].astype(str)
            is_ss = ticker_str.str.endswith(" SS", na=False)
            is_fh = ticker_str.str.endswith(" FH", na=False)
            merged["Local_Rate"] = np.select(
                [is_ss, is_fh],
                [merged["Rates_SE"], merged["Rates_FI"]],
                default=np.nan,
            )
            merged = merged.drop(columns=["Rates_FI", "Rates_SE"], errors="ignore")
        else:
            merged["Local_Rate"] = np.nan

        # Dividend_Yield: produceras av data2 (coalesce i R). Validera; saknas den, lägg till NaN.
        if "Dividend_Yield" not in merged.columns:
            merged["Dividend_Yield"] = np.nan

        out_cols = [
            "date", "ticker", "Quality_Score", "BID_ASK_SPREAD_PCT", "PX_TURN_OVER",
            "SUE", "Amihud", "Vol_Compression", "is_january_prep", "dist_to_sma200",
            "Local_Rate", "Dividend_Yield", "forward_return",
        ]
        out = merged[[c for c in out_cols if c in merged.columns]].copy()

        return out.dropna(subset=["forward_return"]).reset_index(drop=True)
