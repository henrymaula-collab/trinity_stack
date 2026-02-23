"""
Layer 1: Feature Engineering.
Processes raw price and fundamental data into PiT-safe features for Layer 3.
T+2 lag enforced; no look-ahead bias.
Insider cluster signal: disclosure-date only, dynamic transaction threshold.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

T_PLUS_2_DAYS = 2

PRICE_REQUIRED: List[str] = ["date", "ticker", "PX_LAST", "PX_TURN_OVER", "BID_ASK_SPREAD_PCT"]

# Toxicity filter: zero turnover in last 3 days → block new positions for 5 days
TOXICITY_ZERO_LOOKBACK_DAYS: int = 3
TOXICITY_BLOCK_DAYS: int = 5

# Toxic print: VWAP_CP missing or >5% deviance from PX_LAST → forbid execution
TOXIC_PRINT_DEVIANCE_PCT: float = 5.0
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
# Insider data: disclosure_date only (no trade date) — prevents look-ahead bias
INSIDER_REQUIRED: List[str] = ["disclosure_date", "ticker", "transaction_value", "insider_id"]

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
    "insider_cluster_signal",
    "forward_return",
]


def compute_stale_price_mask(price_df: pd.DataFrame) -> pd.Series:
    """
    Identify zero-volume or missing-volume days (stale prices).
    is_stale = (PX_TURN_OVER == 0) | PX_TURN_OVER.isna()
    """
    if "PX_TURN_OVER" not in price_df.columns:
        return pd.Series(False, index=price_df.index)
    turn = price_df["PX_TURN_OVER"]
    return (turn == 0) | turn.isna()


def compute_toxicity_block_mask(
    price_df: pd.DataFrame,
    lookback_days: int = TOXICITY_ZERO_LOOKBACK_DAYS,
    block_days: int = TOXICITY_BLOCK_DAYS,
) -> pd.Series:
    """
    If stock has zero turnover on any of last 3 trading days → block new positions for 5 days.
    Stale pricing creates phantom momentum; ML models overfit to it.
    Returns boolean Series: True = blocked.
    Uses lookback to cover block_days: zero in last (lookback+block_days) → block.
    """
    if "PX_TURN_OVER" not in price_df.columns or "date" not in price_df.columns:
        return pd.Series(False, index=price_df.index)
    px = price_df.copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["ticker", "date"]).reset_index(drop=True)
    zero_turn = (px["PX_TURN_OVER"] == 0) | px["PX_TURN_OVER"].isna()
    blocked = pd.Series(False, index=px.index)

    # Block when any of last N days had zero; N=block_days gives ~5 days block after a zero
    window = block_days
    for ticker, grp in px.groupby("ticker"):
        idx = grp.index
        zero = zero_turn.loc[idx].values
        for i in range(len(zero)):
            start = max(0, i - window)
            if np.any(zero[start : i + 1]):
                blocked.loc[idx[i]] = True
    return blocked.reindex(price_df.index).fillna(False)


def compute_toxic_print_mask(
    price_df: pd.DataFrame,
    deviance_pct: float = TOXIC_PRINT_DEVIANCE_PCT,
) -> pd.Series:
    """
    VWAP_CP missing or deviates >5% from PX_LAST → toxic_print_day, forbid execution.
    Nordic small-cap closing auctions are illiquid/manipulable.
    """
    if "VWAP_CP" not in price_df.columns:
        return pd.Series(True, index=price_df.index)
    px = price_df.copy()
    vwap = px["VWAP_CP"]
    px_last = px["PX_LAST"].replace(0, np.nan)
    missing = vwap.isna()
    deviance = np.abs(vwap - px_last) / px_last
    extreme = deviance > (deviance_pct / 100.0)
    return missing | extreme


def apply_stale_price_handling(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill PX_LAST for continuity; add is_stale_price, toxic_print_day, toxicity_block.
    Stale days excluded from rolling vol/correlation (zero return = zero variance fallacy).
    """
    px = price_df.copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["ticker", "date"]).reset_index(drop=True)
    px["is_stale_price"] = compute_stale_price_mask(px)
    px["toxicity_block"] = compute_toxicity_block_mask(px)
    px["toxic_print_day"] = compute_toxic_print_mask(px)
    px["PX_LAST"] = px.groupby("ticker")["PX_LAST"].ffill()
    return px


def calculate_vol_compression(
    returns: pd.Series,
    is_stale: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Ratio of 20d rolling std to 100d rolling std. shift(1) for no look-ahead.
    Excludes is_stale days from volatility: zero return on stale days would
    falsely signal low-risk (zero variance). Use NaN for stale → rolling ignores.
    """
    r = returns.copy()
    if is_stale is not None:
        r = r.where(~is_stale, np.nan)
    std_20 = r.rolling(20, min_periods=2).std().shift(1)
    std_100 = r.rolling(100, min_periods=20).std().shift(1)
    ratio = std_20 / std_100.replace(0, np.nan)
    return ratio


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


def _rolling_median_turnover(price_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Per-ticker rolling median PX_TURN_OVER. shift(1) to avoid look-ahead."""
    piv = price_df.pivot_table(
        index="date", columns="ticker", values="PX_TURN_OVER"
    ).sort_index()
    med = piv.rolling(window, min_periods=max(1, window // 2)).median().shift(1)
    return med


def _filter_insider_by_turnover(
    insider_df: pd.DataFrame,
    turnover_median: pd.DataFrame,
    pct_threshold: float,
) -> pd.DataFrame:
    """
    Keep transactions >= pct_threshold of rolling 20d median turnover.
    Uses disclosure_date for PiT alignment; turnover at disclosure_date.
    """
    if insider_df.empty or turnover_median.empty:
        return insider_df.iloc[0:0] if not insider_df.empty else insider_df
    ins = insider_df.copy()
    ins["disclosure_date"] = pd.to_datetime(ins["disclosure_date"])
    ins["transaction_value"] = pd.to_numeric(ins["transaction_value"], errors="coerce")
    out = []
    for ticker in turnover_median.columns:
        sub = ins[ins["ticker"] == ticker]
        if sub.empty:
            continue
        med_ser = turnover_median[ticker].dropna()
        if med_ser.empty:
            continue
        med_df = med_ser.reset_index()
        med_df.columns = ["date", "median_turnover"]
        merged = pd.merge_asof(
            sub.sort_values("disclosure_date"),
            med_df,
            left_on="disclosure_date",
            right_on="date",
            direction="backward",
        )
        threshold = pct_threshold * merged["median_turnover"]
        keep = merged["transaction_value"] >= threshold
        out.append(merged.loc[keep, list(ins.columns)])
    if not out:
        return ins.iloc[0:0]
    return pd.concat(out, ignore_index=True)


def compute_insider_cluster_signal(
    filtered_insider: pd.DataFrame,
    dates: pd.Series | pd.DatetimeIndex,
    tickers: pd.Series,
    window_days: int = 60,
    decay_factor: float = 0.05,
) -> pd.Series:
    """
    insider_cluster_signal = decayed_sum * log1p(1 + n_unique_insiders).
    Uses only disclosure_date (no trade date).
    """
    idx = dates.index if hasattr(dates, "index") else range(len(dates))
    if filtered_insider.empty:
        return pd.Series(np.nan, index=idx)
    ins = filtered_insider.copy()
    ins["disclosure_date"] = pd.to_datetime(ins["disclosure_date"])
    results = []
    for d, t in zip(dates, tickers, strict=True):
        d = pd.Timestamp(d)
        lo = d - pd.Timedelta(days=window_days)
        sub = ins[(ins["ticker"] == t) & (ins["disclosure_date"] >= lo) & (ins["disclosure_date"] <= d)]
        if sub.empty:
            results.append(0.0)
            continue
        days_ago = (d - sub["disclosure_date"]).dt.days
        decayed = np.exp(-decay_factor * days_ago.astype(float))
        decayed_sum = float(decayed.sum())
        n_unique = sub["insider_id"].nunique()
        signal = decayed_sum * np.log1p(1 + n_unique)
        results.append(float(signal))
    return pd.Series(results, index=idx)


def _walk_forward_splits(
    dates: pd.Series | pd.DatetimeIndex,
    n_folds: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Walk-forward splits: (train_indices, test_indices) per fold.
    Train is strictly before test; no temporal leakage.
    Indices refer to positions in the input dates array.
    """
    d = pd.Series(dates).dropna()
    if len(d) < 30:
        return []
    uniq = np.sort(d.unique())
    n = len(uniq)
    if n < 3 or n_folds < 2:
        return []
    min_train = n // 2
    fold_size = max(1, (n - min_train) // n_folds)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_folds):
        test_start = min_train + i * fold_size
        test_end = min(n, test_start + fold_size)
        if test_start >= test_end:
            continue
        train_end = test_start
        dvals = d.values if hasattr(d, "values") else d
        train_mask = (dvals >= uniq[0]) & (dvals < uniq[train_end])
        test_mask = (dvals >= uniq[test_start]) & (dvals < uniq[test_end])
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        if len(train_idx) < 20 or len(test_idx) < 5:
            continue
        splits.append((train_idx, test_idx))
    return splits


def optimize_insider_parameters(
    insider_df: pd.DataFrame,
    price_df: pd.DataFrame,
    date_ticker_pairs: pd.DataFrame,
    forward_return: pd.Series,
    window_days_grid: Iterable[int] = (30, 60, 90),
    decay_factor_grid: Iterable[float] = (0.02, 0.05, 0.10, 0.15),
    pct_threshold: float = 0.01,
    use_nested_cv: bool = True,
    n_outer_folds: int = 5,
    n_inner_folds: int = 3,
) -> Dict[str, object]:
    """
    Nested Cross-Validation with Walk-Forward for insider hyperparameters.

    Inner loop: hyperparameter tuning (window_days, decay_factor) on train-only data.
    Outer loop: OOS evaluation. Params selected in inner loop never see outer test data.

    Set use_nested_cv=False for legacy simple grid search (deprecated).
    """
    for col in INSIDER_REQUIRED:
        if col not in insider_df.columns:
            logger.warning("optimize_insider_parameters: insider_df missing %s; skipping", col)
            return {}

    turnover = _rolling_median_turnover(price_df)
    filtered = _filter_insider_by_turnover(insider_df, turnover, pct_threshold)
    dtp = date_ticker_pairs.copy()
    dtp["date"] = pd.to_datetime(dtp["date"])
    fr = forward_return.reindex(dtp.index)
    grid = list(
        (wd, dc) for wd in window_days_grid for dc in decay_factor_grid
    )

    if not use_nested_cv or len(dtp) < 100:
        # Fallback: simple grid (no nested CV)
        results: List[Tuple[Tuple[int, float], float]] = []
        for wd, dc in grid:
            sig = compute_insider_cluster_signal(
                filtered, dtp["date"], dtp["ticker"], wd, dc
            )
            valid = sig.notna() & fr.notna() & (sig != 0)
            if valid.sum() < 10:
                results.append(((wd, dc), np.nan))
                continue
            ic = sig.loc[valid].corr(fr.loc[valid], method="spearman")
            results.append(((wd, dc), float(ic)))
        _warn_insider_fragility(results)
        best = max((r for r in results if not np.isnan(r[1])), key=lambda x: x[1], default=(None, np.nan))
        return {"best_params": best[0], "best_ic": best[1], "all_results": results}

    dates_arr = dtp["date"].values
    outer_splits = _walk_forward_splits(pd.DatetimeIndex(dates_arr), n_outer_folds)
    if not outer_splits:
        return optimize_insider_parameters(
            insider_df, price_df, date_ticker_pairs, forward_return,
            window_days_grid, decay_factor_grid, pct_threshold, use_nested_cv=False,
        )

    outer_ics: List[float] = []
    best_params_per_fold: List[Tuple[int, float]] = []
    for train_idx, test_idx in outer_splits:
        train_dates = pd.DatetimeIndex(dates_arr[train_idx])
        inner_splits = _walk_forward_splits(train_dates, n_inner_folds)
        best_ic_inner = -np.inf
        best_param = None
        for wd, dc in grid:
            ic_inner = []
            for itr, ival in inner_splits:
                inner_valid_idx = train_idx[ival]
                sig = compute_insider_cluster_signal(
                    filtered, dtp["date"], dtp["ticker"], wd, dc
                )
                mask = np.isin(np.arange(len(dtp)), inner_valid_idx)
                s = sig.iloc[mask]
                f = fr.iloc[mask]
                v = s.notna() & f.notna() & (s != 0)
                if v.sum() < 10:
                    continue
                ic = s.loc[v].corr(f.loc[v], method="spearman")
                if not np.isnan(ic):
                    ic_inner.append(float(ic))
            if not ic_inner:
                continue
            mean_ic = float(np.mean(ic_inner))
            if mean_ic > best_ic_inner:
                best_ic_inner = mean_ic
                best_param = (wd, dc)
        if best_param is None:
            continue
        best_params_per_fold.append(best_param)
        wd, dc = best_param
        sig = compute_insider_cluster_signal(
            filtered, dtp["date"], dtp["ticker"], wd, dc
        )
        test_mask = np.isin(np.arange(len(dtp)), test_idx)
        s = sig.iloc[test_mask]
        f = fr.iloc[test_mask]
        v = s.notna() & f.notna() & (s != 0)
        if v.sum() < 5:
            continue
        outer_ic = s.loc[v].corr(f.loc[v], method="spearman")
        if not np.isnan(outer_ic):
            outer_ics.append(float(outer_ic))

    if not outer_ics:
        return optimize_insider_parameters(
            insider_df, price_df, date_ticker_pairs, forward_return,
            window_days_grid, decay_factor_grid, pct_threshold, use_nested_cv=False,
        )
    median_params = (
        int(np.median([p[0] for p in best_params_per_fold])),
        float(np.median([p[1] for p in best_params_per_fold])),
    )
    return {
        "best_params": median_params,
        "best_ic": float(np.mean(outer_ics)),
        "oos_ic_std": float(np.std(outer_ics)) if len(outer_ics) > 1 else 0.0,
        "n_outer_folds": len(outer_ics),
        "all_results": [],  # Nested CV does not return per-param grid
    }


def _warn_insider_fragility(
    results: List[Tuple[Tuple[int, float], float]],
) -> None:
    decay_ics: Dict[float, List[float]] = {}
    for (_, dc), ic in results:
        decay_ics.setdefault(dc, []).append(ic)
    for dc, ics in decay_ics.items():
        ics_clean = [x for x in ics if not np.isnan(x)]
        if len(ics_clean) >= 2 and (max(ics_clean) - min(ics_clean)) > 0.03:
            logger.warning(
                "INSIDER_FRAGILITY: Signal prediktionskraft kollapsar vid små ändringar i decay_factor=%.3f. "
                "IC-spridning: [%.4f, %.4f]. Överväg stabilare parametrar.",
                dc, min(ics_clean), max(ics_clean),
            )


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
        insider_df: pd.DataFrame | None = None,
        insider_window_days: int = 60,
        insider_decay_factor: float = 0.05,
        insider_pct_threshold: float = 0.01,
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

        if raw_price_df["PX_LAST"].isna().all():
            raise ValueError("PX_LAST must have at least some non-NaN values")

        price = apply_stale_price_handling(raw_price_df)
        price["date"] = pd.to_datetime(price["date"])
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

        # VWAP-based returns: Nordic small-cap PX_LAST from closing auctions is manipulable
        if "VWAP_CP" in merged.columns:
            merged["Return"] = merged.groupby("ticker")["VWAP_CP"].pct_change()
            merged.loc[merged["toxic_print_day"], "Return"] = np.nan
        else:
            merged["Return"] = merged.groupby("ticker")["PX_LAST"].pct_change()
        merged.loc[merged["is_stale_price"], "Return"] = np.nan

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
        vol_comp = merged.groupby("ticker").apply(
            lambda g: calculate_vol_compression(g["Return"], g["is_stale_price"])
        )
        merged["Vol_Compression"] = vol_comp.droplevel(0).reindex(merged.index).values

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

        # Insider cluster signal (disclosure_date only; dynamic turnover threshold)
        if insider_df is not None and all(c in insider_df.columns for c in INSIDER_REQUIRED):
            turnover_med = _rolling_median_turnover(price)
            filtered = _filter_insider_by_turnover(
                insider_df, turnover_med, insider_pct_threshold
            )
            merged["insider_cluster_signal"] = compute_insider_cluster_signal(
                filtered,
                merged["date"],
                merged["ticker"],
                window_days=insider_window_days,
                decay_factor=insider_decay_factor,
            )
        else:
            merged["insider_cluster_signal"] = np.nan

        out_cols = [
            "date", "ticker", "Quality_Score", "BID_ASK_SPREAD_PCT", "PX_TURN_OVER",
            "SUE", "Amihud", "Vol_Compression", "is_january_prep", "dist_to_sma200",
            "Local_Rate", "Dividend_Yield", "insider_cluster_signal", "forward_return",
        ]
        out = merged[[c for c in out_cols if c in merged.columns]].copy()

        return out.dropna(subset=["forward_return"]).reset_index(drop=True)
