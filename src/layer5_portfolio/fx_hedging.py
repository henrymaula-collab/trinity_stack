"""
Layer 5: FX Hedging — CIRP forward premium, three backtest tracks, regime analysis.
Empirical hedging cost and regime-dependent hedging strategy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS_YEAR = 252
MONTHS_PER_YEAR = 12
REGIME_RISK_OFF_THRESHOLD = 0.5  # Hedge when regime < 0.5 (Risk-Off)


def infer_currency_from_tickers(tickers: List[str]) -> pd.Series:
    """
    Infer EUR/SEK from ticker suffix. SS -> SEK, FH -> EUR.
    Used when currency_series not provided.
    """
    curr = {}
    for t in tickers:
        curr[t] = "SEK" if " SS" in str(t) or t.endswith(" SS") else "EUR"
    return pd.Series(curr)


def forward_premium_1m_cirp(
    macro_df: pd.DataFrame,
    rates_col: str = "Rates",
    rates_se_col: str = "Rates_SE",
) -> pd.Series:
    """
    1-month Forward Premium (hedging cost) for SEK/EUR via Covered Interest Rate Parity.
    CIRP: (F-S)/S ≈ (r_SEK - r_EUR) * τ for τ = 1/12.
    Rates in decimal (e.g. 0.03 = 3%).
    Returns continuous series index=date, value=forward premium (decimal, per month).
    """
    if rates_col not in macro_df.columns or rates_se_col not in macro_df.columns:
        return pd.Series(dtype=float)
    df = macro_df.copy()
    if "date" in df.columns and df.index.name != "date":
        df = df.set_index("date")
    df = df.sort_index()
    r_eur = df[rates_col]
    r_sek = df[rates_se_col]
    tau = 1.0 / MONTHS_PER_YEAR
    premium = (r_sek - r_eur) * tau
    return premium


def _strategy_returns_from_weights(
    weights_df: pd.DataFrame,
    price_returns: pd.DataFrame,
) -> pd.Series:
    """Compute daily strategy returns from weight records and price returns."""
    if weights_df.empty or price_returns.empty:
        return pd.Series(dtype=float)
    weight_df = weights_df.copy()
    weight_df["date"] = pd.to_datetime(weight_df["date"])
    dates = sorted(weight_df["date"].unique())
    strat_ret = pd.Series(0.0, index=price_returns.index)
    all_dates = price_returns.index

    for i, rb_date in enumerate(dates):
        w_row = weight_df[weight_df["date"] == rb_date]
        curr_weights = dict(zip(w_row["ticker"], w_row["target_weight"]))
        start = rb_date
        end = dates[i + 1] if i + 1 < len(dates) else all_dates.max() + pd.Timedelta(days=1)
        mask = (all_dates > start) & (all_dates < end)
        period = price_returns.loc[mask]
        tickers = [t for t in curr_weights if t in period.columns]
        w_vec = np.array([curr_weights[t] for t in tickers])
        for dt, row in period[tickers].iterrows():
            r = row.values
            if np.any(np.isnan(r)):
                continue
            strat_ret.loc[dt] = float(np.dot(w_vec, r))
    return strat_ret


def _apply_hedging_cost(
    strat_ret: pd.Series,
    weights_df: pd.DataFrame,
    forward_premium: pd.Series,
    regime_series: pd.Series,
    currency_series: pd.Series,
    hedge_mode: str,
) -> pd.Series:
    """
    Deduct forward cost from SEK portion. hedge_mode: 'unhedged' | 'always_hedged' | 'regime_hedged'.
    Cost = sek_weight * forward_premium_1m, prorated daily (≈ /21).
    """
    if hedge_mode == "unhedged":
        return strat_ret.copy()

    weight_df = weights_df.copy()
    weight_df["date"] = pd.to_datetime(weight_df["date"])
    fp_dates = weight_df["date"].unique()
    forward_premium = forward_premium.reindex(fp_dates).ffill().fillna(0)
    regime_series = regime_series.reindex(fp_dates).ffill().fillna(0.5)

    dates = sorted(weight_df["date"].unique())
    out = strat_ret.copy()
    all_dates = strat_ret.index

    for i, rb_date in enumerate(dates):
        w_row = weight_df[weight_df["date"] == rb_date]
        sek_weight = 0.0
        for _, r in w_row.iterrows():
            if currency_series.get(r["ticker"], "EUR") == "SEK":
                sek_weight += r["target_weight"]
        if sek_weight <= 0:
            continue
        fp = float(forward_premium.get(rb_date, 0.0))
        reg = float(regime_series.get(rb_date, 0.5))
        do_hedge = hedge_mode == "always_hedged" or (
            hedge_mode == "regime_hedged" and reg < REGIME_RISK_OFF_THRESHOLD
        )
        if not do_hedge:
            continue
        cost_daily = sek_weight * (fp / 21.0)
        start = rb_date
        end = dates[i + 1] if i + 1 < len(dates) else all_dates.max() + pd.Timedelta(days=1)
        mask = (all_dates > start) & (all_dates < end)
        out.loc[mask] = out.loc[mask] - cost_daily
    return out


def _compute_metrics(returns: pd.Series) -> Dict[str, float]:
    """Sharpe and Max DD from daily returns."""
    ret = returns.dropna()
    if len(ret) < 10 or ret.std() < 1e-12:
        return {"sharpe": np.nan, "max_dd": np.nan}
    sharpe = float((ret.mean() / ret.std()) * np.sqrt(TRADING_DAYS_YEAR))
    equity = (1 + ret).cumprod()
    cummax = equity.cummax()
    dd = (equity - cummax) / cummax
    max_dd = float(dd.min())
    return {"sharpe": sharpe, "max_dd": max_dd}


def run_fx_hedging_report(
    weights_df: pd.DataFrame,
    price_returns: pd.DataFrame,
    macro_df: pd.DataFrame,
    macro_regime_df: pd.DataFrame,
    currency_series: Optional[pd.Series] = None,
    forward_premium: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Run three parallel backtest tracks: Unhedged, Always_Hedged, Regime_Hedged.
    Returns report with Sharpe and Max DD per track, broken by HMM regime.
    """
    if weights_df.empty or price_returns.empty:
        return {"tracks": {}, "by_regime": {}}

    if currency_series is None:
        tickers = weights_df["ticker"].unique().tolist()
        currency_series = infer_currency_from_tickers(tickers)

    if forward_premium is None:
        forward_premium = forward_premium_1m_cirp(macro_df)

    regime_series = macro_regime_df["regime"] if "regime" in macro_regime_df.columns else pd.Series(0.5, index=macro_regime_df.index)

    strat_raw = _strategy_returns_from_weights(weights_df, price_returns)
    if strat_raw.empty:
        return {"tracks": {}, "by_regime": {}}

    ret_unhedged = strat_raw
    ret_always = _apply_hedging_cost(
        strat_raw, weights_df, forward_premium, regime_series, currency_series, "always_hedged"
    )
    ret_regime = _apply_hedging_cost(
        strat_raw, weights_df, forward_premium, regime_series, currency_series, "regime_hedged"
    )

    metrics_unhedged = _compute_metrics(ret_unhedged)
    metrics_always = _compute_metrics(ret_always)
    metrics_regime = _compute_metrics(ret_regime)

    # By regime: align returns with regime
    regime_aligned = regime_series.reindex(ret_unhedged.index).ffill().fillna(0.5)
    risk_on = regime_aligned >= REGIME_RISK_OFF_THRESHOLD
    risk_off = ~risk_on

    by_regime: Dict[str, Dict[str, Dict[str, float]]] = {}
    for label, mask in [("Risk-On", risk_on), ("Risk-Off", risk_off)]:
        if mask.sum() < 5:
            by_regime[label] = {
                "Unhedged": {"sharpe": np.nan, "max_dd": np.nan},
                "Always_Hedged": {"sharpe": np.nan, "max_dd": np.nan},
                "Regime_Hedged": {"sharpe": np.nan, "max_dd": np.nan},
            }
            continue
        r_u = ret_unhedged.loc[mask]
        r_a = ret_always.loc[mask]
        r_r = ret_regime.loc[mask]
        by_regime[label] = {
            "Unhedged": _compute_metrics(r_u),
            "Always_Hedged": _compute_metrics(r_a),
            "Regime_Hedged": _compute_metrics(r_r),
        }

    report: Dict[str, Any] = {
        "tracks": {
            "Unhedged": metrics_unhedged,
            "Always_Hedged": metrics_always,
            "Regime_Hedged": metrics_regime,
        },
        "by_regime": by_regime,
    }
    return report


def format_fx_hedging_report(report: Dict[str, Any]) -> str:
    """
    Format FX hedging report for logging/display.
    Shows Sharpe and Max DD for three tracks, broken by regime.
    """
    lines = [
        "=" * 60,
        "FX Hedging Report — Unhedged vs Always_Hedged vs Regime_Hedged",
        "=" * 60,
        "",
        "Aggregate:",
    ]
    for track, m in report.get("tracks", {}).items():
        sh = m.get("sharpe", np.nan)
        dd = m.get("max_dd", np.nan)
        lines.append(f"  {track:18s} Sharpe={sh:.3f}  MaxDD={dd:.2%}")

    lines.append("")
    lines.append("By HMM Regime:")
    for regime, tracks in report.get("by_regime", {}).items():
        lines.append(f"  {regime}:")
        for track, m in tracks.items():
            sh = m.get("sharpe", np.nan)
            dd = m.get("max_dd", np.nan)
            lines.append(f"    {track:18s} Sharpe={sh:.3f}  MaxDD={dd:.2%}")

    lines.append("")
    return "\n".join(lines)
