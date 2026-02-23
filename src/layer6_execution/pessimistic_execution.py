"""
Layer 6: Pessimistic Execution Engine.

Brutal realism — survival over theoretical profit.
- Momentum tax: BUY at Ask + 0.1%*vol, SELL at Bid - 0.1%*vol; top 10% alpha = price taker
- Dynamic spread stress: 2.5x when regime < 0.5 or V2TX > 25
- 5% participation cap per day; unfilled exposes to overnight risk
- Limit fill-trap: -0.2% when limit fills (adverse selection)
- T+1 execution: signal at t close → execute at t+1 open
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

SLIPPAGE_VOL_PCT: float = 0.001  # 0.1% of daily vol
ALPHA_TOP_PCT: float = 0.10  # Top 10% = price taker (full spread + slippage)
SPREAD_STRESS_MULT: float = 2.5
SPREAD_STRESS_REGIME_THRESHOLD: float = 0.5
SPREAD_STRESS_V2TX_THRESHOLD: float = 25.0
PARTICIPATION_CAP_PCT: float = 0.05  # Max 5% of ADV per day
FILL_TRAP_PCT: float = -0.002  # -0.2% when limit fills (adverse selection)
TRADING_DAYS_YEAR: int = 252


def _stressed_spread(
    spread_pct: float,
    regime_exposure: float,
    v2tx: Optional[float],
) -> float:
    """Apply 2.5x when regime < 0.5 or V2TX > 25."""
    stress = False
    if regime_exposure < SPREAD_STRESS_REGIME_THRESHOLD:
        stress = True
    if v2tx is not None and v2tx > SPREAD_STRESS_V2TX_THRESHOLD:
        stress = True
    return spread_pct * SPREAD_STRESS_MULT if stress else spread_pct


def execution_price_buy(
    mid: float,
    spread_pct: float,
    daily_vol_pct: float,
    alpha_rank_top10: bool,
    regime_exposure: float,
    v2tx: Optional[float],
) -> float:
    """
    BUY: Ask + (0.1% * Daily_Vol).
    Top 10% alpha: price taker, full spread + slippage.
    """
    spread_stressed = _stressed_spread(spread_pct, regime_exposure, v2tx)
    half = spread_stressed / 2.0
    ask = mid + half
    slippage = SLIPPAGE_VOL_PCT * daily_vol_pct
    if alpha_rank_top10:
        return ask + (half + slippage)  # Full spread + slippage
    return ask + slippage


def execution_price_sell(
    mid: float,
    spread_pct: float,
    daily_vol_pct: float,
    alpha_rank_top10: bool,
    regime_exposure: float,
    v2tx: Optional[float],
) -> float:
    """
    SELL: Bid - (0.1% * Daily_Vol).
    """
    spread_stressed = _stressed_spread(spread_pct, regime_exposure, v2tx)
    half = spread_stressed / 2.0
    bid = mid - half
    slippage = SLIPPAGE_VOL_PCT * daily_vol_pct
    if alpha_rank_top10:
        return bid - (half + slippage)
    return bid - slippage


def fill_trap_adjustment(execution_price: float) -> float:
    """Limit fill adverse selection: -0.2% move when filled."""
    return execution_price * (1.0 + FILL_TRAP_PCT)


def participation_cap_fill(
    target_notional: float,
    adv: float,
) -> Tuple[float, float]:
    """
    Max 5% ADV per day. Returns (filled_notional, unfilled_notional).
    """
    max_fill = adv * PARTICIPATION_CAP_PCT
    if max_fill >= target_notional:
        return target_notional, 0.0
    return max_fill, target_notional - max_fill


def compute_friction_and_opportunity_cost(
    backtest_weights: pd.DataFrame,
    raw_prices: pd.DataFrame,
    price_returns: pd.DataFrame,
    alpha_df: pd.DataFrame,
    macro_regime_df: pd.DataFrame,
    macro_raw_df: Optional[pd.DataFrame],
    adv_df: pd.DataFrame,
    portfolio_aum: float = 1e6,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Simulate pessimistic execution over backtest. Returns (strategy_returns_adjusted, metrics).

    T+1: signal at rb_date close → execute at next trading day open.
    Tracks Total_Friction_Cost_BPS and Opportunity_Cost_BPS.
    """
    if backtest_weights.empty or price_returns.empty:
        return pd.Series(dtype=float), {
            "Total_Friction_Cost_BPS": 0.0,
            "Opportunity_Cost_BPS": 0.0,
        }

    weight_df = backtest_weights.copy()
    weight_df["date"] = pd.to_datetime(weight_df["date"])
    rb_dates = sorted(weight_df["date"].unique())
    all_dates = price_returns.index.sort_values()
    px_wide = raw_prices.pivot_table(index="date", columns="ticker", values="PX_LAST")
    spread_wide = raw_prices.pivot_table(
        index="date", columns="ticker", values="BID_ASK_SPREAD_PCT"
    ).reindex(columns=px_wide.columns).fillna(0.01)
    vol_wide = price_returns.rolling(20, min_periods=5).std() * np.sqrt(TRADING_DAYS_YEAR)
    vol_wide = vol_wide.reindex(columns=px_wide.columns).fillna(0.20)

    total_friction_bps = 0.0
    total_opportunity_bps = 0.0
    n_periods = 0

    strat_ret = pd.Series(0.0, index=all_dates)
    prev_weights: Dict[str, float] = {}
    prev_rb = None

    for i, rb_date in enumerate(rb_dates):
        w_row = weight_df[weight_df["date"] == rb_date]
        curr_weights = dict(zip(w_row["ticker"], w_row["target_weight"]))
        tickers = list(set(prev_weights.keys()) | set(curr_weights.keys()))
        if not tickers:
            prev_weights = curr_weights
            prev_rb = rb_date
            continue

        # T+1: first execution date = next trading day after rb_date
        pos = np.searchsorted(all_dates.values, rb_date, side="right")
        if pos >= len(all_dates):
            prev_weights = curr_weights
            prev_rb = rb_date
            continue
        exec_date = pd.Timestamp(all_dates[pos])
        end_date = rb_dates[i + 1] if i + 1 < len(rb_dates) else all_dates.max() + pd.Timedelta(days=1)
        period_dates = all_dates[(all_dates >= exec_date) & (all_dates < end_date)]
        if len(period_dates) == 0:
            prev_weights = curr_weights
            prev_rb = rb_date
            continue

        regime = 0.5
        if not macro_regime_df.empty and exec_date in macro_regime_df.index or (macro_regime_df.index <= exec_date).any():
            asof = macro_regime_df.loc[macro_regime_df.index <= exec_date]
            if not asof.empty:
                regime = float(asof["regime"].iloc[-1])
        v2tx_val = None
        if macro_raw_df is not None and not macro_raw_df.empty and "V2TX" in macro_raw_df.columns:
            asof = macro_raw_df.loc[macro_raw_df.index <= exec_date]
            if not asof.empty:
                v2tx_val = float(asof["V2TX"].iloc[-1])

        # Alpha rank for exec_date
        alpha_cross = alpha_df[(alpha_df["date"] < exec_date)].dropna(subset=["alpha_score"])
        if not alpha_cross.empty:
            last_alpha_date = alpha_cross["date"].max()
            alpha_sub = alpha_cross[alpha_cross["date"] == last_alpha_date]
            alpha_series = alpha_sub.set_index("ticker")["alpha_score"]
            top10_cut = alpha_series.quantile(1 - ALPHA_TOP_PCT)
        else:
            alpha_series = pd.Series(dtype=float)
            top10_cut = np.nan

        # Prices, spread, vol, adv asof exec_date
        px_asof = px_wide.loc[px_wide.index <= exec_date].iloc[-1] if (px_wide.index <= exec_date).any() else None
        if px_asof is None:
            prev_weights = curr_weights
            prev_rb = rb_date
            continue

        spread_asof = spread_wide.loc[spread_wide.index <= exec_date].iloc[-1].reindex(px_asof.index).fillna(0.01) if (spread_wide.index <= exec_date).any() else pd.Series(0.01, index=px_asof.index)
        vol_asof = vol_wide.loc[vol_wide.index <= exec_date].iloc[-1].reindex(px_asof.index).fillna(0.20) if (vol_wide.index <= exec_date).any() else pd.Series(0.20, index=px_asof.index)
        vol_daily = vol_asof / np.sqrt(TRADING_DAYS_YEAR)
        adv_asof = adv_df.loc[adv_df.index <= exec_date].iloc[-1].reindex(px_asof.index) if (adv_df.index <= exec_date).any() and not adv_df.empty else pd.Series(np.nan, index=px_asof.index)

        period_friction_bps = 0.0
        period_opp_bps = 0.0

        for t in tickers:
            if t not in px_asof.index or np.isnan(px_asof.get(t, np.nan)):
                continue
            mid = float(px_asof[t])
            spread = float(spread_asof.get(t, 0.01))
            vol_pct = float(vol_daily.get(t, 0.20 / np.sqrt(TRADING_DAYS_YEAR)))
            adv_t = float(adv_asof.get(t, np.nan)) if t in adv_asof.index else np.nan
            top10 = alpha_series.get(t, 0.0) >= top10_cut if not np.isnan(top10_cut) else False

            prev_w = prev_weights.get(t, 0.0)
            curr_w = curr_weights.get(t, 0.0)
            delta_w = curr_w - prev_w
            if abs(delta_w) < 1e-12:
                continue

            notional = abs(delta_w) * portfolio_aum
            if delta_w > 0:  # BUY
                exec_px = execution_price_buy(mid, spread, vol_pct, top10, regime, v2tx_val)
            else:
                exec_px = execution_price_sell(mid, spread, vol_pct, top10, regime, v2tx_val)

            # Participation cap
            filled_notional = notional
            unfilled = 0.0
            if adv_t > 0:
                filled, unfilled = participation_cap_fill(notional, adv_t)
                filled_notional = filled
                if unfilled > 0:
                    opp_cost_bps = 50.0  # Rough: unfilled exposes to overnight risk
                    period_opp_bps += (unfilled / portfolio_aum) * opp_cost_bps

            # Friction: cost vs mid
            cost_vs_mid_pct = abs(exec_px - mid) / mid if mid > 0 else 0.0
            friction_bps = (filled_notional / portfolio_aum) * cost_vs_mid_pct * 10000
            period_friction_bps += friction_bps

        total_friction_bps += period_friction_bps
        total_opportunity_bps += period_opp_bps
        n_periods += 1

        # Deduct friction from first day return (survival realism)
        friction_ret = period_friction_bps / 10000.0
        opp_ret = period_opp_bps / 10000.0

        # Strategy returns for period (T+1: first day is exec_date)
        tickers_in_ret = [t for t in curr_weights if t in price_returns.columns]
        w_vec = np.array([curr_weights[t] for t in tickers_in_ret])
        first_day_done = False
        for dt in period_dates:
            if dt not in price_returns.index:
                continue
            row = price_returns.loc[dt, tickers_in_ret]
            if np.any(np.isnan(row)):
                continue
            r = row.values
            raw_ret = float(np.dot(w_vec, r))
            if not first_day_done and len(period_dates) > 0:
                raw_ret -= friction_ret + opp_ret
                first_day_done = True
            strat_ret.loc[dt] = raw_ret

        prev_weights = curr_weights
        prev_rb = rb_date

    avg_friction = total_friction_bps / max(n_periods, 1)
    avg_opp = total_opportunity_bps / max(n_periods, 1)
    metrics = {
        "Total_Friction_Cost_BPS": total_friction_bps,
        "Opportunity_Cost_BPS": total_opportunity_bps,
    }
    return strat_ret, metrics
