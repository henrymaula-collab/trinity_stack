"""
Layer 6: Pessimistic Execution Engine.

Step-Function Market Impact (Nordic small-cap microstructure):
- Replace smooth Square-Root Law with discrete Step-Function Impact; liquidity collapses in clusters.
- Liquidity Cliff Penalty: order > 2% of daily volume → add 3× historical Bid/Ask spread.
- Internalization Masking: 30% of PX_TURN_OVER = block trades outside order book; use 70% for participation.

Legacy: Square-Root Law retained as optional fallback.
Dynamic spread stress: 2.5x when regime < 0.5 or V2TX > 25.
T+1 execution: signal at t close → execute at t+1 open.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.layer6_execution.fill_probability_model import (
    PARTICIPATION_CAP_PCT,
    simulate_batch_limit_fills,
)

ALPHA_TOP_PCT: float = 0.10  # Top 10% = price taker (full spread + slippage)
SPREAD_STRESS_MULT: float = 2.5
SPREAD_STRESS_REGIME_THRESHOLD: float = 0.5
SPREAD_STRESS_V2TX_THRESHOLD: float = 25.0
FILL_TRAP_PCT: float = -0.002  # -0.2% when limit fills (adverse selection)
TRADING_DAYS_YEAR: int = 252

# Square-Root Law coefficient (configurable)
MARKET_IMPACT_C_DEFAULT: float = 1.0

# Adverse selection: participation > 15% of intraday liquidity
ADVERSE_SELECTION_THRESHOLD: float = 0.15
ADVERSE_SELECTION_PENALTY_MULT: float = 1.5  # 1.5× slippage when above threshold

# News/sentiment-driven signals: 50 bps penalty first 24h (HFT gets there first)
ADVERSE_SELECTION_NEWS_BPS: float = 50.0

# Step-Function Impact & Nordic microstructure
INTERNALIZATION_MASKING: float = 0.70  # Only 70% of PX_TURN_OVER is order-book liquidity
LIQUIDITY_CLIFF_THRESHOLD_PCT: float = 0.02  # 2% of daily volume
LIQUIDITY_CLIFF_SPREAD_MULT: float = 3.0  # 3× spread penalty when over threshold
USE_STEP_FUNCTION_IMPACT: bool = True

# Execution collapse: reject order when exceeding visible liquidity (forced holding)
EXECUTION_COLLAPSE_PARTICIPATION_THRESHOLD: float = 0.15

# Default AUM levels for capacity stress test (EUR)
DEFAULT_CAPACITY_AUM_LIST: List[float] = [1e6, 5e6, 10e6, 20e6, 50e6]
SHARPE_CAPACITY_LIMIT: float = 1.0  # Strategy non-allocatable below this

# Conviction-scaled limit pricing (for probabilistic fill model)
CONVICTION_DISCOUNT: float = 0.25  # Top alpha
ILLIQUID_DISCOUNT: float = 0.75
NORMAL_DISCOUNT: float = 0.40


def _limit_price(
    mid: float,
    spread_pct: float,
    side: str,
    top10_alpha: bool,
    illiquid: bool,
) -> float:
    """Conviction-scaled limit. BUY: below mid; SELL: above mid."""
    discount = CONVICTION_DISCOUNT if top10_alpha else (ILLIQUID_DISCOUNT if illiquid else NORMAL_DISCOUNT)
    offset = mid * (discount * spread_pct / 100.0)
    if side.upper() == "BUY":
        return mid - offset
    return mid + offset


def available_order_book_volume(
    daily_volume: float,
    internalization_factor: float = INTERNALIZATION_MASKING,
) -> float:
    """
    Internalization Masking: 30% of PX_TURN_OVER = block trades outside order book.
    Participation rate based on only 70% of reported volume.
    """
    if daily_volume <= 0:
        return 0.0
    return daily_volume * internalization_factor


def liquidity_cliff_penalty_pct(
    participation_rate: float,
    spread_pct: float,
    threshold_pct: float = LIQUIDITY_CLIFF_THRESHOLD_PCT,
    mult: float = LIQUIDITY_CLIFF_SPREAD_MULT,
) -> float:
    """
    If order exceeds threshold (e.g. 2% of daily volume), add 3× spread penalty.
    Returns extra slippage as decimal (e.g. 0.003 = 30 bps for 1% spread × 3).
    """
    if participation_rate <= threshold_pct or spread_pct <= 0:
        return 0.0
    return float(mult * spread_pct / 100.0)


def step_function_slippage_pct(
    order_size: float,
    available_volume: float,
    sigma_annual: float,
) -> float:
    """
    Discrete Step-Function Impact: liquidity collapses in clusters.
    Below 0.5%: base. 0.5–1%: 2×. 1–2%: 4×. Above 2%: 8×.
    """
    if available_volume <= 0 or order_size <= 0:
        sigma_daily = sigma_annual / np.sqrt(TRADING_DAYS_YEAR)
        return float(0.001 * sigma_daily)  # nominal fallback
    participation = min(order_size / available_volume, 1.0)
    sigma_daily = sigma_annual / np.sqrt(TRADING_DAYS_YEAR)
    base = float(0.5 * sigma_daily * np.sqrt(0.005))
    if participation <= 0.005:
        mult = 1.0
    elif participation <= 0.01:
        mult = 2.0
    elif participation <= 0.02:
        mult = 4.0
    else:
        mult = 8.0
    return float(base * mult)


def square_root_slippage_pct(
    order_size: float,
    daily_volume: float,
    sigma_annual: float,
    c: float = MARKET_IMPACT_C_DEFAULT,
) -> float:
    """
    Square-Root Law: Slippage = c * σ * sqrt(Order_Size / Daily_Volume).
    Returns slippage as decimal (e.g. 0.001 = 10 bps).
    Legacy fallback when USE_STEP_FUNCTION_IMPACT=False.
    """
    if daily_volume <= 0 or order_size <= 0:
        return 0.0
    participation = order_size / daily_volume
    participation = min(participation, 1.0)  # cap for numerical stability
    sigma_daily = sigma_annual / np.sqrt(TRADING_DAYS_YEAR)
    return float(c * sigma_daily * np.sqrt(participation))


def participation_slippage_multiplier(participation_rate: float) -> float:
    """
    Non-linear slippage factor vs participation. Replaces fixed participation cap.
    Low participation: ~1×. High participation: steep increase.
    """
    if participation_rate <= 0:
        return 1.0
    # Quadratic lift: 1 + (p/0.10)^2 so at 10% participation ~2×, at 20% ~5×
    return float(1.0 + (participation_rate / 0.10) ** 2)


def adverse_selection_multiplier(participation_rate: float) -> float:
    """
    Extra penalty when order exceeds 15% of intraday liquidity.
    """
    if participation_rate <= ADVERSE_SELECTION_THRESHOLD:
        return 1.0
    return ADVERSE_SELECTION_PENALTY_MULT


def is_execution_collapse(
    order_notional: float,
    available_volume: float,
    threshold: float = EXECUTION_COLLAPSE_PARTICIPATION_THRESHOLD,
) -> bool:
    """
    Order exceeds visible liquidity → execution collapse.
    Reject instead of filling at extreme theoretical price.
    """
    if available_volume <= 0:
        return True
    participation = order_notional / available_volume
    return participation > threshold


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
    order_size: float,
    daily_volume: float,
    sigma_annual: float,
    alpha_rank_top10: bool,
    regime_exposure: float,
    v2tx: Optional[float],
    impact_c: float = MARKET_IMPACT_C_DEFAULT,
) -> float:
    """
    BUY: Ask + market impact.
    Step-Function Impact + Internalization Masking (70% available vol) + Liquidity Cliff (3× spread if >2%).
    When order_size/daily_volume unavailable, use nominal slippage.
    """
    spread_stressed = _stressed_spread(spread_pct, regime_exposure, v2tx)
    spread_dec = spread_stressed / 100.0

    if order_size > 0 and daily_volume > 0:
        avail_vol = available_order_book_volume(daily_volume)
        participation = min(order_size / avail_vol, 1.0) if avail_vol > 0 else 1.0
        if USE_STEP_FUNCTION_IMPACT:
            slippage_pct = step_function_slippage_pct(order_size, avail_vol, sigma_annual)
        else:
            slippage_pct = square_root_slippage_pct(
                order_size, daily_volume, sigma_annual, c=impact_c
            )
        slippage_pct *= participation_slippage_multiplier(participation)
        slippage_pct *= adverse_selection_multiplier(participation)
        slippage_pct += liquidity_cliff_penalty_pct(participation, spread_stressed)
    else:
        slippage_pct = 0.001 * (sigma_annual / 0.20)

    if alpha_rank_top10:
        return mid * (1.0 + spread_dec + slippage_pct)
    return mid * (1.0 + spread_dec / 2.0 + slippage_pct)


def execution_price_sell(
    mid: float,
    spread_pct: float,
    order_size: float,
    daily_volume: float,
    sigma_annual: float,
    alpha_rank_top10: bool,
    regime_exposure: float,
    v2tx: Optional[float],
    impact_c: float = MARKET_IMPACT_C_DEFAULT,
) -> float:
    """
    SELL: Bid - market impact.
    Step-Function Impact + Internalization Masking + Liquidity Cliff.
    """
    spread_stressed = _stressed_spread(spread_pct, regime_exposure, v2tx)
    spread_dec = spread_stressed / 100.0

    if order_size > 0 and daily_volume > 0:
        avail_vol = available_order_book_volume(daily_volume)
        participation = min(order_size / avail_vol, 1.0) if avail_vol > 0 else 1.0
        if USE_STEP_FUNCTION_IMPACT:
            slippage_pct = step_function_slippage_pct(order_size, avail_vol, sigma_annual)
        else:
            slippage_pct = square_root_slippage_pct(
                order_size, daily_volume, sigma_annual, c=impact_c
            )
        slippage_pct *= participation_slippage_multiplier(participation)
        slippage_pct *= adverse_selection_multiplier(participation)
        slippage_pct += liquidity_cliff_penalty_pct(participation, spread_stressed)
    else:
        participation = 0.0
        slippage_pct = 0.001 * (sigma_annual / 0.20)

    if alpha_rank_top10:
        return mid * (1.0 - spread_dec - slippage_pct)
    return mid * (1.0 - spread_dec / 2.0 - slippage_pct)


def fill_trap_adjustment(execution_price: float) -> float:
    """Limit fill adverse selection: -0.2% move when filled."""
    return execution_price * (1.0 + FILL_TRAP_PCT)


def _news_catalyst_tickers(
    news_df: pd.DataFrame,
    rb_date: pd.Timestamp,
    tickers: List[str],
) -> set:
    """
    Tickers with news in last 24h: rb_date or rb_date-1 trading day.
    First 24h after news = adverse selection (HFT gets there first).
    """
    if news_df is None or news_df.empty or "date" not in news_df.columns or "ticker" not in news_df.columns:
        return set()
    news_df = news_df.copy()
    news_df["date"] = pd.to_datetime(news_df["date"]).dt.normalize()
    rb_norm = pd.Timestamp(rb_date).normalize()
    window_start = rb_norm - pd.Timedelta(days=2)
    sub = news_df[
        (news_df["ticker"].isin(tickers))
        & (news_df["date"] >= window_start)
        & (news_df["date"] <= rb_norm)
    ]
    return set(sub["ticker"].unique())


def compute_friction_and_opportunity_cost(
    backtest_weights: pd.DataFrame,
    raw_prices: pd.DataFrame,
    price_returns: pd.DataFrame,
    alpha_df: pd.DataFrame,
    macro_regime_df: pd.DataFrame,
    macro_raw_df: Optional[pd.DataFrame],
    adv_df: pd.DataFrame,
    portfolio_aum: float = 1e6,
    impact_c: float = MARKET_IMPACT_C_DEFAULT,
    news_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Simulate pessimistic execution over backtest. Returns (strategy_returns_adjusted, metrics).

    T+1 VWAP: signal at rb_date → execute at VWAP_CP on next trading day (T+1).
    Never use rb_date's PX_LAST (signal latency: we cannot trade at T close).
    News-driven signals: 50 bps adverse selection penalty first 24h.

    Execution collapse: when order exceeds visible liquidity (participation > 15%),
    reject order instead of extreme theoretical fill. Forced holding (inventory risk):
    position kept, mark-to-market daily. Logs Avg_Time_In_Market_Extension_Days.
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
    use_vwap = "VWAP_CP" in raw_prices.columns
    vwap_wide = (
        raw_prices.pivot_table(index="date", columns="ticker", values="VWAP_CP").reindex_like(px_wide)
        if use_vwap
        else None
    )
    spread_wide = raw_prices.pivot_table(
        index="date", columns="ticker", values="BID_ASK_SPREAD_PCT"
    ).reindex(columns=px_wide.columns).fillna(0.01)
    use_probabilistic = "PX_HIGH" in raw_prices.columns and "PX_LOW" in raw_prices.columns
    vol_wide = price_returns.rolling(20, min_periods=5).std() * np.sqrt(TRADING_DAYS_YEAR)
    vol_wide = vol_wide.reindex(columns=px_wide.columns).fillna(0.20)

    total_friction_bps = 0.0
    total_opportunity_bps = 0.0
    n_periods = 0
    rejected_sell_count = 0
    total_extra_holding_days = 0.0

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
        all_dates_idx = pd.DatetimeIndex(all_dates)
        gt = all_dates_idx > rb_date
        if not gt.any():
            prev_weights = curr_weights
            prev_rb = rb_date
            continue
        exec_date = pd.Timestamp(all_dates_idx[gt][0])
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

        # T+1 only: never use rb_date's PX_LAST (signal latency)
        t1_mask = (px_wide.index > rb_date) & (px_wide.index <= exec_date)
        if not t1_mask.any():
            prev_weights = curr_weights
            prev_rb = rb_date
            continue
        px_asof = px_wide.loc[t1_mask].iloc[-1]
        vwap_asof = None
        if vwap_wide is not None:
            common_idx = vwap_wide.index.intersection(px_wide.loc[t1_mask].index)
            if len(common_idx) > 0:
                vwap_asof = vwap_wide.loc[common_idx].iloc[-1].reindex(px_asof.index)
        base_price_asof = vwap_asof.where(vwap_asof.notna()).fillna(px_asof) if vwap_asof is not None else px_asof

        news_catalyst = _news_catalyst_tickers(
            news_df if news_df is not None else pd.DataFrame(), rb_date, tickers
        )

        spread_asof = spread_wide.loc[spread_wide.index <= exec_date].iloc[-1].reindex(px_asof.index).fillna(0.01) if (spread_wide.index <= exec_date).any() else pd.Series(0.01, index=px_asof.index)
        vol_asof = vol_wide.loc[vol_wide.index <= exec_date].iloc[-1].reindex(px_asof.index).fillna(0.20) if (vol_wide.index <= exec_date).any() else pd.Series(0.20, index=px_asof.index)
        sigma_annual_asof = vol_asof
        adv_asof = adv_df.loc[adv_df.index <= exec_date].iloc[-1].reindex(px_asof.index) if (adv_df.index <= exec_date).any() and not adv_df.empty else pd.Series(np.nan, index=px_asof.index)

        period_friction_bps = 0.0
        period_opp_bps = 0.0
        spread_cut_illiquid = float(spread_asof.quantile(0.75)) if len(spread_asof) > 0 else 0.05
        realized_weights: Dict[str, float] = dict(curr_weights)

        if use_probabilistic:
            alpha_strength_map = {}
            if not alpha_series.empty:
                ranks = alpha_series.rank(pct=True)
                n = len(ranks)
                for tick, r in ranks.items():
                    alpha_strength_map[tick] = float(r) if n > 1 else 0.5
            orders_rows = []
            for t in tickers:
                if t not in base_price_asof.index or np.isnan(base_price_asof.get(t, np.nan)):
                    continue
                prev_w = prev_weights.get(t, 0.0)
                curr_w = curr_weights.get(t, 0.0)
                delta_w = curr_w - prev_w
                if abs(delta_w) < 1e-12:
                    continue
                mid = float(base_price_asof[t])
                spread = float(spread_asof.get(t, 0.01))
                top10 = alpha_series.get(t, 0.0) >= top10_cut if not np.isnan(top10_cut) else False
                illiquid = spread >= spread_cut_illiquid
                side = "BUY" if delta_w > 0 else "SELL"
                limit_px = _limit_price(mid, spread, side, top10, illiquid)
                notional = abs(delta_w) * portfolio_aum
                alpha_strength = alpha_strength_map.get(t, 0.5)
                orders_rows.append({
                    "ticker": t,
                    "limit_price": limit_px,
                    "order_notional": notional,
                    "delta_weight": delta_w,
                    "mid": mid,
                    "spread": spread,
                    "alpha_strength": alpha_strength,
                    "news_catalyst": t in news_catalyst,
                    "sigma_ann": float(sigma_annual_asof.get(t, 0.20)),
                    "adv_t": float(adv_asof.get(t, np.nan)) if t in adv_asof.index else np.nan,
                    "top10": top10,
                })
            if orders_rows:
                orders_df = pd.DataFrame(orders_rows)
                filled_orders, overnight_gap_pct = simulate_batch_limit_fills(
                    orders_df, raw_prices, exec_date, all_dates,
                    participation_cap=PARTICIPATION_CAP_PCT,
                )
                for _, r in filled_orders.iterrows():
                    t = r["ticker"]
                    mid = r["mid"]
                    spread = r["spread"]
                    fill_ratio = float(r["fill_ratio"])
                    filled_notional = float(r["filled_notional"])
                    notional = float(r["order_notional"])
                    adv_t = r["adv_t"]
                    avail_vol = available_order_book_volume(adv_t if adv_t and adv_t > 0 else notional * 10.0)
                    if is_execution_collapse(notional, avail_vol):
                        realized_weights[t] = prev_weights.get(t, 0.0)
                        if r["delta_weight"] < 0:
                            rejected_sell_count += 1
                            total_extra_holding_days += len(period_dates)
                        continue
                    if filled_notional <= 0:
                        continue
                    sigma_ann = r["sigma_ann"]
                    adv_t = r["adv_t"]
                    daily_vol = adv_t if adv_t > 0 else filled_notional * 10.0
                    if r["delta_weight"] > 0:
                        exec_px = execution_price_buy(
                            mid, spread, filled_notional, daily_vol, sigma_ann,
                            r["top10"], regime, v2tx_val, impact_c=impact_c,
                        )
                        if r.get("news_catalyst", False):
                            exec_px *= 1.0 + ADVERSE_SELECTION_NEWS_BPS / 10000.0
                    else:
                        exec_px = execution_price_sell(
                            mid, spread, filled_notional, daily_vol, sigma_ann,
                            r["top10"], regime, v2tx_val, impact_c=impact_c,
                        )
                        if r.get("news_catalyst", False):
                            exec_px *= 1.0 - ADVERSE_SELECTION_NEWS_BPS / 10000.0
                    cost_vs_mid_pct = abs(exec_px - mid) / mid if mid > 0 else 0.0
                    period_friction_bps += (filled_notional / portfolio_aum) * cost_vs_mid_pct * 10000
                spill_cost_bps = (overnight_gap_pct / portfolio_aum) * 10000
                period_friction_bps += spill_cost_bps
        else:
            for t in tickers:
                if t not in base_price_asof.index or np.isnan(base_price_asof.get(t, np.nan)):
                    continue
                mid = float(base_price_asof[t])
                spread = float(spread_asof.get(t, 0.01))
                sigma_ann = float(sigma_annual_asof.get(t, 0.20))
                adv_t = float(adv_asof.get(t, np.nan)) if t in adv_asof.index else np.nan
                top10 = alpha_series.get(t, 0.0) >= top10_cut if not np.isnan(top10_cut) else False

                prev_w = prev_weights.get(t, 0.0)
                curr_w = curr_weights.get(t, 0.0)
                delta_w = curr_w - prev_w
                if abs(delta_w) < 1e-12:
                    continue

                notional = abs(delta_w) * portfolio_aum
                daily_vol = adv_t if adv_t > 0 else notional * 10.0
                avail_vol = available_order_book_volume(daily_vol)
                if is_execution_collapse(notional, avail_vol):
                    realized_weights[t] = prev_w
                    if delta_w < 0:
                        rejected_sell_count += 1
                        total_extra_holding_days += len(period_dates)
                    continue

                if delta_w > 0:
                    exec_px = execution_price_buy(
                        mid, spread, notional, daily_vol, sigma_ann, top10, regime, v2tx_val, impact_c=impact_c
                    )
                    if t in news_catalyst:
                        exec_px *= 1.0 + ADVERSE_SELECTION_NEWS_BPS / 10000.0
                else:
                    exec_px = execution_price_sell(
                        mid, spread, notional, daily_vol, sigma_ann, top10, regime, v2tx_val, impact_c=impact_c
                    )
                    if t in news_catalyst:
                        exec_px *= 1.0 - ADVERSE_SELECTION_NEWS_BPS / 10000.0

                filled_notional = notional
                cost_vs_mid_pct = abs(exec_px - mid) / mid if mid > 0 else 0.0
                friction_bps = (filled_notional / portfolio_aum) * cost_vs_mid_pct * 10000
                period_friction_bps += friction_bps

        total_friction_bps += period_friction_bps
        total_opportunity_bps += period_opp_bps
        n_periods += 1

        # Deduct friction from first day return (survival realism)
        friction_ret = period_friction_bps / 10000.0
        opp_ret = period_opp_bps / 10000.0

        # Strategy returns: use realized_weights (inventory risk on rejected orders)
        tickers_in_ret = [t for t in realized_weights if t in price_returns.columns]
        w_vec = np.array([realized_weights[t] for t in tickers_in_ret])
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

        prev_weights = realized_weights
        prev_rb = rb_date

    avg_friction = total_friction_bps / max(n_periods, 1)
    avg_opp = total_opportunity_bps / max(n_periods, 1)
    avg_time_ext = total_extra_holding_days / max(1, rejected_sell_count)
    metrics = {
        "Total_Friction_Cost_BPS": total_friction_bps,
        "Opportunity_Cost_BPS": total_opportunity_bps,
        "Rejected_Sell_Count": rejected_sell_count,
        "Avg_Time_In_Market_Extension_Days": avg_time_ext,
    }
    return strat_ret, metrics


def _net_sharpe_from_returns(returns: pd.Series) -> float:
    """Annualized Sharpe from daily strategy returns."""
    ret = returns.dropna()
    if len(ret) < 10 or ret.std() < 1e-12:
        return np.nan
    return float((ret.mean() / ret.std()) * np.sqrt(TRADING_DAYS_YEAR))


def run_capacity_stress_test(
    run_backtest_fn: Callable[[float], pd.DataFrame],
    raw_prices: pd.DataFrame,
    price_returns: pd.DataFrame,
    alpha_df: pd.DataFrame,
    macro_regime_df: pd.DataFrame,
    adv_df: pd.DataFrame,
    macro_raw_df: Optional[pd.DataFrame] = None,
    base_aum_list: Optional[List[float]] = None,
    impact_c: float = MARKET_IMPACT_C_DEFAULT,
    sharpe_limit: float = SHARPE_CAPACITY_LIMIT,
) -> Dict[str, Any]:
    """
    Iterate backtest over AUM levels; return Capacity Limit where net Sharpe < sharpe_limit.

    Args:
        run_backtest_fn: (aum) -> weights_df. Run backtest with portfolio_aum=aum.
        raw_prices, price_returns, alpha_df, macro_regime_df, adv_df, macro_raw_df: Data.
        base_aum_list: AUM levels to test (default [1M, 5M, 10M, 20M, 50M] EUR).
        impact_c: Square-Root Law coefficient.
        sharpe_limit: Capacity limit = last AUM where net Sharpe >= this (default 1.0).

    Returns:
        Dict with capacity_limit_eur, sharpe_by_aum, breakpoint_index.
    """
    aum_list = base_aum_list or DEFAULT_CAPACITY_AUM_LIST
    sharpe_by_aum: List[Tuple[float, float]] = []
    capacity_limit_eur: Optional[float] = None
    breakpoint_index: Optional[int] = None

    for i, aum in enumerate(aum_list):
        weights_df = run_backtest_fn(aum)
        if weights_df.empty:
            sharpe_by_aum.append((aum, np.nan))
            continue
        strat_ret, _ = compute_friction_and_opportunity_cost(
            weights_df,
            raw_prices,
            price_returns,
            alpha_df,
            macro_regime_df,
            macro_raw_df,
            adv_df,
            portfolio_aum=aum,
            impact_c=impact_c,
        )
        net_sr = _net_sharpe_from_returns(strat_ret)
        sharpe_by_aum.append((aum, net_sr))
        if np.isfinite(net_sr) and net_sr >= sharpe_limit:
            capacity_limit_eur = aum
            breakpoint_index = i

    # Interpolate exact breakpoint if desired (optional)
    result: Dict[str, Any] = {
        "capacity_limit_eur": capacity_limit_eur,
        "sharpe_by_aum": sharpe_by_aum,
        "breakpoint_index": breakpoint_index,
        "sharpe_limit": sharpe_limit,
    }
    return result
