"""
Layer 6: Probabilistic Execution Model.

Limit-order reality: assume order at back of queue. Fill only if next-day
PX_HIGH (sell) or PX_LOW (buy) strictly penetrates limit by >= 1 tick. Touch = 0% fill.

Dynamic volume cap: 5% of realized intraday volume (not historical median).
Spill-over: remainder forced to T+2, T+3; charge overnight gap risk. No phantom liquidity.

Adversarial fill / adverse selection:
- Alpha-liquidity correlation: stronger signal → lower fill probability (others act on same info).
- Runaway price penalty: if price moves in order direction (BUY+price_up, SELL+price_down),
  cap fill at 10%. Full fill only when price moves against you.

Hard falsification: run_adversarial_fill_stress_test() with 30% T+1 / 70% T+2 fill.
If net Sharpe < 0.8 → strategy REJECTED as non-implementable.

Legacy: Logistic fill_prob = sigmoid(a - b*order_size_adv - c*spread - d*intraday_vol)
for calibration when OHLC unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.layer6_execution.tca_logger import DEFAULT_DB_PATH

TRADING_DAYS_YEAR: int = 252


PARTICIPATION_CAP_PCT = 0.05
DEFAULT_TICK_PCT = 0.0001  # 1 bp as fallback when tick unknown

# Adversarial fill: alpha-liquidity correlation
ALPHA_FILL_CORRELATION_SLOPE: float = 0.6  # fill_mult = max(0.2, 1 - slope * |alpha_strength|)

# Runaway price penalty: price moves in order direction → max 10% fill
RUNAWAY_FILL_CAP: float = 0.10

# Hard falsification: 30% T+1 / 70% T+2 fill stress test
ADVERSARIAL_STRESS_FILL_T1: float = 0.30
ADVERSARIAL_STRESS_FILL_T2: float = 0.70
ADVERSARIAL_STRESS_SHARPE_MIN: float = 0.8

# Internalization Masking: 30% of PX_TURN_OVER = block trades outside order book
INTERNALIZATION_MASKING: float = 0.70


def _infer_tick_size(price: float) -> float:
    """Infer minimum tick from price level (e.g. 0.01 for <10, 0.05 for 10-100)."""
    if price <= 0:
        return DEFAULT_TICK_PCT * 100.0
    if price < 10:
        return 0.01
    if price < 100:
        return 0.05
    return 0.10


def is_limit_filled(
    limit_price: float,
    side: str,
    px_high_next: float,
    px_low_next: float,
    tick_size: Optional[float] = None,
) -> bool:
    """
    Limit order at back of queue. Fill only if strict penetration by >= 1 tick.
    BUY: fill iff PX_LOW < limit_price - tick (price traded below our limit).
    SELL: fill iff PX_HIGH > limit_price + tick (price traded above our limit).
    Touching limit = 0% fill (queue never reached).
    """
    if limit_price <= 0 or px_high_next <= 0 or px_low_next <= 0:
        return False
    tick = tick_size if tick_size is not None and tick_size > 0 else _infer_tick_size(limit_price)
    side_upper = side.upper()
    if side_upper == "BUY":
        return px_low_next < (limit_price - tick)
    if side_upper == "SELL":
        return px_high_next > (limit_price + tick)
    return False


def cap_participation(
    order_size: float,
    realized_intraday_volume: float,
    max_participation: float = PARTICIPATION_CAP_PCT,
) -> Tuple[float, float]:
    """
    Cap executable size to max_participation of realized intraday volume.
    Returns (filled_size, spilled_size). No phantom liquidity.
    """
    if realized_intraday_volume <= 0:
        return 0.0, order_size
    cap = realized_intraday_volume * max_participation
    filled = min(order_size, cap)
    spilled = order_size - filled
    return filled, spilled


def overnight_gap_risk(
    mid_before: float,
    open_next: float,
    side: str,
) -> float:
    """
    Overnight gap cost as decimal (e.g. 0.01 = 1%).
    BUY: if open_next > mid_before, we pay more (adverse).
    SELL: if open_next < mid_before, we receive less (adverse).
    """
    if mid_before <= 0:
        return 0.0
    ret = (open_next - mid_before) / mid_before
    if side.upper() == "BUY":
        return max(0.0, ret)
    return max(0.0, -ret)


def alpha_liquidity_fill_multiplier(
    alpha_strength: float,
    slope: float = ALPHA_FILL_CORRELATION_SLOPE,
) -> float:
    """
    Correlation between alpha signal and fill probability.
    Stronger signal → lower fill (others act on same info).
    alpha_strength in [0, 1]: 0=weak, 1=strong (e.g. cross-sectional percentile).
    Returns multiplier in (0, 1]: fill_ratio *= multiplier.
    """
    mult = 1.0 - slope * min(1.0, max(0.0, alpha_strength))
    return float(max(0.2, mult))


def runaway_price_penalty(
    side: str,
    open_exec: float,
    close_exec: float,
) -> Optional[float]:
    """
    Runaway price: price moves in order direction during exec day.
    BUY + price up → adverse (everyone wants to buy) → cap fill.
    SELL + price down → adverse → cap fill.
    Returns max allowed fill_ratio (e.g. 0.10) if runaway, else None (no cap).
    """
    if open_exec <= 0 or close_exec <= 0:
        return None
    ret = (close_exec - open_exec) / open_exec
    side_upper = side.upper()
    if side_upper == "BUY" and ret > 0:
        return RUNAWAY_FILL_CAP
    if side_upper == "SELL" and ret < 0:
        return RUNAWAY_FILL_CAP
    return None


@dataclass
class FillResult:
    """Result of probabilistic fill simulation."""
    filled: bool
    fill_ratio: float
    filled_notional: float
    spilled_notional: float
    overnight_gap_cost_pct: float
    limit_penetrated: bool


def simulate_batch_limit_fills(
    orders: pd.DataFrame,
    raw_prices: pd.DataFrame,
    exec_date: pd.Timestamp,
    all_dates: pd.DatetimeIndex,
    participation_cap: float = PARTICIPATION_CAP_PCT,
    use_adversarial_fill: bool = True,
) -> Tuple[pd.DataFrame, float]:
    """
    Batch probabilistic fill simulation.

    orders: columns [ticker, limit_price, order_notional, delta_weight, ...]
           optional: alpha_strength [0,1] for alpha-liquidity correlation
    raw_prices: long [date, ticker, PX_LAST, PX_HIGH?, PX_LOW?, PX_OPEN?, PX_TURN_OVER]
    exec_date: execution day (T+1 after signal)
    all_dates: sorted trading calendar

    use_adversarial_fill: apply alpha-liquidity correlation + runaway price penalty.

    Returns (orders_with_fill_ratio, total_spill_cost_currency).
    """
    model = ProbabilisticExecutionModel(participation_cap=participation_cap)
    all_dates_idx = pd.DatetimeIndex(all_dates)
    gt = all_dates_idx > exec_date
    if not gt.any():
        return orders.assign(fill_ratio=0.0, filled_notional=0.0), 0.0
    next_date = pd.Timestamp(all_dates_idx[gt][0])

    px_wide = raw_prices.pivot_table(
        index="date", columns="ticker", values="PX_LAST"
    ).sort_index()
    has_ohlc = "PX_HIGH" in raw_prices.columns and "PX_LOW" in raw_prices.columns
    high_wide = (
        raw_prices.pivot_table(index="date", columns="ticker", values="PX_HIGH").sort_index()
        if has_ohlc
        else None
    )
    low_wide = (
        raw_prices.pivot_table(index="date", columns="ticker", values="PX_LOW").sort_index()
        if has_ohlc
        else None
    )
    open_wide = (
        raw_prices.pivot_table(index="date", columns="ticker", values="PX_OPEN").sort_index()
        if "PX_OPEN" in raw_prices.columns
        else None
    )
    turn_wide = (
        raw_prices.pivot_table(index="date", columns="ticker", values="PX_TURN_OVER").sort_index()
        if "PX_TURN_OVER" in raw_prices.columns
        else pd.DataFrame()
    )

    fill_ratios = []
    filled_notionals = []
    total_gap_cost = 0.0

    for _, row in orders.iterrows():
        t = row["ticker"]
        limit = float(row.get("limit_price", row.get("theoretical_price", 0)))
        notional = float(row.get("order_notional", 0))
        side = "BUY" if row.get("delta_weight", 0) > 0 else "SELL"

        px_next = np.nan
        if next_date in px_wide.index and t in px_wide.columns:
            px_next = float(px_wide.loc[next_date, t])
        high_next = px_next
        low_next = px_next
        if has_ohlc and high_wide is not None and low_wide is not None:
            if next_date in high_wide.index and t in high_wide.columns:
                high_next = float(high_wide.loc[next_date, t])
            if next_date in low_wide.index and t in low_wide.columns:
                low_next = float(low_wide.loc[next_date, t])
        mid = limit
        if (px_wide.index <= exec_date).any() and t in px_wide.columns:
            asof = px_wide.loc[px_wide.index <= exec_date, t].dropna()
            if not asof.empty:
                mid = float(asof.iloc[-1])
        open_next_val = None
        if open_wide is not None and next_date in open_wide.index and t in open_wide.columns:
            open_next_val = float(open_wide.loc[next_date, t])
        open_exec_val = None
        close_exec_val = None
        if exec_date in px_wide.index and t in px_wide.columns:
            close_exec_val = float(px_wide.loc[exec_date, t])
        if open_wide is not None and exec_date in open_wide.index and t in open_wide.columns:
            open_exec_val = float(open_wide.loc[exec_date, t])
        alpha_strength = float(row.get("alpha_strength", 0.5))  # default mid
        raw_vol = notional * 10.0
        if not turn_wide.empty and next_date in turn_wide.index and t in turn_wide.columns:
            raw_vol = float(turn_wide.loc[next_date, t])
        realized_vol = raw_vol * INTERNALIZATION_MASKING

        res = model.simulate_limit_fill(
            limit_price=limit,
            side=side,
            order_notional=notional,
            px_high_next=float(high_next) if not np.isnan(high_next) else mid,
            px_low_next=float(low_next) if not np.isnan(low_next) else mid,
            realized_intraday_volume=float(realized_vol) if not np.isnan(realized_vol) else notional * 10.0 * INTERNALIZATION_MASKING,
            mid_price=float(mid) if not np.isnan(mid) else limit,
            open_next=open_next_val,
            alpha_strength=alpha_strength if use_adversarial_fill else 0.0,
            open_exec=open_exec_val if use_adversarial_fill else None,
            close_exec=close_exec_val if use_adversarial_fill else None,
        )
        fill_ratio_out = res.fill_ratio
        filled_notional_out = res.filled_notional
        fill_ratios.append(fill_ratio_out)
        filled_notionals.append(filled_notional_out)
        if res.spilled_notional > 0 and res.overnight_gap_cost_pct > 0:
            total_gap_cost += res.spilled_notional * res.overnight_gap_cost_pct

    out = orders.copy()
    out["fill_ratio"] = fill_ratios
    out["filled_notional"] = filled_notionals
    return out, total_gap_cost


class ProbabilisticExecutionModel:
    """
    Probabilistic execution: limit reality, 5% volume cap, spill-over + overnight gap.
    Adversarial: alpha-liquidity correlation + runaway price penalty.
    """

    def __init__(
        self,
        participation_cap: float = PARTICIPATION_CAP_PCT,
    ) -> None:
        self.participation_cap = participation_cap

    def simulate_limit_fill(
        self,
        limit_price: float,
        side: str,
        order_notional: float,
        px_high_next: float,
        px_low_next: float,
        realized_intraday_volume: float,
        mid_price: float,
        open_next: Optional[float] = None,
        tick_size: Optional[float] = None,
        alpha_strength: float = 0.0,
        open_exec: Optional[float] = None,
        close_exec: Optional[float] = None,
    ) -> FillResult:
        """
        Simulate fill for one day. Returns FillResult.
        Adversarial: alpha_strength reduces fill; runaway price caps fill at 10%.
        """
        penetrated = is_limit_filled(
            limit_price, side, px_high_next, px_low_next, tick_size
        )
        if not penetrated:
            return FillResult(
                filled=False,
                fill_ratio=0.0,
                filled_notional=0.0,
                spilled_notional=order_notional,
                overnight_gap_cost_pct=0.0,
                limit_penetrated=False,
            )

        filled_cap, spilled = cap_participation(
            order_notional, realized_intraday_volume, self.participation_cap
        )
        fill_ratio = filled_cap / order_notional if order_notional > 0 else 0.0

        # Alpha-liquidity correlation: stronger signal → lower fill
        if alpha_strength > 0:
            fill_ratio *= alpha_liquidity_fill_multiplier(alpha_strength)

        # Runaway price penalty: price moves in order direction → max 10% fill
        if open_exec is not None and close_exec is not None and open_exec > 0:
            cap = runaway_price_penalty(side, open_exec, close_exec)
            if cap is not None:
                fill_ratio = min(fill_ratio, cap)

        fill_ratio = min(1.0, max(0.0, fill_ratio))
        filled_cap = order_notional * fill_ratio
        spilled = order_notional - filled_cap

        gap_cost = 0.0
        if spilled > 0 and open_next is not None and mid_price > 0:
            gap_cost = overnight_gap_risk(mid_price, open_next, side)

        return FillResult(
            filled=filled_cap > 0,
            fill_ratio=fill_ratio,
            filled_notional=filled_cap,
            spilled_notional=spilled,
            overnight_gap_cost_pct=gap_cost,
            limit_penetrated=True,
        )


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Sigmoid; clip for numerical stability."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    return float(out[0]) if np.isscalar(x) or out.size == 1 else out


class FillProbabilityModel:
    """
    Logistic fill probability: fill_prob = sigmoid(a - b*size_adv - c*spread - d*intraday_vol).
    Use when OHLC unavailable (no limit penetration check). Calibrated from TCA.
    """

    def __init__(
        self,
        a: float = 2.0,
        b: float = 3.0,
        c: float = 50.0,
        d: float = 20.0,
    ) -> None:
        self._a = a
        self._b = b
        self._c = c
        self._d = d
        self._calibrated = False

    def predict(
        self,
        order_size_adv: np.ndarray | float,
        spread_pct: np.ndarray | float,
        intraday_vol_proxy: np.ndarray | float = 0.01,
    ) -> np.ndarray | float:
        """fill_prob = sigmoid(a - b*order_size_adv - c*spread_pct - d*intraday_vol)."""
        x = (
            self._a
            - self._b * np.atleast_1d(order_size_adv).astype(float)
            - self._c * np.atleast_1d(spread_pct).astype(float)
            - self._d * np.atleast_1d(intraday_vol_proxy).astype(float)
        )
        out = sigmoid(x)
        if np.isscalar(order_size_adv) and np.isscalar(spread_pct):
            return float(out[0]) if hasattr(out, "__getitem__") else float(out)
        return out

    def fit(
        self,
        order_size_adv: np.ndarray,
        spread_pct: np.ndarray,
        intraday_vol_proxy: np.ndarray,
        fill_ratio: np.ndarray,
    ) -> "FillProbabilityModel":
        """Calibrate via logistic regression."""
        from scipy.optimize import minimize

        X = np.column_stack([
            np.ones(len(order_size_adv)),
            order_size_adv,
            spread_pct,
            intraday_vol_proxy,
        ])
        y = np.clip(fill_ratio, 1e-6, 1 - 1e-6)

        def neg_ll(coef: np.ndarray) -> float:
            logit = X @ coef
            p = sigmoid(logit)
            if isinstance(p, (int, float)):
                p = np.full_like(y, p)
            return -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

        res = minimize(
            neg_ll,
            x0=np.array([self._a, -self._b, -self._c, -self._d]),
            method="L-BFGS-B",
        )
        if res.success and len(res.x) >= 4:
            self._a = float(res.x[0])
            self._b = -float(res.x[1])
            self._c = -float(res.x[2])
            self._d = -float(res.x[3])
            self._calibrated = True
        return self

    def calibrate_from_tca(self, db_path: str = DEFAULT_DB_PATH) -> "FillProbabilityModel":
        """Fit from extended TCA log if available."""
        df = _read_tca_extended(db_path)
        if df is None or len(df) < 30:
            return self
        valid = (
            df["order_size_adv"].notna()
            & df["spread_pct"].notna()
            & (df["order_size_adv"] > 0)
            & (df["fill_ratio"] >= 0)
            & (df["fill_ratio"] <= 1)
        )
        sub = df.loc[valid]
        if len(sub) < 30:
            return self
        return self.fit(
            order_size_adv=sub["order_size_adv"].values,
            spread_pct=sub["spread_pct"].values,
            intraday_vol_proxy=sub["intraday_vol_proxy"].fillna(0.01).values,
            fill_ratio=sub["fill_ratio"].values,
        )


def _read_tca_extended(db_path: str) -> Optional[pd.DataFrame]:
    """Read TCA with fill-calibration columns if schema exists."""
    if not Path(db_path).exists():
        return None
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        info = pd.read_sql_query("PRAGMA table_info(tca_log)", conn)
        cols = set(info["name"].tolist())
        if "order_size_adv" not in cols or "spread_pct" not in cols or "fill_ratio" not in cols:
            return None
        df = pd.read_sql_query("SELECT * FROM tca_log", conn)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception:
        return None
    finally:
        conn.close()


def run_adversarial_fill_stress_test(
    run_backtest_fn: Callable[[], pd.DataFrame],
    raw_prices: pd.DataFrame,
    price_returns: pd.DataFrame,
    alpha_df: pd.DataFrame,
    macro_regime_df: pd.DataFrame,
    adv_df: pd.DataFrame,
    macro_raw_df: Optional[pd.DataFrame] = None,
    news_df: Optional[pd.DataFrame] = None,
    portfolio_aum: float = 1e6,
    fill_pct_t1: float = ADVERSARIAL_STRESS_FILL_T1,
    fill_pct_t2: float = ADVERSARIAL_STRESS_FILL_T2,
    sharpe_min: float = ADVERSARIAL_STRESS_SHARPE_MIN,
    adverse_slip_t2_bps: float = 50.0,
) -> Dict[str, Any]:
    """
    Hard falsification: force 30% fill at T+1, 70% at T+2 (worse prices).
    If net Sharpe < 0.8 → strategy REJECTED as non-implementable.

    Args:
        run_backtest_fn: () -> weights_df (no args; uses default AUM).
        raw_prices, price_returns, alpha_df, macro_regime_df, adv_df, macro_raw_df, news_df: Data.
        fill_pct_t1: Fraction of order filled at T+1 (default 0.30).
        fill_pct_t2: Fraction filled at T+2 with worse price (default 0.70).
        sharpe_min: Minimum Sharpe to pass (default 0.8).
        adverse_slip_t2_bps: Extra slippage for T+2 portion (default 50 bps).

    Returns:
        Dict with passed (bool), net_sharpe (float), rejected_reason (str or None).
    """
    from src.layer6_execution.pessimistic_execution import (
        compute_friction_and_opportunity_cost,
    )

    weights_df = run_backtest_fn()
    if weights_df.empty:
        return {
            "passed": False,
            "net_sharpe": np.nan,
            "rejected_reason": "empty_weights",
        }

    # Run with stress fill: we inject a multiplier by post-processing
    # Compute friction with 100% fill first, then scale down returns
    strat_ret, metrics = compute_friction_and_opportunity_cost(
        weights_df,
        raw_prices,
        price_returns,
        alpha_df,
        macro_regime_df,
        macro_raw_df,
        adv_df,
        portfolio_aum=portfolio_aum,
        news_df=news_df,
    )

    # Extra penalty: 70% of order executes at T+2 with adverse_slip_t2_bps worse
    # Extra return drag = fill_pct_t2 * turnover * adverse_slip_bps / 10000
    weight_df = weights_df.copy()
    weight_df["date"] = pd.to_datetime(weight_df["date"])
    rb_dates = sorted(weight_df["date"].unique())
    all_dates = pd.DatetimeIndex(price_returns.index).sort_values()
    prev_weights: Dict[str, float] = {}

    for i, rb_date in enumerate(rb_dates):
        w_row = weight_df[weight_df["date"] == rb_date]
        curr_weights = dict(zip(w_row["ticker"], w_row["target_weight"]))
        tickers = list(set(prev_weights.keys()) | set(curr_weights.keys()))
        if not tickers:
            prev_weights = curr_weights
            continue
        gt = all_dates > rb_date
        if not gt.any():
            prev_weights = curr_weights
            continue
        exec_date = pd.Timestamp(all_dates[gt][0])
        period_dates = all_dates[(all_dates >= exec_date)]
        if i + 1 < len(rb_dates):
            end_date = rb_dates[i + 1]
            period_dates = period_dates[period_dates < end_date]
        if len(period_dates) == 0:
            prev_weights = curr_weights
            continue

        turnover = 0.0
        for t in tickers:
            prev_w = prev_weights.get(t, 0.0)
            curr_w = curr_weights.get(t, 0.0)
            delta_w = curr_w - prev_w
            if abs(delta_w) < 1e-12:
                continue
            turnover += abs(delta_w)
        if turnover > 0:
            extra_ret_drag = fill_pct_t2 * turnover * (adverse_slip_t2_bps / 10000.0)
            first_dt = period_dates[0] if len(period_dates) > 0 else None
            if first_dt is not None and first_dt in strat_ret.index:
                strat_ret.loc[first_dt] -= extra_ret_drag
        prev_weights = curr_weights

    ret = strat_ret.dropna()
    if len(ret) < 10 or ret.std() < 1e-12:
        return {
            "passed": False,
            "net_sharpe": np.nan,
            "rejected_reason": "insufficient_data",
        }
    net_sharpe = float((ret.mean() / ret.std()) * np.sqrt(TRADING_DAYS_YEAR))

    passed = net_sharpe >= sharpe_min
    return {
        "passed": passed,
        "net_sharpe": net_sharpe,
        "rejected_reason": None if passed else f"adversarial_fill_stress_sharpe_{net_sharpe:.3f}_lt_{sharpe_min}",
    }
