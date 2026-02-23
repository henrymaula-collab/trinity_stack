"""
Fragility & Capacity Report: Stress-testing before capital deployment.

Modules:
1. Causal verification: Granger causality (customer→supplier) + Event-study CAR drift
2. Market impact & fill probability (Square Root Law, small-cap penalty, partial fills)
3. Jitter & fragility: data lag +2d, 1% noise injection
4. Liquidity freeze: spread 3×, volume -70%, time-to-exit for 500k EUR
5. Drawdown-constrained Kelly: f* s.t. P(DD > 25%) < 0.01

Does not modify src/ or run_pipeline.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from research._shared import (
    GLOBAL_SEED,
    setup_research_data,
    compute_equity_curve,
    compute_metrics,
)
from research._shared import _NeutralNLP
from src.engine.backtest_loop import BacktestEngine
from src.layer5_portfolio.hrp_clustering import HierarchicalRiskParity
from src.layer5_portfolio.dynamic_vol_target import DynamicVolTargeting

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

# --- Parameters ---
GRILLAN_MAX_LAG = 5
EVENT_WINDOW_DAYS = 5
CAR_START_DAY = 0
CAR_END_DAY = 5
MARKET_IMPACT_ORDER_PCT_OF_VOL = 0.05  # 5% → 2× penalty
FILL_PROB_HIGH_VOL = 0.4
FILL_PROB_NORMAL = 0.85
NOISE_PCT = 0.01
FUNDAMENTAL_LAG_EXTRA_DAYS = 2
LIQUIDITY_FREEZE_POSITION_EUR = 500_000
LIQUIDITY_FREEZE_MAX_IMPACT_PCT = 0.05
LIQUIDITY_FREEZE_SPREAD_MULT = 3.0
LIQUIDITY_FREEZE_VOLUME_MULT = 0.30  # -70%
DRAWDOWN_25PCT_TARGET_PROB = 0.01
DRAWDOWN_SIM_PATHS = 5000
KELLY_SHRINKAGE = 0.5
KELLY_SAFETY = 0.3


# ---------------------------------------------------------------------------
# 1. Granger Causality & Event-Study Drift
# ---------------------------------------------------------------------------


def granger_causality_test(
    customer_ret: np.ndarray,
    supplier_ret: np.ndarray,
    max_lag: int = GRILLAN_MAX_LAG,
) -> Dict[str, Any]:
    """
    Test: does customer return at t-1..t-lag add predictive power for supplier at t?
    Restricted: supplier_ret_t ~ supplier_ret_{t-1..t-lag}
    Unrestricted: supplier_ret_t ~ supplier_ret_{t-1..t-lag} + customer_ret_{t-1..t-lag}
    F-test on nested models.
    """
    n = min(len(customer_ret), len(supplier_ret))
    if n < max_lag * 2 + 10:
        return {"p_value": np.nan, "f_stat": np.nan, "granger_causes": False, "n_obs": n}

    y = supplier_ret[max_lag:]
    # Restricted: only supplier lags
    X_rest = np.column_stack(
        [supplier_ret[max_lag - 1 - j : n - 1 - j] for j in range(max_lag)]
    )
    # Unrestricted: supplier + customer lags
    X_unr = np.column_stack([
        *[supplier_ret[max_lag - 1 - j : n - 1 - j] for j in range(max_lag)],
        *[customer_ret[max_lag - 1 - j : n - 1 - j] for j in range(max_lag)],
    ])

    # Drop rows with NaN
    valid = ~(np.any(np.isnan(X_unr), axis=1) | np.isnan(y))
    y = y[valid]
    X_rest = X_rest[valid]
    X_unr = X_unr[valid]
    if len(y) < 20:
        return {"p_value": np.nan, "f_stat": np.nan, "granger_causes": False, "n_obs": len(y)}

    def rss(X: np.ndarray, yy: np.ndarray) -> float:
        Xc = np.column_stack([np.ones(len(X)), X])
        try:
            beta = np.linalg.lstsq(Xc, yy, rcond=None)[0]
            return float(np.sum((yy - Xc @ beta) ** 2))
        except Exception:
            return np.nan

    rss_rest = rss(X_rest, y)
    rss_unr = rss(X_unr, y)
    if np.isnan(rss_rest) or np.isnan(rss_unr) or rss_unr <= 0:
        return {"p_value": np.nan, "f_stat": np.nan, "granger_causes": False, "n_obs": len(y)}

    q = max_lag
    df1, df2 = q, len(y) - (max_lag * 2 + 1)
    if df2 <= 0:
        return {"p_value": np.nan, "f_stat": np.nan, "granger_causes": False, "n_obs": len(y)}
    f_stat = ((rss_rest - rss_unr) / q) / (rss_unr / df2)
    from scipy import stats
    p_value = 1.0 - stats.f.cdf(f_stat, df1, df2)
    return {
        "p_value": float(p_value),
        "f_stat": float(f_stat),
        "granger_causes": p_value < 0.05,
        "n_obs": len(y),
    }


def event_study_car(
    supplier_returns: pd.DataFrame,
    trigger_dates: pd.DatetimeIndex,
    window: int = EVENT_WINDOW_DAYS,
) -> pd.Series:
    """
    Cumulative abnormal return (CAR) after trigger.
    CAR = sum of supplier returns from day 0 to window-1.
    Uses simple return (no market model); for Nordic small caps market model adds noise.
    """
    car_list: List[float] = []
    dates = supplier_returns.index
    for t0 in trigger_dates:
        if t0 not in dates:
            continue
        pos = dates.get_loc(t0)
        end_pos = min(pos + window, len(dates) - 1)
        if end_pos <= pos:
            continue
        rets = supplier_returns.iloc[pos : end_pos + 1].values.flatten()
        car = float(np.prod(1 + np.nan_to_num(rets, nan=0.0)) - 1.0)
        car_list.append(car)
    return pd.Series(car_list)


def run_granger_and_event_study(
    prices: pd.DataFrame,
    supply_chain: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """Run Granger causality and event-study drift on customer-supplier pairs."""
    out: Dict[str, Any] = {"granger": {}, "event_study": {}}
    if supply_chain is None or supply_chain.empty:
        return {"granger": {"skipped": "no supply chain"}, "event_study": {"skipped": "no supply chain"}}

    px = prices.pivot_table(index="date", columns="ticker", values="PX_LAST")
    px.index = pd.to_datetime(px.index)
    ret = px.pct_change()

    pairs = supply_chain[["supplier_ticker", "customer_ticker"]].drop_duplicates()
    granger_results: List[Dict] = []
    car_all: List[float] = []

    for _, row in pairs.head(50).iterrows():
        supp, cust = row["supplier_ticker"], row["customer_ticker"]
        if supp not in ret.columns or cust not in ret.columns:
            continue
        cust_ret = ret[cust].fillna(0).values
        supp_ret = ret[supp].fillna(0).values
        res = granger_causality_test(cust_ret, supp_ret, max_lag=min(3, GRILLAN_MAX_LAG))
        res["pair"] = f"{cust}->{supp}"
        granger_results.append(res)

        # Event-study: use days where customer had |ret| > 2*rolling_std as trigger
        cust_vol = pd.Series(cust_ret).rolling(60, min_periods=20).std().shift(1)
        vol_safe = cust_vol.fillna(0.01).values
        trigger_mask = np.abs(cust_ret) > 2 * vol_safe
        trigger_positions = np.where(trigger_mask)[0]
        if len(trigger_positions) == 0:
            continue
        trigger_dates = px.index[trigger_positions]
        if len(trigger_dates) > 0:
            supp_ret_df = ret[[supp]]
            car = event_study_car(supp_ret_df, trigger_dates[:30], window=EVENT_WINDOW_DAYS)
            car_all.extend(car.dropna().tolist())

    out["granger"] = {
        "n_pairs_tested": len(granger_results),
        "n_granger_causes": sum(1 for r in granger_results if r.get("granger_causes")),
        "mean_p_value": float(np.nanmean([r["p_value"] for r in granger_results])) if granger_results else np.nan,
        "details": granger_results[:10],
    }
    out["event_study"] = {
        "n_events": len(car_all),
        "car_mean_pct": 100 * float(np.mean(car_all)) if car_all else np.nan,
        "car_median_pct": 100 * float(np.median(car_all)) if car_all else np.nan,
        "drift_exploitable_3_5d": (
            "yes" if car_all and np.mean(car_all) > 0.001 else "no"
        ),
    }
    return out


# ---------------------------------------------------------------------------
# 2. Market Impact & Fill Probability
# ---------------------------------------------------------------------------


def market_impact_sqrt_law(
    sigma: float,
    order_value: float,
    daily_volume: float,
    order_pct_of_vol: float = 0.05,
) -> float:
    """
    ΔP ≈ σ * sqrt(Q/V). Q=order, V=daily volume.
    Small-cap penalty: 2× cost if order > 5% of daily volume.
    Returns estimated price impact in decimal (e.g. 0.01 = 1%).
    """
    if daily_volume <= 0:
        return 0.10  # arbitrary high impact if no volume
    q_over_v = order_value / daily_volume
    impact = sigma * np.sqrt(q_over_v)
    if q_over_v > order_pct_of_vol:
        impact *= 2.0
    return float(impact)


def fill_probability_by_regime(
    vol_regime: str,
    fill_prob_normal: float = FILL_PROB_NORMAL,
    fill_prob_high_vol: float = FILL_PROB_HIGH_VOL,
) -> float:
    """Fill probability: normal vs high-vol (30-50% under stress)."""
    return fill_prob_high_vol if vol_regime == "high_vol" else fill_prob_normal


def simulate_slippage_and_fill(
    returns: pd.Series,
    raw_prices: pd.DataFrame,
    position_eur_per_ticker: float = 50_000,
) -> Dict[str, Any]:
    """
    Estimate impact of market impact and partial fills on strategy returns.
    Uses average daily volume and spread from raw_prices.
    """
    vol = returns.std() * np.sqrt(252) if len(returns) > 20 else 0.20
    if "PX_TURN_OVER" in raw_prices.columns and "BID_ASK_SPREAD_PCT" in raw_prices.columns:
        daily_vol = raw_prices.groupby("date")["PX_TURN_OVER"].sum()
        avg_vol = daily_vol.mean()
        avg_spread_pct = raw_prices.groupby("date")["BID_ASK_SPREAD_PCT"].mean().mean()
    else:
        avg_vol = 1e6
        avg_spread_pct = 0.02
    impact = market_impact_sqrt_law(
        vol, position_eur_per_ticker, avg_vol,
        order_pct_of_vol=MARKET_IMPACT_ORDER_PCT_OF_VOL,
    )
    n_days = len(returns.dropna())
    high_vol_days = int(n_days * 0.10)  # assume 10% high-vol days
    fill_normal = fill_probability_by_regime("normal")
    fill_high = fill_probability_by_regime("high_vol")
    effective_fill = (n_days - high_vol_days) / n_days * fill_normal + high_vol_days / n_days * fill_high
    return {
        "market_impact_roundtrip_pct": 100 * 2 * impact,
        "avg_spread_pct": 100 * avg_spread_pct,
        "effective_fill_rate": effective_fill,
        "implied_cost_per_trade_pct": 100 * (impact * 2 + avg_spread_pct / 2),
    }


# ---------------------------------------------------------------------------
# 3. Jitter & Fragility
# ---------------------------------------------------------------------------


def _run_backtest_with_perturbed_data(
    alpha_df: pd.DataFrame,
    price_returns: pd.DataFrame,
    macro_regime_df: pd.DataFrame,
    news_df: pd.DataFrame,
    raw_prices: pd.DataFrame,
    rebalance_dates: List[pd.Timestamp],
    fundamental_lag_days: int = 0,
    price_noise_pct: float = 0.0,
) -> pd.Series:
    """Run Trinity backtest with optional data perturbations."""
    from research._shared import _NeutralNLP
    from src.engine.backtest_loop import BacktestEngine
    from src.layer5_portfolio.hrp_clustering import HierarchicalRiskParity
    from src.layer5_portfolio.dynamic_vol_target import DynamicVolTargeting

    pr = price_returns.copy()
    if price_noise_pct > 0:
        noise = np.random.randn(*pr.values.shape).astype(float) * (price_noise_pct / 100)
        pr = pr + pd.DataFrame(noise, index=pr.index, columns=pr.columns)
    alpha = alpha_df.copy()
    if fundamental_lag_days != 0:
        alpha["date"] = pd.to_datetime(alpha["date"]) + pd.Timedelta(days=fundamental_lag_days)
    engine = BacktestEngine(
        alpha_model=None,
        nlp_sentinel=_NeutralNLP(),
        hrp_model=HierarchicalRiskParity(),
        vol_targeter=DynamicVolTargeting(),
    )
    bt = engine.run_backtest(
        price_df=pr,
        alpha_df=alpha,
        macro_df=macro_regime_df,
        news_df=news_df,
        rebalance_dates=rebalance_dates,
        currency_series=None,
    )
    eq = compute_equity_curve(bt, pr)
    return eq.pct_change().dropna()


def jitter_fragility_tests(
    alpha_df: pd.DataFrame,
    price_returns: pd.DataFrame,
    macro_regime_df: pd.DataFrame,
    news_df: pd.DataFrame,
    raw_prices: pd.DataFrame,
    rebalance_dates: List[pd.Timestamp],
) -> Dict[str, Any]:
    """
    Data jitter: lag fundamental +2d.
    Noise injection: 1% random noise in price returns.
    Compare baseline vs perturbed Sharpe/CAGR.
    """
    np.random.seed(GLOBAL_SEED)
    baseline_ret = _run_backtest_with_perturbed_data(
        alpha_df, price_returns, macro_regime_df, news_df, raw_prices,
        rebalance_dates, fundamental_lag_days=0, price_noise_pct=0,
    )
    lag_ret = _run_backtest_with_perturbed_data(
        alpha_df, price_returns, macro_regime_df, news_df, raw_prices,
        rebalance_dates, fundamental_lag_days=FUNDAMENTAL_LAG_EXTRA_DAYS, price_noise_pct=0,
    )
    noise_ret = _run_backtest_with_perturbed_data(
        alpha_df, price_returns, macro_regime_df, news_df, raw_prices,
        rebalance_dates, fundamental_lag_days=0, price_noise_pct=100 * NOISE_PCT,
    )
    m_baseline = compute_metrics(
        (1 + baseline_ret).cumprod().reindex(price_returns.index).ffill().bfill()
    )
    m_lag = compute_metrics(
        (1 + lag_ret).cumprod().reindex(price_returns.index).ffill().bfill()
    )
    m_noise = compute_metrics(
        (1 + noise_ret).cumprod().reindex(price_returns.index).ffill().bfill()
    )
    return {
        "baseline": m_baseline,
        "fundamental_lag_plus_2d": m_lag,
        "price_noise_1pct": m_noise,
        "fragile_to_lag": m_lag["sharpe"] < 0.5 * m_baseline["sharpe"] if not np.isnan(m_baseline["sharpe"]) else False,
        "fragile_to_noise": m_noise["sharpe"] < 0.5 * m_baseline["sharpe"] if not np.isnan(m_baseline["sharpe"]) else False,
    }


# ---------------------------------------------------------------------------
# 4. Liquidity Freeze
# ---------------------------------------------------------------------------


def time_to_exit(
    position_eur: float,
    daily_volume_eur: float,
    spread_pct: float,
    max_impact_pct: float = LIQUIDITY_FREEZE_MAX_IMPACT_PCT,
) -> float:
    """
    Rough estimate: days to liquidate position without exceeding max_impact.
    Assumes linear impact: each day we sell volume such that impact = max_impact.
    Simplified: days ≈ position / (daily_volume * fraction we can trade per day).
    Under stress: volume drops, spread widens. We use stressed volume.
    """
    if daily_volume_eur <= 0:
        return 999.0
    # Conservative: sell max 5% of daily volume per day to limit impact
    daily_sellable = daily_volume_eur * 0.05
    if daily_sellable <= 0:
        return 999.0
    days = position_eur / daily_sellable
    return float(np.ceil(days))


def liquidity_freeze_simulation(
    raw_prices: pd.DataFrame,
    position_eur: float = LIQUIDITY_FREEZE_POSITION_EUR,
) -> Dict[str, Any]:
    """
    Simulate: spread 3×, volume -70%.
    Time-to-exit for given position without >5% price impact.
    """
    if "PX_TURN_OVER" not in raw_prices.columns:
        raw_prices = raw_prices.copy()
        raw_prices["PX_TURN_OVER"] = 1e6
    if "BID_ASK_SPREAD_PCT" not in raw_prices.columns:
        raw_prices["BID_ASK_SPREAD_PCT"] = 0.01
    daily_vol = raw_prices.groupby("date")["PX_TURN_OVER"].sum()
    base_vol = daily_vol.mean()
    stressed_vol = base_vol * LIQUIDITY_FREEZE_VOLUME_MULT
    base_spread = raw_prices.groupby("date")["BID_ASK_SPREAD_PCT"].mean().mean()
    stressed_spread = base_spread * LIQUIDITY_FREEZE_SPREAD_MULT
    days_to_exit = time_to_exit(
        position_eur, stressed_vol, stressed_spread, LIQUIDITY_FREEZE_MAX_IMPACT_PCT
    )
    return {
        "position_eur": position_eur,
        "base_daily_vol_eur": float(base_vol),
        "stressed_daily_vol_eur": float(stressed_vol),
        "base_spread_pct": 100 * float(base_spread),
        "stressed_spread_pct": 100 * float(stressed_spread),
        "days_to_exit_stress": float(days_to_exit),
        "position_size_cap_recommended": (
            "reduce" if days_to_exit > 3 else "ok"
        ),
    }


# ---------------------------------------------------------------------------
# 5. Drawdown-Constrained Kelly
# ---------------------------------------------------------------------------


def _kelly_point_estimate(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty or len(r) < 10:
        return 0.0
    mu = float(r.mean())
    sigma2 = float(r.var())
    if sigma2 <= 0:
        return 0.0
    mu_adj = KELLY_SHRINKAGE * mu + (1 - KELLY_SHRINKAGE) * 0.0
    f = (mu_adj / sigma2) * KELLY_SAFETY
    return float(np.clip(f, 0.0, 1.0))


def drawdown_constrained_kelly(
    returns: pd.Series,
    target_dd_pct: float = 25.0,
    target_prob: float = DRAWDOWN_25PCT_TARGET_PROB,
    n_paths: int = DRAWDOWN_SIM_PATHS,
) -> Dict[str, Any]:
    """
    f* = max f in [0, Kelly] s.t. P(Drawdown > 25%) < 0.01.
    Simulate n_paths bootstrap paths, compute drawdown distribution for each f.
    """
    r = returns.dropna().values
    if len(r) < 60:
        return {"kelly_full": np.nan, "f_constrained": np.nan, "dd25_prob_at_kelly": np.nan}
    kelly_f = _kelly_point_estimate(pd.Series(r))
    n = len(r)
    rng = np.random.default_rng(GLOBAL_SEED)
    dd25_probs: Dict[float, float] = {}
    for f in [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, kelly_f]:
        if f > kelly_f and f != kelly_f:
            continue
        f = min(f, kelly_f)
        count_dd25 = 0
        for _ in range(n_paths):
            idx = rng.integers(0, n, size=n)
            path_ret = r[idx] * f
            equity = np.cumprod(1 + path_ret)
            cummax = np.maximum.accumulate(equity)
            dd = (equity - cummax) / cummax
            if np.min(dd) <= -target_dd_pct / 100:
                count_dd25 += 1
        dd25_probs[f] = count_dd25 / n_paths
    f_constrained = 0.0
    for f in sorted(dd25_probs.keys(), reverse=True):
        if dd25_probs[f] <= target_prob:
            f_constrained = f
            break
    return {
        "kelly_full": kelly_f,
        "f_constrained": f_constrained,
        "dd25_prob_at_kelly": dd25_probs.get(kelly_f, np.nan),
        "dd25_probs_by_f": dd25_probs,
    }


# ---------------------------------------------------------------------------
# Main Report
# ---------------------------------------------------------------------------


def run_fragility_and_capacity_report() -> Dict[str, Any]:
    """Run full Fragility & Capacity report."""
    print("=" * 80)
    print("FRAGILITY & CAPACITY REPORT — Stress-test before capital deployment")
    print("=" * 80)

    alpha_df, price_returns, macro_regime_df, news_df, raw_prices, rebalance_dates = (
        setup_research_data()
    )
    engine = BacktestEngine(
        alpha_model=None,
        nlp_sentinel=_NeutralNLP(),
        hrp_model=HierarchicalRiskParity(),
        vol_targeter=DynamicVolTargeting(),
    )
    bt = engine.run_backtest(
        price_df=price_returns,
        alpha_df=alpha_df,
        macro_df=macro_regime_df,
        news_df=news_df,
        rebalance_dates=rebalance_dates,
        currency_series=None,
    )
    equity = compute_equity_curve(bt, price_returns)
    strategy_returns = equity.pct_change().dropna()

    supply_chain: Optional[pd.DataFrame] = None
    tickers = raw_prices["ticker"].unique().tolist() if "ticker" in raw_prices.columns else []
    if len(tickers) >= 2:
        dates = raw_prices["date"].drop_duplicates().sort_values()
        n_links = min(60, len(dates))
        rng = np.random.default_rng(GLOBAL_SEED)
        pair_idx = rng.integers(0, len(tickers) - 1, size=n_links)
        date_idx = rng.integers(0, len(dates), size=n_links)
        supply_chain = pd.DataFrame({
            "date": dates.iloc[date_idx].values,
            "supplier_ticker": [tickers[i] for i in pair_idx],
            "customer_ticker": [tickers[i + 1] for i in pair_idx],
            "revenue_dependency_pct": 20.0,
        })

    report: Dict[str, Any] = {}

    # 1. Granger & Event-study
    print("\n--- 1. Causal verification (Granger + Event-study) ---")
    granger_event = run_granger_and_event_study(raw_prices, supply_chain)
    report["granger_event_study"] = granger_event
    print(f"  Granger: {granger_event.get('granger', {})}")
    print(f"  Event-study CAR: {granger_event.get('event_study', {})}")

    # 2. Market impact & fill
    print("\n--- 2. Market impact & fill probability ---")
    slippage = simulate_slippage_and_fill(strategy_returns, raw_prices)
    report["market_impact_fill"] = slippage
    print(f"  Round-trip impact: {slippage['market_impact_roundtrip_pct']:.2f}%")
    print(f"  Effective fill rate: {slippage['effective_fill_rate']:.2%}")

    # 3. Jitter
    print("\n--- 3. Jitter & fragility ---")
    jitter = jitter_fragility_tests(
        alpha_df, price_returns, macro_regime_df, news_df, raw_prices, rebalance_dates
    )
    report["jitter_fragility"] = jitter
    print(f"  Baseline Sharpe: {jitter['baseline'].get('sharpe', np.nan):.2f}")
    print(f"  After +2d lag:   {jitter['fundamental_lag_plus_2d'].get('sharpe', np.nan):.2f}")
    print(f"  After 1% noise:  {jitter['price_noise_1pct'].get('sharpe', np.nan):.2f}")
    print(f"  Fragile to lag:  {jitter['fragile_to_lag']}, to noise: {jitter['fragile_to_noise']}")

    # 4. Liquidity freeze
    print("\n--- 4. Liquidity freeze simulation ---")
    liq = liquidity_freeze_simulation(raw_prices)
    report["liquidity_freeze"] = liq
    print(f"  Days to exit {liq['position_eur']/1e6:.1f}M (stress): {liq['days_to_exit_stress']:.0f}")
    print(f"  Position cap: {liq['position_size_cap_recommended']}")

    # 5. Drawdown-constrained Kelly
    print("\n--- 5. Drawdown-constrained Kelly ---")
    dd_kelly = drawdown_constrained_kelly(strategy_returns)
    report["drawdown_constrained_kelly"] = dd_kelly
    print(f"  Kelly full: {dd_kelly['kelly_full']:.4f}")
    print(f"  f* (P(DD>25%)<1%): {dd_kelly['f_constrained']:.4f}")
    print(f"  P(DD>25%) at Kelly: {dd_kelly.get('dd25_prob_at_kelly', np.nan):.1%}")

    print("\n" + "=" * 80)
    print("END OF FRAGILITY & CAPACITY REPORT")
    print("=" * 80)
    return report


if __name__ == "__main__":
    run_fragility_and_capacity_report()
