"""
Market Impact & Liquidity Stress: Square Root Law of Market Impact.
Applies synthetic transaction costs to backtest equity curve.
If strategy shows positive alpha after impact, it is viable for real capital.
Does not modify src/ or run_pipeline.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from research._shared import (
    setup_research_data,
    compute_equity_curve,
    compute_metrics,
    _NeutralNLP,
)
from src.engine.backtest_loop import BacktestEngine
from src.layer4_nlp.xlm_roberta_sentinel import NLPSentinel
from src.layer5_portfolio.hrp_clustering import HierarchicalRiskParity
from src.layer5_portfolio.dynamic_vol_target import DynamicVolTargeting


def _square_root_impact(
    order_pct_of_volume: float,
    sigma_annual: float,
    c: float = 1.0,
) -> float:
    """
    Square Root Law: impact = c * sigma * sqrt(Q/V).
    order_pct_of_volume: Q/V ratio (order size / daily volume).
    sigma_annual: annual volatility (e.g. 0.20).
    Returns impact as a return (e.g. 0.001 = 10 bps).
    """
    if order_pct_of_volume <= 0:
        return 0.0
    sigma_daily = sigma_annual / np.sqrt(252)
    return c * sigma_daily * np.sqrt(order_pct_of_volume)


def apply_market_impact(
    backtest_result: pd.DataFrame,
    price_returns: pd.DataFrame,
    raw_prices: pd.DataFrame,
    equity_before_impact: pd.Series,
    impact_coeff: float = 1.0,
) -> pd.Series:
    """
    Apply Square Root Law impact costs to equity curve.
    Deducts impact from returns on each rebalance date.
    """
    if backtest_result.empty or equity_before_impact.empty:
        return equity_before_impact.copy()

    weight_df = backtest_result.copy()
    weight_df["date"] = pd.to_datetime(weight_df["date"])
    rb_dates = sorted(weight_df["date"].unique())

    px = raw_prices.copy()
    px["date"] = pd.to_datetime(px["date"])
    turnover_df = px.pivot_table(index="date", columns="ticker", values="PX_TURN_OVER")
    price_df = px.pivot_table(index="date", columns="ticker", values="PX_LAST")

    ret = equity_before_impact.pct_change().fillna(0)
    impact_by_date: dict[pd.Timestamp, float] = {}
    prev_weights: dict[str, float] = {}

    for rb_date in rb_dates:
        w_row = weight_df[weight_df["date"] == rb_date]
        new_weights = dict(zip(w_row["ticker"], w_row["target_weight"]))

        if prev_weights:
            eq_before = equity_before_impact.loc[equity_before_impact.index <= rb_date]
            pv_val = float(eq_before.iloc[-1]) if not eq_before.empty else 1.0
            total_impact = 0.0

            for t, w_new in new_weights.items():
                w_old = prev_weights.get(t, 0.0)
                delta = abs(w_new - w_old)
                if delta < 1e-9 or t not in turnover_df.columns:
                    continue
                order_dollars = delta * pv_val
                turn_idx = turnover_df.index[turnover_df.index <= rb_date]
                if turn_idx.empty:
                    continue
                vd = turnover_df.loc[turn_idx[-1], t]  # PX_TURN_OVER = daily traded value (fiat)
                if pd.isna(vd) or vd <= 0:
                    continue
                order_pct = min(order_dollars / vd, 1.0)
                sig = price_returns[t].loc[price_returns.index <= rb_date].std() * np.sqrt(252)
                if pd.isna(sig) or sig <= 0:
                    sig = 0.20
                total_impact += _square_root_impact(order_pct, float(sig), impact_coeff)

            next_day = ret.index[ret.index > rb_date]
            if not next_day.empty:
                impact_by_date[next_day[0]] = total_impact

        prev_weights = dict(new_weights)

    ret_adj = ret.copy()
    for dt, imp in impact_by_date.items():
        if dt in ret_adj.index:
            ret_adj.loc[dt] = ret_adj.loc[dt] - imp
    equity_adj = (1 + ret_adj).cumprod()
    return equity_adj


def run_market_impact_study(impact_coeff: float = 1.0) -> None:
    """Run full backtest, apply impact, compare metrics."""
    print("Market Impact Study: Loading data...")
    alpha_df, price_returns, macro_regime_df, news_df, raw_prices, rebalance_dates = (
        setup_research_data()
    )

    try:
        nlp = NLPSentinel()
    except Exception:
        nlp = _NeutralNLP()

    engine = BacktestEngine(
        alpha_model=None,
        nlp_sentinel=nlp,
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

    equity_no_impact = compute_equity_curve(bt, price_returns)
    equity_with_impact = apply_market_impact(
        bt, price_returns, raw_prices, equity_no_impact, impact_coeff
    )

    m_no = compute_metrics(equity_no_impact)
    m_yes = compute_metrics(equity_with_impact)

    print("\n" + "=" * 60)
    print("Market Impact Results (Square Root Law)")
    print("=" * 60)
    print(f"Impact coefficient c = {impact_coeff}")
    print("\nWithout impact:")
    print(f"  Sharpe: {m_no['sharpe']:.3f}  CAGR: {m_no['cagr']:.2%}  MaxDD: {m_no['max_dd']:.2%}")
    print("\nWith impact:")
    print(f"  Sharpe: {m_yes['sharpe']:.3f}  CAGR: {m_yes['cagr']:.2%}  MaxDD: {m_yes['max_dd']:.2%}")
    alpha_survives = m_yes["sharpe"] > 0 and m_yes["cagr"] > 0
    print(f"\nAlpha survives impact: {'Yes' if alpha_survives else 'No'}")


if __name__ == "__main__":
    run_market_impact_study(impact_coeff=1.0)
