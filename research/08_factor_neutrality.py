"""
Factor Neutrality Test — Fama-French 5-Factor + Momentum + Quality + Liquidity.

Regresses strategy returns against established risk factors. If alpha is not
statistically significant after transaction costs, flags strategy as rejected
("repackaged beta").

Adds:
- 6th factor: Liquidity (ILLIQ / Amihud-based cross-sectional factor)
- Decile Drop Test: exclude bottom 10% by 20d rolling median turnover (EUR)
- Falsification: Sharpe drop >30% when illiquid decile excluded → Liquidity Harvesting
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

try:
    import statsmodels.formula.api as smf
except ImportError:
    smf = None  # type: ignore

BASE_FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "UMD", "QMJ"]
REQUIRED_COLS = [
    "strategy_daily_returns",
] + BASE_FACTOR_COLS
ROLLING_WINDOW = 252  # 1 year of trading days
ALPHA_SIGNIFICANCE_LEVEL = 0.05
# Liquidity factor (optional; if present, included in regression)
LIQUIDITY_FACTOR_COL = "ILLIQ"
ROLLING_TURNOVER_DAYS = 20
DECILE_DROP_PCT = 0.10  # Bottom 10% = most illiquid
SHARPE_DROP_FAIL_THRESHOLD = 0.30  # >30% Sharpe drop → Liquidity Harvesting


def run_factor_neutrality_test(
    data: pd.DataFrame,
    transaction_cost_rate: float = 0.001,
    include_liquidity_factor: bool = True,
) -> Dict[str, Any]:
    """
    Run factor neutrality regression and falsification check.

    Args:
        data: DataFrame with columns strategy_daily_returns, Mkt-RF, SMB, HML, UMD, QMJ.
              Optional: ILLIQ (Liquidity factor) — included if present and include_liquidity_factor.
        transaction_cost_rate: Daily cost to deduct (e.g. 0.001 = 10 bps).
        include_liquidity_factor: If True and ILLIQ in data, add as 6th factor.

    Returns:
        Dict with alpha, alpha_p_value, is_rejected, rolling_r_squared, betas (incl. ILLIQ if used).
    """
    if smf is None:
        raise ImportError("statsmodels is required. Install: pip install statsmodels")

    for col in REQUIRED_COLS:
        if col not in data.columns:
            raise ValueError(f"data must contain column '{col}'")

    factor_cols = BASE_FACTOR_COLS.copy()
    if include_liquidity_factor and LIQUIDITY_FACTOR_COL in data.columns:
        factor_cols = factor_cols + [LIQUIDITY_FACTOR_COL]

    df = data[["strategy_daily_returns"] + factor_cols].copy()
    df = df.dropna()
    if len(df) < ROLLING_WINDOW:
        raise ValueError(
            f"data must have at least {ROLLING_WINDOW} rows after dropna for rolling R²"
        )

    net_returns = df["strategy_daily_returns"] - transaction_cost_rate
    formula_terms = ["Q('Mkt-RF')", "SMB", "HML", "UMD", "QMJ"]
    if LIQUIDITY_FACTOR_COL in factor_cols:
        formula_terms.append(LIQUIDITY_FACTOR_COL)
    formula = "net_returns ~ " + " + ".join(formula_terms)
    df_reg = df.copy()
    df_reg["net_returns"] = net_returns

    model = smf.ols(formula, data=df_reg).fit()
    alpha = float(model.params["Intercept"])
    alpha_p_value = float(model.pvalues["Intercept"])
    is_rejected = alpha_p_value > ALPHA_SIGNIFICANCE_LEVEL

    rolling_r2_list = []
    for i in range(ROLLING_WINDOW - 1, len(df_reg)):
        window = df_reg.iloc[i - ROLLING_WINDOW + 1 : i + 1]
        try:
            roll_model = smf.ols(formula, data=window).fit()
            rolling_r2_list.append((window.index[-1], roll_model.rsquared))
        except Exception:
            rolling_r2_list.append((window.index[-1], np.nan))

    rolling_r_squared = pd.Series(
        [r[1] for r in rolling_r2_list],
        index=pd.DatetimeIndex([r[0] for r in rolling_r2_list]),
    )
    betas = {
        c: float(model.params.get(c, model.params.get(f"Q('{c}')", np.nan)))
        for c in factor_cols
    }
    return {
        "alpha": alpha,
        "alpha_p_value": alpha_p_value,
        "is_rejected": is_rejected,
        "rolling_r_squared": rolling_r_squared,
        "betas": betas,
        "rsquared": float(model.rsquared),
    }


def _rolling_median_turnover_eur(
    raw_prices: pd.DataFrame,
    window: int = ROLLING_TURNOVER_DAYS,
) -> pd.DataFrame:
    """
    20-day rolling median turnover (EUR) per ticker per date.
    shift(1) to avoid look-ahead.
    """
    piv = raw_prices.pivot_table(
        index="date", columns="ticker", values="PX_TURN_OVER"
    ).sort_index()
    med = piv.rolling(window, min_periods=max(1, window // 2)).median().shift(1)
    return med


def _illiquid_decile_mask(
    raw_prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Boolean mask: True = ticker is in bottom 10% (most illiquid) on that date.
    Based on 20d rolling median turnover in EUR. Higher turnover = more liquid.
    """
    med = _rolling_median_turnover_eur(raw_prices, ROLLING_TURNOVER_DAYS)
    # Bottom decile = lowest turnover = most illiquid
    threshold = med.quantile(DECILE_DROP_PCT, axis=1)
    is_illiquid = med.le(threshold, axis=0)
    return is_illiquid


def run_decile_drop_test(
    alpha_df: pd.DataFrame,
    raw_prices: pd.DataFrame,
    price_returns: pd.DataFrame,
    macro_regime_df: pd.DataFrame,
    news_df: pd.DataFrame,
    rebalance_dates: List[pd.Timestamp],
) -> Dict[str, Any]:
    """
    Decile Drop Test: OOS backtest with bottom 10% liquidity excluded.
    Falsification: Sharpe drop >30% → Liquidity Harvesting (reject as genuine alpha).
    """
    from research._shared import _NeutralNLP, compute_equity_curve, compute_metrics
    from src.engine.backtest_loop import BacktestEngine
    from src.layer5_portfolio.hrp_clustering import HierarchicalRiskParity
    from src.layer5_portfolio.dynamic_vol_target import DynamicVolTargeting

    if "PX_TURN_OVER" not in raw_prices.columns:
        return {
            "sharpe_full": np.nan,
            "sharpe_decile_drop": np.nan,
            "sharpe_drop_pct": np.nan,
            "liquidity_harvesting": False,
            "skipped": "no PX_TURN_OVER",
        }

    illiquid_mask = _illiquid_decile_mask(raw_prices)
    illiquid_stack = illiquid_mask.stack()
    illiquid_set = {
        (pd.Timestamp(idx[0]), str(idx[1]))
        for idx, v in illiquid_stack.items()
        if v
    }

    alpha_drop = alpha_df.copy()
    alpha_drop["date"] = pd.to_datetime(alpha_drop["date"])
    drop_idx = alpha_drop.apply(
        lambda r: (r["date"], r["ticker"]) in illiquid_set, axis=1
    )
    alpha_drop.loc[drop_idx, "alpha_score"] = np.nan

    engine = BacktestEngine(
        alpha_model=None,
        nlp_sentinel=_NeutralNLP(),
        hrp_model=HierarchicalRiskParity(),
        vol_targeter=DynamicVolTargeting(),
    )

    bt_full = engine.run_backtest(
        price_df=price_returns,
        alpha_df=alpha_df,
        macro_df=macro_regime_df,
        news_df=news_df,
        rebalance_dates=rebalance_dates,
        currency_series=None,
    )
    bt_drop = engine.run_backtest(
        price_df=price_returns,
        alpha_df=alpha_drop,
        macro_df=macro_regime_df,
        news_df=news_df,
        rebalance_dates=rebalance_dates,
        currency_series=None,
    )

    eq_full = compute_equity_curve(bt_full, price_returns)
    eq_drop = compute_equity_curve(bt_drop, price_returns)
    met_full = compute_metrics(eq_full)
    met_drop = compute_metrics(eq_drop)
    sharpe_full = met_full.get("sharpe", np.nan)
    sharpe_drop = met_drop.get("sharpe", np.nan)

    if np.isfinite(sharpe_full) and abs(sharpe_full) > 1e-12:
        sharpe_drop_pct = (sharpe_full - sharpe_drop) / abs(sharpe_full)
    else:
        sharpe_drop_pct = np.nan

    liquidity_harvesting = (
        np.isfinite(sharpe_drop_pct)
        and sharpe_drop_pct > SHARPE_DROP_FAIL_THRESHOLD
    )

    return {
        "sharpe_full": sharpe_full,
        "sharpe_decile_drop": sharpe_drop,
        "sharpe_drop_pct": sharpe_drop_pct,
        "liquidity_harvesting": liquidity_harvesting,
    }


def run_full_factor_and_liquidity_report(
    data: pd.DataFrame,
    transaction_cost_rate: float = 0.001,
) -> Dict[str, Any]:
    """
    Run factor neutrality + Decile Drop Test when backtest data is available.
    Caller can pass data with strategy_daily_returns + factors.
    For Decile Drop, use run_decile_drop_test with research data.
    """
    result = run_factor_neutrality_test(data, transaction_cost_rate)
    result["decile_drop"] = None  # Populate by caller if running backtest
    return result


def run_factor_neutrality_report(
    data: pd.DataFrame,
    transaction_cost_rate: float = 0.001,
) -> None:
    """Run test and print formatted report."""
    result = run_factor_neutrality_test(data, transaction_cost_rate)
    factor_label = "5+UMD+QMJ" + ("+ILLIQ" if LIQUIDITY_FACTOR_COL in result["betas"] else "")

    print("=" * 60)
    print(f"Factor Neutrality Test (Fama-French {factor_label})")
    print("=" * 60)
    print(f"Transaction cost rate: {transaction_cost_rate:.4f} per day")
    print(f"\nAlpha (intercept):     {result['alpha']:.6f}")
    print(f"Alpha p-value:         {result['alpha_p_value']:.4f}")
    print(f"R-squared:             {result['rsquared']:.4f}")
    print(f"\nFactor betas:")
    for f, b in result["betas"].items():
        print(f"  {f:10s} {b:.4f}")
    print(f"\nFalsification:         {'REJECTED (repackaged beta)' if result['is_rejected'] else 'PASSED'}")
    print(f"Rolling R² (252d):     mean={result['rolling_r_squared'].mean():.4f}, std={result['rolling_r_squared'].std():.4f}")
    print("=" * 60)


def run_decile_drop_report() -> Dict[str, Any]:
    """Run Decile Drop Test with research data and print Falsification Report."""
    from research._shared import setup_research_data

    print("Decile Drop Test: Loading research data...")
    alpha_df, price_returns, macro_regime_df, news_df, raw_prices, rebalance_dates = (
        setup_research_data()
    )
    result = run_decile_drop_test(
        alpha_df, raw_prices, price_returns, macro_regime_df, news_df, rebalance_dates
    )
    if result.get("skipped"):
        print(f"  Skipped: {result['skipped']}")
        return result

    print("=" * 70)
    print("Decile Drop Test — Liquidity Anomaly Detection")
    print("=" * 70)
    print(f"  Baseline Sharpe (full universe):     {result['sharpe_full']:.4f}")
    print(f"  Sharpe (bottom 10% illiquid excluded): {result['sharpe_decile_drop']:.4f}")
    drop_pct = result.get("sharpe_drop_pct", np.nan)
    if np.isfinite(drop_pct):
        print(f"  Sharpe change: {drop_pct:.1%} (positive = worse without illiquid)")
    print()
    if result["liquidity_harvesting"]:
        print("  >>> LIQUIDITY HARVESTING: REJECTED as genuine alpha strategy <<<")
        print("  Falsification: Sharpe drops >30% when illiquid decile excluded.")
    else:
        print("  PASS: Strategy not classified as Liquidity Harvesting.")
    print("=" * 70)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--decile-drop", action="store_true", help="Run Decile Drop Test (Liquidity Controls)")
    args = parser.parse_args()

    if args.decile_drop:
        run_decile_drop_report()
        sys.exit(0)

    # Demo with synthetic data (factor neutrality, incl. optional ILLIQ)
    np.random.seed(42)
    n = 504  # ~2 years
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    factors = pd.DataFrame(
        {
            "Mkt-RF": np.random.randn(n) * 0.01,
            "SMB": np.random.randn(n) * 0.005,
            "HML": np.random.randn(n) * 0.005,
            "UMD": np.random.randn(n) * 0.005,
            "QMJ": np.random.randn(n) * 0.005,
            LIQUIDITY_FACTOR_COL: np.random.randn(n) * 0.003,
        },
        index=dates,
    )
    # Strategy = 0.5*Mkt + noise + small alpha
    strategy = 0.5 * factors["Mkt-RF"] + np.random.randn(n) * 0.005 + 0.0001
    data = pd.DataFrame(
        {"strategy_daily_returns": strategy.values},
        index=dates,
    ).join(factors)
    data.index.name = None
    data = data.reset_index(drop=True)

    run_factor_neutrality_report(data, transaction_cost_rate=0.0005)
