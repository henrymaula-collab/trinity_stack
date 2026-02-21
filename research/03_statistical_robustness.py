"""
Statistical Robustness & Deflated Sharpe Ratio.
Block bootstraps strategy returns to compute Probabilistic Sharpe Ratio (PSR)
and Deflated Sharpe Ratio (DSR). Academic standard for avoiding data mining bias.
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
from research._shared import GLOBAL_SEED


def strategy_returns(
    backtest_result: pd.DataFrame,
    price_returns: pd.DataFrame,
) -> pd.Series:
    """Compute daily strategy returns from weight records and price returns."""
    if backtest_result.empty or price_returns.empty:
        return pd.Series(dtype=float)

    weight_df = backtest_result.copy()
    weight_df["date"] = pd.to_datetime(weight_df["date"])
    dates = sorted(weight_df["date"].unique())
    price_returns = price_returns.sort_index()

    all_dates = price_returns.index
    strat_ret = pd.Series(0.0, index=all_dates)

    for i, rb_date in enumerate(dates):
        w_row = weight_df[weight_df["date"] == rb_date]
        prev_weights = dict(zip(w_row["ticker"], w_row["target_weight"]))
        start = rb_date
        end = dates[i + 1] if i + 1 < len(dates) else all_dates.max() + pd.Timedelta(days=1)
        mask = (all_dates > start) & (all_dates < end)
        period = price_returns.loc[mask]
        tickers = [t for t in prev_weights if t in period.columns]
        w_vec = np.array([prev_weights[t] for t in tickers])
        for dt, row in period[tickers].iterrows():
            r = row.values
            if np.any(np.isnan(r)):
                continue
            strat_ret.loc[dt] = float(np.dot(w_vec, r))

    return strat_ret


def block_bootstrap(
    returns: pd.Series,
    n_paths: int = 10_000,
    block_size: int = 21,
    random_state: int = GLOBAL_SEED,
) -> np.ndarray:
    """
    Block bootstrap of return series. Preserves autocorrelation.
    Returns array of shape (n_paths,) of annualized Sharpe ratios.
    """
    rng = np.random.default_rng(random_state)
    ret = returns.dropna().values
    n = len(ret)
    if n < block_size * 2:
        return np.full(n_paths, np.nan)

    n_blocks = (n + block_size - 1) // block_size
    sharpes = np.zeros(n_paths)

    for i in range(n_paths):
        blocks = []
        for _ in range(n_blocks):
            start = rng.integers(0, n - block_size + 1) if n > block_size else 0
            blocks.append(ret[start : start + block_size])
        boot = np.concatenate(blocks)[:n]
        if boot.std() < 1e-12:
            sharpes[i] = np.nan
        else:
            sharpes[i] = (boot.mean() / boot.std()) * np.sqrt(252)

    return sharpes


def probabilistic_sharpe_ratio(sharpes: np.ndarray) -> float:
    """PSR: proportion of bootstrap Sharpe > 0."""
    valid = sharpes[np.isfinite(sharpes)]
    if len(valid) == 0:
        return np.nan
    return float(np.mean(valid > 0))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    sharpes: np.ndarray,
    n_trials: int = 100,
) -> float:
    """
    DSR: adjust observed Sharpe for multiple testing.
    n_trials: effective number of independent strategy trials (conservative).
    """
    valid = sharpes[np.isfinite(sharpes)]
    if len(valid) < 10:
        return np.nan
    mean_sr = np.mean(valid)
    std_sr = np.std(valid)
    if std_sr < 1e-12:
        return observed_sharpe
    expected_max_sr = mean_sr + std_sr * np.sqrt(2 * np.log(n_trials))
    return float(observed_sharpe - expected_max_sr)


def run_robustness_study(
    n_bootstrap: int = 10_000,
    block_size: int = 21,
    n_trials: int = 100,
) -> None:
    """Run block bootstrap, compute PSR and DSR."""
    print("Statistical Robustness: Loading data...")
    alpha_df, price_returns, macro_regime_df, news_df, _, rebalance_dates = (
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

    strat_ret = strategy_returns(bt, price_returns)
    equity = (1 + strat_ret).cumprod()
    m = compute_metrics(equity)
    observed_sharpe = m["sharpe"]

    print(f"Block bootstrapping ({n_bootstrap} paths, block={block_size})...")
    sharpes = block_bootstrap(strat_ret, n_paths=n_bootstrap, block_size=block_size)
    psr = probabilistic_sharpe_ratio(sharpes)
    dsr = deflated_sharpe_ratio(observed_sharpe, sharpes, n_trials=n_trials)

    print("\n" + "=" * 60)
    print("Statistical Robustness Results")
    print("=" * 60)
    print(f"Observed Sharpe:     {observed_sharpe:.3f}")
    print(f"PSR (P(SR>0)):      {psr:.2%}")
    print(f"DSR (deflated):     {dsr:.3f}")
    print(f"Bootstrap SR mean:  {np.nanmean(sharpes):.3f}")
    print(f"Bootstrap SR std:   {np.nanstd(sharpes):.3f}")
    print(f"\nStatistically robust (PSR > 95%): {'Yes' if psr > 0.95 else 'No'}")


if __name__ == "__main__":
    run_robustness_study(n_bootstrap=5000, block_size=21, n_trials=100)
