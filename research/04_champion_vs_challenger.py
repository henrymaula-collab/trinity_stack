"""
Champion vs. Challenger: Rigorous comparison of Trinity Full Stack vs Trinity Lite.
Ergodicity & survival metrics: CAGR, Max DD, Omega Ratio, Probability of Ruin,
Terminal Wealth 10th percentile. Follows .cursorrules strictly.
Does not modify src/ or run_pipeline.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

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
    _NeutralNLP,
)
from src.engine.backtest_loop import BacktestEngine
from src.layer4_nlp.xlm_roberta_sentinel import NLPSentinel
from src.layer5_portfolio.hrp_clustering import HierarchicalRiskParity
from src.layer5_portfolio.dynamic_vol_target import DynamicVolTargeting

RUIN_THRESHOLD: float = -0.35
Bootstrap_N: int = 3000
Block_size: int = 21


# ---- Challenger: Inverse Volatility with 5% max cap ----

class InverseVolatilityWithCap:
    """
    Simple inverse-volatility allocation with 5% per-asset cap.
    No HRP, no currency isolation. Survival-optimized.
    """

    def __init__(self, max_weight: float = 0.05):
        self.max_weight = max_weight

    def allocate(
        self, returns_df: pd.DataFrame, currency_series: pd.Series
    ) -> pd.Series:
        cov = returns_df.cov()
        iv = 1.0 / np.diag(cov)
        iv = np.clip(iv, 0, np.inf)
        if iv.sum() <= 0:
            n = len(returns_df.columns)
            return pd.Series(1.0 / n, index=returns_df.columns)
        w = pd.Series(iv / iv.sum(), index=returns_df.columns)
        # Cap each weight at max_weight, renormalize
        w = w.clip(upper=self.max_weight)
        s = w.sum()
        if s <= 0:
            n = len(returns_df.columns)
            return pd.Series(1.0 / n, index=returns_df.columns)
        return w / s


# ---- Challenger: Aggressive regime scaling ----

def _build_aggressive_regime_df(
    macro_regime_df: pd.DataFrame,
    raw_prices: pd.DataFrame,
    macro_df: pd.DataFrame,
    sigma_threshold: float = 1.5,
    sma_window: int = 63,
) -> pd.DataFrame:
    """
    Build regime series that cuts exposure (regime=2) when:
    - Cross-sectional avg spread > mean + 1.5*std (shifted), or
    - V2TX > SMA(V2TX, 63) indicating elevated vol / trend broken.
    """
    dates = macro_regime_df.index
    out = macro_regime_df[["regime"]].copy()

    # Spread stress: date-level avg spread vs rolling baseline
    px = raw_prices.copy()
    px["date"] = pd.to_datetime(px["date"])
    daily_spread = px.groupby("date")["BID_ASK_SPREAD_PCT"].mean()
    roll_mean = daily_spread.rolling(window=sma_window, min_periods=21).mean().shift(1)
    roll_std = daily_spread.rolling(window=sma_window, min_periods=21).std().shift(1)
    threshold = roll_mean + sigma_threshold * roll_std
    spread_stress = (daily_spread > threshold).reindex(dates).fillna(False)

    # SMA trend broken: V2TX above its SMA => elevated vol (stress)
    v2tx = macro_df["V2TX"]
    v2tx_sma = v2tx.rolling(window=sma_window, min_periods=21).mean().shift(1)
    trend_broken = (v2tx > v2tx_sma).reindex(dates).fillna(False)

    # Either condition -> regime 2 (full cut)
    cut = spread_stress | trend_broken
    out.loc[cut, "regime"] = 2
    return out


# ---- Backtest runner that supports regime override ----

def _run_backtest(
    alpha_df: pd.DataFrame,
    price_returns: pd.DataFrame,
    macro_df: pd.DataFrame,
    news_df: pd.DataFrame,
    rebalance_dates: List[pd.Timestamp],
    nlp: Any,
    hrp: Any,
    vol_targeter: Any,
) -> pd.DataFrame:
    engine = BacktestEngine(
        alpha_model=None,
        nlp_sentinel=nlp,
        hrp_model=hrp,
        vol_targeter=vol_targeter,
    )
    return engine.run_backtest(
        price_df=price_returns,
        alpha_df=alpha_df,
        macro_df=macro_df,
        news_df=news_df,
        rebalance_dates=rebalance_dates,
        currency_series=None,
    )


# ---- Strategy returns and block bootstrap ----

def _strategy_returns(
    backtest_result: pd.DataFrame,
    price_returns: pd.DataFrame,
) -> pd.Series:
    """Daily strategy returns from weight records."""
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


def _block_bootstrap_paths(
    returns: pd.Series,
    n_paths: int = 3000,
    block_size: int = 21,
    random_state: int = GLOBAL_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Block bootstrap returns. For each path compute Max DD and Terminal Wealth.
    Returns (max_dds, terminal_wealths) each shape (n_paths,).
    """
    rng = np.random.default_rng(random_state)
    ret = returns.dropna().values
    n = len(ret)
    if n < block_size * 2:
        return np.full(n_paths, np.nan), np.full(n_paths, np.nan)

    n_blocks = (n + block_size - 1) // block_size
    max_dds = np.zeros(n_paths)
    terminal_wealths = np.zeros(n_paths)

    for i in range(n_paths):
        blocks = []
        for _ in range(n_blocks):
            start = rng.integers(0, max(1, n - block_size + 1))
            blocks.append(ret[start : start + block_size])
        boot = np.concatenate(blocks)[:n]
        eq = np.cumprod(1.0 + boot)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / np.where(peak > 0, peak, 1.0)
        max_dds[i] = float(np.min(dd))
        terminal_wealths[i] = float(eq[-1]) if len(eq) > 0 else np.nan

    return max_dds, terminal_wealths


def _prob_ruin(max_dds: np.ndarray, threshold: float = RUIN_THRESHOLD) -> float:
    """Percentage of paths with Max DD worse than threshold."""
    valid = max_dds[np.isfinite(max_dds)]
    if len(valid) == 0:
        return np.nan
    return float(np.mean(valid < threshold))


def _terminal_wealth_p10(terminal_wealths: np.ndarray) -> float:
    """10th percentile terminal wealth (bad luck scenario)."""
    valid = terminal_wealths[np.isfinite(terminal_wealths)]
    if len(valid) == 0:
        return np.nan
    return float(np.percentile(valid, 10))


# ---- Main run ----

def run_champion_vs_challenger() -> pd.DataFrame:
    """Run Champion (Full Stack) vs Challenger (Trinity Lite) and return comparative table."""
    print("Champion vs Challenger: Loading data and building alpha...")
    alpha_df, price_returns, macro_regime_df, news_df, raw_prices, rebalance_dates = (
        setup_research_data()
    )

    # Challenger: aggressive regime overlay
    macro_df_full = _research_load_macro()
    aggressive_regime_df = _build_aggressive_regime_df(
        macro_regime_df, raw_prices, macro_df_full
    )

    try:
        nlp_full = NLPSentinel()
    except Exception:
        print("WARNING: NLPSentinel init failed. Using NeutralNLP for Champion.")
        nlp_full = _NeutralNLP()

    results: List[Dict[str, Any]] = []

    # ---- Champion (Full Stack) ----
    print("\n--- Champion (Full Stack) ---")
    bt_champ = _run_backtest(
        alpha_df, price_returns, macro_regime_df, news_df, rebalance_dates,
        nlp=nlp_full,
        hrp=HierarchicalRiskParity(),
        vol_targeter=DynamicVolTargeting(),
    )
    eq_champ = compute_equity_curve(bt_champ, price_returns)
    m_champ = compute_metrics(eq_champ)
    ret_champ = _strategy_returns(bt_champ, price_returns)
    max_dds_champ, tw_champ = _block_bootstrap_paths(
        ret_champ, n_paths=Bootstrap_N, block_size=Block_size
    )

    results.append({
        "System": "Champion (Full Stack)",
        "CAGR": m_champ["cagr"],
        "Max_DD": m_champ["max_dd"],
        "Omega_Ratio": m_champ["omega_ratio"],
        "Prob_Ruin": _prob_ruin(max_dds_champ),
        "Terminal_Wealth_P10": _terminal_wealth_p10(tw_champ),
    })
    print(f"  CAGR={m_champ['cagr']:.2%}, MaxDD={m_champ['max_dd']:.2%}, Omega={m_champ['omega_ratio']:.3f}")
    print(f"  Prob(Ruin): {_prob_ruin(max_dds_champ):.2%}, TW P10: {_terminal_wealth_p10(tw_champ):.4f}")

    # ---- Challenger (Trinity Lite) ----
    print("\n--- Challenger (Trinity Lite) ---")
    bt_chall = _run_backtest(
        alpha_df, price_returns, aggressive_regime_df, news_df, rebalance_dates,
        nlp=_NeutralNLP(),
        hrp=InverseVolatilityWithCap(max_weight=0.05),
        vol_targeter=DynamicVolTargeting(),
    )
    eq_chall = compute_equity_curve(bt_chall, price_returns)
    m_chall = compute_metrics(eq_chall)
    ret_chall = _strategy_returns(bt_chall, price_returns)
    max_dds_chall, tw_chall = _block_bootstrap_paths(
        ret_chall, n_paths=Bootstrap_N, block_size=Block_size
    )

    results.append({
        "System": "Challenger (Trinity Lite)",
        "CAGR": m_chall["cagr"],
        "Max_DD": m_chall["max_dd"],
        "Omega_Ratio": m_chall["omega_ratio"],
        "Prob_Ruin": _prob_ruin(max_dds_chall),
        "Terminal_Wealth_P10": _terminal_wealth_p10(tw_chall),
    })
    print(f"  CAGR={m_chall['cagr']:.2%}, MaxDD={m_chall['max_dd']:.2%}, Omega={m_chall['omega_ratio']:.3f}")
    print(f"  Prob(Ruin): {_prob_ruin(max_dds_chall):.2%}, TW P10: {_terminal_wealth_p10(tw_chall):.4f}")

    # Save daily returns for tearsheet (05_tearsheet_and_benchmarks.py)
    ret_df = pd.DataFrame({
        "Champion": ret_champ,
        "Challenger": ret_chall,
    }).dropna(how="all")
    ret_path = _PROJECT_ROOT / "research" / "strategy_returns.parquet"
    ret_df.to_parquet(ret_path, index=True)
    print(f"\nSaved daily returns to {ret_path}")

    return pd.DataFrame(results)


def _research_load_macro() -> pd.DataFrame:
    """Load macro data for aggressive regime overlay (mirrors _shared internals)."""
    import research._shared as _sh
    return _sh._load_parquet_or_mock("macro")


if __name__ == "__main__":
    df = run_champion_vs_challenger()
    print("\n" + "=" * 70)
    print("Champion vs Challenger Results")
    print("=" * 70)
    print(df.to_string(index=False))
    out_path = _PROJECT_ROOT / "research" / "champion_vs_challenger.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print("\nKeep the current system until mathematics says otherwise.")
