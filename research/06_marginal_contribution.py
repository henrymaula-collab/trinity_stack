"""
Shapley-Ablation: Ergodisk marginalbidrag med Inferential Hardness.

Block Bootstrap (12-mo), 95% CI, Stability Score, Regime-Specific Shapley,
Refined Fractional Kelly, Final Verdict (DELETION logic).
PhD-proof per peer-review.

Does not modify src/ or run_pipeline.py.
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from research._shared import (
    GLOBAL_SEED,
    LOO_COMPONENTS,
    setup_research_data,
    setup_research_data_with_components,
    compute_equity_curve,
    compute_metrics,
    _NeutralNLP,
)
from src.engine.backtest_loop import BacktestEngine
from src.layer4_nlp.xlm_roberta_sentinel import NLPSentinel
from src.layer5_portfolio.hrp_clustering import HierarchicalRiskParity
from src.layer5_portfolio.dynamic_vol_target import DynamicVolTargeting

LAYERS = ["HMM_Regime", "NLP_Sentinel", "HRP_Clustering", "Vol_Target", "Debt_Wall"]
STRESS_WINDOWS = [
    ("GFC_2008", "2008-01-01", "2008-12-31"),
    ("Euro_2011", "2011-01-01", "2011-12-31"),
    ("Covid_2020", "2020-01-01", "2020-12-31"),
    ("Bear_2022", "2022-01-01", "2022-12-31"),
]
ES_QUANTILE = 0.05
REDUNDANCY_CORR_THRESHOLD = 0.70
BOOTSTRAP_N = 1000
# Bonferroni: alpha/n for n layers (model selection correction)
MULTIPLE_TEST_ALPHA = 0.05 / 5
BLOCK_SIZE_DAYS = 252  # 12 months
STABILITY_MIN_PCT = 65.0
# Ergodicity: require Crisis phi_G 97.5% CI < 0 (not just median)
KELLY_SHRINKAGE = 0.5
KELLY_SAFETY = 0.3


class EqualWeightHRP:
    def allocate(
        self, returns_df: pd.DataFrame, currency_series: pd.Series
    ) -> pd.Series:
        n = len(returns_df.columns)
        return pd.Series(1.0 / n, index=returns_df.columns)


class PassThroughVolTarget:
    def apply_targets(
        self,
        hrp_weights: pd.Series,
        cov_matrix: pd.DataFrame,
        nlp_multipliers: pd.Series,
        regime_exposure: float,
        historical_volatilities: pd.Series,
        portfolio_drawdown: float = 0.0,
        target_vol_annual: float | None = None,
    ) -> pd.Series:
        w = hrp_weights * nlp_multipliers
        s = w.sum()
        if s <= 0:
            return hrp_weights * 0.0
        return w / s * regime_exposure


def _strategy_returns(
    backtest_result: pd.DataFrame,
    price_returns: pd.DataFrame,
) -> pd.Series:
    if backtest_result.empty or price_returns.empty:
        return pd.Series(dtype=float)
    weight_df = backtest_result.copy()
    weight_df["date"] = pd.to_datetime(weight_df["date"])
    dates = sorted(weight_df["date"].unique())
    price_returns = price_returns.sort_index()
    all_dates = price_returns.index
    strat_ret = pd.Series(0.0, index=all_dates)
    sorted_dates = np.sort(all_dates.unique())
    for i, rb_date in enumerate(dates):
        w_row = weight_df[weight_df["date"] == rb_date]
        prev_weights = dict(zip(w_row["ticker"], w_row["target_weight"]))
        # T+1: signal at close of rb_date → execute at open of next trading day.
        # Do not credit T+1 return (overnight close→open not captured).
        pos = np.searchsorted(sorted_dates, rb_date, side="right") - 1
        if pos < 0:
            pos = 0
        t_plus_2_pos = min(pos + 2, len(sorted_dates) - 1)
        start = pd.Timestamp(sorted_dates[t_plus_2_pos])
        end = dates[i + 1] if i + 1 < len(dates) else all_dates.max() + pd.Timedelta(days=1)
        mask = (all_dates >= start) & (all_dates < end)
        period = price_returns.loc[mask]
        tickers = [t for t in prev_weights if t in period.columns]
        if not tickers:
            continue
        w_vec = np.array([prev_weights[t] for t in tickers])
        for dt, row in period[tickers].iterrows():
            r = row.values
            if np.any(np.isnan(r)):
                continue
            strat_ret.loc[dt] = float(np.dot(w_vec, r))
    return strat_ret


def _apply_debt_wall_blacklist(
    backtest_result: pd.DataFrame,
    tickers_universe: List[str],
    n_exclude: int = 2,
    seed: int = GLOBAL_SEED,
) -> pd.DataFrame:
    """
    Exogenous mock: exclude random tickers (uncorrelated with alpha).
    With real data, use Debt Wall's Cash Runway Deficit logic instead.
    """
    rng = np.random.default_rng(seed)
    tickers = [t for t in tickers_universe if t in backtest_result["ticker"].unique()]
    if len(tickers) <= n_exclude:
        return backtest_result
    blacklist = set(rng.choice(tickers, size=n_exclude, replace=False))
    out = backtest_result.copy()
    out.loc[out["ticker"].isin(blacklist), "target_weight"] = 0.0
    for rb_date in out["date"].unique():
        mask = out["date"] == rb_date
        sub = out.loc[mask]
        s = sub["target_weight"].sum()
        if s > 0:
            out.loc[mask, "target_weight"] = out.loc[mask, "target_weight"] / s
    return out


def _run_backtest(
    alpha_df: pd.DataFrame,
    price_returns: pd.DataFrame,
    macro_regime_df: pd.DataFrame,
    news_df: pd.DataFrame,
    rebalance_dates: List[pd.Timestamp],
    layer_config: FrozenSet[str],
    nlp_full: Any,
) -> pd.DataFrame:
    use_hmm = "HMM_Regime" in layer_config
    use_nlp = "NLP_Sentinel" in layer_config
    use_hrp = "HRP_Clustering" in layer_config
    use_vol = "Vol_Target" in layer_config
    use_debt = "Debt_Wall" in layer_config

    macro_df = macro_regime_df if use_hmm else _neutral_regime(macro_regime_df)
    nlp = nlp_full if use_nlp else _NeutralNLP()
    hrp = HierarchicalRiskParity() if use_hrp else EqualWeightHRP()
    vol = DynamicVolTargeting() if use_vol else PassThroughVolTarget()

    engine = BacktestEngine(
        alpha_model=None,
        nlp_sentinel=nlp,
        hrp_model=hrp,
        vol_targeter=vol,
    )
    bt = engine.run_backtest(
        price_df=price_returns,
        alpha_df=alpha_df,
        macro_df=macro_df,
        news_df=news_df,
        rebalance_dates=rebalance_dates,
        currency_series=None,
    )
    if use_debt:
        tickers_universe = alpha_df["ticker"].unique().tolist()
        bt = _apply_debt_wall_blacklist(bt, tickers_universe)
    return bt


def _neutral_regime(macro_regime_df: pd.DataFrame) -> pd.DataFrame:
    out = macro_regime_df.copy()
    out["regime"] = 1.0
    return out


def _compute_turnover(backtest_result: pd.DataFrame) -> float:
    """
    Average rebalance turnover: sum(|w_new - w_prev|) per date, averaged.
    One-sided (gross) turnover.
    """
    if backtest_result.empty or len(backtest_result) < 2:
        return 0.0
    weight_df = backtest_result.copy()
    weight_df["date"] = pd.to_datetime(weight_df["date"])
    dates = sorted(weight_df["date"].unique())
    prev_weights: Dict[str, float] = {}
    turnovers = []
    for rb_date in dates:
        w_row = weight_df[weight_df["date"] == rb_date]
        curr = dict(zip(w_row["ticker"], w_row["target_weight"]))
        all_tickers = set(prev_weights) | set(curr)
        tot = 0.0
        for t in all_tickers:
            w_prev = prev_weights.get(t, 0.0)
            w_curr = curr.get(t, 0.0)
            tot += abs(w_curr - w_prev)
        turnovers.append(tot)
        prev_weights = curr
    return float(np.mean(turnovers)) if turnovers else 0.0


def _run_backtest_loo(
    alpha_df: pd.DataFrame,
    price_returns: pd.DataFrame,
    macro_regime_df: pd.DataFrame,
    news_df: pd.DataFrame,
    rebalance_dates: List[pd.Timestamp],
    enabled_components: FrozenSet[str],
    nlp_full: Any,
) -> pd.DataFrame:
    """Backtest with LOO component toggles (HMM, HRP). Insider/Meta affect alpha_df upstream."""
    base = {"NLP_Sentinel", "Vol_Target", "Debt_Wall"}
    config = set(base)
    if "HMM_Regime" in enabled_components:
        config.add("HMM_Regime")
    if "HRP_Clustering" in enabled_components:
        config.add("HRP_Clustering")
    return _run_backtest(
        alpha_df, price_returns, macro_regime_df, news_df, rebalance_dates,
        frozenset(config), nlp_full,
    )


def _geometric_growth(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return np.nan
    return float(np.mean(np.log(1.0 + r)))


def _expected_shortfall_95(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty or len(r) < 5:
        return np.nan
    q = r.quantile(ES_QUANTILE)
    tail = r[r <= q]
    return float(tail.mean()) if len(tail) > 0 else np.nan


def _subset_returns(
    returns: pd.Series,
    start: str,
    end: str,
) -> pd.Series:
    idx = pd.to_datetime(returns.index)
    mask = (idx >= start) & (idx <= end)
    return returns.loc[mask]


def _shapley_weight(n: int, s: int) -> float:
    if s < 0 or s >= n:
        return 0.0
    return math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)


def _draw_block_indices(
    n: int,
    block_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one block-bootstrap index sequence. Same for ALL configs (Shapley structure)."""
    if n < block_days * 2:
        return np.arange(n)
    n_blocks = (n + block_days - 1) // block_days
    indices = []
    for _ in range(n_blocks):
        start = rng.integers(0, max(1, n - block_days + 1))
        indices.extend(range(start, start + block_days))
    return np.array(indices[:n])


def _apply_block_indices(
    returns: pd.Series,
    indices: np.ndarray,
) -> pd.Series:
    """Resample returns using given indices. Same length and index as input."""
    r = returns.sort_index()
    vals = r.values.astype(float)
    n = len(vals)
    if n == 0 or len(indices) == 0:
        return returns
    idx = np.clip(indices[:n], 0, n - 1)
    boot_vals = vals[idx]
    return pd.Series(boot_vals, index=r.index[: len(boot_vals)])


def _compute_shapley_from_metrics(
    metrics_per_config: Dict[FrozenSet[str], Dict[str, float]],
    all_subsets: List[Tuple[str, ...]],
    n_layers: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    shapley_G = {k: 0.0 for k in LAYERS}
    shapley_ES = {k: 0.0 for k in LAYERS}
    for subset in all_subsets:
        for layer in LAYERS:
            if layer in subset:
                continue
            s = frozenset(subset)
            s_with = s | {layer}
            G_S = metrics_per_config.get(s, {}).get("G", np.nan)
            G_Sk = metrics_per_config.get(s_with, {}).get("G", np.nan)
            ES_S = metrics_per_config.get(s, {}).get("ES95", np.nan)
            ES_Sk = metrics_per_config.get(s_with, {}).get("ES95", np.nan)
            w = _shapley_weight(n_layers, len(subset))
            dG = (G_Sk - G_S) if not (np.isnan(G_S) or np.isnan(G_Sk)) else 0.0
            dES = (ES_Sk - ES_S) if not (np.isnan(ES_S) or np.isnan(ES_Sk)) else 0.0
            shapley_G[layer] += w * dG
            shapley_ES[layer] += w * dES
    return shapley_G, shapley_ES


def _refined_fractional_kelly_point_estimate(
    returns: pd.Series,
    shrinkage: float = KELLY_SHRINKAGE,
    safety: float = KELLY_SAFETY,
) -> float:
    """
    Kelly f* ≈ μ/σ² (arithmetic mean for formula; G for evaluation only).
    Small-return approximation: f* ≈ μ/σ². Using G≈μ−½σ² would bias.
    """
    r = returns.dropna()
    if r.empty or len(r) < 10:
        return 0.0
    mu = float(r.mean())
    sigma2 = float(r.var())
    if sigma2 <= 0:
        return 0.0
    mu_adj = shrinkage * mu + (1 - shrinkage) * 0.0
    skew = float(r.skew()) if hasattr(r, "skew") else 0.0
    kurt = float(r.kurtosis()) if hasattr(r, "kurtosis") else 0.0
    excess_var_penalty = 1.0 + 0.5 * abs(skew) + 0.1 * max(0, kurt - 3)
    sigma2_adj = sigma2 * excess_var_penalty
    f = (mu_adj / sigma2_adj) * safety
    return float(np.clip(f, 0.0, 1.0))


def _rolling_kelly_weights(
    returns: pd.Series,
    window: int = 252,
    shrinkage: float = KELLY_SHRINKAGE,
    safety: float = KELLY_SAFETY,
) -> pd.Series:
    """Rolling Kelly for truthful backtest. Each day uses only past data."""
    r = returns.dropna().sort_index()
    if len(r) < window:
        return pd.Series(dtype=float)
    out = pd.Series(index=r.index, dtype=float)
    for i in range(window, len(r) + 1):
        window_ret = r.iloc[i - window : i]
        out.iloc[i - 1] = _refined_fractional_kelly_point_estimate(
            window_ret, shrinkage, safety
        )
    return out


def run_shapley_ablation() -> pd.DataFrame:
    # Load data once (no I/O inside bootstrap loop)
    print("Shapley-Ablation (Inferential Hardness): Loading data...")
    alpha_df, price_returns, macro_regime_df, news_df, raw_prices, rebalance_dates = (
        setup_research_data()
    )

    try:
        nlp_full = NLPSentinel()
    except Exception:
        print("WARNING: NLPSentinel init failed. Using NeutralNLP.")
        nlp_full = _NeutralNLP()

    n_layers = len(LAYERS)
    all_subsets = list(
        itertools.chain.from_iterable(
            itertools.combinations(LAYERS, k) for k in range(n_layers + 1)
        )
    )

    print(f"Running {len(all_subsets)} combinatorial backtests...")
    results: Dict[FrozenSet[str], pd.Series] = {}
    for subset in all_subsets:
        config = frozenset(subset)
        bt = _run_backtest(
            alpha_df, price_returns, macro_regime_df, news_df, rebalance_dates,
            config, nlp_full,
        )
        ret = _strategy_returns(bt, price_returns)
        results[config] = ret

    full_config = frozenset(LAYERS)
    full_returns = results[full_config]

    # Regime masks (align macro_regime_df with return dates)
    regime_series = macro_regime_df["regime"].reindex(full_returns.index).ffill().bfill()
    daily_spread = (
        raw_prices.groupby("date")["BID_ASK_SPREAD_PCT"].mean()
        .reindex(full_returns.index)
        .ffill()
        .bfill()
    )
    spread_p90 = daily_spread.quantile(0.90)
    low_liq_mask = (daily_spread >= spread_p90).reindex(full_returns.index).fillna(False)
    crisis_mask = (regime_series < 0.33).fillna(False)
    bull_mask = (regime_series > 0.66).fillna(False)

    rng = np.random.default_rng(GLOBAL_SEED)
    bootstrap_phi_G: Dict[str, List[float]] = {k: [] for k in LAYERS}
    bootstrap_phi_ES: Dict[str, List[float]] = {k: [] for k in LAYERS}
    regime_phi_G: Dict[str, Dict[str, List[float]]] = {
        k: {"Crisis": [], "Bull": [], "Low_Liquidity": []} for k in LAYERS
    }

    print(f"Block Bootstrap: N={BOOTSTRAP_N}, block={BLOCK_SIZE_DAYS} days (single index per iter)...")
    ref_ret = results[full_config]
    n_obs = len(ref_ret.dropna())
    for b in range(BOOTSTRAP_N):
        indices = _draw_block_indices(n_obs, BLOCK_SIZE_DAYS, rng)
        boot_results: Dict[FrozenSet[str], pd.Series] = {}
        for config, ret in results.items():
            boot_results[config] = _apply_block_indices(ret, indices)

        metrics_b: Dict[FrozenSet[str], Dict[str, float]] = {}
        for config, ret in boot_results.items():
            metrics_b[config] = {
                "G": _geometric_growth(ret),
                "ES95": _expected_shortfall_95(ret),
            }

        phi_G_b, phi_ES_b = _compute_shapley_from_metrics(
            metrics_b, all_subsets, n_layers
        )
        for layer in LAYERS:
            bootstrap_phi_G[layer].append(phi_G_b[layer])
            bootstrap_phi_ES[layer].append(phi_ES_b[layer])

        for regime_name, mask in [
            ("Crisis", crisis_mask),
            ("Bull", bull_mask),
            ("Low_Liquidity", low_liq_mask),
        ]:
            if mask.sum() < 20:
                continue
            metrics_reg: Dict[FrozenSet[str], Dict[str, float]] = {}
            for config, ret in boot_results.items():
                sub = ret.loc[mask]
                metrics_reg[config] = {"G": _geometric_growth(sub), "ES95": _expected_shortfall_95(sub)}
            phi_G_r, _ = _compute_shapley_from_metrics(metrics_reg, all_subsets, n_layers)
            for layer in LAYERS:
                regime_phi_G[layer][regime_name].append(phi_G_r[layer])

    phi_G_arr = {k: np.array(v) for k, v in bootstrap_phi_G.items()}
    phi_ES_arr = {k: np.array(v) for k, v in bootstrap_phi_ES.items()}

    ci_alpha_lo = 100 * (MULTIPLE_TEST_ALPHA / 2)
    ci_alpha_hi = 100 * (1 - MULTIPLE_TEST_ALPHA / 2)
    ci_G_lo = {k: float(np.percentile(phi_G_arr[k], ci_alpha_lo)) for k in LAYERS}
    ci_G_hi = {k: float(np.percentile(phi_G_arr[k], ci_alpha_hi)) for k in LAYERS}
    ci_ES_lo = {k: float(np.percentile(phi_ES_arr[k], ci_alpha_lo)) for k in LAYERS}
    ci_ES_hi = {k: float(np.percentile(phi_ES_arr[k], ci_alpha_hi)) for k in LAYERS}
    stability_pct = {k: 100.0 * (phi_G_arr[k] > 0).mean() for k in LAYERS}
    phi_G_mean = {k: float(phi_G_arr[k].mean()) for k in LAYERS}
    phi_ES_mean = {k: float(phi_ES_arr[k].mean()) for k in LAYERS}

    phi_G_regime: Dict[str, Dict[str, float]] = {
        k: {"Crisis": np.nan, "Bull": np.nan, "Low_Liquidity": np.nan} for k in LAYERS
    }
    for layer in LAYERS:
        for regime_name in ["Crisis", "Bull", "Low_Liquidity"]:
            arr = regime_phi_G[layer].get(regime_name, [])
            phi_G_regime[layer][regime_name] = float(np.median(arr)) if len(arr) >= 10 else np.nan

    ci_crisis_97_5: Dict[str, float] = {}
    for layer in LAYERS:
        arr = regime_phi_G[layer].get("Crisis", [])
        ci_crisis_97_5[layer] = float(np.percentile(arr, 97.5)) if len(arr) >= 10 else np.nan
    ci_lowliq_97_5: Dict[str, float] = {}
    for layer in LAYERS:
        arr = regime_phi_G[layer].get("Low_Liquidity", [])
        ci_lowliq_97_5[layer] = float(np.percentile(arr, 97.5)) if len(arr) >= 10 else np.nan
    ergodic_penalty: Dict[str, bool] = {k: False for k in LAYERS}
    for layer in LAYERS:
        if not np.isnan(ci_crisis_97_5[layer]) and ci_crisis_97_5[layer] < 0:
            ergodic_penalty[layer] = True
        if not np.isnan(ci_lowliq_97_5[layer]) and ci_lowliq_97_5[layer] < 0:
            ergodic_penalty[layer] = True

    residuals: Dict[str, pd.Series] = {}
    for layer in LAYERS:
        config_without = full_config - {layer}
        ret_full = results[full_config]
        ret_without = results.get(config_without, pd.Series(dtype=float))
        common = ret_full.index.intersection(ret_without.index)
        if len(common) < 10:
            residuals[layer] = pd.Series(dtype=float)
            continue
        eps = ret_full.reindex(common).fillna(0) - ret_without.reindex(common).fillna(0)
        residuals[layer] = eps

    redundant_pairs: List[Tuple[str, str]] = []
    for i, a in enumerate(LAYERS):
        for j, b in enumerate(LAYERS):
            if i >= j:
                continue
            ra, rb = residuals.get(a), residuals.get(b)
            if ra is None or rb is None or ra.empty or rb.empty:
                continue
            common = ra.index.intersection(rb.index).drop_duplicates()
            if len(common) < 10:
                continue
            c = ra.reindex(common).fillna(0).corr(
                rb.reindex(common).fillna(0), method="spearman"
            )
            if not np.isnan(c) and abs(c) > REDUNDANCY_CORR_THRESHOLD:
                redundant_pairs.append((a, b))

    kelly_f = _refined_fractional_kelly_point_estimate(full_returns)

    verdict_delete: Dict[str, List[str]] = {k: [] for k in LAYERS}
    for layer in LAYERS:
        if ci_G_lo[layer] <= 0 <= ci_G_hi[layer]:
            verdict_delete[layer].append("CI_includes_zero")
        if stability_pct[layer] < STABILITY_MIN_PCT:
            verdict_delete[layer].append("stability_<65pct")
        if ergodic_penalty[layer]:
            verdict_delete[layer].append("ergodicity_penalty")
        for a, b in redundant_pairs:
            if a == layer or b == layer:
                other = b if a == layer else a
                if phi_G_mean[layer] < phi_G_mean[other]:
                    verdict_delete[layer].append(f"redundant_vs_{other}")

    summary_rows = []
    for layer in LAYERS:
        summary_rows.append({
            "Layer": layer,
            "phi_G_mean": phi_G_mean[layer],
            "phi_G_CI_lo": ci_G_lo[layer],
            "phi_G_CI_hi": ci_G_hi[layer],
            "phi_ES_mean": phi_ES_mean[layer],
            "phi_ES_CI_lo": ci_ES_lo[layer],
            "phi_ES_CI_hi": ci_ES_hi[layer],
            "stability_pct": stability_pct[layer],
            "phi_G_Crisis": phi_G_regime[layer]["Crisis"],
            "phi_G_Crisis_CI97.5": ci_crisis_97_5[layer],
            "phi_G_Bull": phi_G_regime[layer]["Bull"],
            "phi_G_LowLiq": phi_G_regime[layer]["Low_Liquidity"],
            "phi_G_LowLiq_CI97.5": ci_lowliq_97_5[layer],
            "ergodic_penalty": ergodic_penalty[layer],
            "redundant_with": ", ".join(
                b if a == layer else a for a, b in redundant_pairs if a == layer or b == layer
            ) or "—",
            "VERDICT_DELETE": "; ".join(verdict_delete[layer]) if verdict_delete[layer] else "KEEP",
        })
    summary_df = pd.DataFrame(summary_rows)

    kelly_row = {
        "Layer": "Refined_Kelly_f*",
        "phi_G_mean": kelly_f,
        "phi_G_CI_lo": np.nan,
        "phi_G_CI_hi": np.nan,
        "phi_ES_mean": np.nan,
        "phi_ES_CI_lo": np.nan,
        "phi_ES_CI_hi": np.nan,
        "stability_pct": np.nan,
        "phi_G_Crisis": np.nan,
        "phi_G_Crisis_CI97.5": np.nan,
        "phi_G_Bull": np.nan,
        "phi_G_LowLiq": np.nan,
        "phi_G_LowLiq_CI97.5": np.nan,
        "ergodic_penalty": False,
        "redundant_with": "—",
        "VERDICT_DELETE": "—",
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([kelly_row])], ignore_index=True)

    print("\n" + "=" * 110)
    print(f"SHAPLEY-ABLATION: CI at {100*(1-MULTIPLE_TEST_ALPHA):.0f}% (Bonferroni n={n_layers})")
    print("=" * 110)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    print(summary_df.to_string(index=False))
    print("\n--- Final Verdict (DELETION if any flag) ---")
    for layer in LAYERS:
        flags = verdict_delete[layer]
        status = "DELETE" if flags else "KEEP"
        print(f"  {layer}: {status}" + (f" — {'; '.join(flags)}" if flags else ""))
    print(f"\nRefined Fractional Kelly f* (point-estimate; use rolling in backtest) = {kelly_f:.4f}")

    out_path = _PROJECT_ROOT / "research" / "shapley_results.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved summary to {out_path}")

    boot_df = pd.DataFrame({
        **{f"phi_G_{k}": bootstrap_phi_G[k] for k in LAYERS},
        **{f"phi_ES_{k}": bootstrap_phi_ES[k] for k in LAYERS},
    })
    boot_path = _PROJECT_ROOT / "research" / "bootstrap_results.parquet"
    boot_df.to_parquet(boot_path, index=False)
    print(f"Saved bootstrap distributions to {boot_path}")

    return summary_df


# --- Leave-One-Out Marginal Contribution ---
LOO_COMPONENT_NAMES = ["Insider", "Meta_Labeling", "HMM_Regime", "HRP_Clustering"]
MARGINAL_SHARPE_PCT_THRESHOLD = 5.0  # Falsification: <5% Sharpe contribution → disable


def run_loo_marginal_contribution() -> pd.DataFrame:
    """
    Leave-One-Out backtest: isolate effect of Insider, Meta_Labeling, HMM_Regime, HRP_Clustering.
    Compute marginal delta in Sharpe, Max DD, turnover. Apply falsification rule.
    """
    print("LOO Marginal Contribution: Loading data...")
    try:
        nlp_full = NLPSentinel()
    except Exception:
        print("WARNING: NLPSentinel init failed. Using NeutralNLP.")
        nlp_full = _NeutralNLP()

    full_enabled = frozenset(LOO_COMPONENT_NAMES)
    alpha_full, price_returns, macro_regime_df, news_df, raw_prices, rebalance_dates = (
        setup_research_data_with_components(full_enabled)
    )

    bt_full = _run_backtest_loo(
        alpha_full, price_returns, macro_regime_df, news_df, rebalance_dates,
        full_enabled, nlp_full,
    )
    ret_full = _strategy_returns(bt_full, price_returns)
    equity_full = compute_equity_curve(bt_full, price_returns)
    met_full = compute_metrics(equity_full)
    sharpe_full = met_full["sharpe"] if not np.isnan(met_full["sharpe"]) else 0.0
    max_dd_full = met_full["max_dd"] if not np.isnan(met_full["max_dd"]) else 0.0
    turnover_full = _compute_turnover(bt_full)

    configs: Dict[str, Dict[str, float]] = {
        "Full": {
            "sharpe": sharpe_full,
            "max_dd": max_dd_full,
            "turnover": turnover_full,
        }
    }

    for excluded in LOO_COMPONENT_NAMES:
        enabled = full_enabled - {excluded}
        alpha_df, _, _, _, _, _ = setup_research_data_with_components(enabled)
        bt = _run_backtest_loo(
            alpha_df, price_returns, macro_regime_df, news_df, rebalance_dates,
            enabled, nlp_full,
        )
        ret = _strategy_returns(bt, price_returns)
        equity = compute_equity_curve(bt, price_returns)
        met = compute_metrics(equity)
        sharpe = met["sharpe"] if not np.isnan(met["sharpe"]) else 0.0
        max_dd = met["max_dd"] if not np.isnan(met["max_dd"]) else 0.0
        turnover = _compute_turnover(bt)
        configs[f"Without_{excluded}"] = {"sharpe": sharpe, "max_dd": max_dd, "turnover": turnover}

    rows = []
    disabled_recommend: List[str] = []
    for comp in LOO_COMPONENT_NAMES:
        key = f"Without_{comp}"
        s_full = configs["Full"]["sharpe"]
        s_wo = configs[key]["sharpe"]
        dd_full = configs["Full"]["max_dd"]
        dd_wo = configs[key]["max_dd"]
        to_full = configs["Full"]["turnover"]
        to_wo = configs[key]["turnover"]

        delta_sharpe = s_full - s_wo
        delta_dd = dd_full - dd_wo
        delta_turnover = turnover_full - configs[key]["turnover"]

        sharpe_pct = 100.0 * (delta_sharpe / abs(s_full)) if abs(s_full) > 1e-12 else 0.0
        cannibalizes = s_wo > s_full

        verdict = "KEEP"
        if sharpe_pct < MARGINAL_SHARPE_PCT_THRESHOLD and sharpe_pct >= 0:
            verdict = "DISABLE (<5% Sharpe contribution)"
            disabled_recommend.append(comp)
        elif cannibalizes:
            verdict = "DISABLE (cannibalizes alpha)"
            disabled_recommend.append(comp)

        rows.append({
            "Component": comp,
            "Sharpe_Full": s_full,
            "Sharpe_Without": s_wo,
            "Delta_Sharpe": delta_sharpe,
            "Sharpe_Contribution_Pct": sharpe_pct,
            "MaxDD_Full": dd_full,
            "MaxDD_Without": dd_wo,
            "Delta_MaxDD": delta_dd,
            "Turnover_Full": to_full,
            "Turnover_Without": to_wo,
            "Delta_Turnover": delta_turnover,
            "Cannibalizes": cannibalizes,
            "VERDICT": verdict,
        })

    report_df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("LOO MARGINAL CONTRIBUTION: Signal Interaction Evaluation")
    print("=" * 100)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(report_df.to_string(index=False))
    print("\n--- Falsification Rule: Components to programmatically disable ---")
    if disabled_recommend:
        for c in disabled_recommend:
            print(f"  {c}: DISABLE")
    else:
        print("  None (all components pass)")
    print(f"\nThreshold: marginal Sharpe contribution < {MARGINAL_SHARPE_PCT_THRESHOLD}% → disable")
    print("Cannibalization: Sharpe(without X) > Sharpe(full) → disable X")

    out_path = _PROJECT_ROOT / "research" / "loo_marginal_contribution.csv"
    report_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    return report_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--loo", action="store_true", help="Run LOO marginal contribution (default: Shapley ablation)")
    args = parser.parse_args()
    if args.loo:
        run_loo_marginal_contribution()
    else:
        run_shapley_ablation()
