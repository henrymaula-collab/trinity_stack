"""
Ablation Study: Systematic layer removal to prove marginal benefit.
Each run adds one layer. Output: Sharpe, CAGR, Max DD, Omega Ratio per configuration.
Does not modify src/ or run_pipeline.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from research._shared import (
    setup_research_data,
    compute_equity_curve,
    compute_metrics,
)
from src.engine.backtest_loop import BacktestEngine
from src.layer4_nlp.xlm_roberta_sentinel import NLPSentinel
from src.layer5_portfolio.hrp_clustering import HierarchicalRiskParity
from src.layer5_portfolio.dynamic_vol_target import DynamicVolTargeting


# ---- Stub implementations (research-only, for ablation) ----

from research._shared import _NeutralNLP


class _NaiveEqualWeight:
    """Naive allocation: equal weight per ticker."""

    def allocate(
        self, returns_df: pd.DataFrame, currency_series: pd.Series
    ) -> pd.Series:
        n = len(returns_df.columns)
        return pd.Series(1.0 / n, index=returns_df.columns)


def _neutral_regime_df(macro_regime_df: pd.DataFrame) -> pd.DataFrame:
    """Force regime=0 (bull) for all dates."""
    out = macro_regime_df.copy()
    out["regime"] = 0
    return out


# ---- Ablation runs ----

CONFIGS = [
    {
        "name": "Baseline (Momentum + Naive EW)",
        "hrp": _NaiveEqualWeight(),
        "nlp": _NeutralNLP(),
        "macro_df_key": "neutral",
    },
    {
        "name": "+ HRP",
        "hrp": HierarchicalRiskParity(),
        "nlp": _NeutralNLP(),
        "macro_df_key": "neutral",
    },
    {
        "name": "+ Macro Regime",
        "hrp": HierarchicalRiskParity(),
        "nlp": _NeutralNLP(),
        "macro_df_key": "real",
    },
    {
        "name": "+ NLP (Full Stack)",
        "hrp": HierarchicalRiskParity(),
        "nlp": None,  # use real NLPSentinel
        "macro_df_key": "real",
    },
]


def run_ablation() -> pd.DataFrame:
    """Run ablation study and return metrics table."""
    print("Ablation Study: Loading data and building alpha...")
    alpha_df, price_returns, macro_regime_df, news_df, _, rebalance_dates = (
        setup_research_data()
    )

    macro_neutral = _neutral_regime_df(macro_regime_df)
    try:
        nlp_real = NLPSentinel()
    except Exception:
        print("WARNING: NLPSentinel init failed (transformers). Using NeutralNLP for Run 4.")
        nlp_real = _NeutralNLP()

    results = []

    for cfg in CONFIGS:
        nlp = cfg["nlp"] if cfg["nlp"] is not None else nlp_real
        macro = macro_neutral if cfg["macro_df_key"] == "neutral" else macro_regime_df

        engine = BacktestEngine(
            alpha_model=None,
            nlp_sentinel=nlp,
            hrp_model=cfg["hrp"],
            vol_targeter=DynamicVolTargeting(),
        )
        bt = engine.run_backtest(
            price_df=price_returns,
            alpha_df=alpha_df,
            macro_df=macro,
            news_df=news_df,
            rebalance_dates=rebalance_dates,
            currency_series=None,
        )
        equity = compute_equity_curve(bt, price_returns)
        m = compute_metrics(equity)
        results.append({
            "config": cfg["name"],
            "sharpe": m["sharpe"],
            "cagr": m["cagr"],
            "max_dd": m["max_dd"],
            "Omega_Ratio": m["omega_ratio"],
        })
        print(f"  {cfg['name']}: Sharpe={m['sharpe']:.3f}, CAGR={m['cagr']:.2%}, MaxDD={m['max_dd']:.2%}, Omega={m['omega_ratio']:.3f}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = run_ablation()
    print("\n" + "=" * 60)
    print("Ablation Results")
    print("=" * 60)
    print(df.to_string(index=False))
    out_path = _PROJECT_ROOT / "research" / "ablation_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
