"""
Research shared utilities. Imports production code; does not modify src/ or run_pipeline.py.
Provides data loading and backtest harness for ablation, market impact, and robustness studies.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Duplicate data loading (no run_pipeline import to avoid NLPSentinel/transformers)
GLOBAL_SEED = 42
DATA_RAW = _PROJECT_ROOT / "data" / "raw"


def _load_parquet_or_mock(name: str) -> pd.DataFrame:
    path = DATA_RAW / f"{name}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        if df.empty:
            raise ValueError(f"Parquet {path} is empty")
        return df
    np.random.seed(GLOBAL_SEED)
    n_days, n_tickers = 400, 12
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")
    if name == "prices":
        records = []
        for t in range(n_tickers):
            base_px = 100.0
            for d in dates:
                ret = np.random.randn() * 0.02
                base_px = max(1.0, base_px * (1 + ret))
                records.append({
                    "date": pd.Timestamp(d), "ticker": f"T_{t:02d}",
                    "PX_LAST": base_px, "PX_VOLUME": max(100, int(np.random.lognormal(10, 1))),
                    "BID_ASK_SPREAD_PCT": np.random.uniform(0.005, 0.04),
                })
        return pd.DataFrame(records)
    if name == "fundamentals":
        report_dates = pd.date_range("2021-03-01", periods=20, freq="2MS")
        records = []
        for rd in report_dates:
            for t in range(n_tickers):
                actual = np.random.randn() * 0.5 + 1.0
                records.append({
                    "report_date": pd.Timestamp(rd), "ticker": f"T_{t:02d}",
                    "ACTUAL_EPS": actual, "CONSENSUS_EPS": actual + np.random.randn() * 0.2,
                    "EPS_STD": max(0.1, np.abs(np.random.randn()) * 0.3),
                    "ROIC": np.random.uniform(0.02, 0.25),
                    "Piotroski_F": int(np.clip(np.random.randint(0, 9), 0, 9)),
                    "Accruals": np.random.randn() * 0.05,
                })
        return pd.DataFrame(records)
    if name == "macro":
        v2tx = 15 + np.cumsum(np.random.randn(n_days) * 2)
        breadth = 0.5 + np.cumsum(np.random.randn(n_days) * 0.1)
        rates = 0.01 + np.cumsum(np.random.randn(n_days) * 0.001)
        return pd.DataFrame({"V2TX": v2tx, "Breadth": breadth, "Rates": rates}, index=dates)
    if name == "news":
        records = []
        for _ in range(50):
            records.append({
                "ticker": f"T_{np.random.randint(0, n_tickers):02d}",
                "date": np.random.choice(dates),
                "text": "Neutral update.",
            })
        return pd.DataFrame(records)
    raise ValueError(f"Unknown data name: {name}")


def _research_load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    prices = _load_parquet_or_mock("prices")
    fundamentals = _load_parquet_or_mock("fundamentals")
    macro = _load_parquet_or_mock("macro")
    news = _load_parquet_or_mock("news")
    prices = prices.sort_values(["date", "ticker"]).reset_index(drop=True)
    fundamentals = fundamentals.sort_values(["report_date", "ticker"]).reset_index(drop=True)
    return prices, fundamentals, macro, news


def _build_price_returns_wide(raw_prices: pd.DataFrame) -> pd.DataFrame:
    px = raw_prices.copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["ticker", "date"])
    px["ret"] = px.groupby("ticker")["PX_LAST"].pct_change()
    wide = px.pivot_table(index="date", columns="ticker", values="ret")
    wide = wide.dropna(how="all").dropna(how="any")
    if wide.empty:
        raise ValueError("price_df (returns) is empty after dropping NaNs")
    return wide


def _build_macro_regime_series(macro_df: pd.DataFrame, detector) -> pd.DataFrame:
    macro_df = macro_df.sort_index()
    regimes = []
    for i in range(len(macro_df)):
        r = detector.predict_regime(macro_df.iloc[: i + 1])
        regimes.append(r)
    return pd.DataFrame({"regime": regimes}, index=macro_df.index)
from src.layer1_data.feature_engineering import FeatureEngineer
from src.layer2_regime.hmm_macro_student_t import MacroRegimeDetector
from src.layer3_alpha.signal_generators import LightGBMGenerator, MomentumGenerator
from src.layer3_alpha.ic_weighted_ensemble import AlphaEnsemble
from src.layer5_portfolio.hrp_clustering import HierarchicalRiskParity
from src.layer5_portfolio.dynamic_vol_target import DynamicVolTargeting
from src.engine.backtest_loop import BacktestEngine


class _NeutralNLP:
    """NLP bypass for research: get_risk_multiplier always returns 1.0."""

    def get_risk_multiplier(
        self, text: str | float | None, days_since_event: int = 0
    ) -> float:
        return 1.0


def setup_research_data() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    List[pd.Timestamp],
]:
    """
    Load data and build alpha_df, price_returns, macro_regime_df, news_df, raw_prices.
    Returns (alpha_df, price_returns, macro_regime_df, news_df, raw_prices, rebalance_dates).
    """
    raw_prices, raw_fundamentals, macro_df, news_df = _research_load_data()
    feature_engineer = FeatureEngineer()
    macro_detector = MacroRegimeDetector(random_state=GLOBAL_SEED)
    macro_detector.fit(macro_df)
    macro_regime_df = _build_macro_regime_series(macro_df, macro_detector)

    features_df = feature_engineer.build_features(raw_prices, raw_fundamentals)
    momentum_gen = MomentumGenerator()
    lgbm_gen = LightGBMGenerator()
    alpha_ensemble = AlphaEnsemble()

    signal_mom = momentum_gen.generate(raw_prices)
    lgbm_gen.train_walk_forward(features_df, target_col="forward_return")
    signal_lgbm = lgbm_gen.predict(features_df)

    merged = features_df.drop(columns=["signal_mom", "signal_lgbm"], errors="ignore").merge(
        signal_mom.rename("signal_mom"),
        left_on=["date", "ticker"],
        right_index=True,
        how="left",
    )
    merged = merged.merge(
        signal_lgbm.rename("signal_lgbm"),
        left_on=["date", "ticker"],
        right_index=True,
        how="left",
    )
    merged["signal_mom"] = merged["signal_mom"].fillna(0.5)
    merged["signal_lgbm"] = merged["signal_lgbm"].fillna(0.5)
    alpha_df = alpha_ensemble.fit_transform(merged)

    price_returns = _build_price_returns_wide(raw_prices)
    rebalance_dates = pd.date_range(
        price_returns.index.min() + pd.Timedelta(days=300),
        price_returns.index.max(),
        freq="BME",
    ).tolist()

    return alpha_df, price_returns, macro_regime_df, news_df, raw_prices, rebalance_dates


def compute_equity_curve(
    backtest_result: pd.DataFrame,
    price_df: pd.DataFrame,
) -> pd.Series:
    """Compute daily equity curve from weight records and price returns."""
    if backtest_result.empty:
        return pd.Series(dtype=float)

    weight_df = backtest_result.copy()
    weight_df["date"] = pd.to_datetime(weight_df["date"])
    dates = sorted(weight_df["date"].unique())
    price_df = price_df.sort_index()

    equity = 1.0
    peak = 1.0
    equity_series: List[Tuple[pd.Timestamp, float]] = []

    for i, rb_date in enumerate(dates):
        w_row = weight_df[weight_df["date"] == rb_date]
        prev_weights: Dict[str, float] = dict(
            zip(w_row["ticker"], w_row["target_weight"])
        )
        start = rb_date
        end = dates[i + 1] if i + 1 < len(dates) else price_df.index.max() + pd.Timedelta(days=1)
        mask = (price_df.index > start) & (price_df.index < end)
        period = price_df.loc[mask]
        tickers = [t for t in prev_weights if t in period.columns]
        w_vec = np.array([prev_weights[t] for t in tickers])
        for dt, row in period[tickers].iterrows():
            r = row.values
            if np.any(np.isnan(r)):
                continue
            daily_ret = float(np.dot(w_vec, r))
            equity *= 1.0 + daily_ret
            peak = max(peak, equity)
            equity_series.append((dt, equity))

    if not equity_series:
        return pd.Series(dtype=float)
    return pd.Series(
        [e for _, e in equity_series],
        index=pd.DatetimeIndex([d for d, _ in equity_series]),
    ).sort_index()


def calculate_omega_ratio(returns: pd.Series, mar: float = 0.0) -> float:
    """
    Omega Ratio: sum of positive excess returns / |sum of negative excess returns|.
    MAR=0.0 => 0% daily threshold (equivalent to 0% annualized).
    """
    excess = returns - mar
    pos_sum = excess[excess > 0].sum()
    neg_sum = excess[excess < 0].sum()
    if neg_sum == 0 or np.isclose(abs(neg_sum), 0.0):
        return 999.0 if pos_sum > 0 else np.nan
    return float(pos_sum / abs(neg_sum))


def compute_metrics(equity: pd.Series) -> Dict[str, float]:
    """Sharpe, CAGR, Max Drawdown, Omega Ratio from equity curve."""
    if equity.empty or len(equity) < 2:
        return {"sharpe": np.nan, "cagr": np.nan, "max_dd": np.nan, "omega_ratio": np.nan}

    ret = equity.pct_change().dropna()
    if ret.std() < 1e-12:
        return {"sharpe": np.nan, "cagr": np.nan, "max_dd": np.nan, "omega_ratio": np.nan}

    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years < 1e-6:
        years = 1.0

    sharpe = (ret.mean() / ret.std()) * np.sqrt(252)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = float(drawdown.min())
    omega_ratio = calculate_omega_ratio(ret, mar=0.0)

    return {"sharpe": float(sharpe), "cagr": float(cagr), "max_dd": max_dd, "omega_ratio": omega_ratio}
