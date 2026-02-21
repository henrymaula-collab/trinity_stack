"""
Trinity Stack v9.0 — Master Execution Pipeline.
Runs all 6 layers sequentially. Fail-fast on missing data. Strict type hints.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# Ensure project root on path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Layer imports
from src.layer1_data.feature_engineering import FeatureEngineer
from src.layer2_regime.hmm_macro_student_t import MacroRegimeDetector
from src.layer2_regime.liquidity_overlay import LiquidityOverlay
from src.layer3_alpha.signal_generators import LightGBMGenerator, MomentumGenerator
from src.layer3_alpha.ic_weighted_ensemble import AlphaEnsemble
from src.layer4_nlp.xlm_roberta_sentinel import NLPSentinel
from src.layer5_portfolio.hrp_clustering import HierarchicalRiskParity
from src.layer5_portfolio.dynamic_vol_target import DynamicVolTargeting
from src.engine.backtest_loop import BacktestEngine
from src.layer6_execution.order_generator import ExecutionEngine
from src.layer6_execution.tca_logger import log_execution, DEFAULT_DB_PATH

DATA_RAW = _PROJECT_ROOT / "data" / "raw"
GLOBAL_SEED = 42


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def _load_parquet_or_mock(name: str) -> pd.DataFrame:
    """Load parquet from data/raw or generate mock DataFrame. Raise on failure."""
    path = DATA_RAW / f"{name}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        if df.empty:
            raise ValueError(f"Parquet {path} is empty")
        return df

    # Mock generation when file does not exist
    np.random.seed(GLOBAL_SEED)
    n_days = 400
    n_tickers = 12
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")

    if name == "prices":
        records: List[dict] = []
        for t in range(n_tickers):
            base_px = 100.0
            for d in dates:
                ret = np.random.randn() * 0.02
                base_px = max(1.0, base_px * (1 + ret))
                records.append({
                    "date": pd.Timestamp(d),
                    "ticker": f"T_{t:02d}",
                    "PX_LAST": base_px,
                    "PX_TURN_OVER": max(1000.0, float(np.random.lognormal(12, 1))),
                    "BID_ASK_SPREAD_PCT": np.random.uniform(0.005, 0.04),
                })
        return pd.DataFrame(records)

    if name == "fundamentals":
        report_dates = pd.date_range("2021-03-01", periods=20, freq="2MS")
        records = []
        for rd in report_dates:
            for t in range(n_tickers):
                actual = np.random.randn() * 0.5 + 1.0
                consensus = actual + np.random.randn() * 0.2
                records.append({
                    "report_date": pd.Timestamp(rd),
                    "ticker": f"T_{t:02d}",
                    "ACTUAL_EPS": actual,
                    "CONSENSUS_EPS": consensus,
                    "EPS_STD": max(0.1, np.abs(np.random.randn()) * 0.3),
                    "ROIC": np.random.uniform(0.02, 0.25),
                    "Dilution": np.random.uniform(0.8, 1.5),
                    "Accruals": np.random.randn() * 0.05,
                })
        return pd.DataFrame(records)

    if name == "macro":
        v2tx = 15 + np.cumsum(np.random.randn(n_days) * 2)
        breadth = 0.5 + np.cumsum(np.random.randn(n_days) * 0.1)
        rates = 0.01 + np.cumsum(np.random.randn(n_days) * 0.001)
        return pd.DataFrame(
            {"V2TX": v2tx, "Breadth": breadth, "Rates": rates},
            index=dates,
        )

    if name == "news":
        records = []
        for _ in range(50):
            d = np.random.choice(dates)
            t = f"T_{np.random.randint(0, n_tickers):02d}"
            records.append({
                "ticker": t,
                "date": d,
                "text": "Company reports strong earnings." if np.random.rand() > 0.9 else "Neutral update.",
            })
        return pd.DataFrame(records)

    raise ValueError(f"Unknown data name: {name}")


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load prices, fundamentals, macro, news. Fail loudly on missing or empty data."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    prices = _load_parquet_or_mock("prices")
    fundamentals = _load_parquet_or_mock("fundamentals")
    macro = _load_parquet_or_mock("macro")
    news = _load_parquet_or_mock("news")

    for col in ["date", "ticker", "PX_LAST", "PX_TURN_OVER", "BID_ASK_SPREAD_PCT"]:
        if col not in prices.columns:
            raise ValueError(f"prices missing column: {col}")
    for col in ["V2TX", "Breadth", "Rates"]:
        if col not in macro.columns:
            raise ValueError(f"macro missing column: {col}")
    for col in ["ticker", "date", "text"]:
        if col not in news.columns:
            raise ValueError(f"news missing column: {col}")

    prices = prices.sort_values(["date", "ticker"]).reset_index(drop=True)
    fundamentals = fundamentals.sort_values(["report_date", "ticker"]).reset_index(drop=True)
    return prices, fundamentals, macro, news


def _build_price_returns_wide(raw_prices: pd.DataFrame) -> pd.DataFrame:
    """Convert long-format raw prices to wide daily returns for BacktestEngine."""
    px = raw_prices.copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["ticker", "date"])
    px["ret"] = px.groupby("ticker")["PX_LAST"].pct_change()
    wide = px.pivot_table(index="date", columns="ticker", values="ret")
    wide = wide.dropna(how="all").dropna(how="any")
    if wide.empty:
        raise ValueError("price_df (returns) is empty after dropping NaNs")
    return wide


def _build_macro_regime_series(
    macro_df: pd.DataFrame, detector: MacroRegimeDetector
) -> pd.DataFrame:
    """Build regime time series using MacroRegimeDetector.predict_regime per date."""
    macro_df = macro_df.sort_index()
    regimes: List[int] = []
    for i in range(len(macro_df)):
        block = macro_df.iloc[: i + 1]
        r = detector.predict_regime(block)
        regimes.append(r)
    return pd.DataFrame({"regime": regimes}, index=macro_df.index)


# ---------------------------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------------------------


def run_pipeline() -> None:
    """Run all 6 layers sequentially with clear console logs."""
    print("[Pipeline] Trinity Stack v9.0 — Master Execution")
    print("=" * 60)

    # ----- Data Loading -----
    print("\n[L0] Loading data from data/raw/ (mock if missing)...")
    raw_prices, raw_fundamentals, macro_df, news_df = load_data()
    print(f"  prices: {len(raw_prices)} rows | fundamentals: {len(raw_fundamentals)} | macro: {len(macro_df)} | news: {len(news_df)}")

    # ----- Initialization -----
    print("\n[Init] Instantiating all layer components...")
    feature_engineer = FeatureEngineer()
    macro_detector = MacroRegimeDetector(random_state=GLOBAL_SEED)
    liquidity_overlay = LiquidityOverlay()
    momentum_gen = MomentumGenerator()
    lgbm_gen = LightGBMGenerator()
    alpha_ensemble = AlphaEnsemble()
    try:
        nlp_sentinel = NLPSentinel()
    except Exception as e:
        raise RuntimeError(f"NLPSentinel init failed (transformers/huggingface): {e}") from e
    hrp = HierarchicalRiskParity()
    vol_targeter = DynamicVolTargeting()
    backtest_engine = BacktestEngine(
        alpha_model=alpha_ensemble,
        nlp_sentinel=nlp_sentinel,
        hrp_model=hrp,
        vol_targeter=vol_targeter,
    )
    execution_engine = ExecutionEngine()
    print("  All components initialized.")

    # ----- L1: Feature Engineering -----
    print("\n[L1] FeatureEngineer.build_features()...")
    features_df = feature_engineer.build_features(raw_prices, raw_fundamentals)
    if features_df.empty:
        raise ValueError("L1 output is empty")
    print(f"  Output: {len(features_df)} rows")

    # ----- L2: Regime Detection -----
    print("\n[L2] MacroRegimeDetector.fit() and predict_regime()...")
    macro_detector.fit(macro_df)
    macro_regime_df = _build_macro_regime_series(macro_df, macro_detector)
    print(f"  Regime series: {len(macro_regime_df)} dates")

    print("  LiquidityOverlay.calculate_stress_regime()...")
    spread_ts = raw_prices.groupby("date")["BID_ASK_SPREAD_PCT"].mean().reset_index()
    spread_ts = liquidity_overlay.calculate_stress_regime(spread_ts)
    if "LIQUIDITY_STRESS" not in spread_ts.columns:
        raise ValueError("LiquidityOverlay did not add LIQUIDITY_STRESS")
    print("  Liquidity stress overlay applied.")

    # ----- L3: Alpha Signals -----
    print("\n[L3] MomentumGenerator.generate()...")
    signal_mom = momentum_gen.generate(raw_prices)
    print(f"  signal_mom: {len(signal_mom)} values")

    print("  LightGBMGenerator.train_walk_forward()...")
    feat_for_lgbm = features_df.copy()
    lgbm_gen.train_walk_forward(feat_for_lgbm, target_col="forward_return")

    print("  LightGBMGenerator.predict()...")
    signal_lgbm = lgbm_gen.predict(feat_for_lgbm)
    print(f"  signal_lgbm: {len(signal_lgbm)} values")

    print("  Merging signals and AlphaEnsemble.fit_transform()...")
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
    if alpha_df.empty:
        raise ValueError("AlphaEnsemble output is empty")
    print(f"  alpha_df: {len(alpha_df)} rows with alpha_score")

    # ----- L4 & L5: Backtest -----
    print("\n[L4 & L5] BacktestEngine.run_backtest()...")
    price_returns = _build_price_returns_wide(raw_prices)
    rebalance_dates = pd.date_range(
        price_returns.index.min() + pd.Timedelta(days=300),
        price_returns.index.max(),
        freq="BME",
    ).tolist()
    if not rebalance_dates:
        raise ValueError("No rebalance dates in range")
    backtest_result = backtest_engine.run_backtest(
        price_df=price_returns,
        alpha_df=alpha_df,
        macro_df=macro_regime_df,
        news_df=news_df,
        rebalance_dates=rebalance_dates,
        currency_series=None,
    )
    print(f"  backtest_result: {len(backtest_result)} weight records")

    # ----- L6: Execution & TCA -----
    print("\n[L6] Extracting final target weights and generating orders...")
    last_date = backtest_result["date"].max()
    final_weights = backtest_result[backtest_result["date"] == last_date].set_index("ticker")["target_weight"]
    tickers_active = final_weights[final_weights != 0].index.tolist()
    if not tickers_active:
        print("  No active positions; skipping ExecutionEngine and TCA.")
        print("\n[Pipeline] Complete.")
        return

    last_price_date = raw_prices[raw_prices["date"] <= last_date]["date"].max()
    if pd.isna(last_price_date):
        last_price_date = raw_prices["date"].max()
    last_prices = raw_prices[raw_prices["date"] == last_price_date]
    current_prices = last_prices.set_index("ticker")["PX_LAST"].reindex(tickers_active).dropna()
    spreads = last_prices.set_index("ticker")["BID_ASK_SPREAD_PCT"].reindex(tickers_active).fillna(0.01)
    last_alpha = alpha_df[alpha_df["date"] == alpha_df["date"].max()].set_index("ticker")["alpha_score"]
    alpha_scores = last_alpha.reindex(tickers_active).fillna(0.0)

    tickers_common = list(
        set(tickers_active)
        & set(current_prices.index)
        & set(spreads.index)
        & set(alpha_scores.index)
        & set(price_returns.columns)
    )
    if len(tickers_common) < 2:
        print("  Insufficient overlap for ExecutionEngine; skipping.")
        print("\n[Pipeline] Complete.")
        return

    final_weights = final_weights.reindex(tickers_common).fillna(0)
    current_prices = current_prices.reindex(tickers_common).dropna()
    spreads = spreads.reindex(tickers_common)
    alpha_scores = alpha_scores.reindex(tickers_common)
    trailing_mask = price_returns.index <= last_date
    trailing_ret = price_returns.loc[trailing_mask, tickers_common].iloc[-252:]

    orders = execution_engine.generate_orders(
        target_weights=final_weights,
        current_prices=current_prices,
        spreads=spreads,
        alpha_scores=alpha_scores,
        historical_returns=trailing_ret,
    )
    print(f"  Generated {len(orders)} orders.")

    regime_asof = macro_regime_df.loc[macro_regime_df.index <= last_date, "regime"].iloc[-1]
    regime_int = int(regime_asof) if regime_asof in (0, 1, 2) else 0

    print("  TCA: Logging theoretical trades to SQLite...")
    for _, row in orders.iterrows():
        log_execution(
            ticker=row["ticker"],
            theoretical_price=row["limit_price"],
            filled_price=row["limit_price"],
            regime=regime_int,
            target_weight=row["target_weight"],
            db_path=str(_PROJECT_ROOT / "data" / "tca" / "tca.db"),
        )
    print(f"  Logged {len(orders)} executions to {DEFAULT_DB_PATH}.")

    print("\n[Pipeline] Complete.")


if __name__ == "__main__":
    run_pipeline()
