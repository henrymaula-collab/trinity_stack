"""
Vectorized Backtest for Supply-Chain Lead-Lag Strategy.

Loads mock data (schema: date, supplier_ticker, customer_ticker, revenue_dependency_pct),
runs signal engine, executes trades, and computes geometric return and hit rate.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy_supply_chain.data_pipeline import (
    REQUIRED_SCHEMA,
    build_unified_dataset,
    load_prices_from_parquet,
    load_supply_chain,
)
from strategy_supply_chain.execution_logic import build_trades
from strategy_supply_chain.signal_engine import generate_signals

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def generate_mock_supply_chain(
    start: str = "2020-01-01",
    end: str = "2023-12-31",
    n_links: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Create mock supply-chain dataset for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    dates = dates[dates.dayofweek < 5][:n_links * 2]
    dates = rng.choice(dates, size=min(n_links, len(dates)), replace=False)
    dates = pd.to_datetime(sorted(dates))
    suppliers = [f"SUPP{i}" for i in rng.integers(1, 15, size=len(dates))]
    customers = [f"CUST{j}" for j in rng.integers(1, 10, size=len(dates))]
    revenue_pct = rng.uniform(5, 80, size=len(dates))
    return pd.DataFrame({
        "date": dates,
        "supplier_ticker": suppliers,
        "customer_ticker": customers,
        "revenue_dependency_pct": revenue_pct,
    })


def generate_mock_prices(
    tickers: list[str],
    start: str = "2020-01-01",
    end: str = "2023-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Create mock daily prices with random walk returns."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="B")
    rows = []
    for t in tickers:
        log_ret = rng.standard_normal(len(dates)) * 0.01
        px = 100 * np.exp(np.cumsum(log_ret))
        for i, d in enumerate(dates):
            rows.append({"date": d, "ticker": t, "PX_LAST": px[i]})
    df = pd.DataFrame(rows)
    df["Return"] = df.groupby("ticker")["PX_LAST"].pct_change().groupby(df["ticker"]).shift(1)
    return df


def run_backtest(
    supply_chain: pd.DataFrame,
    prices: pd.DataFrame,
    earnings: pd.DataFrame | None = None,
    sigma: float = 2.0,
) -> dict:
    """
    Run full pipeline: unified dataset -> signals -> trades -> metrics.

    Returns:
        Dict with keys: geometric_return, hit_rate, n_trades, trade_returns
    """
    unified = build_unified_dataset(supply_chain, prices, earnings)
    signals = generate_signals(unified, sigma=sigma)
    trades = build_trades(signals, prices, earnings)

    if trades.empty:
        return {"geometric_return": 0.0, "hit_rate": 0.0, "n_trades": 0, "trade_returns": []}

    # Compute per-trade returns (entry->exit)
    px = prices.pivot(index="date", columns="ticker", values="PX_LAST")
    px.index = pd.to_datetime(px.index)
    trade_returns = []
    for _, row in trades.iterrows():
        supp = row["supplier_ticker"]
        if supp not in px.columns:
            trade_returns.append(np.nan)
            continue
        entry_px = px.loc[px.index >= row["entry_date"], supp]
        exit_px = px.loc[px.index <= row["exit_date"], supp]
        if entry_px.empty or exit_px.empty:
            trade_returns.append(np.nan)
            continue
        p0 = entry_px.iloc[0]
        p1 = exit_px.iloc[-1]
        if p0 != 0:
            ret = (p1 - p0) / p0
        else:
            ret = np.nan
        trade_returns.append(ret)
    trade_returns = np.array([r for r in trade_returns if not np.isnan(r)])

    if len(trade_returns) == 0:
        return {"geometric_return": 0.0, "hit_rate": 0.0, "n_trades": len(trades), "trade_returns": []}

    n = len(trade_returns)
    hit_rate = (trade_returns > 0).sum() / n
    gross_return = np.prod(1 + trade_returns)
    geometric_return = gross_return ** (1 / n) - 1 if n > 0 else 0.0

    return {
        "geometric_return": float(geometric_return),
        "hit_rate": float(hit_rate),
        "n_trades": n,
        "trade_returns": trade_returns.tolist(),
    }


def main() -> None:
    """Self-contained backtest with mock data."""
    logging.info("Generating mock supply-chain and prices...")
    supply_chain = generate_mock_supply_chain(n_links=80)
    tickers = list(set(supply_chain["supplier_ticker"].tolist() + supply_chain["customer_ticker"].tolist()))
    prices = generate_mock_prices(tickers)
    earnings = pd.DataFrame(columns=["ticker", "earnings_date"])

    logging.info("Running backtest...")
    result = run_backtest(supply_chain, prices, earnings, sigma=1.5)

    print("\n--- Supply-Chain Lead-Lag Backtest Results ---")
    print(f"  Geometric return (per trade): {result['geometric_return']:.4f}")
    print(f"  Hit rate:                     {result['hit_rate']:.2%}")
    print(f"  Number of trades:             {result['n_trades']}")


if __name__ == "__main__":
    main()
