"""
Debt Wall Backtest Engine.

Processes mock corporate finance data, executes short positions when deficit
threshold breached, closes on equity issue announcement or maturity date.
Calculates Omega Ratio and CAGR.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy_debt_wall.arbitrage_engine import generate_signals
from strategy_debt_wall.data_pipeline import REQUIRED_SCHEMA, load_debt_wall_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def compute_omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    Omega Ratio = sum(gains above threshold) / sum(losses below threshold).

    Higher = better risk-adjusted performance.
    """
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns < threshold]
    if losses.sum() == 0:
        return np.inf if gains.sum() > 0 else 1.0
    return float(gains.sum() / losses.sum())


def compute_cagr(equity_curve: np.ndarray, years: float) -> float:
    """CAGR from equity curve (start=1, end=last value)."""
    if len(equity_curve) < 2 or years <= 0:
        return 0.0
    total_return = equity_curve[-1] / equity_curve[0]
    return float(total_return ** (1 / years) - 1)


def generate_mock_debt_wall_data(
    start: str = "2020-01-01",
    end: str = "2023-12-31",
    n_rows: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Mock corporate finance data for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="M")
    dates = rng.choice(dates, size=min(n_rows, len(dates)), replace=True)
    dates = pd.to_datetime(sorted(dates))
    tickers = [f"T{i}" for i in rng.integers(1, 25, size=len(dates))]
    market_cap = rng.uniform(50e6, 500e6, size=len(dates))
    cash = rng.uniform(0, 100e6, size=len(dates))
    ttm_fcf = rng.uniform(-30e6, 20e6, size=len(dates))
    next_12m_debt = rng.uniform(0, 150e6, size=len(dates))
    short_interest = rng.uniform(0, 0.15, size=len(dates))
    # Maturity 0–9 months from observation (some within 6m for triggers)
    maturity_delta = pd.to_timedelta(rng.integers(30, 270, size=len(dates)), unit="D")
    maturity_date = dates + maturity_delta
    return pd.DataFrame({
        "date": dates,
        "ticker": tickers,
        "market_cap": market_cap,
        "cash_equivalents": cash,
        "ttm_fcf": ttm_fcf,
        "next_12m_debt_maturity": next_12m_debt,
        "short_interest": short_interest,
        "maturity_date": maturity_date,
    })


def generate_mock_prices(
    tickers: list[str],
    start: str = "2020-01-01",
    end: str = "2023-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Mock daily prices (for short PnL)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="B")
    rows = []
    for t in tickers:
        log_ret = rng.standard_normal(len(dates)) * 0.012
        px = 100 * np.exp(np.cumsum(log_ret))
        for i, d in enumerate(dates):
            rows.append({"date": d, "ticker": t, "PX_LAST": px[i]})
    return pd.DataFrame(rows)


def run_backtest(
    debt_data: pd.DataFrame,
    prices: pd.DataFrame,
    maturity_dates: pd.DataFrame | None = None,
    dilution_threshold: float = 0.10,
) -> dict:
    """
    Run debt wall backtest: signals -> short trades -> PnL -> Omega, CAGR.

    Exit: equity issue announcement (mock: flag) or maturity date passed.
    """
    df = debt_data.copy()
    if maturity_dates is not None and not maturity_dates.empty:
        df = df.merge(maturity_dates[["ticker", "maturity_date"]], on="ticker", how="left")
    maturity_col = "maturity_date" if "maturity_date" in df.columns else None
    signals = generate_signals(df, dilution_threshold=dilution_threshold, maturity_date_col=maturity_col)
    triggers = signals[signals["trigger"]].copy()
    if triggers.empty:
        return {"omega_ratio": 0.0, "cagr": 0.0, "n_trades": 0, "returns": []}

    px = prices.pivot(index="date", columns="ticker", values="PX_LAST")
    px.index = pd.to_datetime(px.index)

    returns_list = []
    for _, row in triggers.iterrows():
        ticker = row["ticker"]
        entry_date = pd.to_datetime(row["date"])
        if ticker not in px.columns:
            continue
        ser = px[ticker].loc[px.index >= entry_date]
        if ser.empty or len(ser) < 2:
            continue
        p0 = ser.iloc[0]
        # Exit: 20 days (mock) or at maturity; for mock we use 20-day hold
        exit_idx = min(20, len(ser) - 1)
        p1 = ser.iloc[exit_idx]
        # Short: profit when price falls
        ret = (p0 - p1) / p0 if p0 != 0 else 0.0
        returns_list.append(ret)

    if not returns_list:
        return {"omega_ratio": 0.0, "cagr": 0.0, "n_trades": len(triggers), "returns": []}

    returns = np.array(returns_list)
    years = (pd.to_datetime(prices["date"]).max() - pd.to_datetime(prices["date"]).min()).days / 365.25
    equity = np.cumprod(1 + returns)
    omega = compute_omega_ratio(returns)
    cagr = compute_cagr(np.concatenate([[1.0], equity]), max(years, 0.01))

    return {
        "omega_ratio": float(omega),
        "cagr": float(cagr),
        "n_trades": len(returns),
        "returns": returns.tolist(),
    }


def main() -> None:
    """Self-contained backtest with mock data."""
    logging.info("Generating mock debt wall data...")
    debt_data = generate_mock_debt_wall_data(n_rows=150)
    tickers = debt_data["ticker"].unique().tolist()
    prices = generate_mock_prices(tickers)
    maturity_dates = None

    logging.info("Running debt wall backtest...")
    result = run_backtest(debt_data, prices, maturity_dates, dilution_threshold=0.10)

    print("\n--- Debt Wall Arbitrage Backtest Results ---")
    print(f"  Omega Ratio:  {result['omega_ratio']:.4f}")
    print(f"  CAGR:         {result['cagr']:.2%}")
    print(f"  Trades:       {result['n_trades']}")


if __name__ == "__main__":
    main()
