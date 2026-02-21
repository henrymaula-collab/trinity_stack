"""
Tearsheet & Benchmarks: Professional visualization for Champion vs Challenger.
Naive 1/N benchmark, equity curves, underwater drawdowns, statistics table.
Follows .cursorrules. Requires: pip install matplotlib seaborn.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Optional: matplotlib/seaborn
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    _PLOTTING_AVAILABLE = True
except ImportError:
    _PLOTTING_AVAILABLE = False

RETURNS_PATH = _PROJECT_ROOT / "research" / "strategy_returns.parquet"
PRICES_PATH = _PROJECT_ROOT / "data" / "raw" / "prices.parquet"
OUTPUT_PDF = _PROJECT_ROOT / "research" / "strategy_tearsheet.pdf"

TRADING_DAYS = 252


def _load_prices() -> pd.DataFrame:
    """Load prices (real or mock)."""
    import research._shared as _sh
    _sh.DATA_RAW.mkdir(parents=True, exist_ok=True)
    return _sh._load_parquet_or_mock("prices")


def build_naive_1n_benchmark(prices: pd.DataFrame) -> pd.Series:
    """
    Naive 1/N benchmark: cross-sectional mean of daily returns per day.
    No ML, momentum, or macro filters. Proves whether Layer 3 adds Alpha.
    """
    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["ticker", "date"])
    px["ret"] = px.groupby("ticker")["PX_LAST"].pct_change()
    wide = px.pivot_table(index="date", columns="ticker", values="ret")
    # Cross-sectional mean per day (equal weight)
    naive = wide.mean(axis=1).dropna()
    return naive


def load_strategy_returns() -> pd.DataFrame:
    """Load Champion and Challenger daily returns from 04 output."""
    if not RETURNS_PATH.exists():
        raise FileNotFoundError(
            f"Strategy returns not found at {RETURNS_PATH}. "
            "Run research/04_champion_vs_challenger.py first."
        )
    df = pd.read_parquet(RETURNS_PATH)
    if df.index.name is None and "date" in df.columns:
        df = df.set_index("date")
    return df


def merge_returns(
    strategy_returns: pd.DataFrame,
    naive_1n: pd.Series,
    index_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Merge Champion, Challenger, 1/N (and optional index) into one DataFrame aligned by date."""
    merged = strategy_returns.copy()
    merged["Naive_1N"] = naive_1n
    if index_returns is not None:
        merged["Index"] = index_returns
    return merged.dropna(how="all").sort_index()


def compute_tearsheet_stats(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Annualized Return, Volatility, Max Drawdown, Omega Ratio per series."""
    rows = []
    for col in returns_df.columns:
        ret = returns_df[col].dropna()
        if ret.empty or len(ret) < 2:
            rows.append({
                "Series": col,
                "Ann_Return": np.nan,
                "Ann_Vol": np.nan,
                "Max_DD": np.nan,
                "Omega_Ratio": np.nan,
            })
            continue
        equity = (1 + ret).cumprod()
        years = (equity.index[-1] - equity.index[0]).days / 365.25
        years = max(years, 1e-6)
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
        ann_vol = ret.std() * np.sqrt(TRADING_DAYS)
        cummax = equity.cummax()
        dd = (equity - cummax) / cummax
        max_dd = float(dd.min())
        excess = ret - 0.0
        pos = excess[excess > 0].sum()
        neg = excess[excess < 0].sum()
        omega = pos / abs(neg) if neg != 0 and not np.isclose(abs(neg), 0) else (999.0 if pos > 0 else np.nan)
        rows.append({
            "Series": col,
            "Ann_Return": cagr,
            "Ann_Vol": ann_vol,
            "Max_DD": max_dd,
            "Omega_Ratio": omega,
        })
    return pd.DataFrame(rows)


def _drawdown_series(equity: pd.Series) -> pd.Series:
    """Underwater plot: (equity - cummax) / cummax."""
    cummax = equity.cummax()
    dd = (equity - cummax) / np.where(cummax > 0, cummax, np.nan)
    return dd.fillna(0)


def plot_tearsheet(merged: pd.DataFrame, output_path: Path) -> None:
    """2-panel tearsheet: cumulative log wealth (top), underwater drawdowns (bottom)."""
    if not _PLOTTING_AVAILABLE:
        print("matplotlib/seaborn not installed. Skipping tearsheet plot.")
        return

    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})

    colors = {"Champion": "#2ecc71", "Challenger": "#3498db", "Naive_1N": "#95a5a6", "Index": "#e74c3c"}
    for col in merged.columns:
        ret = merged[col].dropna()
        if ret.empty:
            continue
        equity = (1 + ret).cumprod()
        log_wealth = np.log(equity)
        c = colors.get(col, None)
        ax1.plot(log_wealth.index, log_wealth.values, label=col, color=c, linewidth=1.5)
        dd = _drawdown_series(equity)
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.5, color=c, label=col)

    ax1.set_ylabel("Cumulative Log Wealth (ln)")
    ax1.set_title("Equity Curves")
    ax1.legend(loc="upper left", frameon=True)
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.set_title("Underwater Plot (Drawdowns)")
    ax2.legend(loc="lower left", frameon=True)
    ax2.set_ylim(bottom=-1.05, top=0.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Tearsheet saved to {output_path}")


def run_tearsheet_and_benchmarks(
    index_returns_path: Path | None = None,
    output_pdf: Path = OUTPUT_PDF,
) -> pd.DataFrame:
    """
    Main entry: build 1/N benchmark, load strategies, merge, plot, print stats.
    index_returns_path: optional path to index daily returns (date index, single column).
    """
    print("Tearsheet & Benchmarks: Loading data...")
    prices = _load_prices()
    naive_1n = build_naive_1n_benchmark(prices)
    strategy_returns = load_strategy_returns()

    index_ret = None
    if index_returns_path is not None and index_returns_path.exists():
        idx_df = pd.read_parquet(index_returns_path) if index_returns_path.suffix == ".parquet" else pd.read_csv(index_returns_path, parse_dates=[0], index_col=0)
        if idx_df.shape[1] >= 1:
            index_ret = idx_df.iloc[:, 0]

    merged = merge_returns(strategy_returns, naive_1n, index_ret)
    stats = compute_tearsheet_stats(merged)

    print("\n" + "=" * 70)
    print("Comparative Statistics (Annualized Return, Volatility, Max DD, Omega)")
    print("=" * 70)
    print(stats.to_string(index=False))

    plot_tearsheet(merged, output_pdf)
    return stats


if __name__ == "__main__":
    run_tearsheet_and_benchmarks()
