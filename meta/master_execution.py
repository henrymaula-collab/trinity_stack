"""
Master Allocator — Orchestrates Trinity, Supply-Chain, and Debt Wall.

Fetches:
  - Trinity: Target Weights (core portfolio)
  - Supply-Chain: Tactical Target Weights (short-horizon bets)
  - Debt Wall: Blacklist (tickers to exclude / veto)

Merges weights, zeros out blacklisted tickers, outputs single order file for Nordnet.

Kill-switch rules (when live metrics provided):
  - Edge collapse: rolling_60d_hit_rate < 0.45 → disable strategy
  - Liquidity stress: median_spread_zscore > 2 → reduce gross 50%
  - Model deviation: live_slippage > 2 * backtest_slippage → halt new entries
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Literal

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

TACTICAL_PER_TICKER: float = 0.04  # 4% per active supply-chain ticker
TACTICAL_CAP: float = 0.20  # Max 20% tactical allocation

# Kill-switch thresholds
KILL_EDGE_HIT_RATE_THRESHOLD: float = 0.45
KILL_SPREAD_ZSCORE_THRESHOLD: float = 2.0
KILL_SLIPPAGE_MULTIPLIER: float = 2.0
KILL_GROSS_REDUCTION: float = 0.50


def _compute_tactical_share(supply_chain_weights: pd.Series) -> float:
    """Dynamic tactical share: 4% per active ticker, capped at 20%."""
    active = supply_chain_weights[supply_chain_weights > 0]
    if active.empty:
        return 0.0
    return min(TACTICAL_CAP, len(active) * TACTICAL_PER_TICKER)


def merge_weights(
    trinity_weights: pd.Series,
    supply_chain_weights: pd.Series,
    core_share: float,
    tactical_share: float,
) -> pd.Series:
    """
    Combine core (Trinity) and tactical (Supply-Chain) weights.

    combined = core_share * trinity + tactical_share * supply_chain
    Renormalizes to sum to 1.0. core_share + tactical_share should equal 1.0.
    """
    all_tickers = (
        set(trinity_weights.index) | set(supply_chain_weights.index)
    )
    w_core = trinity_weights.reindex(all_tickers, fill_value=0.0) * core_share
    w_tact = supply_chain_weights.reindex(all_tickers, fill_value=0.0) * tactical_share
    combined = w_core + w_tact
    s = combined.sum()
    if s <= 0:
        return pd.Series(0.0, index=all_tickers)
    return combined / s


def _kill_switch_state(
    rolling_60d_hit_rate: Optional[float],
    median_spread_zscore: Optional[float],
    live_slippage_bps: Optional[float],
    backtest_slippage_bps: Optional[float],
) -> Literal["NORMAL", "REDUCED", "HALT"]:
    """
    NORMAL: no action.
    REDUCED: reduce gross exposure by 50% (liquidity stress).
    HALT: return zero weights (edge collapse or model deviation).
    """
    if rolling_60d_hit_rate is not None and rolling_60d_hit_rate < KILL_EDGE_HIT_RATE_THRESHOLD:
        logging.warning(
            "Kill-switch: edge collapse (hit_rate=%.2f < %.2f). HALT.",
            rolling_60d_hit_rate, KILL_EDGE_HIT_RATE_THRESHOLD,
        )
        return "HALT"
    if live_slippage_bps is not None and backtest_slippage_bps is not None and backtest_slippage_bps > 0:
        if live_slippage_bps > KILL_SLIPPAGE_MULTIPLIER * backtest_slippage_bps:
            logging.warning(
                "Kill-switch: model deviation (live_slippage=%.0f bps > 2× backtest=%.0f). HALT.",
                live_slippage_bps, backtest_slippage_bps,
            )
            return "HALT"
    if median_spread_zscore is not None and median_spread_zscore > KILL_SPREAD_ZSCORE_THRESHOLD:
        logging.warning(
            "Kill-switch: liquidity stress (spread_zscore=%.1f > %.1f). REDUCED.",
            median_spread_zscore, KILL_SPREAD_ZSCORE_THRESHOLD,
        )
        return "REDUCED"
    return "NORMAL"


def apply_blacklist(weights: pd.Series, blacklist: set[str]) -> pd.Series:
    """Zero out blacklisted tickers and renormalize."""
    w = weights.copy()
    for t in blacklist:
        if t in w.index:
            w[t] = 0.0
    s = w.sum()
    if s <= 0:
        return w
    return w / s


def to_nordnet_order_format(
    weights: pd.Series,
    output_path: str | Path,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Write order file suitable for Nordnet import.

    Default columns: ticker, target_weight.
    """
    df = pd.DataFrame({"ticker": weights.index, "target_weight": weights.values})
    df = df[df["target_weight"] > 0].reset_index(drop=True)
    if columns:
        df = df[[c for c in columns if c in df.columns]]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


class MasterAllocator:
    """
    Orchestrates Trinity (Core), Supply-Chain (Tactical), Debt Wall (Veto).

    Tactical share is dynamic: 4% per active supply-chain ticker, capped at 20%.
    core_share = 1.0 - tactical_share.
    """

    def execute(
        self,
        trinity_weights: pd.Series | Callable[[], pd.Series],
        supply_chain_weights: pd.Series | Callable[[], pd.Series],
        debt_wall_blacklist: set[str] | list[str] | Callable[[], set[str] | list[str]],
        output_path: Optional[str | Path] = None,
        rolling_60d_hit_rate: Optional[float] = None,
        median_spread_zscore: Optional[float] = None,
        live_slippage_bps: Optional[float] = None,
        backtest_slippage_bps: Optional[float] = None,
    ) -> pd.Series:
        """
        Fetch weights from all three, merge, apply veto, return final weights.

        Args:
            trinity_weights: Core target weights (ticker -> weight).
            supply_chain_weights: Tactical target weights.
            debt_wall_blacklist: Tickers to exclude (veto).
            output_path: If set, write Nordnet order file to this path.

        Returns:
            Final target weights (blacklist applied, renormalized).
        """
        w_trinity = trinity_weights() if callable(trinity_weights) else trinity_weights
        w_supply = supply_chain_weights() if callable(supply_chain_weights) else supply_chain_weights
        blacklist = debt_wall_blacklist() if callable(debt_wall_blacklist) else debt_wall_blacklist
        blacklist = set(blacklist)

        if w_trinity.empty and w_supply.empty:
            raise ValueError("Both Trinity and Supply-Chain weights are empty")

        state = _kill_switch_state(
            rolling_60d_hit_rate,
            median_spread_zscore,
            live_slippage_bps,
            backtest_slippage_bps,
        )
        if state == "HALT":
            all_tickers = set(w_trinity.index) | set(w_supply.index)
            return pd.Series(0.0, index=list(all_tickers))

        tactical_share = _compute_tactical_share(
            w_supply if not w_supply.empty else pd.Series(dtype=float)
        )
        core_share = 1.0 - tactical_share

        combined = merge_weights(
            w_trinity if not w_trinity.empty else pd.Series(dtype=float),
            w_supply if not w_supply.empty else pd.Series(dtype=float),
            core_share,
            tactical_share,
        )
        final = apply_blacklist(combined, blacklist)
        if state == "REDUCED":
            final = final * (1.0 - KILL_GROSS_REDUCTION)

        blocked = [t for t in blacklist if t in combined.index and combined[t] > 0]
        if blocked:
            logging.info("Debt Wall veto: blocked %d tickers: %s", len(blocked), blocked[:10])

        if output_path:
            orders = to_nordnet_order_format(final, output_path)
            logging.info("Wrote order file to %s (%d positions)", output_path, len(orders))

        return final
