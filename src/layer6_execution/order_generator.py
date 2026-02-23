"""
Layer 6: Conviction Scaled Limit Pricing + Pessimistic Execution Engine.

Mechanical market stop-losses removed. Risk handled ex-ante via position limits,
regime scalar, and HRP. In illiquid small-caps, market stops cause gap-down
liquidity traps.

Pessimistic mode: BUY at Ask + 0.1%*vol, SELL at Bid - 0.1%*vol; top 10% alpha
= price taker; 2.5x spread when regime < 0.5 or V2TX > 25.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.layer6_execution.pessimistic_execution import (
    execution_price_buy,
    execution_price_sell,
)

ALPHA_TOP_PCT: float = 0.05
ALPHA_TOP_PCT_PESSIMISTIC: float = 0.10
SPREAD_ILLIQUID_PCT: float = 0.30
CONVICTION_DISCOUNT: float = 0.25
ILLIQUID_DISCOUNT: float = 0.75
NORMAL_DISCOUNT: float = 0.40
TRADING_DAYS_YEAR: int = 252


class ExecutionEngine:
    """
    Generate limit and stop-loss orders from target weights.
    Conviction-scaled limit pricing; 95% ES for resting stop-losses.
    """

    def generate_orders(
        self,
        target_weights: pd.Series,
        current_prices: pd.Series,
        spreads: pd.Series,
        alpha_scores: pd.Series,
    ) -> pd.DataFrame:
        """
        Produce orders: limit_price (conviction-scaled). No stop-loss.

        Omit assets with zero target weight.
        Raises ValueError on index mismatch or missing data.
        """
        self._validate_inputs(target_weights, current_prices, spreads, alpha_scores)
        tickers = target_weights[target_weights != 0].index.tolist()
        if not tickers:
            return pd.DataFrame(columns=["ticker", "target_weight", "limit_price"])

        common = (
            set(tickers)
            & set(current_prices.index)
            & set(spreads.index)
            & set(alpha_scores.index)
        )
        tickers = [t for t in tickers if t in common]
        if not tickers:
            raise ValueError("No overlap between target_weights and price/spread/alpha")

        alpha_cut = alpha_scores.reindex(tickers).quantile(1 - ALPHA_TOP_PCT)
        spread_cut = spreads.reindex(tickers).quantile(1 - SPREAD_ILLIQUID_PCT)

        rows = []
        for t in tickers:
            mid = float(current_prices[t])
            spread = float(spreads[t])
            alpha = alpha_scores.reindex([t]).iloc[0]
            spr = spreads.reindex([t]).iloc[0]

            if alpha >= alpha_cut:
                discount = CONVICTION_DISCOUNT
            elif spr >= spread_cut:
                discount = ILLIQUID_DISCOUNT
            else:
                discount = NORMAL_DISCOUNT

            limit_price = mid - (discount * spread)

            rows.append({
                "ticker": t,
                "target_weight": float(target_weights[t]),
                "limit_price": limit_price,
            })

        return pd.DataFrame(rows)

    def generate_orders_pessimistic(
        self,
        target_weights: pd.Series,
        current_prices: pd.Series,
        spreads: pd.Series,
        alpha_scores: pd.Series,
        daily_vol_pct: Optional[pd.Series] = None,
        regime_exposure: float = 0.5,
        v2tx: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Pessimistic execution: BUY at Ask + 0.1%*vol, SELL at Bid - 0.1%*vol.
        Top 10% alpha = price taker (full spread + slippage).
        Spread stress 2.5x when regime < 0.5 or V2TX > 25.
        """
        self._validate_inputs(target_weights, current_prices, spreads, alpha_scores)
        tickers = target_weights[target_weights != 0].index.tolist()
        if not tickers:
            return pd.DataFrame(columns=["ticker", "target_weight", "limit_price", "execution_price_pessimistic"])

        common = set(tickers) & set(current_prices.index) & set(spreads.index) & set(alpha_scores.index)
        tickers = [t for t in tickers if t in common]
        if not tickers:
            raise ValueError("No overlap between target_weights and price/spread/alpha")

        alpha_cut = alpha_scores.reindex(tickers).quantile(1 - ALPHA_TOP_PCT_PESSIMISTIC)
        rows = []
        for t in tickers:
            mid = float(current_prices[t])
            spread = float(spreads[t])
            vol = float(daily_vol_pct.get(t, 0.20 / np.sqrt(TRADING_DAYS_YEAR))) if daily_vol_pct is not None and not daily_vol_pct.empty else 0.20 / np.sqrt(TRADING_DAYS_YEAR)
            top10 = alpha_scores.reindex([t]).iloc[0] >= alpha_cut
            exec_px = execution_price_buy(mid, spread, vol, top10, regime_exposure, v2tx)
            rows.append({
                "ticker": t,
                "target_weight": float(target_weights[t]),
                "limit_price": exec_px,
                "execution_price_pessimistic": exec_px,
            })
        return pd.DataFrame(rows)

    def _validate_inputs(
        self,
        target_weights: pd.Series,
        current_prices: pd.Series,
        spreads: pd.Series,
        alpha_scores: pd.Series,
    ) -> None:
        if target_weights.empty:
            raise ValueError("target_weights must be non-empty")
        if current_prices.empty or current_prices.isna().any():
            raise ValueError("current_prices must be non-empty and free of NaNs")
        if spreads.empty or (spreads < 0).any():
            raise ValueError("spreads must be non-empty and non-negative")
        if alpha_scores.empty or alpha_scores.isna().any():
            raise ValueError("alpha_scores must be non-empty and free of NaNs")
