"""
Layer 6: Conviction Scaled Limit Pricing and 95% Expected Shortfall stop-loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ALPHA_TOP_PCT: float = 0.05
SPREAD_ILLIQUID_PCT: float = 0.30
CONVICTION_DISCOUNT: float = 0.25
ILLIQUID_DISCOUNT: float = 0.75
NORMAL_DISCOUNT: float = 0.40
ES_QUANTILE: float = 0.05


def _expected_shortfall_95(returns: pd.Series) -> float:
    """95% Expected Shortfall (CVaR): mean of worst 5% of returns."""
    if returns.empty or returns.isna().all():
        return 0.0
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    threshold = clean.quantile(ES_QUANTILE)
    tail = clean[clean <= threshold]
    if len(tail) == 0:
        return 0.0
    return float(tail.mean())


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
        historical_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Produce orders: limit_price (conviction-scaled), stop_loss_price (95% ES).

        Omit assets with zero target weight.
        Raises ValueError on index mismatch or missing data.
        """
        self._validate_inputs(
            target_weights, current_prices, spreads, alpha_scores, historical_returns
        )
        tickers = target_weights[target_weights != 0].index.tolist()
        if not tickers:
            return pd.DataFrame(
                columns=["ticker", "target_weight", "limit_price", "stop_loss_price"]
            )

        common = (
            set(tickers)
            & set(current_prices.index)
            & set(spreads.index)
            & set(alpha_scores.index)
            & set(historical_returns.columns)
        )
        tickers = [t for t in tickers if t in common]
        if not tickers:
            raise ValueError("No overlap between target_weights and price/spread/alpha/returns")

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
            rets = historical_returns[t].dropna()
            es = _expected_shortfall_95(rets)
            stop_loss_price = mid * (1.0 - abs(es))

            rows.append({
                "ticker": t,
                "target_weight": float(target_weights[t]),
                "limit_price": limit_price,
                "stop_loss_price": stop_loss_price,
            })

        return pd.DataFrame(rows)

    def _validate_inputs(
        self,
        target_weights: pd.Series,
        current_prices: pd.Series,
        spreads: pd.Series,
        alpha_scores: pd.Series,
        historical_returns: pd.DataFrame,
    ) -> None:
        if target_weights.empty:
            raise ValueError("target_weights must be non-empty")
        if current_prices.empty or current_prices.isna().any():
            raise ValueError("current_prices must be non-empty and free of NaNs")
        if spreads.empty or (spreads < 0).any():
            raise ValueError("spreads must be non-empty and non-negative")
        if alpha_scores.empty or alpha_scores.isna().any():
            raise ValueError("alpha_scores must be non-empty and free of NaNs")
        if historical_returns.empty:
            raise ValueError("historical_returns must be non-empty")
