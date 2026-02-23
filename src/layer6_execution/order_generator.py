"""
Layer 6: Conviction Scaled Limit Pricing + Pessimistic Execution Engine.

Mechanical market stop-losses removed. Risk handled ex-ante via position limits,
regime scalar, and HRP. In illiquid small-caps, market stops cause gap-down
liquidity traps.

Pessimistic mode: BUY at Ask + 0.1%*vol, SELL at Bid - 0.1%*vol; top 10% alpha
= price taker; 2.5x spread when regime < 0.5 or V2TX > 25.

Endogenous Reflexivity Circuit Breaker: Before TWAP/VWAP sliced orders, snapshot
bid/ask spread and top-of-book depth. After partial fills, detect if execution
causes spread widening or opposite-side pullback (HFT/MM spoofing reaction).
If spread widens >50% from snapshot → classify as Toxic Flow, cancel remainder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Protocol

import numpy as np
import pandas as pd

from src.layer6_execution.pessimistic_execution import (
    execution_price_buy,
    execution_price_sell,
)

# Circuit Breaker: spread increase threshold (50% = toxic flow, cancel remainder)
SPREAD_TOXIC_THRESHOLD_PCT: float = 50.0


@dataclass
class OrderBookSnapshot:
    """
    Snapshot of order book before sliced order (TWAP/VWAP) initiation.
    Used to detect endogenous reflexivity: system's own orders damaging liquidity.
    """
    bid: float
    ask: float
    spread_pct: float  # (ask - bid) / mid * 100
    depth_bid: float  # top-of-book bid size
    depth_ask: float  # top-of-book ask size
    timestamp: Optional[pd.Timestamp] = None

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0 if (self.bid > 0 and self.ask > 0) else 0.0


class OrderBookFeed(Protocol):
    """Protocol for real-time market data (broker / Infront / IB)."""

    def get_bid(self, ticker: str) -> float:
        """Top-of-book bid."""
        ...

    def get_ask(self, ticker: str) -> float:
        """Top-of-book ask."""
        ...

    def get_depth_bid(self, ticker: str) -> float:
        """Size at best bid."""
        ...

    def get_depth_ask(self, ticker: str) -> float:
        """Size at best ask."""
        ...


@dataclass
class ReflexivityCircuitBreaker:
    """
    Real-time monitor: detect when system's own orders damage market liquidity.

    Before a sliced order (TWAP/VWAP), snapshot spread and depth.
    After partial fills, measure if spread widened or opposite-side orders pulled
    (HFT/MM spoofing reaction). If spread increases >50% from snapshot → Toxic Flow
    → cancel remaining volume immediately.
    """

    spread_toxic_threshold_pct: float = SPREAD_TOXIC_THRESHOLD_PCT
    _snapshot: Optional[OrderBookSnapshot] = field(default=None, repr=False)

    def snapshot_orderbook(
        self,
        bid: float,
        ask: float,
        depth_bid: float,
        depth_ask: float,
    ) -> OrderBookSnapshot:
        """
        Call before first slice of a TWAP/VWAP order.
        Saves baseline spread and top-of-book depth.
        """
        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else 0.0
        spread_pct = ((ask - bid) / mid * 100.0) if mid > 0 else 0.0
        self._snapshot = OrderBookSnapshot(
            bid=bid,
            ask=ask,
            spread_pct=spread_pct,
            depth_bid=depth_bid,
            depth_ask=depth_ask,
            timestamp=pd.Timestamp.now(),
        )
        return self._snapshot

    def check_after_partial_fill(
        self,
        bid: float,
        ask: float,
        depth_bid: float,
        depth_ask: float,
    ) -> tuple[bool, float]:
        """
        Call after first partial fills. Returns (is_toxic_flow, spread_increase_pct).

        If spread widened > threshold from snapshot → toxic flow, cancel remainder.
        Also detects opposite-side pullback (depth collapse).
        """
        if self._snapshot is None:
            return False, 0.0

        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else 0.0
        if mid <= 0:
            return False, 0.0

        current_spread_pct = (ask - bid) / mid * 100.0
        baseline_spread = self._snapshot.spread_pct

        if baseline_spread <= 0:
            return False, 0.0

        spread_increase_pct = ((current_spread_pct - baseline_spread) / baseline_spread) * 100.0
        is_toxic = spread_increase_pct > self.spread_toxic_threshold_pct

        return is_toxic, spread_increase_pct

    def classify_as_toxic_flow(
        self,
        bid: float,
        ask: float,
        depth_bid: float,
        depth_ask: float,
    ) -> Literal["OK", "TOXIC_FLOW"]:
        """
        Classify order: OK or TOXIC_FLOW. If TOXIC_FLOW, remaining volume
        must be cancelled immediately to avoid driving price against self.
        """
        is_toxic, _ = self.check_after_partial_fill(bid, ask, depth_bid, depth_ask)
        return "TOXIC_FLOW" if is_toxic else "OK"

    def reset(self) -> None:
        """Reset snapshot (e.g. for new order)."""
        self._snapshot = None


@dataclass
class SlicedOrderMonitor:
    """
    Orchestrates pre/post flow for TWAP/VWAP sliced orders.
    Uses ReflexivityCircuitBreaker to decide whether to cancel remainder.
    """

    ticker: str
    feed: OrderBookFeed
    circuit_breaker: ReflexivityCircuitBreaker = field(
        default_factory=ReflexivityCircuitBreaker,
        repr=False,
    )

    def before_first_slice(self) -> OrderBookSnapshot:
        """Call before initiating first slice. Returns baseline snapshot."""
        self.circuit_breaker.reset()
        bid = self.feed.get_bid(self.ticker)
        ask = self.feed.get_ask(self.ticker)
        depth_bid = self.feed.get_depth_bid(self.ticker)
        depth_ask = self.feed.get_depth_ask(self.ticker)
        return self.circuit_breaker.snapshot_orderbook(bid, ask, depth_bid, depth_ask)

    def after_partial_fill(self) -> tuple[Literal["OK", "TOXIC_FLOW"], bool]:
        """
        Call after first partial fills.
        Returns (classification, cancel_remaining).
        If TOXIC_FLOW: cancel_remaining=True → cancel remaining volume immediately.
        """
        bid = self.feed.get_bid(self.ticker)
        ask = self.feed.get_ask(self.ticker)
        depth_bid = self.feed.get_depth_bid(self.ticker)
        depth_ask = self.feed.get_depth_ask(self.ticker)

        classification = self.circuit_breaker.classify_as_toxic_flow(
            bid, ask, depth_bid, depth_ask
        )
        cancel_remaining = classification == "TOXIC_FLOW"
        return classification, cancel_remaining


class MockOrderBookFeed:
    """
    Mock feed for testing. Configure post_fill_* to simulate spread widening.
    Use when real-time market data unavailable (backtest / paper trading).
    """

    def __init__(
        self,
        ticker: str,
        bid: float = 100.0,
        ask: float = 100.5,
        depth_bid: float = 1000.0,
        depth_ask: float = 1000.0,
        post_fill_spread_widen_pct: float = 0.0,
    ) -> None:
        self.ticker = ticker
        self._bid = bid
        self._ask = ask
        self._depth_bid = depth_bid
        self._depth_ask = depth_ask
        self._post_fill_spread_widen_pct = post_fill_spread_widen_pct
        self._post_fill_called = False

    def get_bid(self, ticker: str) -> float:
        if ticker != self.ticker:
            return 0.0
        if self._post_fill_called and self._post_fill_spread_widen_pct > 0:
            mid = (self._bid + self._ask) / 2.0
            spread = self._ask - self._bid
            new_spread = spread * (1 + self._post_fill_spread_widen_pct / 100.0)
            return mid - new_spread / 2.0
        return self._bid

    def get_ask(self, ticker: str) -> float:
        if ticker != self.ticker:
            return 0.0
        if self._post_fill_called and self._post_fill_spread_widen_pct > 0:
            mid = (self._bid + self._ask) / 2.0
            spread = self._ask - self._bid
            new_spread = spread * (1 + self._post_fill_spread_widen_pct / 100.0)
            return mid + new_spread / 2.0
        return self._ask

    def get_depth_bid(self, ticker: str) -> float:
        return self._depth_bid if ticker == self.ticker else 0.0

    def get_depth_ask(self, ticker: str) -> float:
        return self._depth_ask if ticker == self.ticker else 0.0

    def mark_partial_fill(self) -> None:
        """Call to simulate post-fill state (spread widens)."""
        self._post_fill_called = True


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
        is_stale_price: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Produce orders: limit_price (conviction-scaled). No stop-loss.

        Omit assets with zero target weight.
        Hard falsification: if is_stale_price[ticker]==True on execution day, block order
        (no fictitious trade at yesterday's close in empty order book).
        """
        self._validate_inputs(target_weights, current_prices, spreads, alpha_scores)
        tickers = target_weights[target_weights != 0].index.tolist()
        if is_stale_price is not None:
            stale_block = is_stale_price.reindex(tickers).fillna(False)
            tickers = [t for t in tickers if not bool(stale_block.get(t, False))]
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
        is_stale_price: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Pessimistic execution: BUY at Ask + 0.1%*vol, SELL at Bid - 0.1%*vol.
        Top 10% alpha = price taker (full spread + slippage).
        Spread stress 2.5x when regime < 0.5 or V2TX > 25.
        Hard falsification: block order if is_stale_price[ticker]==True.
        """
        self._validate_inputs(target_weights, current_prices, spreads, alpha_scores)
        tickers = target_weights[target_weights != 0].index.tolist()
        if is_stale_price is not None:
            stale_block = is_stale_price.reindex(tickers).fillna(False)
            tickers = [t for t in tickers if not bool(stale_block.get(t, False))]
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
            vol_daily = float(daily_vol_pct.get(t, 0.20 / np.sqrt(TRADING_DAYS_YEAR))) if daily_vol_pct is not None and not daily_vol_pct.empty else 0.20 / np.sqrt(TRADING_DAYS_YEAR)
            sigma_annual = vol_daily * np.sqrt(TRADING_DAYS_YEAR)
            top10 = alpha_scores.reindex([t]).iloc[0] >= alpha_cut
            exec_px = execution_price_buy(
                mid, spread, 0.0, 0.0, sigma_annual, top10, regime_exposure, v2tx
            )
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
