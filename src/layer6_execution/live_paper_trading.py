"""
Layer 6: Live Paper Trading & Implementation Shortfall Tracker.

Bridges Trinity from historical simulation to production measurement.
- Broker connector interface (Interactive Brokers / Infront pluggable)
- Micro-lot execution (1 share per signal for validation)
- implementation_shortfall.db: theoretical arrival (VWAP/PX_LAST) vs realized fill
- Hard production rule: if rolling 30d realized slippage > simulated pessimistic,
  pause strategy from capital scaling.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Protocol, Tuple

import pandas as pd

DEFAULT_IMPL_SHORTFALL_DB: str = "data/tca/implementation_shortfall.db"
ROLLING_DAYS: int = 30
CAPITAL_SCALING_PAUSE_THRESHOLD: float = 1.0  # realized/simulated > 1 => pause


@dataclass
class OrderRequest:
    """Micro-lot order request (1 share per signal for validation)."""
    ticker: str
    side: str  # BUY | SELL
    quantity: int
    order_type: str  # MARKET | LIMIT
    limit_price: Optional[float] = None


@dataclass
class FillReport:
    """Result of order execution."""
    ticker: str
    side: str
    quantity: int
    filled_price: float
    timestamp: pd.Timestamp
    order_id: Optional[str] = None


class BrokerConnector(Protocol):
    """Protocol for broker API (IB, Infront, etc.)."""

    def submit_order(self, req: OrderRequest) -> str:
        """Submit order; return order_id."""
        ...

    def get_fill(self, order_id: str) -> Optional[FillReport]:
        """Get fill report if filled."""
        ...

    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        ...


class PaperBroker:
    """
    Paper broker: no real execution.
    Simulates fill at theoretical_price for micro-lot validation.
    """

    def __init__(self, fill_at_theoretical: bool = True) -> None:
        self._fill_at_theoretical = fill_at_theoretical
        self._orders: Dict[str, Tuple[OrderRequest, Optional[float]]] = {}

    def submit_order(
        self,
        req: OrderRequest,
        theoretical_price: Optional[float] = None,
    ) -> str:
        order_id = f"PAPER_{pd.Timestamp.now().value}"
        self._orders[order_id] = (req, theoretical_price)
        return order_id

    def get_fill(
        self,
        order_id: str,
        realized_price_override: Optional[float] = None,
    ) -> Optional[FillReport]:
        if order_id not in self._orders:
            return None
        req, theoretical = self._orders[order_id]
        px = realized_price_override if realized_price_override is not None else (theoretical or 0.0)
        if px <= 0:
            return None
        return FillReport(
            ticker=req.ticker,
            side=req.side,
            quantity=req.quantity,
            filled_price=px,
            timestamp=pd.Timestamp.now(),
            order_id=order_id,
        )

    def cancel_order(self, order_id: str) -> bool:
        return order_id in self._orders


def _ensure_impl_shortfall_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS impl_shortfall_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            side TEXT NOT NULL,
            theoretical_arrival_price REAL NOT NULL,
            filled_price REAL NOT NULL,
            quantity INTEGER NOT NULL,
            realized_slippage_bps REAL NOT NULL,
            simulated_pessimistic_bps REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_impl_shortfall_ts ON impl_shortfall_log(timestamp)
    """)
    conn.commit()


def log_implementation_shortfall(
    ticker: str,
    side: str,
    theoretical_arrival_price: float,
    filled_price: float,
    quantity: int,
    simulated_pessimistic_bps: float,
    timestamp: Optional[pd.Timestamp] = None,
    db_path: str = DEFAULT_IMPL_SHORTFALL_DB,
) -> None:
    """
    Log implementation shortfall: theoretical (VWAP/PX_LAST) vs realized fill.
    realized_slippage_bps = |filled - theoretical| / theoretical * 10000
    """
    if theoretical_arrival_price <= 0 or filled_price <= 0:
        raise ValueError("Prices must be positive")
    realized_bps = abs(filled_price - theoretical_arrival_price) / theoretical_arrival_price * 10000.0
    ts = (timestamp or pd.Timestamp.now()).isoformat()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        _ensure_impl_shortfall_schema(conn)
        conn.execute(
            "INSERT INTO impl_shortfall_log "
            "(timestamp, ticker, side, theoretical_arrival_price, filled_price, quantity, "
            "realized_slippage_bps, simulated_pessimistic_bps) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ts, ticker, side.upper(), theoretical_arrival_price, filled_price, quantity,
                realized_bps, simulated_pessimistic_bps,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def read_implementation_shortfall(db_path: str = DEFAULT_IMPL_SHORTFALL_DB) -> pd.DataFrame:
    """Read full implementation shortfall log."""
    if not Path(db_path).exists():
        return pd.DataFrame(columns=[
            "timestamp", "ticker", "side", "theoretical_arrival_price", "filled_price",
            "quantity", "realized_slippage_bps", "simulated_pessimistic_bps",
        ])
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM impl_shortfall_log", conn)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    finally:
        conn.close()


def rolling_slippage_comparison(
    db_path: str = DEFAULT_IMPL_SHORTFALL_DB,
    window_days: int = ROLLING_DAYS,
) -> Tuple[float, float, bool]:
    """
    Rolling 30d realized vs simulated pessimistic slippage.

    Returns:
        (avg_realized_bps, avg_simulated_bps, should_pause_capital_scaling)
    """
    df = read_implementation_shortfall(db_path)
    if df.empty or len(df) < 5:
        return 0.0, 0.0, False
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=window_days)
    recent = df[df["timestamp"] >= cutoff]
    if recent.empty:
        return 0.0, 0.0, False
    avg_realized = float(recent["realized_slippage_bps"].mean())
    avg_simulated = float(recent["simulated_pessimistic_bps"].mean())
    if avg_simulated <= 0:
        should_pause = avg_realized > 0
    else:
        ratio = avg_realized / avg_simulated
        should_pause = ratio > CAPITAL_SCALING_PAUSE_THRESHOLD
    return avg_realized, avg_simulated, should_pause


def is_capital_scaling_paused(db_path: str = DEFAULT_IMPL_SHORTFALL_DB) -> bool:
    """
    Hard production rule: if rolling 30d realized > simulated pessimistic,
    pause strategy from capital scaling.
    """
    _, _, should_pause = rolling_slippage_comparison(db_path)
    return should_pause


class LivePaperTrader:
    """
    Orchestrates micro-lot paper trading and implementation shortfall tracking.
    """

    def __init__(
        self,
        broker: Optional[BrokerConnector] = None,
        db_path: str = DEFAULT_IMPL_SHORTFALL_DB,
        micro_lot_size: int = 1,
    ) -> None:
        self._broker = broker or PaperBroker()
        self._db_path = db_path
        self._micro_lot_size = micro_lot_size

    def execute_micro_lot(
        self,
        ticker: str,
        side: str,
        theoretical_price: float,
        simulated_pessimistic_bps: float,
        limit_price: Optional[float] = None,
    ) -> Optional[FillReport]:
        """
        Submit micro-lot (1 share) order. Log implementation shortfall on fill.
        """
        req = OrderRequest(
            ticker=ticker,
            side=side.upper(),
            quantity=self._micro_lot_size,
            order_type="LIMIT" if limit_price else "MARKET",
            limit_price=limit_price,
        )
        if isinstance(self._broker, PaperBroker):
            order_id = self._broker.submit_order(req, theoretical_price)
            fill = self._broker.get_fill(order_id)
        else:
            order_id = self._broker.submit_order(req)
            fill = self._broker.get_fill(order_id)
        if fill is not None:
            log_implementation_shortfall(
                ticker=fill.ticker,
                side=fill.side,
                theoretical_arrival_price=theoretical_price,
                filled_price=fill.filled_price,
                quantity=fill.quantity,
                simulated_pessimistic_bps=simulated_pessimistic_bps,
                timestamp=fill.timestamp,
                db_path=self._db_path,
            )
        return fill

    def check_capital_scaling_pause(self) -> bool:
        """Returns True if capital scaling should be paused."""
        return is_capital_scaling_paused(self._db_path)
