"""
Layer 6: TCA (Transaction Cost Analysis) SQLite logger.
Logs theoretical vs filled prices, timestamp, regime label.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_DB_PATH: str = "data/tca/tca.db"


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tca_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            theoretical_price REAL NOT NULL,
            filled_price REAL NOT NULL,
            regime INTEGER NOT NULL,
            target_weight REAL NOT NULL
        )
    """)
    conn.commit()


def log_execution(
    ticker: str,
    theoretical_price: float,
    filled_price: float,
    regime: int,
    target_weight: float,
    timestamp: Optional[pd.Timestamp] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Append one TCA record. Fail on invalid inputs."""
    if theoretical_price <= 0 or filled_price <= 0:
        raise ValueError("Prices must be positive")
    if regime not in (0, 1, 2):
        raise ValueError("regime must be 0, 1, or 2")

    ts = (timestamp or pd.Timestamp.now()).isoformat()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        conn.execute(
            "INSERT INTO tca_log (timestamp, ticker, theoretical_price, filled_price, regime, target_weight) VALUES (?, ?, ?, ?, ?, ?)",
            (ts, ticker, theoretical_price, filled_price, regime, target_weight),
        )
        conn.commit()
    finally:
        conn.close()


def read_tca(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Read full TCA log. Returns empty DataFrame if no records."""
    if not Path(db_path).exists():
        return pd.DataFrame(
            columns=["timestamp", "ticker", "theoretical_price", "filled_price", "regime", "target_weight"]
        )
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT timestamp, ticker, theoretical_price, filled_price, regime, target_weight FROM tca_log", conn)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    finally:
        conn.close()
