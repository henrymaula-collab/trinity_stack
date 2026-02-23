"""
Layer 6: TCA (Transaction Cost Analysis) SQLite logger.
Logs theoretical vs filled prices, timestamp, regime label.
Extended schema for fill-probability calibration: order_size_adv, spread_pct,
intraday_vol_proxy, time_to_fill, fill_ratio.
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
    _ensure_fill_columns(conn)


def _ensure_fill_columns(conn: sqlite3.Connection) -> None:
    """Add fill-calibration columns if missing (migration)."""
    info = conn.execute("PRAGMA table_info(tca_log)").fetchall()
    cols = {r[1] for r in info}
    for col, typ in [
        ("order_size_adv", "REAL"),
        ("spread_pct", "REAL"),
        ("intraday_vol_proxy", "REAL"),
        ("time_to_fill_sec", "REAL"),
        ("fill_ratio", "REAL"),
    ]:
        if col not in cols:
            try:
                conn.execute(f"ALTER TABLE tca_log ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError:
                pass
    conn.commit()


def log_execution(
    ticker: str,
    theoretical_price: float,
    filled_price: float,
    regime: int,
    target_weight: float,
    timestamp: Optional[pd.Timestamp] = None,
    db_path: str = DEFAULT_DB_PATH,
    order_size_adv: Optional[float] = None,
    spread_pct: Optional[float] = None,
    intraday_vol_proxy: Optional[float] = None,
    time_to_fill_sec: Optional[float] = None,
    fill_ratio: Optional[float] = None,
) -> None:
    """Append one TCA record. Fail on invalid inputs. Optional fill-calibration fields."""
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
            "INSERT INTO tca_log (timestamp, ticker, theoretical_price, filled_price, regime, target_weight, "
            "order_size_adv, spread_pct, intraday_vol_proxy, time_to_fill_sec, fill_ratio) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ts, ticker, theoretical_price, filled_price, regime, target_weight,
                order_size_adv, spread_pct, intraday_vol_proxy, time_to_fill_sec, fill_ratio,
            ),
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
