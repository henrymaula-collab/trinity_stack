"""
Trinity Stack v9.0 — Layer 6 Dashboard.
Streamlit UI for TCA logs and regime visibility. SQLite-backed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.layer6_execution.tca_logger import DEFAULT_DB_PATH, read_tca

st.set_page_config(page_title="Trinity Stack TCA", layout="wide")

st.title("Trinity Stack v9.0 — TCA Dashboard")

db_path = Path(DEFAULT_DB_PATH)
if not db_path.exists():
    st.warning(f"No TCA database at {DEFAULT_DB_PATH}. Run executions to populate.")
    st.stop()

df = read_tca(str(db_path))
if df.empty:
    st.info("TCA log is empty.")
    st.stop()

df["slippage_bps"] = 10_000 * (df["filled_price"] - df["theoretical_price"]) / df["theoretical_price"]
df["regime_label"] = df["regime"].map({0: "Bull", 1: "Neutral", 2: "Crisis"})

st.subheader("TCA Log")
st.dataframe(
    df[["timestamp", "ticker", "theoretical_price", "filled_price", "slippage_bps", "regime_label", "target_weight"]],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Slippage by Regime (bps)")
by_regime = df.groupby("regime_label")["slippage_bps"].agg(["mean", "std", "count"])
st.dataframe(by_regime, use_container_width=True, hide_index=True)
