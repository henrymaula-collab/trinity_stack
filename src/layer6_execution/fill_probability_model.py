"""
Layer 6: Empirical Fill Probability Model.

Replaces static assumptions (0.85/0.4) with logistic model:
  fill_prob = sigmoid(a - b*(order_size/ADV) - c*spread_pct - d*intraday_vol_proxy)

Calibrated from historical TCA. Uses prior when insufficient data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.layer6_execution.tca_logger import read_tca, DEFAULT_DB_PATH


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid; clip for numerical stability."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class FillProbabilityModel:
    """
    Predict fill probability from order/ liquidity features.
    fill_prob = sigmoid(a - b*size_adv - c*spread_pct - d*intraday_vol)
    """

    def __init__(
        self,
        a: float = 2.0,
        b: float = 3.0,
        c: float = 50.0,
        d: float = 20.0,
    ) -> None:
        """
        Prior coefficients. Positive b,c,d reduce fill prob.
        a: intercept; b: order_size/ADV; c: spread_pct; d: intraday_vol.
        """
        self._a = a
        self._b = b
        self._c = c
        self._d = d
        self._calibrated = False

    def predict(
        self,
        order_size_adv: np.ndarray | float,
        spread_pct: np.ndarray | float,
        intraday_vol_proxy: np.ndarray | float = 0.01,
    ) -> np.ndarray | float:
        """
        fill_prob = sigmoid(a - b*order_size_adv - c*spread_pct - d*intraday_vol).
        """
        x = (
            self._a
            - self._b * np.atleast_1d(order_size_adv).astype(float)
            - self._c * np.atleast_1d(spread_pct).astype(float)
            - self._d * np.atleast_1d(intraday_vol_proxy).astype(float)
        )
        out = sigmoid(x)
        if np.isscalar(order_size_adv) and np.isscalar(spread_pct):
            return float(out[0])
        return out

    def fit(
        self,
        order_size_adv: np.ndarray,
        spread_pct: np.ndarray,
        intraday_vol_proxy: np.ndarray,
        fill_ratio: np.ndarray,
    ) -> "FillProbabilityModel":
        """
        Calibrate via logistic regression (scipy.optimize or sklearn).
        fill_ratio: 0..1 observed fill.
        """
        from scipy.optimize import minimize

        X = np.column_stack([
            np.ones(len(order_size_adv)),
            order_size_adv,
            spread_pct,
            intraday_vol_proxy,
        ])
        y = np.clip(fill_ratio, 1e-6, 1 - 1e-6)

        def neg_log_likelihood(coef: np.ndarray) -> float:
            logit = X @ coef
            p = sigmoid(logit)
            return -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

        # Linear predictor: a - b*x1 - c*x2 - d*x3 = X @ [a, -b, -c, -d]
        res = minimize(
            neg_log_likelihood,
            x0=np.array([self._a, -self._b, -self._c, -self._d]),
            method="L-BFGS-B",
        )
        if res.success and len(res.x) >= 4:
            self._a = float(res.x[0])
            self._b = -float(res.x[1])
            self._c = -float(res.x[2])
            self._d = -float(res.x[3])
            self._calibrated = True
        return self

    def calibrate_from_tca(self, db_path: str = DEFAULT_DB_PATH) -> "FillProbabilityModel":
        """Fit from extended TCA log if available."""
        df = _read_tca_extended(db_path)
        if df is None or len(df) < 30:
            return self
        valid = (
            df["order_size_adv"].notna()
            & df["spread_pct"].notna()
            & (df["order_size_adv"] > 0)
            & (df["fill_ratio"] >= 0)
            & (df["fill_ratio"] <= 1)
        )
        sub = df.loc[valid]
        if len(sub) < 30:
            return self
        return self.fit(
            order_size_adv=sub["order_size_adv"].values,
            spread_pct=sub["spread_pct"].values,
            intraday_vol_proxy=sub["intraday_vol_proxy"].fillna(0.01).values,
            fill_ratio=sub["fill_ratio"].values,
        )


def _read_tca_extended(db_path: str) -> Optional[pd.DataFrame]:
    """Read TCA with fill-calibration columns if schema exists."""
    if not Path(db_path).exists():
        return None
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        info = pd.read_sql_query("PRAGMA table_info(tca_log)", conn)
        cols = set(info["name"].tolist())
        if "order_size_adv" not in cols or "spread_pct" not in cols or "fill_ratio" not in cols:
            return None
        df = pd.read_sql_query("SELECT * FROM tca_log", conn)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception:
        return None
    finally:
        conn.close()
