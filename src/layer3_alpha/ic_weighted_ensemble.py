from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROLLING_DAYS = 126
QUALITY_BOTTOM_PCT = 0.25
SPREAD_MAX = 0.03
WEIGHT_MIN = 0.3
WEIGHT_MAX = 0.7

REQUIRED_COLS: List[str] = [
    "date",
    "Quality_Score",
    "BID_ASK_SPREAD_PCT",
    "PX_TURN_OVER",
    "signal_lgbm",
    "signal_mom",
    "forward_return",
]


class AlphaEnsemble:
    """
    Layer 3 alpha ensemble with hard filters and IC stability weighting.
    Combines signal_lgbm and signal_mom with weights bounded [0.3, 0.7].
    """

    def __init__(
        self,
        rolling_days: int = ROLLING_DAYS,
        quality_bottom_pct: float = QUALITY_BOTTOM_PCT,
        spread_max: float = SPREAD_MAX,
        weight_min: float = WEIGHT_MIN,
        weight_max: float = WEIGHT_MAX,
    ) -> None:
        self.rolling_days = rolling_days
        self.quality_bottom_pct = quality_bottom_pct
        self.spread_max = spread_max
        self.weight_min = weight_min
        self.weight_max = weight_max

    def _validate_input(self, df: pd.DataFrame) -> None:
        for col in REQUIRED_COLS:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        if df["date"].isna().any():
            raise ValueError("NaNs in date")
        if df["forward_return"].isna().all():
            raise ValueError("forward_return is all NaN")

    def _apply_hard_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.dropna(subset=["Quality_Score", "BID_ASK_SPREAD_PCT", "PX_TURN_OVER"])
        df = df[df["PX_TURN_OVER"] > 0]
        df = df[df["BID_ASK_SPREAD_PCT"] <= self.spread_max]
        by_date = df.groupby("date", as_index=False)
        quantile = by_date["Quality_Score"].transform(
            lambda x: x.quantile(self.quality_bottom_pct)
        )
        df = df[df["Quality_Score"] > quantile]
        return df.reset_index(drop=True)

    def _compute_cross_sectional_ic_series(
        self, df: pd.DataFrame, signal_col: str
    ) -> pd.Series:
        ics = []
        dates = df["date"].unique()
        for d in dates:
            block = df[df["date"] == d]
            sig = block[signal_col].values
            ret = block["forward_return"].values
            valid = ~(np.isnan(sig) | np.isnan(ret))
            if valid.sum() < 3:
                ics.append((d, np.nan))
            else:
                r, _ = spearmanr(sig[valid], ret[valid])
                ics.append((d, r if not np.isnan(r) else np.nan))
        ic_df = pd.DataFrame(ics, columns=["date", "ic"])
        return ic_df.set_index("date")["ic"].sort_index()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies hard filters, computes IC weights with shift(1), and returns
        df with alpha_score column. Raises ValueError on NaN propagation.
        """
        self._validate_input(df)
        filtered = self._apply_hard_filters(df)

        ic_mom = self._compute_cross_sectional_ic_series(filtered, "signal_mom")
        ic_lgbm = self._compute_cross_sectional_ic_series(filtered, "signal_lgbm")

        min_periods = max(2, self.rolling_days // 2)
        rolling_ic_mom = ic_mom.rolling(
            window=self.rolling_days, min_periods=min_periods
        ).mean()
        rolling_ic_lgbm = ic_lgbm.rolling(
            window=self.rolling_days, min_periods=min_periods
        ).mean()
        rolling_std_mom = ic_mom.rolling(
            window=self.rolling_days, min_periods=min_periods
        ).std()
        rolling_std_lgbm = ic_lgbm.rolling(
            window=self.rolling_days, min_periods=min_periods
        ).std()

        ic_mom_shifted = rolling_ic_mom.shift(1).values
        ic_lgbm_shifted = rolling_ic_lgbm.shift(1).values
        std_mom_shifted = rolling_std_mom.shift(1).values
        std_lgbm_shifted = rolling_std_lgbm.shift(1).values

        raw_w_mom = np.where(
            (std_mom_shifted > 1e-10) & np.isfinite(ic_mom_shifted),
            ic_mom_shifted / std_mom_shifted,
            np.nan,
        )
        raw_w_lgbm = np.where(
            (std_lgbm_shifted > 1e-10) & np.isfinite(ic_lgbm_shifted),
            ic_lgbm_shifted / std_lgbm_shifted,
            np.nan,
        )

        total = np.nansum([raw_w_mom, raw_w_lgbm], axis=0)
        total = np.where(total > 0, total, np.nan)
        w_mom = np.where(np.isfinite(total), raw_w_mom / total, np.nan)
        w_lgbm = np.where(np.isfinite(total), raw_w_lgbm / total, np.nan)

        w_mom = np.clip(w_mom, self.weight_min, self.weight_max)
        w_lgbm = np.clip(w_lgbm, self.weight_min, self.weight_max)
        total_clipped = w_mom + w_lgbm
        total_clipped = np.where(total_clipped > 1e-10, total_clipped, np.nan)
        w_mom = np.where(np.isfinite(total_clipped), w_mom / total_clipped, np.nan)
        w_lgbm = np.where(np.isfinite(total_clipped), w_lgbm / total_clipped, np.nan)

        weight_df = pd.DataFrame(
            {
                "date": ic_mom.index,
                "w_mom": w_mom,
                "w_lgbm": w_lgbm,
            }
        )

        merged = filtered.merge(weight_df, on="date", how="left")
        merged["alpha_score"] = (
            merged["w_mom"].values * merged["signal_mom"].values
            + merged["w_lgbm"].values * merged["signal_lgbm"].values
        )

        weights_valid = merged["w_mom"].notna() & merged["w_lgbm"].notna()
        post_burnin_nan = weights_valid & merged["alpha_score"].isna()
        if post_burnin_nan.any():
            raise ValueError("NaN propagated to alpha_score after burn-in")

        return merged
