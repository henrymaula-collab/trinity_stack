"""
Layer 3: Alpha signal generators.
Cross-Sectional Momentum and LightGBM models for AlphaEnsemble.
Point-in-time safe; no look-ahead bias.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

# Reproducibility (Trinity Stack v9.0)
GLOBAL_SEED = 42

MOM_WINDOW = 126
MOM_SKIP_DAYS = 21

LGBM_FEATURE_CANDIDATES: List[str] = [
    "Quality_Score",
    "SUE",
    "Amihud",
    "Vol_Compression",
    "is_january_prep",
    "dist_to_sma200",
]


def _rank_normalize(series: pd.Series, grouper: pd.Series) -> pd.Series:
    """Cross-sectional rank per group, normalized to [0, 1]. NaN preserved."""
    ranks = series.groupby(grouper).rank(method="average", na_option="keep")
    counts = series.groupby(grouper).transform("count")
    out = (ranks - 1) / (counts - 1).replace(0, np.nan)
    return out.clip(0, 1)


class MomentumGenerator:
    """
    Cross-sectional 6-month momentum with short-term reversal exclusion.
    126 trading days, excluding most recent 21 days.
    """

    def __init__(
        self,
        window_days: int = MOM_WINDOW,
        skip_days: int = MOM_SKIP_DAYS,
    ) -> None:
        self.window_days = window_days
        self.skip_days = skip_days

    def generate(self, price_df: pd.DataFrame) -> pd.Series:
        """
        Compute 6-month momentum, rank cross-sectionally, normalize to [0, 1].

        Args:
            price_df: Long format with columns [date, ticker] and a price column.
                      Use 'PX_LAST' or 'price' (first found). Index alignment preserved.

        Returns:
            Series aligned with (date, ticker) containing signal_mom in [0, 1].
        """
        price_col = "PX_LAST" if "PX_LAST" in price_df.columns else "price"
        if price_col not in price_df.columns:
            raise ValueError("price_df must contain 'PX_LAST' or 'price'")
        if "date" not in price_df.columns or "ticker" not in price_df.columns:
            raise ValueError("price_df must contain 'date' and 'ticker'")

        df = price_df[["date", "ticker", price_col]].copy()
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        # Momentum: (price[t-21] / price[t-126-21]) - 1
        # Excludes last 21 days; uses 126-day lookback before that
        start_shift = self.window_days + self.skip_days  # 147
        end_shift = self.skip_days  # 21

        price_start = df.groupby("ticker")[price_col].shift(start_shift)
        price_end = df.groupby("ticker")[price_col].shift(end_shift)
        mom = (price_end / price_start) - 1
        mom = mom.replace([np.inf, -np.inf], np.nan)

        df["mom_raw"] = mom
        df = df.dropna(subset=["mom_raw"])

        if df.empty:
            return pd.Series(dtype=float)

        signal = _rank_normalize(df["mom_raw"], df["date"])
        out = signal.copy()
        out.index = pd.MultiIndex.from_frame(df[["date", "ticker"]])
        return out


PURGE_DAYS_DEFAULT = 5
FORWARD_RETURN_HORIZON_DEFAULT = 1  # Layer 1 uses shift(-1) = 1-day forward


class LightGBMGenerator:
    """
    LightGBM regressor for forward return prediction.
    Expanding window walk-forward; purge_days must be >= forward_return_horizon.
    """

    def __init__(
        self,
        random_state: int = GLOBAL_SEED,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        forward_return_horizon: int = FORWARD_RETURN_HORIZON_DEFAULT,
        purge_days: Optional[int] = None,
    ) -> None:
        self.forward_return_horizon = forward_return_horizon
        self.purge_days = (
            purge_days
            if purge_days is not None
            else max(forward_return_horizon, PURGE_DAYS_DEFAULT)
        )
        if self.purge_days < forward_return_horizon:
            raise ValueError(
                f"purge_days ({self.purge_days}) must be >= forward_return_horizon ({forward_return_horizon}) "
                "to prevent target overlap with test period"
            )
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._model: Optional[object] = None
        self._train_date_max: Optional[pd.Timestamp] = None
        self._models_by_date: dict = {}

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        available = [c for c in LGBM_FEATURE_CANDIDATES if c in df.columns]
        if not available:
            raise ValueError(
                f"features_df must contain at least one of {LGBM_FEATURE_CANDIDATES}"
            )
        return available

    def train_walk_forward(
        self,
        features_df: pd.DataFrame,
        target_col: str = "forward_return",
    ) -> None:
        """
        Expanding window walk-forward with purge period.
        For each pred_date: train on data with date < (pred_date - purge_days).
        Karantänperioden eliminerar läckage från rapporteringsfördröjningar.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required for LightGBMGenerator")

        feat_cols = self._get_feature_cols(features_df)
        if target_col not in features_df.columns:
            raise ValueError(f"features_df must contain '{target_col}'")
        if "date" not in features_df.columns or "ticker" not in features_df.columns:
            raise ValueError("features_df must contain 'date' and 'ticker'")

        df = features_df.dropna(
            subset=feat_cols + [target_col, "date", "ticker"]
        ).copy()
        df["date"] = pd.to_datetime(df["date"])
        dates = sorted(df["date"].unique())

        self._models_by_date.clear()
        cutoff_delta = pd.offsets.BDay(self.purge_days)

        for i, pred_date in enumerate(dates):
            train_cutoff = pd.Timestamp(pred_date) - cutoff_delta
            train_mask = df["date"] < train_cutoff
            train_df = df.loc[train_mask]
            if len(train_df) < 10:
                continue

            X_train = train_df[feat_cols]
            y_train = train_df[target_col]

            model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                verbose=-1,
                force_col_wise=True,
            )
            model.fit(X_train, y_train)
            self._models_by_date[pred_date] = (model, feat_cols)

        self._train_date_max = max(self._models_by_date.keys()) if self._models_by_date else None

    def predict(self, features_df: pd.DataFrame) -> pd.Series:
        """
        Predict using stored walk-forward models.
        For each date, use the latest model trained on data before that date.
        Output ranked cross-sectionally per date, normalized to [0, 1].
        """
        if not self._models_by_date:
            raise ValueError("Call train_walk_forward before predict")

        feat_cols = self._get_feature_cols(features_df)
        df = features_df.dropna(subset=feat_cols + ["date", "ticker"]).copy()
        df["date"] = pd.to_datetime(df["date"])
        dates_sorted = sorted(self._models_by_date.keys())

        preds = np.full(len(df), np.nan)
        for i, (idx, row) in enumerate(df.iterrows()):
            d = row["date"]
            model_date = None
            for md in dates_sorted:
                if md <= d:
                    model_date = md
                else:
                    break
            if model_date is None:
                continue
            model, cols = self._models_by_date[model_date]
            if not all(c in row.index for c in cols):
                continue
            x = row[cols].values.reshape(1, -1)
            preds[i] = float(model.predict(x)[0])

        df["pred_raw"] = preds
        df = df.dropna(subset=["pred_raw"])
        if df.empty:
            return pd.Series(dtype=float)

        signal = _rank_normalize(df["pred_raw"], df["date"])
        out = signal.copy()
        out.index = pd.MultiIndex.from_frame(df[["date", "ticker"]])
        return out
