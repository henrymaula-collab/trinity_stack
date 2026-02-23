"""
Layer 3: Alpha signal generators.
Cross-Sectional Momentum and LightGBM models for AlphaEnsemble.
Meta-Labeler: regime-neutral, profit-weighted, stratified eval.
Point-in-time safe; no look-ahead bias.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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
    "Dividend_Yield",  # Direktavkastning; kvalitets-/värdefaktor
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


# ---------------------------------------------------------------------------
# Meta-Labeling (Regime-Neutral, Profit-Weighted)
# ---------------------------------------------------------------------------

REGIME_RISK_ON_THRESHOLD = 0.5  # regime >= 0.5 = Risk-On, < 0.5 = Risk-Off


def _meta_walk_forward_splits(
    dates: pd.Series,
    n_folds: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Walk-forward (train_idx, test_idx) by date. Indices into the input series."""
    d = dates.dropna()
    if len(d) < 30:
        return []
    uniq = np.sort(d.unique())
    n = len(uniq)
    if n < 3 or n_folds < 2:
        return []
    min_train = n // 2
    fold_size = max(1, (n - min_train) // n_folds)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    dvals = d.values
    for i in range(n_folds):
        test_start = min_train + i * fold_size
        test_end = min(n, test_start + fold_size)
        if test_start >= test_end:
            continue
        train_end = test_start
        train_mask = (dvals >= uniq[0]) & (dvals < uniq[train_end])
        test_mask = (dvals >= uniq[test_start]) & (dvals < uniq[test_end])
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        if len(train_idx) < 20 or len(test_idx) < 5:
            continue
        splits.append((train_idx, test_idx))
    return splits


def _regime_neutralize(
    df: pd.DataFrame,
    cols: List[str],
    grouper: str = "date",
) -> pd.DataFrame:
    """
    Cross-sectional Z-scores within each grouper (e.g. date).
    Prevents Meta from learning 'don't trade in bear markets'.
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        grp = out.groupby(grouper)[c]
        mean = grp.transform("mean")
        std = grp.transform("std")
        out[f"{c}_z"] = np.where(std > 1e-10, (out[c] - mean) / std, 0.0)
    return out


def _build_meta_target(
    features_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    forward_return_col: str = "forward_return",
) -> Tuple[pd.Series, pd.Series]:
    """
    is_profitable = (forward_return - transaction_cost) > 0.
    Transaction cost = median BID_ASK_SPREAD_PCT from prices (decimal).
    """
    if "BID_ASK_SPREAD_PCT" not in prices_df.columns:
        raise ValueError("prices_df must contain BID_ASK_SPREAD_PCT")
    median_spread_pct = prices_df["BID_ASK_SPREAD_PCT"].median()
    cost_decimal = median_spread_pct / 100.0  # 1.5% -> 0.015

    if forward_return_col not in features_df.columns:
        raise ValueError(f"features_df must contain '{forward_return_col}'")

    net_profit = features_df[forward_return_col] - cost_decimal
    is_profitable = (net_profit > 0).astype(int)
    return is_profitable, net_profit


class MetaLabeler:
    """
    Meta-model: predicts is_profitable (binary) given regime-neutralized features.
    Target horizon must match primary model's forward_return_horizon.
    Profit-weighted loss; stratified evaluation by Risk-On / Risk-Off.
    """

    def __init__(
        self,
        forward_return_horizon: int = FORWARD_RETURN_HORIZON_DEFAULT,
        random_state: int = GLOBAL_SEED,
        n_estimators: int = 80,
        max_depth: int = 4,
        learning_rate: float = 0.08,
    ) -> None:
        self.forward_return_horizon = forward_return_horizon
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._model: Optional[object] = None
        self._meta_feature_cols: List[str] = []

    def _prepare_meta_data(
        self,
        features_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        regime_df: pd.DataFrame,
    ) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        """Prepare X, y, weights and meta_cols for training. Returns (X, y, weights, meta_cols)."""
        is_profitable, net_profit = _build_meta_target(features_df, prices_df)
        df = self._get_meta_features(features_df).copy()
        df["is_profitable"] = is_profitable
        df["net_profit"] = net_profit
        df["date"] = pd.to_datetime(features_df["date"])
        df["ticker"] = features_df["ticker"].values

        meta_cols = [c for c in df.columns if c.endswith("_z")]
        if not meta_cols:
            raise ValueError("No regime-neutralized features available")

        regime_df = regime_df.copy()
        regime_df.index = pd.to_datetime(regime_df.index)
        df = df.merge(
            regime_df[["regime"]],
            left_on="date",
            right_index=True,
            how="left",
        )
        valid = (
            df["is_profitable"].notna()
            & df[meta_cols].notna().all(axis=1)
            & df["regime"].notna()
        )
        df = df.loc[valid].reset_index(drop=True)
        if len(df) < 20:
            return None, None, None, meta_cols

        X = df[meta_cols]
        y = df["is_profitable"].astype(int).values
        weights = np.maximum(1e-6, np.abs(df["net_profit"].values))
        return X, y, weights, meta_cols

    def _check_horizon_match(self, primary_horizon: int) -> None:
        if self.forward_return_horizon != primary_horizon:
            raise ValueError(
                f"Meta target horizon ({self.forward_return_horizon}) must equal "
                f"primary forward_return_horizon ({primary_horizon})"
            )

    def _get_meta_features(
        self,
        df: pd.DataFrame,
        base_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        base = base_cols or [
            c for c in LGBM_FEATURE_CANDIDATES if c in df.columns
        ]
        if "alpha_score" in df.columns:
            base = base + ["alpha_score"]
        elif "signal_lgbm" in df.columns:
            base = base + ["signal_lgbm"]
        base = [c for c in base if c in df.columns]
        return _regime_neutralize(df, base, grouper="date")

    def _train_single(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        weights: np.ndarray,
        n_estimators: int,
        max_depth: int,
        learning_rate: float,
    ) -> object:
        """Train a single LightGBM model with given hyperparams. Returns booster."""
        import lightgbm as lgb

        def _profit_weighted_binary_objective(
            preds: np.ndarray, train_data: "lgb.Dataset"
        ) -> Tuple[np.ndarray, np.ndarray]:
            labels = train_data.get_label()
            w = train_data.get_weight()
            if w is None or len(w) == 0:
                w = np.ones_like(labels)
            p = 1.0 / (1.0 + np.exp(-preds))
            grad = w * (p - labels)
            hess = w * p * (1.0 - p)
            hess = np.clip(hess, 1e-16, None)
            return grad, hess

        train_data = lgb.Dataset(X, label=y, weight=weights)
        params = {
            "objective": "binary",
            "metric": "None",
            "boosting_type": "gbdt",
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "random_state": self.random_state,
            "verbose": -1,
            "force_col_wise": True,
        }
        return lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
            fobj=_profit_weighted_binary_objective,
        )

    def train(
        self,
        features_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        regime_df: pd.DataFrame,
        primary_forward_return_horizon: int = FORWARD_RETURN_HORIZON_DEFAULT,
    ) -> None:
        """
        Train Meta-model on is_profitable (net of median spread).
        Uses regime-neutralized features + profit-weighted sample weights.
        """
        self._check_horizon_match(primary_forward_return_horizon)
        X, y, weights, meta_cols = self._prepare_meta_data(
            features_df, prices_df, regime_df
        )
        if X is None or len(X) < 20:
            return
        self._meta_feature_cols = meta_cols
        self._model = self._train_single(
            X, y, weights,
            self.n_estimators, self.max_depth, self.learning_rate,
        )

    def predict_proba(self, features_df: pd.DataFrame) -> pd.Series:
        """Predict P(profitable). Requires prior train(). Booster returns logits -> sigmoid."""
        if self._model is None:
            raise ValueError("Call train before predict_proba")
        df = self._get_meta_features(features_df)
        meta_cols = [c for c in self._meta_feature_cols if c in df.columns]
        if not meta_cols:
            return pd.Series(np.nan, index=features_df.index)
        X = df[meta_cols].fillna(0)
        logits = self._model.predict(X)
        proba = 1.0 / (1.0 + np.exp(-logits))
        return pd.Series(proba, index=features_df.index)

    def train_nested_cv(
        self,
        features_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        regime_df: pd.DataFrame,
        primary_forward_return_horizon: int = FORWARD_RETURN_HORIZON_DEFAULT,
        n_outer_folds: int = 5,
        n_inner_folds: int = 3,
        param_grid: Optional[Dict[str, List]] = None,
    ) -> Dict[str, object]:
        """
        Nested Cross-Validation with Walk-Forward for Meta hyperparameters.

        Inner loop: tune n_estimators, max_depth, learning_rate on train-only data.
        Outer loop: OOS evaluation. Params selected in inner never see outer test.
        """
        self._check_horizon_match(primary_forward_return_horizon)
        X, y, weights, meta_cols = self._prepare_meta_data(
            features_df, prices_df, regime_df
        )
        if X is None or len(X) < 50:
            logger.warning("MetaLabeler: insufficient data for nested CV; falling back to train()")
            self.train(features_df, prices_df, regime_df, primary_forward_return_horizon)
            return {}

        grid = param_grid or {
            "n_estimators": [40, 80, 120],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.05, 0.08, 0.12],
        }
        df = features_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        dates = df["date"].reindex(X.index).ffill().bfill()
        outer_splits = _meta_walk_forward_splits(dates, n_outer_folds)
        if not outer_splits:
            self.train(features_df, prices_df, regime_df, primary_forward_return_horizon)
            return {}

        outer_aucs: List[float] = []
        best_per_fold: List[Tuple[int, int, float]] = []
        for train_idx, test_idx in outer_splits:
            train_dates = dates.iloc[train_idx].reset_index(drop=True)
            inner_splits = _meta_walk_forward_splits(train_dates, n_inner_folds)
            best_auc = -1.0
            best_params = (self.n_estimators, self.max_depth, self.learning_rate)
            for ne in grid["n_estimators"]:
                for md in grid["max_depth"]:
                    for lr in grid["learning_rate"]:
                        aucs = []
                        for itr, ival in inner_splits:
                            itr_full = train_idx[itr]
                            ival_full = train_idx[ival]
                            Xtr = X.iloc[itr_full]
                            Xva = X.iloc[ival_full]
                            ytr = y[itr_full]
                            yva = y[ival_full]
                            wtr = weights[itr_full]
                            model = self._train_single(
                                Xtr, ytr, wtr, ne, md, lr
                            )
                            pred = 1.0 / (1.0 + np.exp(-model.predict(Xva)))
                            from sklearn.metrics import roc_auc_score
                            if len(np.unique(yva)) < 2:
                                continue
                            auc = roc_auc_score(yva, pred)
                            aucs.append(auc)
                        if not aucs:
                            continue
                        mean_auc = float(np.mean(aucs))
                        if mean_auc > best_auc:
                            best_auc = mean_auc
                            best_params = (ne, md, lr)
            best_per_fold.append(best_params)
            ne, md, lr = best_params
            model = self._train_single(
                X.iloc[train_idx], y[train_idx], weights[train_idx],
                ne, md, lr,
            )
            pred = 1.0 / (1.0 + np.exp(-model.predict(X.iloc[test_idx])))
            from sklearn.metrics import roc_auc_score
            if len(np.unique(y[test_idx])) >= 2:
                outer_auc = roc_auc_score(y[test_idx], pred)
                outer_aucs.append(outer_auc)

        self._meta_feature_cols = meta_cols
        ne_med = int(np.median([p[0] for p in best_per_fold]))
        md_med = int(np.median([p[1] for p in best_per_fold]))
        lr_med = float(np.median([p[2] for p in best_per_fold]))
        self._model = self._train_single(X, y, weights, ne_med, md_med, lr_med)
        self.n_estimators = ne_med
        self.max_depth = md_med
        self.learning_rate = lr_med

        return {
            "best_params": (ne_med, md_med, lr_med),
            "oos_auc_mean": float(np.mean(outer_aucs)) if outer_aucs else np.nan,
            "oos_auc_std": float(np.std(outer_aucs)) if len(outer_aucs) > 1 else 0.0,
        }

    def evaluate_stratified(
        self,
        features_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        regime_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Stratified Precision/Recall by Risk-On vs Risk-Off.
        Logs and returns metrics.
        """
        is_profitable, _ = _build_meta_target(features_df, prices_df)
        proba = self.predict_proba(features_df)
        regime_df = regime_df.copy()
        regime_df.index = pd.to_datetime(regime_df.index)

        merged = features_df[["date", "ticker"]].copy()
        merged["date"] = pd.to_datetime(merged["date"])
        merged["y_true"] = is_profitable.values
        merged["y_prob"] = proba.values
        merged = merged.merge(
            regime_df[["regime"]],
            left_on="date",
            right_index=True,
            how="left",
        )
        merged = merged.dropna(subset=["y_true", "y_prob", "regime"])
        pred = (merged["y_prob"] >= 0.5).astype(int)

        def _prec_rec(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            return prec, rec

        risk_on = merged["regime"] >= REGIME_RISK_ON_THRESHOLD
        risk_off = ~risk_on

        out: Dict[str, float] = {}
        for name, mask in [("Risk-On", risk_on), ("Risk-Off", risk_off)]:
            if mask.sum() < 5:
                out[f"{name}_Precision"] = np.nan
                out[f"{name}_Recall"] = np.nan
                continue
            yt = merged.loc[mask, "y_true"].values
            yp = pred.loc[mask].values
            prec, rec = _prec_rec(yt, yp)
            out[f"{name}_Precision"] = prec
            out[f"{name}_Recall"] = rec
            logger.info("Meta %s: Precision=%.4f Recall=%.4f (n=%d)", name, prec, rec, mask.sum())

        return out
