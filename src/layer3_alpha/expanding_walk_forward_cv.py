"""
Layer 3: Expanding Walk-Forward Cross-Validation.
Point-in-time safe splits with purge period to prevent leakage.
"""

from __future__ import annotations

from typing import Generator, List, Tuple

import numpy as np
import pandas as pd


class ExpandingWalkForwardCV:
    """
    Expanding-window walk-forward CV with purge period.
    For each fold: train on data with date < (test_start - purge), test on test block.
    No look-ahead; purge_days eliminates leakage from reporting lags.
    """

    def __init__(
        self,
        purge_days: int = 5,
        min_train_samples: int = 126,
        test_size: int = 21,
    ) -> None:
        """
        Args:
            purge_days: Business days between train end and test start. MUST be >=
                        target horizon (e.g. 1 for 1-day forward_return, 21 for 21-day).
                        Otherwise target overlap causes look-ahead bias.
            min_train_samples: Minimum rows in training set (skip early folds).
            test_size: Number of dates per test block (default 21 ≈ 1 month).
        """
        self.purge_days = purge_days
        self.min_train_samples = min_train_samples
        self.test_size = test_size

    def split(
        self,
        X: pd.DataFrame,
        date_col: str = "date",
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yield (train_indices, test_indices) for each fold.

        Args:
            X: DataFrame with date_col. Must be sorted by date.

        Yields:
            (train_idx, test_idx) as numpy arrays of row indices.
        """
        if date_col not in X.columns:
            raise ValueError(f"X must contain '{date_col}'")
        df = X.copy()
        df = df.sort_values(date_col).reset_index(drop=True)
        dates = df[date_col].unique()
        dates = np.sort(dates)
        n_dates = len(dates)

        for i in range(0, n_dates, self.test_size):
            test_end_idx = min(i + self.test_size, n_dates)
            test_dates = dates[i:test_end_idx]
            if len(test_dates) == 0:
                break

            # Train: all dates before (first_test_date - purge_days)
            first_test_date = pd.Timestamp(test_dates[0])
            if isinstance(first_test_date, str):
                first_test_date = pd.to_datetime(first_test_date)
            cutoff = first_test_date - pd.offsets.BDay(self.purge_days)

            train_mask = df[date_col] < cutoff
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(df[date_col].isin(test_dates))[0]

            if len(train_idx) < self.min_train_samples:
                continue
            if len(test_idx) == 0:
                continue
            yield train_idx, test_idx

    def get_n_splits(self, X: pd.DataFrame, date_col: str = "date") -> int:
        """Return number of folds for a given DataFrame."""
        return sum(1 for _ in self.split(X, date_col))
