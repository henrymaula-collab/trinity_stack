from __future__ import annotations

import numpy as np
import pandas as pd

DECAY_DAYS = 60


class SoftDecayHysteresis:
    """
    Applies a linearly decaying penalty to sell-threshold from T+1 to day 60
    to prevent portfolio churn. Penalty starts high and decays to zero.
    """

    def __init__(self, decay_days: int = DECAY_DAYS) -> None:
        if decay_days < 1:
            raise ValueError("decay_days must be >= 1")
        self.decay_days = decay_days

    def compute_sell_penalty(self, days_held: int | np.ndarray) -> float | np.ndarray:
        """
        Returns penalty factor in [0, 1]. High at T+1, zero at day 60+.
        Linear decay: penalty(d) = max(0, 1 - (d - 1) / (decay_days - 1)) for d in [1, decay_days].
        """
        scalar = np.isscalar(days_held) or (
            isinstance(days_held, np.ndarray) and days_held.ndim == 0
        )
        d = np.asarray(days_held, dtype=np.float64)
        if np.any(d < 1):
            raise ValueError("days_held must be >= 1")
        denom = max(1, self.decay_days - 1)
        penalty = np.clip(1.0 - (d - 1.0) / denom, 0.0, 1.0)
        penalty = np.where(d > self.decay_days, 0.0, penalty)
        return float(penalty) if scalar else penalty

    def apply(
        self,
        df: pd.DataFrame,
        days_held_col: str = "days_held",
        base_sell_threshold_col: str | None = None,
        output_col: str = "sell_threshold",
    ) -> pd.DataFrame:
        """
        Applies soft decay hysteresis. Adds output_col with adjusted sell threshold.
        If base_sell_threshold_col is provided, output = base * (1 - penalty).
        Otherwise outputs the penalty factor (0 to 1) as the threshold modifier.
        """
        if days_held_col not in df.columns:
            raise ValueError(f"Missing column: {days_held_col}")
        if df[days_held_col].isna().any():
            raise ValueError(f"NaNs in {days_held_col}")

        penalty = self.compute_sell_penalty(df[days_held_col].values)
        penalty_arr = np.asarray(penalty)

        if base_sell_threshold_col is not None:
            if base_sell_threshold_col not in df.columns:
                raise ValueError(f"Missing column: {base_sell_threshold_col}")
            out = df[base_sell_threshold_col].values * (1.0 - penalty_arr)
        else:
            out = 1.0 - penalty_arr

        result = df.copy()
        result[output_col] = out
        return result
