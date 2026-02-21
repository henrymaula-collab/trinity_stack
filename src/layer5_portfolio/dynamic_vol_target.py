"""
Layer 5: Dynamic volatility targeting and regime scaling.
Applies NLP multipliers, macro regime exposure, and vol target with leverage cap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS: int = 252
MAX_LEVERAGE: float = 1.5
TARGET_VOL_MULTIPLIER: float = 0.80
DRAWDOWN_THRESHOLD: float = -0.10
DRAWDOWN_CUT: float = 0.80

REGIME_EXPOSURE: dict[int, float] = {
    0: 1.0,
    1: 0.5,
    2: 0.0,
}


def _align_indices(
    hrp_weights: pd.Series,
    cov_matrix: pd.DataFrame,
    nlp_multipliers: pd.Series,
    historical_volatilities: pd.Series,
) -> None:
    """Raise ValueError if any index mismatch or NaN present."""
    if hrp_weights.isna().any() or nlp_multipliers.isna().any() or cov_matrix.isna().any().any():
        raise ValueError("NaNs present in hrp_weights, nlp_multipliers, or cov_matrix")
    if historical_volatilities.isna().any() or (historical_volatilities <= 0).any():
        raise ValueError("historical_volatilities must be strictly positive and non-NaN")
    assets = set(hrp_weights.index)
    if set(nlp_multipliers.index) != assets:
        raise ValueError(
            "Index mismatch: hrp_weights and nlp_multipliers must share identical index"
        )
    cov_assets = set(cov_matrix.index) | set(cov_matrix.columns)
    if assets != cov_assets:
        raise ValueError(
            "Index mismatch: cov_matrix must have index/columns matching hrp_weights"
        )
    if set(historical_volatilities.index) != assets:
        raise ValueError(
            "Index mismatch: historical_volatilities must match hrp_weights index"
        )


class DynamicVolTargeting:
    """
    Apply NLP penalties, macro regime scaling, and volatility targeting.
    Fail loudly on index mismatch.
    """

    def apply_targets(
        self,
        hrp_weights: pd.Series,
        cov_matrix: pd.DataFrame,
        nlp_multipliers: pd.Series,
        macro_regime: int,
        historical_volatilities: pd.Series,
        portfolio_drawdown: float = 0.0,
        target_vol_annual: float | None = None,
    ) -> pd.Series:
        """
        Step 1: Multiply by nlp_multipliers, renormalize.
        Step 2: Scale by regime exposure (0->1.0, 1->0.5, 2->0.0).
        Step 3: Target vol = 0.80 * median(historical_volatilities) if not overridden.
        Step 4: Scale to target vol, cap leverage at 1.5x.
        Step 5: If portfolio_drawdown <= -0.10, cut gross exposure by 20% (multiply by 0.80).
        """
        _align_indices(hrp_weights, cov_matrix, nlp_multipliers, historical_volatilities)
        if macro_regime not in REGIME_EXPOSURE:
            raise ValueError(f"macro_regime must be 0, 1, or 2; got {macro_regime}")

        w = hrp_weights * nlp_multipliers
        s = w.sum()
        if s <= 0:
            raise ValueError("Sum of hrp_weights * nlp_multipliers must be positive")
        w = w / s

        regime_scale = REGIME_EXPOSURE[macro_regime]
        w = w * regime_scale
        if regime_scale == 0:
            return w

        if target_vol_annual is None:
            target_vol_annual = TARGET_VOL_MULTIPLIER * float(
                historical_volatilities.reindex(w.index).median()
            )
        if target_vol_annual <= 0:
            raise ValueError("target_vol_annual must be positive")

        effective_target_vol = target_vol_annual * regime_scale
        var = float(np.dot(w.values, np.dot(cov_matrix.loc[w.index, w.index].values, w.values)))
        vol_annual = np.sqrt(var * TRADING_DAYS)
        if vol_annual <= 0:
            raise ValueError("Ex-ante portfolio volatility is non-positive")
        scale = effective_target_vol / vol_annual
        w = w * scale

        gross = w.abs().sum()
        if gross > MAX_LEVERAGE:
            w = w * (MAX_LEVERAGE / gross)

        if portfolio_drawdown <= DRAWDOWN_THRESHOLD:
            w = w * DRAWDOWN_CUT

        return w
