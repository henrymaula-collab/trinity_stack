from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

REQUIRED_MACRO_COLS: List[str] = ["V2TX", "Breadth", "Rates"]
PERSISTENCE_DAYS = 3
CONFIDENCE_THRESHOLD = 0.70

# Target exposures per state: 0=Normal, 1=Uncertain, 2=Crisis
STATE_EXPOSURE: tuple[float, float, float] = (1.0, 0.5, 0.0)


class MacroRegimeDetector:
    """
    Layer 2 macro regime detection using Gaussian HMM on V2TX, Breadth, Rates (Bund).
    Enforces 3-day persistence. (Liquidity Stress override is handled externally).
    """

    def __init__(
        self,
        n_states: int = 3,
        persistence_days: int = PERSISTENCE_DAYS,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        random_state: int = 42,
    ) -> None:
        self.n_states = n_states
        self.persistence_days = persistence_days
        self.confidence_threshold = confidence_threshold
        self._hmm = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            random_state=random_state,
            n_iter=100,
            tol=1e-4,
        )
        self._fitted = False

    def fit(self, X: pd.DataFrame | np.ndarray) -> MacroRegimeDetector:
        arr = self._validate_and_extract(X)
        self._hmm.fit(arr)
        self._reorder_states_by_variance()
        self._fitted = True
        return self

    def _reorder_states_by_variance(self) -> None:
        """
        Deterministic state mapping to prevent label switching.
        EM outputs unordered states; force 0=lowest var (Bull), 1=mid, 2=highest (Crisis).
        Uses trace of covariance; tie-break by V2TX (col 0) mean for stability.
        """
        variances = np.array(
            [np.trace(self._hmm.covars_[k]) for k in range(self.n_states)]
        )
        means_v2tx = np.array([self._hmm.means_[k, 0] for k in range(self.n_states)])
        perm = np.lexsort((variances, means_v2tx))  # primary: V2TX mean (crisis=high), secondary: var
        self._hmm.means_ = self._hmm.means_[perm]
        self._hmm.covars_ = self._hmm.covars_[perm]
        self._hmm.transmat_ = self._hmm.transmat_[perm, :][:, perm]
        self._hmm.startprob_ = self._hmm.startprob_[perm]

    def _validate_and_extract(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            for col in REQUIRED_MACRO_COLS:
                if col not in X.columns:
                    raise ValueError(f"Missing required column: {col}")
            arr = X[REQUIRED_MACRO_COLS].values
        elif isinstance(X, np.ndarray):
            if X.ndim != 2 or X.shape[1] != len(REQUIRED_MACRO_COLS):
                raise ValueError(
                    f"X must be (n_samples, {len(REQUIRED_MACRO_COLS)}), got {X.shape}"
                )
            arr = X.astype(np.float64)
        else:
            raise ValueError("X must be pd.DataFrame or np.ndarray")

        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise ValueError("X contains NaN or inf")
        return arr

    def predict_regime(self, X: pd.DataFrame | np.ndarray) -> float:
        """
        Return continuous regime_exposure_scalar (0.0 to 1.0).

        Uses predict_proba and dot product with state target exposures
        (Normal=1.0, Uncertain=0.5, Crisis=0.0). Replaces discrete integer output.
        """
        if not self._fitted:
            raise ValueError("Detector must be fitted before predict_regime")

        arr = self._validate_and_extract(X)
        if len(arr) == 0:
            raise ValueError("X has no samples")

        posteriors = self._hmm.predict_proba(arr)
        smoothed = self._apply_persistence_smoothing(posteriors)
        exposure = float(np.dot(smoothed, np.array(STATE_EXPOSURE)))
        return max(0.0, min(1.0, exposure))

    def _apply_persistence_smoothing(self, posteriors: np.ndarray) -> np.ndarray:
        """Average posteriors over persistence window; return last smoothed prob vector."""
        if len(posteriors) == 0:
            raise ValueError("posteriors is empty")
        window = posteriors[-self.persistence_days :]
        return window.mean(axis=0).astype(np.float64)