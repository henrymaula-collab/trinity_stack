from __future__ import annotations

from collections import deque
from typing import List

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

REQUIRED_MACRO_COLS: List[str] = ["V2TX", "Breadth", "Rates"]
PERSISTENCE_DAYS = 3
CONFIDENCE_THRESHOLD = 0.70


class MacroRegimeDetector:
    """
    Layer 2 macro regime detection using Gaussian HMM on V2TX, Breadth, Rates.
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
        self._fitted = True
        return self

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

    def predict_regime(self, X: pd.DataFrame | np.ndarray) -> int:
        if not self._fitted:
            raise ValueError("Detector must be fitted before predict_regime")

        arr = self._validate_and_extract(X)
        if len(arr) == 0:
            raise ValueError("X has no samples")

        posteriors = self._hmm.predict_proba(arr)
        regime = self._apply_persistence(posteriors)
        return regime

    def _apply_persistence(self, posteriors: np.ndarray) -> int:
        n = len(posteriors)
        if n == 0:
            raise ValueError("posteriors is empty")

        current_regime = int(np.argmax(posteriors[0]))
        window: deque[np.ndarray] = deque(maxlen=self.persistence_days)

        for i in range(n):
            window.append(posteriors[i])
            if len(window) < self.persistence_days:
                current_regime = int(np.argmax(posteriors[i]))
                continue

            qualified = [
                s
                for s in range(self.n_states)
                if all(w[s] > self.confidence_threshold for w in window)
            ]
            if qualified:
                current_regime = max(qualified, key=lambda s: posteriors[i][s])

        return current_regime