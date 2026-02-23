"""
Layer 3: Combinatorial Purged Cross-Validation (CPCV).
Institutional significance: PBO, Deflated Sharpe, IQR of Sharpe distribution.
Purging and embargo based on forward_return_horizon.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Institutional threshold: PBO > 20% => non-allocatable
PBO_NON_ALLOCATABLE_THRESHOLD = 0.20

MIN_PATHS = 1_000


@dataclass
class CPCVResult:
    """Result of CPCV evaluation."""

    sharpe_q1: float
    sharpe_median: float
    sharpe_q3: float
    sharpe_iqr: float  # q3 - q1
    pbo: float
    dsr: float
    is_allocatable: bool  # True iff PBO <= 20%
    n_paths: int
    train_sharpes: np.ndarray = field(repr=False)
    test_sharpes: np.ndarray = field(repr=False)

    def __str__(self) -> str:
        return (
            f"CPCVResult(Sharpe IQR=[{self.sharpe_q1:.3f}, {self.sharpe_median:.3f}, {self.sharpe_q3:.3f}], "
            f"PBO={self.pbo:.1%}, DSR={self.dsr:.3f}, allocatable={self.is_allocatable})"
        )


def _annualized_sharpe(returns: np.ndarray, trading_days: int = 252) -> float:
    """Annualized Sharpe from daily returns."""
    if returns.size < 2 or returns.std() < 1e-12:
        return np.nan
    return float((returns.mean() / returns.std()) * np.sqrt(trading_days))


def _generate_segment_bounds(
    n_samples: int,
    n_segments: int,
) -> List[Tuple[int, int]]:
    """Partition [0, n_samples) into n_segments roughly equal segments."""
    base = n_samples // n_segments
    extra = n_samples % n_segments
    bounds = []
    start = 0
    for i in range(n_segments):
        size = base + (1 if i < extra else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def generate_cpcv_splits(
    n_samples: int,
    n_segments: int = 16,
    n_test_segments: int = 4,
    forward_return_horizon: int = 1,
    embargo_multiple: float = 1.0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate combinatorial purged splits.

    Args:
        n_samples: Total number of samples (e.g. trading days).
        n_segments: Number of segments to partition timeline.
        n_test_segments: Segments used for test in each path.
        forward_return_horizon: Purging removes train samples whose target overlaps test.
                               Embargo = forward_return_horizon * embargo_multiple.
        embargo_multiple: Embargo = horizon * this (default 1.0).

    Returns:
        List of (train_indices, test_indices) per path.
    """
    purge_days = forward_return_horizon
    embargo_days = max(1, int(forward_return_horizon * embargo_multiple))

    bounds = _generate_segment_bounds(n_samples, n_segments)

    paths = []
    for test_seg_indices in combinations(range(n_segments), n_test_segments):
        train_seg_indices = [i for i in range(n_segments) if i not in test_seg_indices]

        test_idx = []
        for j in sorted(test_seg_indices):
            test_idx.extend(range(bounds[j][0], bounds[j][1]))
        test_idx = np.array(test_idx)
        test_set = set(test_idx)

        train_idx = []
        for j in train_seg_indices:
            s, e = bounds[j]
            for i in range(s, e):
                # Purge: target [i+1, i+horizon] must not overlap test
                overlap = any(i + 1 <= j <= i + forward_return_horizon for j in test_set)
                if overlap:
                    continue
                # Embargo: exclude if within embargo_days before any test sample
                emb = any(0 < j - i <= embargo_days for j in test_set)
                if emb:
                    continue
                train_idx.append(i)

        train_idx = np.array(sorted(set(train_idx)))
        if len(train_idx) < 10 or len(test_idx) < 5:
            continue
        paths.append((train_idx, test_idx))

    return paths


def probability_of_backtest_overfitting(
    train_sharpes: np.ndarray,
    test_sharpes: np.ndarray,
) -> float:
    """
    PBO: proportion of top-train paths whose test SR < median test SR of bottom-train paths.
    Lopez de Prado / Bailey et al.
    """
    valid = np.isfinite(train_sharpes) & np.isfinite(test_sharpes)
    train_sr = train_sharpes[valid]
    test_sr = test_sharpes[valid]
    if len(train_sr) < 10:
        return np.nan

    mid = len(train_sr) // 2
    order = np.argsort(train_sr)
    top_half = order[mid:]
    bottom_half = order[:mid]

    median_test_bottom = np.median(test_sr[bottom_half])
    count_worse = np.sum(test_sr[top_half] < median_test_bottom)
    pbo = count_worse / len(top_half)
    return float(pbo)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    sharpe_distribution: np.ndarray,
    n_trials: Optional[int] = None,
) -> float:
    """
    DSR: adjust observed Sharpe for multiple testing / selection bias.
    DSR = observed - E[max SR under null].
    Uses distribution of test Sharpes from CPCV; n_trials = number of parameter combos.
    """
    valid = sharpe_distribution[np.isfinite(sharpe_distribution)]
    if len(valid) < 10:
        return np.nan
    if n_trials is None:
        n_trials = len(valid)

    mean_sr = np.mean(valid)
    std_sr = np.std(valid)
    if std_sr < 1e-12:
        return float(observed_sharpe)
    # Expected max SR under null (multiple testing)
    expected_max_sr = mean_sr + std_sr * np.sqrt(2 * np.log(max(2, n_trials)))
    dsr = float(observed_sharpe - expected_max_sr)
    return dsr


def make_returns_evaluate_fn(
    returns: np.ndarray,
) -> Callable[[np.ndarray, np.ndarray], Tuple[float, float]]:
    """Build evaluate_fn from precomputed daily returns. Indices = sample indices."""

    def fn(train_idx: np.ndarray, test_idx: np.ndarray) -> Tuple[float, float]:
        train_ret = returns[train_idx]
        test_ret = returns[test_idx]
        return _annualized_sharpe(train_ret), _annualized_sharpe(test_ret)

    return fn


def run_cpcv(
    evaluate_fn: Callable[[np.ndarray, np.ndarray], Tuple[float, float]],
    n_samples: int,
    forward_return_horizon: int = 1,
    n_segments: int = 16,
    n_test_segments: int = 4,
    n_paths_min: int = MIN_PATHS,
    n_trials_dsr: Optional[int] = None,
) -> CPCVResult:
    """
    Run CPCV with at least n_paths_min paths.

    Args:
        evaluate_fn: (train_idx, test_idx) -> (train_sharpe, test_sharpe).
                     Caller runs backtest/strategy on each split.
        n_samples: Total samples (trading days).
        forward_return_horizon: For purge and embargo.
        n_segments: Timeline segments.
        n_test_segments: Test segments per path. C(n_segments, n_test_segments) >= n_paths_min.
        n_paths_min: Minimum paths (default 1000).
        n_trials_dsr: Effective trials for DSR (default = n_paths).

    Returns:
        CPCVResult with IQR, PBO, DSR, is_allocatable.
    """
    paths = generate_cpcv_splits(
        n_samples,
        n_segments=n_segments,
        n_test_segments=n_test_segments,
        forward_return_horizon=forward_return_horizon,
    )

    if len(paths) < n_paths_min:
        # Increase segments to get more combinations
        for n_seg in range(n_segments + 1, 24):
            paths = generate_cpcv_splits(
                n_samples,
                n_segments=n_seg,
                n_test_segments=min(n_test_segments, n_seg // 2),
                forward_return_horizon=forward_return_horizon,
            )
            if len(paths) >= n_paths_min:
                break

    paths = paths[: max(n_paths_min, len(paths))]
    train_srs = []
    test_srs = []

    for train_idx, test_idx in paths:
        try:
            ts, vs = evaluate_fn(train_idx, test_idx)
            train_srs.append(ts)
            test_srs.append(vs)
        except Exception as e:
            logger.warning("CPCV path failed: %s", e)
            continue

    train_srs = np.array(train_srs)
    test_srs = np.array(test_srs)
    if len(test_srs) < 20:
        raise ValueError(
            f"CPCV: too few valid paths ({len(test_srs)}). Increase n_samples or reduce n_test_segments."
        )

    q1 = float(np.nanpercentile(test_srs, 25))
    median = float(np.nanmedian(test_srs))
    q3 = float(np.nanpercentile(test_srs, 75))
    iqr = q3 - q1

    pbo = probability_of_backtest_overfitting(train_srs, test_srs)
    dsr = deflated_sharpe_ratio(
        median,
        test_srs,
        n_trials=n_trials_dsr or len(test_srs),
    )

    is_allocatable = pbo <= PBO_NON_ALLOCATABLE_THRESHOLD if np.isfinite(pbo) else False
    if pbo > PBO_NON_ALLOCATABLE_THRESHOLD and np.isfinite(pbo):
        logger.warning(
            "CPCV: PBO=%.1f%% > 20%%. System flaggad som icke-allokerbart.",
            pbo * 100,
        )

    return CPCVResult(
        sharpe_q1=q1,
        sharpe_median=median,
        sharpe_q3=q3,
        sharpe_iqr=iqr,
        pbo=pbo,
        dsr=dsr,
        is_allocatable=is_allocatable,
        n_paths=len(test_srs),
        train_sharpes=train_srs,
        test_sharpes=test_srs,
    )
