"""
Layer 5: Hierarchical Risk Parity (HRP) portfolio allocation.
López de Prado algorithm: correlation distance, linkage, quasi-diagonalization, recursive bisection.
FX volatility penalty in distance; currency isolation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform


def _correl_dist(
    corr: pd.DataFrame,
    fx_vol_penalty: float = 0.0,
    sek_tickers: Optional[list] = None,
) -> np.ndarray:
    """
    Distance matrix from correlation: d = sqrt((1 - rho) / 2).
    If fx_vol_penalty > 0 and sek_tickers given: add penalty for pairs involving SEK
    (FX volatility as straffaktor in avståndsberäkning).
    """
    dist = np.sqrt((1 - corr) / 2).values
    if fx_vol_penalty <= 0 or not sek_tickers:
        return dist
    sek_set = set(sek_tickers)
    labels = corr.index.tolist()
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            if i != j and (li in sek_set or lj in sek_set):
                dist[i, j] += fx_vol_penalty
                dist[j, i] += fx_vol_penalty
    return dist


def _get_ivp(cov: pd.DataFrame) -> np.ndarray:
    """Inverse-variance portfolio weights."""
    ivp = 1.0 / np.diag(cov)
    return ivp / ivp.sum()


def _get_cluster_var(cov: pd.DataFrame, items: list) -> float:
    """Variance of inverse-variance-weighted portfolio over items."""
    cov_sub = cov.loc[items, items]
    w = _get_ivp(cov_sub).reshape(-1, 1)
    return float(np.dot(np.dot(w.T, cov_sub), w)[0, 0])


def _get_quasi_diag(link: np.ndarray) -> list[int]:
    """Quasi-diagonal ordering from linkage tree (0-based indices)."""
    return sch.leaves_list(link).tolist()


def _get_rec_bipart(cov: pd.DataFrame, sort_labels: list) -> pd.Series:
    """Top-down recursive bisection HRP allocation. sort_labels: asset names in quasi-diag order."""

    w = pd.Series(0.0, index=sort_labels)

    def bisect(items: list, weight: float) -> None:
        if len(items) == 1:
            w[items[0]] = weight
            return
        mid = len(items) // 2
        left, right = items[:mid], items[mid:]
        var_l = _get_cluster_var(cov, left)
        var_r = _get_cluster_var(cov, right)
        alpha = var_r / (var_l + var_r)
        bisect(left, weight * alpha)
        bisect(right, weight * (1.0 - alpha))

    bisect(sort_labels, 1.0)
    return w


def _hrp_weights_for_returns(
    returns_df: pd.DataFrame,
    fx_vol_penalty: float = 0.0,
    sek_tickers: Optional[list] = None,
) -> pd.Series:
    """Run HRP on a single returns block. Returns weights summing to 1.0."""
    if returns_df.empty or len(returns_df.columns) == 0:
        return pd.Series(dtype=float)
    cov = returns_df.cov()
    corr = returns_df.corr()
    dist = _correl_dist(corr, fx_vol_penalty, sek_tickers)
    condensed = squareform(dist, checks=False)
    link = sch.linkage(condensed, method="single")
    sort_ix = _get_quasi_diag(link)
    sort_labels = corr.index[sort_ix].tolist()
    w = _get_rec_bipart(cov, sort_labels)
    w = w.reindex(returns_df.columns).fillna(0.0)
    w = w / w.sum()
    return w


class HierarchicalRiskParity:
    """
    HRP allocation with currency isolation. Split by EUR/SEK, run HRP per group,
    50% capital to each currency bloc, combine to unified weights summing to 1.0.
    Optional FX volatility penalty in distance for SEK bloc.
    """

    def allocate(
        self,
        returns_df: pd.DataFrame,
        currency_series: pd.Series,
        fx_vol_penalty: float = 0.0,
    ) -> pd.Series:
        """
        Compute HRP weights with currency isolation.
        currency_series: ticker -> 'EUR' or 'SEK'. Must cover all columns of returns_df.
        fx_vol_penalty: Straffaktor for SEK-tickers i avståndsmatrisen (default 0).
        Returns weights summing to 1.0.
        """
        if returns_df.isna().any().any():
            raise ValueError("returns_df contains NaNs")
        tickers = returns_df.columns.tolist()
        missing = set(tickers) - set(currency_series.index)
        if missing:
            raise ValueError(f"currency_series missing tickers: {missing}")
        invalid = set(currency_series.loc[tickers]) - {"EUR", "SEK"}
        if invalid:
            raise ValueError(f"currency_series must be 'EUR' or 'SEK'; got {invalid}")

        eur_tickers = [t for t in tickers if currency_series[t] == "EUR"]
        sek_tickers = [t for t in tickers if currency_series[t] == "SEK"]

        weights = pd.Series(0.0, index=tickers)

        if eur_tickers:
            eur_ret = returns_df[eur_tickers]
            eur_w = _hrp_weights_for_returns(eur_ret)
            weights.loc[eur_tickers] = eur_w.values * 0.5

        if sek_tickers:
            sek_ret = returns_df[sek_tickers]
            sek_w = _hrp_weights_for_returns(
                sek_ret,
                fx_vol_penalty=fx_vol_penalty,
                sek_tickers=sek_tickers if fx_vol_penalty > 0 else None,
            )
            weights.loc[sek_tickers] = sek_w.values * 0.5

        s = weights.sum()
        if s <= 0:
            raise ValueError("No valid weights after currency split")
        return weights / s
