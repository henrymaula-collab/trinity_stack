"""
Layer 2: Reflexivity & Liquidity Decay Formalization.

Replaces static step-function with dynamic liquidity decay and endogenous risk.
- L(t) = L_baseline * exp(-α * V_trinity / ADV)
- α (Resilience Factor) from bid/ask bounce frequency (thin Nordic stocks = high α)
- Reflexive Alpha Decay: participation > 5% → alpha erosion (feedback to Layer 3)
- Endogenous Risk: Self-Impact metric; >15% of daily trend → Predatory Trading Alert
- Circuit Breaker: Alert → Passive Stealth or Halt
- Reflexivity Sensitivity: Break-Even α (Sharpe → 0) = Fragility Point
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS_YEAR: int = 252
REFLEXIVITY_THRESHOLD_PCT: float = 5.0  # participation > 5% of order book depth
PREDATORY_ALERT_THRESHOLD_PCT: float = 15.0  # Trinity > 15% of daily trend
ALPHA_DECAY_EXPONENT: float = 2.0  # steepness of alpha erosion above threshold
BOUNCE_ROLLING_DAYS: int = 21  # for bid/ask bounce estimation
ALPHA_MIN: float = 0.1  # min resilience (liquid names)
ALPHA_MAX: float = 3.0  # max resilience (thin names)
INTERNALIZATION_MASKING: float = 0.70


class ExecutionMode(Enum):
    """Circuit breaker: execution mode when Predatory Alert triggers."""

    AGGRESSIVE_FILL = "aggressive"  # default
    PASSIVE_STEALTH = "passive_stealth"  # hidden orders
    HALT = "halt"  # stop execution, let book reset


# ---------------------------------------------------------------------------
# Dynamic Liquidity Decay
# ---------------------------------------------------------------------------


def liquidity_decay(
    v_trinity: float,
    adv: float,
    alpha: float,
    l_baseline: Optional[float] = None,
) -> float:
    """
    L(t) = L_baseline * exp(-α * V_trinity / ADV)

    Available liquidity decays as Trinity's cumulative trade volume increases.
    Higher α (thin stocks) → faster decay.
    """
    if adv <= 0:
        return 0.0
    if l_baseline is None:
        l_baseline = adv  # assume baseline = ADV
    ratio = v_trinity / adv
    return float(l_baseline * np.exp(-alpha * ratio))


def participation_effective(
    order_size: float,
    l_available: float,
) -> float:
    """Effective participation = order / available liquidity (capped at 1)."""
    if l_available <= 0:
        return 1.0
    return min(order_size / l_available, 1.0)


# ---------------------------------------------------------------------------
# Resilience Factor α from Bid/Ask Bounce
# ---------------------------------------------------------------------------


def derive_resilience_factor(
    spread_series: pd.Series,
    rolling_days: int = BOUNCE_ROLLING_DAYS,
    alpha_min: float = ALPHA_MIN,
    alpha_max: float = ALPHA_MAX,
) -> float:
    """
    α from bid/ask bounce frequency. Higher spread volatility = thinner = higher α.

    Proxy: coefficient of variation of spread changes (bounce = spread oscillates).
    Thin Nordic stocks: single trade 'shocks' the book → high CV → high α.
    """
    if spread_series.empty or len(spread_series) < rolling_days:
        return (alpha_min + alpha_max) / 2.0

    changes = spread_series.diff().dropna()
    if len(changes) < 2:
        return (alpha_min + alpha_max) / 2.0

    cv = np.abs(changes).mean() / (spread_series.mean() + 1e-12)
    # Map CV to [alpha_min, alpha_max]; high CV → high α
    # Normalize: typical CV 0.1–1.0 → alpha
    cv_clipped = np.clip(cv, 0.05, 2.0)
    alpha = alpha_min + (alpha_max - alpha_min) * (cv_clipped - 0.05) / 1.95
    return float(np.clip(alpha, alpha_min, alpha_max))


def derive_resilience_wide(
    raw_prices: pd.DataFrame,
    spread_col: str = "BID_ASK_SPREAD_PCT",
    rolling_days: int = BOUNCE_ROLLING_DAYS,
) -> pd.DataFrame:
    """Resilience factor α per ticker, index=date, columns=ticker."""
    if spread_col not in raw_prices.columns:
        return pd.DataFrame()

    px = raw_prices.copy()
    px["date"] = pd.to_datetime(px["date"])
    wide = px.pivot_table(index="date", columns="ticker", values=spread_col)

    alphas = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
    for c in wide.columns:
        s = wide[c].dropna()
        if len(s) < rolling_days:
            alphas[c] = (ALPHA_MIN + ALPHA_MAX) / 2.0
            continue
        rolled = s.rolling(rolling_days, min_periods=rolling_days // 2)
        # Rolling α per date (point-in-time)
        for i in range(rolling_days - 1, len(s)):
            idx = s.index[i]
            block = s.iloc[: i + 1]
            alphas.loc[idx, c] = derive_resilience_factor(block, rolling_days)

    return alphas.ffill().fillna((ALPHA_MIN + ALPHA_MAX) / 2.0)


# ---------------------------------------------------------------------------
# Reflexive Alpha Decay (feedback to Layer 3)
# ---------------------------------------------------------------------------


def reflexive_alpha_decay(
    alpha_scores: pd.Series,
    participation_rates: pd.Series,
    threshold_pct: float = REFLEXIVITY_THRESHOLD_PCT,
    decay_exponent: float = ALPHA_DECAY_EXPONENT,
) -> pd.Series:
    """
    Alpha Erosion: when participation > threshold, decay alpha.
    By 40% filled, market has 'sniffed' intent; remaining 60% priced in.
    """
    threshold = threshold_pct / 100.0
    decayed = alpha_scores.copy()
    for t in alpha_scores.index:
        if t not in participation_rates.index:
            continue
        p = participation_rates[t]
        if p <= threshold:
            continue
        excess = (p - threshold) / (1.0 - threshold + 1e-12)
        decay_factor = np.exp(-decay_exponent * excess)
        decayed[t] = alpha_scores[t] * decay_factor
    return decayed


# ---------------------------------------------------------------------------
# Endogenous Risk Engine
# ---------------------------------------------------------------------------


@dataclass
class SelfImpactResult:
    """Output of self-impact computation."""

    self_impact_pct: float  # % of daily move caused by Trinity
    predatory_alert: bool
    execution_mode: ExecutionMode


def compute_self_impact(
    trinity_notional: float,
    adv: float,
    daily_price_move_pct: float,
    participation_to_impact_coef: float = 1.5,
) -> SelfImpactResult:
    """
    Self-Impact: % of daily price move caused by Trinity's orders.

    Model: contribution ≈ participation * sqrt(participation) * coef.
    If Trinity responsible for >15% of daily trend → Predatory Alert.
    """
    if adv <= 0 or abs(daily_price_move_pct) < 1e-8:
        return SelfImpactResult(
            self_impact_pct=0.0,
            predatory_alert=False,
            execution_mode=ExecutionMode.AGGRESSIVE_FILL,
        )

    participation = trinity_notional / adv
    participation = min(participation, 1.0)

    # Rough impact: participation drives sqrt(participation) of move (square-root law)
    trinity_contribution_pct = (
        100.0
        * participation_to_impact_coef
        * np.sqrt(participation)
        * (1.0 if daily_price_move_pct >= 0 else -1.0)
    )
    total_move_abs = abs(daily_price_move_pct)
    if total_move_abs < 1e-8:
        self_impact_pct = 0.0
    else:
        self_impact_pct = min(
            100.0, abs(trinity_contribution_pct) / total_move_abs * 100.0
        )

    predatory_alert = self_impact_pct > PREDATORY_ALERT_THRESHOLD_PCT
    mode = (
        ExecutionMode.PASSIVE_STEALTH
        if predatory_alert
        else ExecutionMode.AGGRESSIVE_FILL
    )

    return SelfImpactResult(
        self_impact_pct=self_impact_pct,
        predatory_alert=predatory_alert,
        execution_mode=mode,
    )


def check_predatory_alert(
    self_impact_pct: float,
    threshold_pct: float = PREDATORY_ALERT_THRESHOLD_PCT,
) -> Tuple[bool, ExecutionMode]:
    """Trigger circuit breaker: transition to Passive Stealth or Halt."""
    alert = self_impact_pct > threshold_pct
    mode = ExecutionMode.PASSIVE_STEALTH if alert else ExecutionMode.AGGRESSIVE_FILL
    return alert, mode


# ---------------------------------------------------------------------------
# Reflexivity Sensitivity Report (PhD Validation)
# ---------------------------------------------------------------------------


@dataclass
class ReflexivitySensitivityReport:
    """Break-Even α and fragility diagnostics."""

    break_even_alpha: float
    alpha_at_sharpe_zero: Optional[float]
    baseline_sharpe: float
    fragility_point_detected: bool
    alphas_tested: List[float]
    sharpes_by_alpha: Dict[float, float]


def reflexivity_sensitivity_report(
    strategy_returns_fn: Callable[[float], pd.Series],
    alpha_grid: Optional[List[float]] = None,
    sharpe_threshold: float = 0.0,
) -> ReflexivitySensitivityReport:
    """
    Break-Even Resilience: α at which Sharpe drops to zero.
    Identifies Fragility Point of Trinity Stack.

    strategy_returns_fn(alpha: float) -> pd.Series of daily returns.
    """
    if alpha_grid is None:
        alpha_grid = [0.0, 0.1, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]

    sharpes: Dict[float, float] = {}
    for a in alpha_grid:
        try:
            ret = strategy_returns_fn(a)
            if ret.empty or ret.std() < 1e-12:
                sharpes[a] = np.nan
            else:
                sr = (ret.mean() / ret.std()) * np.sqrt(TRADING_DAYS_YEAR)
                sharpes[a] = float(sr)
        except Exception:
            sharpes[a] = np.nan

    baseline = sharpes.get(0.0, sharpes.get(min(alpha_grid)))
    if baseline is None:
        baseline = sharpes.get(alpha_grid[0], np.nan)
    if np.isnan(baseline) and sharpes:
        valid = [v for v in sharpes.values() if np.isfinite(v)]
        baseline = valid[0] if valid else np.nan

    # Find α where Sharpe crosses zero
    sorted_alphas = sorted(sharpes.keys())
    alpha_at_zero: Optional[float] = None
    for i, a in enumerate(sorted_alphas):
        s = sharpes[a]
        if not np.isfinite(s):
            continue
        if s <= sharpe_threshold:
            if i == 0:
                alpha_at_zero = a
            else:
                a_prev = sorted_alphas[i - 1]
                s_prev = sharpes[a_prev]
                if np.isfinite(s_prev) and s_prev > sharpe_threshold:
                    # Linear interpolation
                    frac = (sharpe_threshold - s_prev) / (s - s_prev)
                    alpha_at_zero = a_prev + frac * (a - a_prev)
            break

    break_even = alpha_at_zero if alpha_at_zero is not None else float("inf")
    fragility = (
        np.isfinite(break_even)
        and break_even < max(alpha_grid)
        and baseline is not None
        and np.isfinite(baseline)
        and baseline > 0
    )

    return ReflexivitySensitivityReport(
        break_even_alpha=break_even,
        alpha_at_sharpe_zero=alpha_at_zero,
        baseline_sharpe=float(baseline) if np.isfinite(baseline) else np.nan,
        fragility_point_detected=fragility,
        alphas_tested=alpha_grid,
        sharpes_by_alpha=sharpes,
    )


def format_reflexivity_report(report: ReflexivitySensitivityReport) -> str:
    """Format report for backtest output."""
    lines = [
        "--- Reflexivity Sensitivity (PhD Validation) ---",
        f"  Baseline Sharpe (α→0):    {report.baseline_sharpe:.4f}",
        f"  Break-Even α (Sharpe→0): {report.break_even_alpha:.4f}",
        f"  Fragility Point:         {'DETECTED' if report.fragility_point_detected else 'Not detected'}",
        "",
        "  Sharpe by α:",
    ]
    for a in sorted(report.sharpes_by_alpha.keys()):
        s = report.sharpes_by_alpha[a]
        val = f"{s:.4f}" if np.isfinite(s) else "nan"
        lines.append(f"    α={a:.2f}  →  Sharpe={val}")
    return "\n".join(lines)


def strategy_returns_at_resilience(
    baseline_returns: pd.Series,
    total_friction_bps: float,
    alpha: float,
    scale: float = 0.35,
) -> pd.Series:
    """
    Approximate strategy returns when market resilience = α.
    Higher α → liquidity decays faster → worse fills → more drag.
    ret(α) = ret_baseline - daily_drag(α), where drag ∝ α.
    """
    out = baseline_returns.copy()
    n = len(out.dropna())
    if n < 2:
        return out
    daily_drag = (total_friction_bps / 10000.0) * scale * alpha / max(n, 1)
    out = out - daily_drag
    return out


def run_reflexivity_sensitivity(
    compute_strat_ret_fn: Callable[[], Tuple[pd.Series, Dict[str, float]]],
    alpha_grid: Optional[List[float]] = None,
) -> ReflexivitySensitivityReport:
    """
    Run reflexivity sensitivity using baseline friction + scaling.
    compute_strat_ret_fn() returns (strategy_returns, metrics).
    Uses strategy_returns_at_resilience for α > 0.
    """
    baseline_ret, metrics = compute_strat_ret_fn()
    total_friction = metrics.get("Total_Friction_Cost_BPS", 0.0)

    def _returns_fn(alpha: float) -> pd.Series:
        if alpha <= 0:
            return baseline_ret
        return strategy_returns_at_resilience(
            baseline_ret, total_friction, alpha,
        )

    return reflexivity_sensitivity_report(_returns_fn, alpha_grid=alpha_grid)
