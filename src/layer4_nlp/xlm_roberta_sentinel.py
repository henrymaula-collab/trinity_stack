"""
Layer 4 Event-Driven NLP Sentinel.
Position sizing penalty multipliers via financial sentiment analysis.
Pipeline loaded once (singleton) to prevent OOM during backtest loops.
"""

from __future__ import annotations

import math
from typing import ClassVar, Optional

from transformers import Pipeline, pipeline

DECAY_TAU: float = 6.67  # ~20/3 trading days for ~95% recovery
RECOVERY_DAYS: int = 20

CRITICAL_KEYWORDS: tuple[str, ...] = (
    "dilution",
    "bankruptcy",
    "fraud",
    "litigation",
    "default",
    "restructuring",
)

NEGATIVE_THRESHOLD: float = 0.85
PENALTY_MULTIPLIER: float = 0.5


def _is_invalid_input(text: str | float | None) -> bool:
    """Return True if input should be treated as safe (return 1.0)."""
    if text is None:
        return True
    if isinstance(text, float) and math.isnan(text):
        return True
    if isinstance(text, str) and not text.strip():
        return True
    return False


def _has_critical_keyword(text: str) -> bool:
    """Case-insensitive scan for CRITICAL_KEYWORDS."""
    lower = text.lower()
    return any(kw in lower for kw in CRITICAL_KEYWORDS)


class NLPSentinel:
    """
    Singleton NLP sentinel for corporate action sentiment.
    Pipeline loaded exactly once to prevent OOM in backtest loops.
    """

    _pipeline: ClassVar[Optional[Pipeline]] = None
    _model_name: str = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

    def __init__(self) -> None:
        if NLPSentinel._pipeline is None:
            NLPSentinel._pipeline = pipeline(
                "sentiment-analysis",
                model=NLPSentinel._model_name,
                truncation=True,
                max_length=512,
            )

    def get_risk_multiplier(
        self, text: str | float | None, days_since_event: int = 0
    ) -> float:
        """
        Base penalty 0.5 if critical keyword or negative sentiment > 0.85.
        Exponential decay: multiplier = 1.0 - (0.5 * exp(-days_since_event / decay_tau)).
        If days_since_event > 20, return 1.0. Invalid input -> 1.0.
        """
        if _is_invalid_input(text):
            return 1.0

        if days_since_event > RECOVERY_DAYS:
            return 1.0

        penalty_triggered = False
        txt: str = text if isinstance(text, str) else str(text)
        if _has_critical_keyword(txt):
            penalty_triggered = True
        else:
            try:
                assert NLPSentinel._pipeline is not None
                result = NLPSentinel._pipeline(txt[:4000])[0]
                label = str(result.get("label", "")).lower()
                score = float(result.get("score", 0.0))
                if label == "negative" and score > NEGATIVE_THRESHOLD:
                    penalty_triggered = True
            except Exception:
                pass

        if not penalty_triggered:
            return 1.0

        decay = math.exp(-days_since_event / DECAY_TAU)
        return 1.0 - (PENALTY_MULTIPLIER * decay)
