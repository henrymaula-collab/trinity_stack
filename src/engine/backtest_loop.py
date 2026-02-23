"""
Backtest engine orchestrating Layers 2-5 over rebalance dates.
Strict point-in-time semantics; fail-fast on missing data or index mismatch.
Supports Pessimistic Execution Engine: T+1 realism, friction and opportunity cost metrics.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import pandas as pd

TRAILING_DAYS: int = 252
ROLLING_VOL_DAYS: int = 1260  # 5 years
CRISIS_EXPOSURE_THRESHOLD: float = 0.01  # regime_exposure < this => full exit
TRADING_DAYS_YEAR: int = 252


class _HRPProtocol(Protocol):
    def allocate(
        self, returns_df: pd.DataFrame, currency_series: pd.Series
    ) -> pd.Series:
        ...


class _VolTargetProtocol(Protocol):
    def apply_targets(
        self,
        hrp_weights: pd.Series,
        cov_matrix: pd.DataFrame,
        nlp_multipliers: pd.Series,
        regime_exposure: float,
        historical_volatilities: pd.Series,
        portfolio_drawdown: float = 0.0,
        target_vol_annual: float | None = None,
    ) -> pd.Series:
        ...


class _NLPSentinelProtocol(Protocol):
    def get_risk_multiplier(
        self, text: str | float | None, days_since_event: int = 0
    ) -> float:
        ...


class BacktestEngine:
    """
    Orchestrates Layers 2-5 over time.
    Accepts pre-instantiated layer objects. Does not import layer logic.
    """

    def __init__(
        self,
        alpha_model: Any,
        nlp_sentinel: _NLPSentinelProtocol,
        hrp_model: _HRPProtocol,
        vol_targeter: _VolTargetProtocol,
        trailing_days: int = TRAILING_DAYS,
    ) -> None:
        self._alpha_model = alpha_model
        self._nlp_sentinel = nlp_sentinel
        self._hrp_model = hrp_model
        self._vol_targeter = vol_targeter
        self._trailing_days = trailing_days

    def run_backtest(
        self,
        price_df: pd.DataFrame,
        alpha_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        news_df: pd.DataFrame,
        rebalance_dates: List[pd.Timestamp],
        currency_series: Optional[pd.Series] = None,
        adv_series: Optional[pd.DataFrame] = None,
        portfolio_aum: Optional[float] = None,
        raw_prices: Optional[pd.DataFrame] = None,
        macro_raw_df: Optional[pd.DataFrame] = None,
        use_pessimistic_execution: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, float]]]:
        """
        Iterate through rebalance_dates, run Layers 2-5, return concatenated weights.

        Args:
            price_df: Daily returns, index=date, columns=ticker.
            alpha_df: Alpha signals with columns [date, ticker, alpha_score, ...].
            macro_df: Regime time series, index=date, column 'regime' (float 0.0–1.0).
            news_df: News with columns [ticker, date, text].
            rebalance_dates: List of rebalance dates.
            currency_series: Ticker -> 'EUR' or 'SEK'. If None, mock (alternating by ticker).

        Returns:
            DataFrame with columns [date, ticker, target_weight].
        """
        self._validate_inputs(
            price_df, alpha_df, macro_df, news_df, rebalance_dates
        )
        price_df = price_df.sort_index()
        macro_df = macro_df.sort_index()

        records: List[dict[str, Any]] = []
        sorted_dates = sorted(rebalance_dates)

        cum_equity = 1.0
        peak_equity = 1.0
        prev_weights: Dict[str, float] = {}
        last_rb_date: Optional[pd.Timestamp] = None

        for rb_date in sorted_dates:
            # Compute portfolio_drawdown from previous period
            portfolio_drawdown = 0.0
            if last_rb_date is not None and prev_weights:
                cum_equity, peak_equity = self._update_equity(
                    price_df, last_rb_date, rb_date, prev_weights, cum_equity, peak_equity
                )
                if peak_equity > 0:
                    portfolio_drawdown = (cum_equity - peak_equity) / peak_equity

            # Layer 2: Macro regime (continuous exposure)
            regime_exposure = self._get_regime_asof(macro_df, rb_date)
            if regime_exposure < CRISIS_EXPOSURE_THRESHOLD:
                tickers = self._extract_tickers(alpha_df, rb_date)
                prev_weights = {t: 0.0 for t in tickers}
                last_rb_date = rb_date
                for t in tickers:
                    records.append({"date": rb_date, "ticker": t, "target_weight": 0.0})
                continue

            # Layer 3: Alpha cross-section
            tickers = self._extract_tickers(alpha_df, rb_date)
            if not tickers:
                last_rb_date = rb_date
                continue

            tickers_in_price = [t for t in tickers if t in price_df.columns]
            if len(tickers_in_price) < 2:
                last_rb_date = rb_date
                continue

            returns_trailing = self._trailing_returns(
                price_df, rb_date, tickers_in_price
            )
            if returns_trailing is None or len(returns_trailing.columns) < 2:
                last_rb_date = rb_date
                continue

            final_tickers = returns_trailing.columns.tolist()
            currency = self._resolve_currency(currency_series, final_tickers)

            # Layer 4: NLP multipliers with days_since_event
            nlp_multipliers = self._compute_nlp_multipliers(
                news_df, final_tickers, rb_date
            )

            # Layer 5: HRP + vol target + capacity penalty
            cov_matrix = returns_trailing.cov()
            hrp_weights = self._hrp_model.allocate(returns_trailing, currency)
            hist_vols = self._rolling_vol(price_df, rb_date, final_tickers)
            adv = None
            if adv_series is not None and not adv_series.empty:
                asof = adv_series[adv_series.index < rb_date]
                if not asof.empty:
                    adv = asof.iloc[-1].reindex(final_tickers)
            final_weights = self._vol_targeter.apply_targets(
                hrp_weights,
                cov_matrix,
                nlp_multipliers,
                regime_exposure,
                historical_volatilities=hist_vols,
                portfolio_drawdown=portfolio_drawdown,
                adv_series=adv,
                portfolio_aum=portfolio_aum,
            )

            prev_weights = dict(final_weights)
            last_rb_date = rb_date
            for ticker, w in final_weights.items():
                records.append({"date": rb_date, "ticker": ticker, "target_weight": float(w)})

        if not records:
            weights_df = pd.DataFrame(columns=["date", "ticker", "target_weight"])
            if use_pessimistic_execution and raw_prices is not None:
                self._last_pessimistic_metrics = {"Total_Friction_Cost_BPS": 0.0, "Opportunity_Cost_BPS": 0.0}
                return weights_df, self._last_pessimistic_metrics
            return weights_df

        weights_df = pd.DataFrame(records)
        if use_pessimistic_execution and raw_prices is not None:
            from src.layer6_execution.pessimistic_execution import compute_friction_and_opportunity_cost

            adv_for_sim = adv_series if adv_series is not None and not adv_series.empty else pd.DataFrame()
            if adv_for_sim.empty and "PX_TURN_OVER" in raw_prices.columns:
                px = raw_prices.pivot_table(index="date", columns="ticker", values="PX_TURN_OVER")
                adv_for_sim = px.rolling(20, min_periods=1).mean()
            _, metrics = compute_friction_and_opportunity_cost(
                weights_df,
                raw_prices,
                price_df,
                alpha_df,
                macro_df,
                macro_raw_df,
                adv_for_sim,
                portfolio_aum=portfolio_aum or 1e6,
            )
            self._last_pessimistic_metrics = metrics
            return weights_df, metrics
        return weights_df

    def _validate_inputs(
        self,
        price_df: pd.DataFrame,
        alpha_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        news_df: pd.DataFrame,
        rebalance_dates: List[pd.Timestamp],
    ) -> None:
        if price_df.empty or price_df.isna().any().any():
            raise ValueError("price_df must be non-empty and free of NaNs")
        for col in ["date", "ticker", "alpha_score"]:
            if col not in alpha_df.columns:
                raise ValueError(f"alpha_df missing required column: {col}")
        if "regime" not in macro_df.columns:
            raise ValueError("macro_df missing required column: regime")
        for col in ["ticker", "date", "text"]:
            if col not in news_df.columns:
                raise ValueError(f"news_df missing required column: {col}")
        if not rebalance_dates:
            raise ValueError("rebalance_dates must be non-empty")

    def _get_regime_asof(self, macro_df: pd.DataFrame, rb_date: pd.Timestamp) -> float:
        asof = macro_df.loc[macro_df.index < rb_date]
        if asof.empty:
            raise ValueError(f"No macro regime data before {rb_date}")
        regime = float(asof["regime"].iloc[-1])
        if not (0.0 <= regime <= 1.0):
            raise ValueError(f"Invalid regime_exposure {regime}; must be in [0, 1]")
        return regime

    def _extract_tickers(self, alpha_df: pd.DataFrame, rb_date: pd.Timestamp) -> List[str]:
        sub = alpha_df[(alpha_df["date"] < rb_date)].dropna(subset=["alpha_score"])
        if sub.empty:
            return []
        last_date = sub["date"].max()
        cross = sub[sub["date"] == last_date]
        return cross["ticker"].unique().tolist()

    def _trailing_returns(
        self,
        price_df: pd.DataFrame,
        rb_date: pd.Timestamp,
        tickers: List[str],
    ) -> Optional[pd.DataFrame]:
        mask = price_df.index < rb_date
        window = price_df.loc[mask, tickers].iloc[-self._trailing_days :]
        if window.empty or len(window) < 2:
            return None
        # Exclude tickers with incomplete data (no ffill/bfill: IPO / short history
        # would fake zero vol and mislead HRP overweighting)
        complete = window.dropna(axis=1, how="any")
        if len(complete.columns) < 2:
            return None
        return complete

    def _compute_nlp_multipliers(
        self,
        news_df: pd.DataFrame,
        tickers: List[str],
        rb_date: pd.Timestamp,
    ) -> pd.Series:
        mults: dict[str, float] = {}
        for t in tickers:
            news_sub = news_df[
                (news_df["ticker"] == t) & (news_df["date"] < rb_date)
            ].sort_values("date", ascending=False)
            if news_sub.empty:
                mults[t] = 1.0
            else:
                row = news_sub.iloc[0]
                text = row["text"]
                event_date = pd.Timestamp(row["date"])
                base_mult = self._nlp_sentinel.get_risk_multiplier(text, 0)
                if base_mult >= 1.0:
                    mults[t] = 1.0
                else:
                    days_since = len(
                        pd.bdate_range(
                            event_date + pd.Timedelta(days=1),
                            rb_date,
                            inclusive="left",
                        )
                    )
                    mults[t] = self._nlp_sentinel.get_risk_multiplier(
                        text, days_since
                    )
        return pd.Series(mults)

    def _update_equity(
        self,
        price_df: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
        prev_weights: Dict[str, float],
        cum_equity: float,
        peak_equity: float,
    ) -> tuple[float, float]:
        mask = (price_df.index > start) & (price_df.index < end)
        period = price_df.loc[mask]
        tickers = [t for t in prev_weights if t in period.columns]
        w_vec = np.array([prev_weights[t] for t in tickers])
        for _, row in period[tickers].iterrows():
            r = row.values
            if np.any(np.isnan(r)):
                continue
            daily_ret = float(np.dot(w_vec, r))
            cum_equity *= 1.0 + daily_ret
            peak_equity = max(peak_equity, cum_equity)
        return cum_equity, peak_equity

    def _resolve_currency(
        self,
        currency_series: Optional[pd.Series],
        tickers: List[str],
    ) -> pd.Series:
        if currency_series is not None:
            curr = currency_series.reindex(tickers)
            if curr.isna().any():
                raise ValueError(
                    f"currency_series missing or NaN for tickers: {curr[curr.isna()].index.tolist()}"
                )
            return curr
        # Mock: alternate EUR/SEK by sorted ticker
        mock = {
            t: ("EUR" if i % 2 == 0 else "SEK")
            for i, t in enumerate(sorted(tickers))
        }
        return pd.Series(mock).reindex(tickers)

    def _rolling_vol(
        self,
        price_df: pd.DataFrame,
        rb_date: pd.Timestamp,
        tickers: List[str],
    ) -> pd.Series:
        mask = price_df.index < rb_date
        window = price_df.loc[mask, tickers].iloc[-ROLLING_VOL_DAYS:]
        vols = window.std() * np.sqrt(TRADING_DAYS_YEAR)
        if (vols <= 0).any() or vols.isna().any():
            raise ValueError(
                "historical_volatilities must be strictly positive; "
                "insufficient data or zero-vol assets"
            )
        return vols
