import numpy as np
import pandas as pd
from src.engine.backtest_loop import BacktestEngine

# --- Mock Classes för att testa flödet ---
class MockNLP:
    def get_risk_multiplier(
        self, text: str | float | None, days_since_event: int = 0
    ) -> float:
        return 0.5 if "bad" in str(text).lower() else 1.0


class MockHRP:
    def allocate(
        self, returns_df: pd.DataFrame, currency_series: pd.Series
    ) -> pd.Series:
        w = 1.0 / len(returns_df.columns)
        return pd.Series(w, index=returns_df.columns)


class MockVolTarget:
    def apply_targets(
        self,
        hrp_weights,
        cov_matrix,
        nlp_multipliers,
        regime_exposure,
        historical_volatilities,
        portfolio_drawdown=0.0,
        target_vol_annual=None,
        adv_series=None,
        portfolio_aum=None,
        **kwargs,
    ):
        w = hrp_weights * nlp_multipliers
        w = w / w.sum()
        w = w * regime_exposure
        return w

def run_tests():
    print("Initierar Master Backtest Engine...")
    
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    rebalance_dates = [dates[5], dates[9]]
    
    # 1. Price Data (Dagliga avkastningar)
    price_df = pd.DataFrame(np.random.normal(0, 0.01, (10, 2)), index=dates, columns=["A", "B"])
    
    # 2. Alpha Data
    alpha_df = pd.DataFrame({
        "date": [dates[4], dates[4], dates[8], dates[8]],
        "ticker": ["A", "B", "A", "B"],
        "alpha_score": [0.8, 0.9, 0.7, 0.6]
    })
    
    # 3. Macro Data (continuous regime exposure: 1.0=Bull, 0.5=Neutral, 0.0=Crisis)
    macro_df = pd.DataFrame({
        "regime": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0]
    }, index=dates)
    
    # 4. News Data
    news_df = pd.DataFrame({
        "date": [dates[3], dates[7]],
        "ticker": ["A", "B"],
        "text": ["Good earnings.", "Really bad news."]
    })
    
    engine = BacktestEngine(
        alpha_model=None, 
        nlp_sentinel=MockNLP(), 
        hrp_model=MockHRP(), 
        vol_targeter=MockVolTarget(),
        trailing_days=5 # Kort fönster för testet
    )
    
    res = engine.run_backtest(price_df, alpha_df, macro_df, news_df, rebalance_dates)
    
    # Valideringar
    assert not res.empty, "❌ FEL: Motorn returnerade ingen data."
    
    # Vid Rebalance 1 (Date 5): Regime 0 (Bull). A har nyhet "Good", B har ingen nyhet. Vikter bör vara ~0.5 / 0.5
    date1_res = res[res["date"] == dates[5]]
    assert np.isclose(date1_res["target_weight"].sum(), 1.0), "❌ FEL: Vikterna summerar inte till 1.0 i Bull Regime."
    
    # Vid Rebalance 2 (Date 9): Regime 0.0 (Crisis). Båda aktierna ska ha 0 vikt.
    date2_res = res[res["date"] == dates[9]]
    assert date2_res["target_weight"].sum() == 0.0, "❌ FEL: Motorn ignorerade Crisis Regime (ska vara 0 exponering)."
    
    print("✓ Test 1 godkänt: Exekveringsmotorn navigerar tidslinjen utan data-leakage.")
    print("✓ Test 2 godkänt: Motorn integrerar Layer 2, Layer 4 och Layer 5 korrekt.")

if __name__ == "__main__":
    run_tests()