import numpy as np
import pandas as pd
from src.layer3_alpha.ic_weighted_ensemble import AlphaEnsemble
from src.layer3_alpha.hysteresis_turnover import SoftDecayHysteresis

def create_mock_layer3_data(n_dates=80, n_tickers=20):
    """
    Genererar 80 dagars data (behövs >63 för att komma förbi burn-in perioden).
    """
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
    records = []
    
    for d in dates:
        for t in range(n_tickers):
            records.append({
                "date": d,
                "ticker": f"TICK_{t}",
                "Quality_Score": np.random.uniform(0, 100),
                "BID_ASK_SPREAD_PCT": np.random.uniform(0.005, 0.05), # Medvetet > 0.03 för att testa filter
                "PX_VOLUME": np.random.choice([0, 1000, 5000]),       # Medvetet nollor för att testa filter
                "signal_lgbm": np.random.normal(0, 1),
                "signal_mom": np.random.normal(0, 1),
                "forward_return": np.random.normal(0, 0.02)
            })
    return pd.DataFrame(records)

def run_tests():
    print("Skapar syntetisk marknadsdata för Layer 3...")
    df = create_mock_layer3_data()

    # --- TEST 1: Alpha Ensemble (IC Weights & Filters) ---
    print("\nTest 1: Validerar AlphaEnsemble (Filters & Weights)...")
    ensemble = AlphaEnsemble(rolling_days=10, quality_bottom_pct=0.25, spread_max=0.03)
    
    res = ensemble.fit_transform(df)
    
    # 1.1 Validera Hard Filters
    assert res["BID_ASK_SPREAD_PCT"].max() <= 0.03, "❌ FEL: Spread-filtret släppte igenom illikvida aktier."
    assert res["PX_VOLUME"].min() > 0, "❌ FEL: Volym-filtret släppte igenom otillgängliga aktier."
    
    # 1.2 Validera Boundaries [0.3, 0.7] (Kollar raderna efter burn-in perioden på 63 dgr)
    valid_weights = res.dropna(subset=["w_mom", "w_lgbm"])
    if not valid_weights.empty:
        assert valid_weights["w_mom"].between(0.29, 0.71).all(), "❌ FEL: Momentum-vikten bröt [0.3, 0.7] gränsen."
        assert valid_weights["w_lgbm"].between(0.29, 0.71).all(), "❌ FEL: LightGBM-vikten bröt [0.3, 0.7] gränsen."
    
    print("✓ Test 1 godkänt: Koden filtrerar skräp och håller modellvikterna inom institutionella ramar.")


    # --- TEST 2: Soft Decay Hysteresis (Turnover Control) ---
    print("\nTest 2: Validerar Soft Decay Hysteresis...")
    hysteresis = SoftDecayHysteresis(decay_days=60)
    
    df_hyst = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "days_held": [1, 30, 65],
        "base_sell": [0.10, 0.10, 0.10] # Bas-tröskel för att sälja
    })
    
    res_hyst = hysteresis.apply(df_hyst, days_held_col="days_held", base_sell_threshold_col="base_sell", output_col="adj_sell")
    
    # Dag 1: Straff = 100%. adj_sell bör vara 0 (vi vägrar sälja pga transaktionskostnad)
    assert np.isclose(res_hyst.loc[0, "adj_sell"], 0.0), "❌ FEL: Hysteresis släppte igenom dag 1-försäljning."
    # Dag 65: Straff = 0%. adj_sell bör vara tillbaka på 0.10
    assert np.isclose(res_hyst.loc[2, "adj_sell"], 0.10), "❌ FEL: Hysteresis straffar en aktie efter dag 60."
    
    print("✓ Test 2 godkänt: Omsättningströgheten fungerar. Systemet kommer ej agera bruskänsligt.")

if __name__ == "__main__":
    run_tests()