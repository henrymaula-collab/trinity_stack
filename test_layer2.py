import numpy as np
import pandas as pd
from src.layer2_regime.hmm_macro_student_t import MacroRegimeDetector, CRISIS_STATE

def create_mock_macro_data(n_samples=100) -> pd.DataFrame:
    """Genererar syntetisk makrodata."""
    np.random.seed(42)
    return pd.DataFrame({
        "V2TX": np.random.normal(20, 5, n_samples),
        "Breadth": np.random.normal(0, 1, n_samples),
        "Rates": np.random.normal(3, 0.5, n_samples)
    })

def run_tests():
    detector = MacroRegimeDetector(n_states=3, persistence_days=3, confidence_threshold=0.70)
    df_macro = create_mock_macro_data(100)
    
    # 1. Test Fit & Basic Predict
    detector.fit(df_macro)
    assert detector._fitted, "Modellen markerades inte som fitted."
    
    # Baseline prediction utan spread-data
    base_regime = detector.predict_regime(df_macro)
    assert base_regime in [0, 1, 2], f"Ogiltig regim returnerad: {base_regime}"
    print("✓ Test 1: HMM Fit och grundläggande prediktion fungerar.")

    # 2. Test Liquidity Override (Crisis Forcing)
    mu_spread = 2.0
    sigma_spread = 0.5
    # Spread på 3.0 är > (2.0 + 1.5 * 0.5 = 2.75)
    crisis_spread = 3.0 
    
    override_regime = detector.predict_regime(
        df_macro, 
        current_spread=crisis_spread, 
        historical_spread_mean=mu_spread, 
        historical_spread_std=sigma_spread
    )
    assert override_regime == CRISIS_STATE, "Liquidity Override tvingade inte fram CRISIS_STATE (2)."
    print("✓ Test 2: Liquidity Stress Override fungerar matematiskt.")

    # 3. Test Fail-Fast Validering
    df_corrupt = df_macro.copy()
    df_corrupt.loc[0, "V2TX"] = np.nan
    try:
        detector.predict_regime(df_corrupt)
        print("❌ Test 3 Misslyckades: Modellen accepterade NaN-värden.")
    except ValueError as e:
        assert "NaN" in str(e)
        print("✓ Test 3: Fail-Fast blockerar korrupt data (NaN/Inf).")

if __name__ == "__main__":
    run_tests()