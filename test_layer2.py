import numpy as np
import pandas as pd
from src.layer2_regime.hmm_macro_student_t import MacroRegimeDetector

def create_mock_macro_data(n_samples=100) -> pd.DataFrame:
    """Genererar syntetisk makrodata."""
    np.random.seed(42)
    return pd.DataFrame({
        "V2TX": np.random.normal(20, 5, n_samples),
        "Breadth": np.random.normal(0, 1, n_samples),
        "Rates": np.random.normal(3, 0.5, n_samples),
    })

def run_tests():
    detector = MacroRegimeDetector(n_states=3, persistence_days=3, confidence_threshold=0.70)
    df_macro = create_mock_macro_data(100)

    # 1. Test Fit & Basic Predict (continuous regime exposure 0.0–1.0)
    detector.fit(df_macro)
    assert detector._fitted, "Modellen markerades inte som fitted."

    regime_exposure = detector.predict_regime(df_macro)
    assert isinstance(regime_exposure, (int, float)), f"Förväntat float, fick {type(regime_exposure)}"
    assert 0.0 <= regime_exposure <= 1.0, f"Regime exposure måste vara i [0, 1], fick {regime_exposure}"
    print("✓ Test 1: HMM Fit och kontinuerlig regime exposure (predict_proba) fungerar.")

    # 2. Test Fail-Fast Validering
    df_corrupt = df_macro.copy()
    df_corrupt.loc[0, "V2TX"] = np.nan
    try:
        detector.predict_regime(df_corrupt)
        print("❌ Test 2 Misslyckades: Modellen accepterade NaN-värden.")
    except ValueError as e:
        assert "NaN" in str(e)
        print("✓ Test 2: Fail-Fast blockerar korrupt data (NaN/Inf).")

if __name__ == "__main__":
    run_tests()
