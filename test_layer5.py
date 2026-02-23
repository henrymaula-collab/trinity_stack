import numpy as np
import pandas as pd
from src.layer5_portfolio.hrp_clustering import HierarchicalRiskParity
from src.layer5_portfolio.dynamic_vol_target import DynamicVolTargeting

def create_mock_returns(n_assets=4, n_days=252) -> pd.DataFrame:
    """Genererar syntetiska dagliga avkastningar för testning."""
    np.random.seed(42)
    vols = [0.01, 0.02, 0.015, 0.03]
    data = {f"TICK_{i}": np.random.normal(0, vols[i], n_days) for i in range(n_assets)}
    return pd.DataFrame(data)


def run_tests():
    print("Initierar test för Layer 5: Portfolio Construction...\n")

    returns_df = create_mock_returns()
    cov_matrix = returns_df.cov()
    currency = pd.Series(
        ["EUR", "SEK", "EUR", "SEK"],
        index=["TICK_0", "TICK_1", "TICK_2", "TICK_3"],
    )
    hist_vols = returns_df.std() * np.sqrt(252)

    # --- TEST 1: Hierarchical Risk Parity (Currency Isolation) ---
    hrp = HierarchicalRiskParity()
    hrp_weights = hrp.allocate(returns_df, currency)
    assert np.isclose(hrp_weights.sum(), 1.0), "❌ FEL: HRP-vikterna summerar inte till 1.0."
    assert all(hrp_weights >= 0), "❌ FEL: Negativa vikter upptäcktes i HRP."
    print("✓ Test 1 godkänt: HRP med currency isolation konvergerar.")

    # --- TEST 2: Dynamic Volatility Targeting & Regime Scaling ---
    vol_targeter = DynamicVolTargeting()
    nlp_multipliers = pd.Series(1.0, index=hrp_weights.index)
    nlp_multipliers["TICK_3"] = 0.5
    target_vol = 0.15

    w_bull = vol_targeter.apply_targets(
        hrp_weights,
        cov_matrix,
        nlp_multipliers,
        regime_exposure=1.0,
        historical_volatilities=hist_vols,
        target_vol_annual=target_vol,
    )
    var_bull = np.dot(w_bull.values, np.dot(cov_matrix.loc[w_bull.index, w_bull.index].values, w_bull.values))
    vol_bull = np.sqrt(var_bull * 252)
    assert np.isclose(vol_bull, target_vol), f"❌ FEL: Bull-volatilitet blev {vol_bull:.4f}, förväntat {target_vol}"

    w_neutral = vol_targeter.apply_targets(
        hrp_weights,
        cov_matrix,
        nlp_multipliers,
        regime_exposure=0.5,
        historical_volatilities=hist_vols,
        target_vol_annual=target_vol,
    )
    var_neutral = np.dot(w_neutral.values, np.dot(cov_matrix.loc[w_neutral.index, w_neutral.index].values, w_neutral.values))
    vol_neutral = np.sqrt(var_neutral * 252)
    assert np.isclose(vol_neutral, target_vol * 0.5), f"❌ FEL: Neutral-volatilitet blev {vol_neutral:.4f}"

    w_crisis = vol_targeter.apply_targets(
        hrp_weights,
        cov_matrix,
        nlp_multipliers,
        regime_exposure=0.0,
        historical_volatilities=hist_vols,
    )
    assert w_crisis.sum() == 0.0, "❌ FEL: Portföljen likviderades inte i Crisis Regime."

    # Drawdown overlay
    w_dd = vol_targeter.apply_targets(
        hrp_weights,
        cov_matrix,
        nlp_multipliers,
        regime_exposure=1.0,
        historical_volatilities=hist_vols,
        portfolio_drawdown=-0.12,
    )
    assert np.isclose(w_dd.abs().sum(), w_bull.abs().sum() * 0.8), "❌ FEL: Drawdown-overlay 80% saknas."

    print("✓ Test 2 godkänt: Dynamisk volatilitetsstyrning, regime och drawdown-overlay.")

    # --- TEST 3: Fail-Fast ---
    try:
        bad_nlp = nlp_multipliers.copy()
        bad_nlp.index = ["A", "B", "C", "D"]
        vol_targeter.apply_targets(
            hrp_weights, cov_matrix, bad_nlp, regime_exposure=1.0,
            historical_volatilities=hist_vols,
        )
        print("❌ FEL: Koden fångade inte Index Mismatch.")
    except ValueError as e:
        assert "Index mismatch" in str(e)
        print("✓ Test 3 godkänt: Fail-Fast blockerar asymmetrisk data.")



if __name__ == "__main__":
    run_tests()