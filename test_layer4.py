import numpy as np
from src.layer4_nlp.xlm_roberta_sentinel import NLPSentinel

def run_tests():
    print("Initierar NLP Sentinel (laddar modell, tar ca 10-30 sekunder)...")
    sentinel1 = NLPSentinel()
    
    # Test 1: Singleton Memory Check
    sentinel2 = NLPSentinel()
    assert sentinel1._pipeline is sentinel2._pipeline, "❌ FEL: Modellen laddades in två gånger i minnet."
    print("✓ Test 1 godkänt: Singleton förhindrar minnesläckage.")

    # Test 2: Edge Cases (Tomma strängar & NaNs)
    assert sentinel1.get_risk_multiplier(None) == 1.0
    assert sentinel1.get_risk_multiplier(np.nan) == 1.0
    assert sentinel1.get_risk_multiplier("   ") == 1.0
    print("✓ Test 2 godkänt: Felaktig indata kraschar inte pipelinen.")

    # Test 3: Critical Keyword Override
    text_keyword = "The company announced a massive share dilution today."
    assert sentinel1.get_risk_multiplier(text_keyword) == 0.5, "❌ FEL: Nyckelord (dilution) utlöste inte risk_multiplier 0.5."
    print("✓ Test 3 godkänt: Nyckelord halverar positionen direkt.")

    # Test 4: Sentiment Analysis (Negative)
    text_negative = "Earnings completely collapsed this quarter, missing all analyst expectations by a wide margin."
    assert sentinel1.get_risk_multiplier(text_negative) == 0.5, "❌ FEL: Starkt negativ text ignorerades."
    print("✓ Test 4 godkänt: Negativt sentiment fångas upp av FinBERT.")

    # Test 5: Sentiment Analysis (Positive/Neutral)
    text_positive = "Revenue grew by 20% and margins expanded."
    assert sentinel1.get_risk_multiplier(text_positive) == 1.0, "❌ FEL: Positiv text straffades."
    print("✓ Test 5 godkänt: Positiv data ger risk_multiplier 1.0.")

if __name__ == "__main__":
    run_tests()