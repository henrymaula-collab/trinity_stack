import pandas as pd
import numpy as np
from pathlib import Path
from src.layer1_data.triple_lag_engine import TripleLagEngine

def create_synthetic_data(file_path: Path, corrupt_dates: bool = False):
    """Skapar en syntetisk tidslinje med EPS och prisdata."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="B")
    
    if corrupt_dates:
        dates = list(dates)
        # Byter plats på två datum för att simulera korrupt data
        dates[5], dates[6] = dates[6], dates[5]

    df = pd.DataFrame({
        "date": dates,
        "PX_LAST": np.linspace(100, 150, 100),
        "PX_TURN_OVER": np.random.uniform(1000, 50000, 100),
        "ACTUAL_EPS": np.linspace(1.0, 2.0, 100),
        "CONSENSUS_EPS": np.linspace(0.9, 1.9, 100),
        "ROIC": np.random.uniform(0.05, 0.15, 100)
    })
    
    df.to_parquet(file_path, index=False)

def run_tests():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    clean_file = raw_dir / "TEST_CLEAN.parquet"
    corrupt_file = raw_dir / "TEST_CORRUPT.parquet"
    
    engine = TripleLagEngine()

    print("Test 1: Verifierar Triple-Lag Integrity...")
    create_synthetic_data(clean_file, corrupt_dates=False)
    df_clean = pd.read_parquet(clean_file)
    df_processed = engine.process_asset(df_clean, "TEST_CLEAN")
    
    # Pris från dag T, ROIC och EPS från dag T-2
    assert "ROIC_lag2" in df_processed.columns, "ROIC laggades inte."
    assert "ROIC" not in df_processed.columns, "Ursprunglig ROIC raderades inte (Dataläcka)."
    print("✓ Test 1 godkänt. T+2 lag isolerad.")

    print("Test 2: Verifierar Fail-Fast Monotonicity...")
    create_synthetic_data(corrupt_file, corrupt_dates=True)
    df_corrupt = pd.read_parquet(corrupt_file)
    
    try:
        engine.process_asset(df_corrupt, "TEST_CORRUPT")
        print("❌ Test 2 misslyckades. Motorn accepterade korrupta datum.")
    except ValueError as e:
        assert "Dates are not monotonically increasing" in str(e)
        print("✓ Test 2 godkänt. Motorn kraschade korrekt vid korrupta datum.")

    # Städning
    clean_file.unlink()
    corrupt_file.unlink()

if __name__ == "__main__":
    run_tests()
