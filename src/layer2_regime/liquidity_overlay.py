import pandas as pd
import numpy as np

class LiquidityOverlay:
    """
    Layer 2 (Micro): Liquidity Regime Override.
    Mekanisk detektering av spread-expansion över 1.5 sigma.
    """
    def __init__(self, sigma_threshold: float = 1.5, rolling_window: int = 63):
        self.sigma_threshold = sigma_threshold
        self.rolling_window = rolling_window  # 63 handelsdagar = ca 3 månader

    def calculate_stress_regime(self, df: pd.DataFrame, spread_col: str = "BID_ASK_SPREAD_PCT") -> pd.DataFrame:
        """
        Utvärderar spread-kolumnen och returnerar en uppdaterad DataFrame med en boolean mask.
        """
        if spread_col not in df.columns:
            raise ValueError(f"Kritisk data saknas: {spread_col} existerar inte.")
        
        df = df.copy()
        
        # Rullande parametrar laggas med T-1 för att utgöra gårdagens historiska baseline
        roll_mean = df[spread_col].rolling(window=self.rolling_window, min_periods=21).mean().shift(1)
        roll_std = df[spread_col].rolling(window=self.rolling_window, min_periods=21).std().shift(1)
        
        threshold = roll_mean + (self.sigma_threshold * roll_std)
        
        # Generera binär stressindikator. Hantera NaN (början av serien) som False.
        df['LIQUIDITY_STRESS'] = (df[spread_col] > threshold).fillna(False)
        
        return df