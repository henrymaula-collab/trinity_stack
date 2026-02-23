# Meta-Portfolio — Master Allocator

Tre strategier, ett samlat ordereflöde. Källkoden slås inte ihop; varje strategi förblir i sin egen mapp.

## Roller

| Strategi | Kapital | Funktion |
|----------|---------|----------|
| **Trinity Stack** (Kärnan) | 80% | Systematisk basportfölj. Högkvalitativt momentum, makroskydd. |
| **Supply-Chain Lead-Lag** (Spjutet) | 0–20% (dynamisk) | 4% per aktiv ticker, max 20%. Taktiska bet med kort horisont. |
| **Debt Wall** (Skölden) | Veto | Blockering. Nollar positioner i bolag med skuldförfall. |

## Användning

```python
from meta.master_execution import MasterAllocator
import pandas as pd

# Trinity: target weights från run_pipeline / BacktestEngine
trinity_weights = pd.Series({"T_01": 0.15, "T_02": 0.12, ...})  # eller callable

# Supply-Chain: taktiska vikter från build_trades (conviction → weight)
supply_weights = pd.Series({"SUPP3": 0.08, "SUPP7": 0.05, ...})  # eller callable

# Debt Wall: blacklist från generate_signals (trigger=True)
blacklist = {"T_05", "SUPP2"}  # eller callable

allocator = MasterAllocator()
final_weights = allocator.execute(
    trinity_weights,
    supply_chain_weights=supply_weights,
    debt_wall_blacklist=blacklist,
    output_path="data/orders/nordnet_orders.csv",
    # Kill-switch (valfritt, från live/TCA):
    # rolling_60d_hit_rate=0.52,
    # median_spread_zscore=1.2,
    # live_slippage_bps=15,
    # backtest_slippage_bps=8,
)
```

## Wiring från varje strategi

### Trinity Stack
- `run_pipeline` → `backtest_result` → sista rebalance-datum → `final_weights = backtest_result[date==last].set_index("ticker")["target_weight"]`

### Supply-Chain
- `build_trades` → aktiva positioner med `conviction` → normalisera till vikter (sum=1)

### Debt Wall
- `generate_signals` → `signals[signals["trigger"]]["ticker"].tolist()` → `set(...)` = blacklist

## Kill-Switch

När live-metriker anges till `execute()`: edge collapse (hit_rate < 0.45) → HALT; liquidity stress (spread_zscore > 2) → REDUCE 50%; model deviation (live_slippage > 2×backtest) → HALT.

## Struktur

- `meta/master_execution.py` — MasterAllocator, merge_weights, apply_blacklist, kill-switch, to_nordnet_order_format
- Varje strategi förblir i `src/`, `strategy_supply_chain/`, `strategy_debt_wall/`
- Om Supply-Chain stängs av: `supply_chain_weights=pd.Series()` → tactical_share=0 automatiskt
