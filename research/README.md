# Research — Academic Hypothesis Testing

Isolated research environment for validating strategy robustness. **Does not modify `src/` or `run_pipeline.py`.** Imports production code and manipulates inputs/execution to gather statistical evidence.

## Scripts

| Script | Purpose |
|--------|---------|
| `01_ablation_study.py` | Systematic layer removal. Sharpe, CAGR, Max DD per add-layer run. |
| `02_market_impact.py` | Square Root Law of Market Impact. Synthetic transaction costs on equity curve. |
| `03_statistical_robustness.py` | Block bootstrap. Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR). |

## Run

```bash
cd trinity_stack
python research/01_ablation_study.py
python research/02_market_impact.py
python research/03_statistical_robustness.py
```

## Output

- **01**: `research/ablation_results.csv` — metrics table
- **02/03**: Console output
