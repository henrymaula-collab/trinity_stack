# Research — Academic Hypothesis Testing

Isolated research environment for validating strategy robustness. **Does not modify `src/` or `run_pipeline.py`.** Imports production code and manipulates inputs/execution to gather statistical evidence.

## Scripts

| Script | Purpose |
|--------|---------|
| `01_ablation_study.py` | Systematic layer removal. Sharpe, CAGR, Max DD per add-layer run. |
| `02_market_impact.py` | Square Root Law of Market Impact. Synthetic transaction costs on equity curve. |
| `03_statistical_robustness.py` | Block bootstrap. Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR). |
| `04_champion_vs_challenger.py` | Champion (Full Stack) vs Challenger (Trinity Lite). Ergodicity & survival metrics. |
| `05_tearsheet_and_benchmarks.py` | Tearsheet visualization. Naive 1/N benchmark, equity curves, drawdowns, stats table. |
| `06_marginal_contribution.py` | **Shapley-Ablation.** φ_k^G, φ_k^ES per lager. Stress-test, redundanscheck. Ergodisk "domstol". |
| `07_fragility_and_capacity.py` | **Fragility & Capacity.** Granger/event-study, market impact, jitter/fragility, liquidity freeze, drawdown-constrained Kelly, HMM regime fragility. |
| `08_factor_neutrality.py` | **Factor Neutrality.** Falsification: Sharpe drop >30% when illiquid decile excluded → Liquidity Harvesting (reject as genuine alpha). |

## Run

```bash
cd trinity_stack
python research/01_ablation_study.py
python research/02_market_impact.py
python research/03_statistical_robustness.py
python research/04_champion_vs_challenger.py   # Run before 05
python research/05_tearsheet_and_benchmarks.py # Requires matplotlib, seaborn
python research/06_marginal_contribution.py   # Shapley-Ablation
python research/07_fragility_and_capacity.py  # Fragility & Capacity
python research/08_factor_neutrality.py       # Factor Neutrality
```

**Shared utilities:** `_shared.py` provides data loading, backtest harness, `compute_metrics()`. CPCV (`src/layer3_alpha/cpcv.py`) and FX hedging (`src/layer5_portfolio/fx_hedging.py`) are used by 07 and pipeline.

## Output

- **01**: `research/ablation_results.csv` — metrics table
- **02/03**: Console output
- **04**: `research/champion_vs_challenger.csv`, `research/strategy_returns.parquet`
- **05**: `research/strategy_tearsheet.pdf` — equity curves + underwater drawdowns
- **06**: `research/shapley_results.csv` (CI, Stability, Regime-Shapley, VERDICT_DELETE), `research/bootstrap_results.parquet` (bootstrap-fördelningar)
- **07**: Konsolutskrift — Granger/event-study, market impact & fill, jitter, liquidity freeze, drawdown-constrained Kelly, HMM regime fragility
- **08**: Konsolutskrift — Sharpe full vs illiquid-decile-excluded; Liquidity Harvesting verdict
