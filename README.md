# Trinity Stack — Portfolio Management System

Institutional-grade monthly rebalanced trading system for Nordic Small Caps. Six-layer architecture with strict point-in-time semantics, no look-ahead bias, and fail-fast validation.

---

## Features

- **Point-in-time safe** — All features use strictly past data; fundamental data lagged appropriately
- **No survivorship bias** — Universe includes delisted securities; monthly snapshots reflect true tradable set
- **Reproducible** — Fixed global random seed; deterministic ML and clustering
- **Fail-fast** — Raises on missing data, NaN propagation, or dimension mismatch

---

## Architecture

```
Layer 1: Data & Microstructure   → Feature matrix (PiT safe)
Layer 2: Regime Detection        → Macro + liquidity regime
Layer 3: Alpha Generation        → Momentum + LightGBM ensemble
Layer 4: Event-Driven NLP        → Position sizing penalties
Layer 5: Portfolio Construction  → HRP + vol targeting
Layer 6: Execution & TCA         → Orders + SQLite logging
```

| Layer | Module | Output |
|-------|--------|--------|
| **L1** | Feature engineering | Quality Score, SUE, Amihud, Vol Compression, toxicity filters (toxicity_block, toxic_print_day), insider cluster, VWAP-based returns |
| **L2** | Macro HMM + Liquidity overlay + Liquidity decay model | Regime state, liquidity stress; L(t)=L₀·e^(-α·V/ADV); Reflexive Alpha Decay; Endogenous Risk / Predatory Alert; Reflexivity Sensitivity report |
| **L3** | Momentum + LightGBM + IC ensemble + CPCV | Alpha scores (ranked), Deflated Sharpe |
| **L4** | NLP Sentinel | Risk multipliers (negative event penalties) |
| **L5** | HRP + dynamic vol target + capacity penalty + FX hedging | Target weights; FX hedging report (Unhedged / Always / Regime) |
| **L6** | Execution engine + TCA + fill-probability model + Reflexivity Circuit Breaker + Live paper trading | Limit orders; pessimistic execution (T+1 VWAP, step-function impact); Reflexivity Circuit Breaker (spread >50% → Toxic Flow); implementation_shortfall.db; capital scaling pause |

---

## Layer Details (High-Level)

**Layer 1** — PEAD/SUE, quality composite (Z(ROIC) - Z(Dilution)), Amihud illiquidity, volatility compression, seasonality, trend-shield. Toxicity filters: zero turnover 5d → block 5d; VWAP missing or >5% deviance from PX_LAST → toxic_print_day. VWAP-based returns when available. Insider cluster signal (optional).

**Layer 2** — Macro HMM (Student-t) on V2TX, Breadth, Rates; liquidity overlay (spread >1.5σ). Liquidity decay model: L(t)=L₀·e^(-α·V/ADV), α from bid/ask bounce; Reflexive Alpha Decay (participation >5%); Self-Impact metric; Predatory Alert >15%; Reflexivity Sensitivity (break-even α).

**Layer 3** — Hard filters (quality, spread, volume); cross-sectional momentum (short-term reversal excluded); LightGBM on fundamentals/microstructure; IC stability weighting [0.3, 0.7]; turnover hysteresis; CPCV for Deflated Sharpe.

**Layer 4** — Transformer-based sentiment on corporate actions; penalty for extreme negative events; exponential recovery decay over 20 days.

**Layer 5** — Currency-isolated HRP; inverse volatility intra-cluster; dynamic vol target (80% of 5y median); drawdown overlay (-10% → cut 20%); capacity penalty; FX hedging report (Unhedged / Always Hedged / Regime Hedged).

**Layer 6** — Conviction-scaled limit pricing; pessimistic execution (T+1 VWAP, step-function impact, execution collapse >15% participation); FillProbabilityModel (participation cap 5%, spill-over); Reflexivity Circuit Breaker (spread widening >50% → Toxic Flow, cancel remainder); live paper trading + implementation_shortfall.db; capital scaling pause when realized slippage > simulated.

---

## Installation

```bash
git clone https://github.com/your-org/trinity_stack.git
cd trinity_stack

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Quick Start

### Run full pipeline

```bash
python run_pipeline.py
```

Loads data from `data/raw/` (or generates mock data if missing), runs all six layers, and logs theoretical trades to SQLite.

### Launch TCA dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

---

## Project Structure

```
trinity_stack/
├── run_pipeline.py          # Master orchestrator (L1–L6)
├── requirements.txt
├── research/                # Academic hypothesis testing (isolated)
│   ├── 01_ablation_study.py
│   ├── 02_market_impact.py
│   ├── 03_statistical_robustness.py
│   ├── 04_champion_vs_challenger.py
│   ├── 05_tearsheet_and_benchmarks.py
│   ├── 06_marginal_contribution.py
│   ├── 07_fragility_and_capacity.py
│   ├── 08_factor_neutrality.py
│   ├── _shared.py
│   └── README.md
├── docs/
│   ├── SYSTEM_ARCHITECTURE_AND_VALIDATION.md
│   ├── BLOOMBERG_BLPAPI_DATA_INSTRUCTIONS.md
│   ├── data1_BLOOMBERG.R    # Prices + FX (PiT universe, session-safe batches)
│   ├── data2_BLOOMBERG.R    # Fundamentals + Macro + News (PiT bdh only, Dilution)
│   └── data_CAPITAL_IQ.R    # Supply chain + Debt wall (optional)
├── data/
│   ├── raw/                 # Parquet: prices, fundamentals, macro, news
│   └── tca/                 # SQLite TCA + implementation_shortfall.db
├── dashboard/
│   └── streamlit_app.py     # TCA dashboard
├── src/
│   ├── layer1_data/         # Feature engineering, toxicity filters
│   ├── layer2_regime/       # HMM, liquidity overlay, liquidity_decay_model
│   ├── layer3_alpha/        # Momentum, LightGBM, IC ensemble, cpcv
│   ├── layer4_nlp/          # NLP sentinel
│   ├── layer5_portfolio/    # HRP, vol targeting, fx_hedging
│   ├── layer6_execution/    # Order generator, TCA, fill-probability, reflexivity circuit breaker, live_paper_trading
│   └── engine/              # Backtest loop
└── test_*.py                # Layer tests
```

---

## Data

Place parquet files in `data/raw/` with standard market data schemas (price, volume, spreads, fundamentals, macro series, news). If files are missing, the pipeline generates mock data.

**Bloomberg/CapIQ scripts:** Run `docs/data1_BLOOMBERG.R` first (prices + FX), then `docs/data2_BLOOMBERG.R` (fundamentals + macro). Data1 uses session-timestamped batch files to avoid overwriting when resuming after quota limits. Data2 uses PiT fundamentals via `bdh()` only; Dilution computed from EQY_SH_OUT lag 4.

---

## Configuration

All thresholds, windows, and parameters are configurable at instantiation. The system is designed for institutional production use with sensible defaults.

---

## License

All Rights Reserved
