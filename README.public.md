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
| **L1** | Feature engineering | Quality Score, SUE, Amihud, Vol Compression, toxicity filters, VWAP returns |
| **L2** | Macro HMM + Liquidity overlay + Liquidity decay model | Regime state, liquidity stress; reflexivity (L(t)=L₀·e^(-α·V/ADV)); Reflexivity Sensitivity |
| **L3** | Momentum + LightGBM + IC ensemble + CPCV | Alpha scores (ranked) |
| **L4** | NLP Sentinel | Risk multipliers |
| **L5** | HRP + dynamic vol target + capacity penalty + FX hedging | Target weights |
| **L6** | Execution engine + TCA + fill-probability + Reflexivity Circuit Breaker + Live paper trading | Limit orders; implementation_shortfall.db; capital scaling pause |

---

## Layer Details (High-Level)

**Layer 1** — PEAD/SUE, quality composite, Amihud, volatility compression, toxicity filters (toxicity_block, toxic_print_day), VWAP-based returns.

**Layer 2** — HMM on macro; liquidity overlay; liquidity decay (L(t)=L₀·e^(-α·V/ADV)); Reflexive Alpha Decay; Self-Impact; Predatory Alert; Reflexivity Sensitivity report.

**Layer 3** — Hard filters; momentum + LightGBM; IC stability weighting; turnover hysteresis; CPCV.

**Layer 4** — Transformer-based sentiment; penalty for extreme negative events; exponential recovery.

**Layer 5** — Currency-isolated HRP; dynamic vol target; drawdown overlay; capacity penalty; FX hedging report.

**Layer 6** — Conviction-scaled limit; pessimistic execution (T+1 VWAP); FillProbabilityModel; Reflexivity Circuit Breaker; live paper trading + implementation_shortfall.db. No mechanical market stop-losses.

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
├── research/                # 01–08 scripts + _shared
├── docs/
│   ├── BLOOMBERG_BLPAPI_DATA_INSTRUCTIONS.md
│   ├── data1_BLOOMBERG.R
│   ├── data2_BLOOMBERG.R
│   └── data_CAPITAL_IQ.R
├── data/raw/, data/tca/
├── dashboard/
├── src/layer1_data/, layer2_regime/, layer3_alpha/, layer4_nlp/, layer5_portfolio/, layer6_execution/
└── engine/
```

---

## Data

Place parquet files in `data/raw/` with standard market data schemas (price, volume, spreads, fundamentals, macro series, news). If files are missing, the pipeline generates mock data.

---

## Configuration

All thresholds, windows, and parameters are configurable at instantiation. The system is designed for institutional production use with sensible defaults.

---

## License

All Rights Reserved
