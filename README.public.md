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
| **L1** | Feature engineering | Quality Score, SUE, Amihud, Vol Compression, seasonality features, trend metrics |
| **L2** | Macro HMM + Liquidity overlay | Regime state (multi-state), liquidity stress indicator |
| **L3** | Momentum + LightGBM + IC ensemble | Alpha scores (ranked) |
| **L4** | NLP Sentinel | Risk multipliers (negative event penalties) |
| **L5** | HRP + dynamic vol target | Target weights |
| **L6** | Execution engine + TCA | Limit orders, stop-loss, SQLite logs |

---

## Layer Details (High-Level)

**Layer 1** — PEAD/SUE, quality composite (ROIC, Piotroski, Accruals), Amihud illiquidity, volatility compression, seasonality, trend-shield features.

**Layer 2** — HMM on macro factors; persistence rule for regime stability; liquidity overlay on spread expansion beyond historical norm.

**Layer 3** — Hard filters (quality, spread, volume); cross-sectional momentum (with short-term reversal exclusion); LightGBM on fundamentals and microstructure; IC stability weighting; turnover hysteresis.

**Layer 4** — Transformer-based sentiment on corporate actions; penalty for extreme negative events with exponential recovery decay.

**Layer 5** — Currency-isolated HRP; inverse volatility intra-cluster; dynamic vol target; drawdown overlay.

**Layer 6** — Conviction-scaled limit pricing; Expected Shortfall stop-loss; TCA logging.

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
├── data/
│   ├── raw/                 # Parquet: prices, fundamentals, macro, news
│   └── tca/                 # SQLite TCA log
├── dashboard/
│   └── streamlit_app.py     # TCA dashboard
├── src/
│   ├── layer1_data/         # Feature engineering
│   ├── layer2_regime/       # Macro + liquidity regime
│   ├── layer3_alpha/        # Momentum, LightGBM, ensemble
│   ├── layer4_nlp/          # NLP sentinel
│   ├── layer5_portfolio/    # HRP, vol targeting
│   ├── layer6_execution/    # Order generator, TCA logger
│   └── engine/              # Backtest loop
└── test_*.py                # Layer tests
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
