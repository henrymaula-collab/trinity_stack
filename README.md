# Trinity Stack v9.0

Institutional-grade monthly rebalanced trading system for Nordic Small Caps (OMXH/OMXS). Six-layer architecture with strict point-in-time semantics, no look-ahead bias, and fail-fast validation.

---

## Features

- **Point-in-time safe** — All features use strictly past data; fundamental data lagged T+2 minimum
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
| **L1** | Feature engineering | Quality Score, SUE, Amihud, Vol Compression, January effect, SMA distance |
| **L2** | Macro HMM + Liquidity overlay | Regime state (0/1/2), liquidity stress |
| **L3** | Momentum + LightGBM + IC ensemble | Alpha scores (ranked) |
| **L4** | NLP Sentinel | Risk multipliers (negative event penalties) |
| **L5** | HRP + dynamic vol target | Target weights |
| **L6** | Execution engine + TCA | Limit orders, stop-loss, SQLite logs |

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

Loads data from `data/raw/` (or generates mock data if missing), runs all six layers, and logs theoretical trades to `data/tca/tca.db`.

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

Place parquet files in `data/raw/`:

| File | Columns |
|------|---------|
| `prices.parquet` | date, ticker, PX_LAST, PX_VOLUME, BID_ASK_SPREAD_PCT |
| `fundamentals.parquet` | report_date, ticker, ACTUAL_EPS, CONSENSUS_EPS, EPS_STD, ROIC, Piotroski_F, Accruals |
| `macro.parquet` | index=date, V2TX, Breadth, Rates |
| `news.parquet` | ticker, date, text |

If files are missing, the pipeline generates mock data.

---

## Configuration

- **Random seed:** `GLOBAL_SEED = 42`
- **Regime:** 3-day persistence; 70% confidence threshold
- **Vol target:** 80% of 5-year median realized vol
- **Drawdown cut:** −10% DD → reduce gross exposure by 20%

---

## License

All Rights Reserved
