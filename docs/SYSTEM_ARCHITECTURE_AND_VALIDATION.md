# Trinity Stack — Systemarkitektur och Validering

Fördjupad dokumentation av hela systemet för förståelse och validering med olika agenter.

---

## 1. Översikt

Systemet består av **tre separata strategier** som agerar tillsammans via ett **meta-lager**, men vars källkod förblir strikt isolerad. Detta möjliggör oberoende forskning, optimering och stresstest — om en strategi slutar fungera kan den stängas av utan att påverka de andra.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        META-PORTFOLIO (Master Allocator)                 │
│  Slår ihop vikter, applicerar blacklist, skriver orderefil till Nordnet  │
└─────────────────────────────────────────────────────────────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ TRINITY STACK   │  │ SUPPLY-CHAIN    │  │ DEBT WALL       │
│ (Kärnan) 80%   │  │ (Spjutet) 20%   │  │ (Skölden) Veto  │
│                 │  │                 │  │                 │
│ Target Weights  │  │ Tactical Weights│  │ Blacklist       │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## 2. Trinity Stack (Kärnan) — 6 Lagers Arkitektur

**Funktion:** Systematisk basportfölj. Skannar marknaden dagligen, bygger bred portfölj av högkvalitativt momentum, skyddar mot makrokrascher.

**Kapitalandel:** 80 % (konfigurerbar)

### 2.1 Exekveringsordning

```
Layer 1 → Layer 2 → Layer 3 → Layer 4 → Layer 5 → Layer 6
```

Ingen layer får importera logik från en annan layer; endast output (DataFrame/Parquet) får flöda vidare.

### 2.2 Layer 1: Data & Microstructure (`src/layer1_data/`)

**Output:** Ren månatlig feature-matris (point-in-time säker).

**Huvudkomponenter:**
- `FeatureEngineer` — bygger features från råpriser och fundamentals
- `TripleLagEngine` — T+2 lag på fundamentaldata (eliminerar look-ahead bias)

**Features:**
- **PEAD/SUE:** `(Actual EPS - Consensus EPS) / std(EPS Forecast Error)`
- **Quality Composite:** `Z(ROIC) - Z(Dilution)` där Dilution = EQY_SH_OUT_t / EQY_SH_OUT_{t-1y}
- **Amihud Illiquidity:** `|Return| / Volume` (likviditetsmått)
- **Cross-Sectional Volatility Compression**

**Kritiskt:** Alla rolling-statistik använder `.shift(1)`. Fundamentaldata lagras minst T+2 handelsdagar.

### 2.3 Layer 2: Regime Detection (`src/layer2_regime/`)

**Output:** Kontinuerlig regime_exposure (float 0.0–1.0).

**Huvudkomponenter:**
- `MacroRegimeDetector` — Gaussian HMM på V2TX, Breadth, Rates. Använder `predict_proba()` i stället för diskret `predict()`. Kontinuerlig exposure = dot product av state-sannolikheter och target exposures (Normal=1.0, Uncertain=0.5, Crisis=0.0).
- `LiquidityOverlay` — Om spread > 1.5σ från historiskt medelvärde → Liquidity Stress

**Continuous Regime Scaling:** Regime exposure är en float 0.0–1.0 som används direkt som skalningsfaktor (inget diskret 0/1/2-steg).

### 2.4 Layer 3: Alpha Generation (`src/layer3_alpha/`)

**Output:** Rankade alphasignaler (cross-sectionell).

**Hard Filters:** Kasta lägsta Quality Score-kvartilen, spread > 3 %, otillräcklig volym.

**Ensemble:** Momentum + LightGBM med IC-stabilitetsviktning. Model weight = IC / std(IC), begränsad till [0.3, 0.7]. Modellen tränas uteslutande via en strictly out-of-sample 'Expanding Walk-Forward Cross-Validation' för att förhindra look-ahead bias och data-läckage mellan tränings- och test-set.

**Soft Decay Hysteresis:** Sell-threshold startar högt på T+1 och fasar ut linjärt över 60 dagar (ingen framtidsrank).

### 2.5 Layer 4: Event-Driven NLP (`src/layer4_nlp/`)

**Output:** Position sizing penalty-multiplicators (0–1).

**Modell:** XLM-RoBERTa (NER) skannar corporate actions/nyheter.

**Logik:** Översta 5 % extrema negativa händelser → maxvikt reduceras med 50 %. Exponentiell återhämtning över 20 handelsdagar.

### 2.6 Layer 5: Portfolio Construction (`src/layer5_portfolio/`)

**Output:** Slutliga målvikter (target weights).

**Metod:** Valuta-isolerad Hierarchical Risk Parity (HRP) med inverse volatility intra-kluster.

**Dynamic Vol Target:** 80 % av 5-årigt median realiserad volatilitet. Skalar ner gross exposure om realiserad > target.

**Drawdown Overlay:** Vid portfölj-DD ≤ -10 % → skär gross exposure med 20 %.

**Capacity Penalty:** När ADV och portfolio_aum anges: `capacity_penalty = min(1, k * ADV / position_size)`. Förhindrar överallokering till illikvida namn.

### 2.7 Layer 6: Execution & TCA (`src/layer6_execution/`, `dashboard/`)

**Output:** Limit orders, SQLite TCA-loggar. **Inga mekaniska market stop-losses.**

**Limit-prissättning (conviction-skalad):**
- Top 5 % Alpha Rank: Limit = MidPrice - (0.25 × Spread)
- Normal likviditet: Limit = MidPrice - (0.40 × Spread)
- Bottom 30 % likviditet: Limit = MidPrice - (0.75 × Spread)

**Risk:** Hanteras ex-ante via positionsgränser, regime-skalär och HRP. I illikvida småbolag orsakar mekaniska market stops gap-down likviditetsfällor — därför är Expected Shortfall/stop-loss borttagen.

**TCA:** Loggar teoretiskt vs fyllt pris i SQLite med timestamp och regime-label. Utökat schema för fill-probability-kalibrering: `order_size_adv`, `spread_pct`, `intraday_vol_proxy`, `time_to_fill_sec`, `fill_ratio`.

**FillProbabilityModel:** Logistisk modell `fill_prob = sigmoid(a - b×order_size_adv - c×spread_pct - d×intraday_vol)`. Kalibreras från historisk TCA när `fill_ratio` observeras.

**Pessimistic Execution Engine:** Survival over theoretical profit. BUY at Ask + 0.1%×vol, SELL at Bid - 0.1%×vol. Top 10% alpha = price taker. Spread stress 2.5× when regime < 0.5 or V2TX > 25. 5% participation cap per day. T+1 execution. Output: `Total_Friction_Cost_BPS`, `Opportunity_Cost_BPS`.

### 2.8 Trinity-dataström

```
data/raw/*.parquet (prices, fundamentals, macro, news)
    → Layer 1: features_df
    → Layer 2: regime_df, liquidity_stress
    → Layer 3: alpha_df
    → Layer 4: nlp_multipliers (in-backtest)
    → Layer 5: target_weights
    → Layer 6: orders + TCA
```

**Körning:** `python run_pipeline.py`

---

## 3. Supply-Chain Lead-Lag (Spjutet)

**Funktion:** Taktiska bet med extremt hög övertygelse, kort horisont (dagar–veckor). Letar specifika, isolerade händelser (kundens earnings surprise) oberoende av marknadstrend.

**Kapitalandel:** Dynamisk 0–20 % (4 % per aktiv supply-chain-ticker, max 20 %)

**Teoretisk grund:** Illikvida nordiska småbolagsleverantörer underreagerar på earnings surprises eller kursrallys hos globala megacap-kunder pga informationsasymmetri och bristande analytikertäckning.

### 3.1 Moduler (`strategy_supply_chain/`)

| Fil | Syfte |
|-----|-------|
| `data_pipeline.py` | Laddar supply-chain-data (date, supplier_ticker, customer_ticker, revenue_dependency_pct), priser, earnings |
| `signal_engine.py` | Cross-asset trigger: Kundens avkastning > 2σ av 60-d vol ELLER earnings surprise > threshold → BUY leverantör |
| `execution_logic.py` | Entry T+1 open, exit 3 dagar före leverantörens earnings eller efter max 20 handelsdagar |
| `backtest_network.py` | Vektorerad backtest: geometrisk avkastning, hit rate |

### 3.2 Signal-logik

- **Trigger 1:** `customer_return > 2 × rolling_60d_volatility` (shifted, ingen look-ahead)
- **Trigger 2:** `customer_eps_surprise > threshold` (om tillgänglig)
- **Conviction:** Skalas med `revenue_dependency_pct / 100`

### 3.3 Output för Master Allocator

Supply-Chain producerar **taktiska target weights** (supplier_ticker → conviction-skalad vikt). Normalisera aktiva positioner till summa 1 innan inmatning till MasterAllocator.

### 3.4 Valideringspunkter

- Kontrollera att ingen `customer_return` eller `customer_vol_60d` använder framtida data (shift(1)).
- Verifiera att entry endast sker T+1 efter signal.
- Verifiera att exit sker ≥3 dagar före supplier earnings.

---

## 4. Debt Wall Arbitrage (Skölden)

**Funktion:** Veto-motor. Behöver inte eget kapital för blankning i long-only-portfölj. Blockerar farliga köp.

**Exempel:** Trinity vill köpa 5 % av ett bolag med starkt momentum. Debt Wall ser att bolaget har massiv skuld som förfaller om 4 månader och tom kassa → blockerar köpet.

### 4.1 Moduler (`strategy_debt_wall/`)

| Fil | Syfte |
|-----|-------|
| `data_pipeline.py` | Laddar skulddatar (date, ticker, market_cap, cash_equivalents, ttm_fcf, next_12m_debt_maturity, short_interest) |
| `arbitrage_engine.py` | Cash Runway Deficit, SHORT/EXCLUDE-signal |
| `cost_of_borrowing.py` | Overlay: om rf >> skuldsränta → högre conviction (refinansiering blir dyr) |
| `backtest_debt_wall.py` | Backtest för shorts: Omega Ratio, CAGR |

### 4.2 Arbitrage-logik

**Cash Runway Deficit:**
```
Deficit = next_12m_debt_maturity - (cash_equivalents + (ttm_fcf × 0.5))
```

**Trigger (SHORT / EXCLUDE):**
- `Deficit > market_cap × 0.10` (implicerar >10 % utspädning för att överleva)
- Maturity < 6 månader (om maturity_date finns)

### 4.3 Output för Master Allocator

Debt Wall producerar en **blacklist**: `set(ticker)` för alla bolag som triggat. MasterAllocator nollar deras vikter.

### 4.4 Valideringspunkter

- Kontrollera att Deficit använder endast point-in-time fundamentaldata.
- Verifiera att blacklist endast innehåller tickers med trigger=True.

---

## 5. Meta-Portfolio (Master Allocator)

**Plats:** `meta/master_execution.py`

### 5.1 Flöde

1. Hämta `trinity_weights` (pd.Series, ticker → weight)
2. Hämta `supply_chain_weights` (pd.Series)
3. Hämta `debt_wall_blacklist` (set[str] eller list[str])
4. Slå ihop: `combined = core_share × trinity + tactical_share × supply_chain`
5. Applicera blacklist: sätt vikt = 0 för alla tickers i blacklist, renormalisera
6. Skriv orderefil till Nordnet (CSV: ticker, target_weight)

### 5.2 Konfiguration

- **Dynamisk taktisk allocation:** `tactical_share` beräknas från antal aktiva supply-chain-signaler: 4 % per ticker, max 20 %. Om inga aktiva signaler → 0 %.
- `core_share` = 1.0 - `tactical_share` (enforced)

### 5.3 Kill-Switch (valfria live-metriker)

När `execute()` anropas med live-mått kan följande regler trigga:

| Regel | Tröskel | Åtgärd |
|-------|---------|--------|
| Edge collapse | `rolling_60d_hit_rate < 0.45` | HALT — returnera nollvikter |
| Liquidity stress | `median_spread_zscore > 2` | REDUCED — skär gross exposure 50 % |
| Model deviation | `live_slippage_bps > 2 × backtest_slippage_bps` | HALT — returnera nollvikter |

Parametrar: `rolling_60d_hit_rate`, `median_spread_zscore`, `live_slippage_bps`, `backtest_slippage_bps`. Om ej angivna → ingen kill-switch-utvärdering.

### 5.4 Wiring

| Strategi | Output-format | Hur man hämtar |
|----------|---------------|----------------|
| Trinity | pd.Series (ticker → weight) | Sista rebalance-datum från `backtest_result`: `backtest_result[date==last].set_index("ticker")["target_weight"]` |
| Supply-Chain | pd.Series | Från `build_trades`: konvertera conviction till vikter, normalisera till sum=1 |
| Debt Wall | set[str] | `set(signals[signals["trigger"]]["ticker"])` |

---

## 6. Validering med olika agenter

### 6.1 Checklista för Look-Ahead Bias

En validerande agent bör söka efter:

- [ ] Användning av `.shift(1)` på alla rolling-statistik (returns, volatility, etc.)
- [ ] Ingen feature som använder datum efter observationstidpunkten
- [ ] Fundamentaldata lagrad minst T+2 handelsdagar
- [ ] `reference_date`-validering i data pipelines (raise på framtida datum)

**Grep-exempel:**
```
rg "\.shift\(" --type py
rg "rolling\(" --type py -A 2
```

### 6.2 Checklista för Layer-isolering

- [ ] Ingen `from src.layerX import` i `src.layerY` (X ≠ Y)
- [ ] Ingen import mellan `strategy_supply_chain` och `strategy_debt_wall`
- [ ] `meta/` importerar inte internt av strategierna; tar endast emot pd.Series/set

### 6.3 Checklista för Reproducerbarhet

- [ ] Global random seed (t.ex. 42) i alla ML-/clustering-anrop
- [ ] Inga `np.random` utan `np.random.seed` eller `rng = np.random.default_rng(42)`

### 6.4 Testscenarier för Master Allocator

1. **Endast Trinity:** `supply_chain_weights = pd.Series()`, `tactical_share = 0`
2. **Endast Supply-Chain:** `trinity_weights = pd.Series()`, `core_share = 0` (eller vice versa beroende på tolking)
3. **Blacklist aktiv:** En ticker i både Trinity och blacklist → slutvikt = 0
4. **Tomma inputs:** `trinity_weights` och `supply_chain_weights` båda tomma → förväntat: ValueError

### 6.5 Körbara tester

```bash
# Trinity pipeline
python run_pipeline.py

# Supply-Chain backtest
python -m strategy_supply_chain.backtest_network

# Debt Wall backtest
python -m strategy_debt_wall.backtest_debt_wall
```

### 6.6 Prompt för validerande agent

> "Validera att Trinity Stack, Supply-Chain och Debt Wall följer .cursorrules: ingen look-ahead bias, point-in-time data, layer isolation, fail-fast på NaN/dimension mismatch. Granska varje modul för shift(1) på rolling stats, T+2 på fundamentals, och att meta/master_execution endast konsumerar output utan att importera strategilogik."

---

## 7. Filstruktur (komplett)

```
trinity_stack/
├── run_pipeline.py              # Trinity L1–L6 orkestrering
├── meta/
│   ├── __init__.py
│   ├── master_execution.py      # Master Allocator
│   └── README.md
├── strategy_supply_chain/
│   ├── __init__.py
│   ├── data_pipeline.py
│   ├── signal_engine.py
│   ├── execution_logic.py
│   └── backtest_network.py
├── strategy_debt_wall/
│   ├── __init__.py
│   ├── data_pipeline.py
│   ├── arbitrage_engine.py
│   ├── cost_of_borrowing.py
│   └── backtest_debt_wall.py
├── src/
│   ├── layer1_data/
│   ├── layer2_regime/
│   ├── layer3_alpha/
│   ├── layer4_nlp/
│   ├── layer5_portfolio/
│   ├── layer6_execution/
│   └── engine/
├── dashboard/
│   └── streamlit_app.py
├── data/
│   ├── raw/                     # Parquet: prices, fundamentals, macro, news
│   ├── processed/
│   └── tca/                     # SQLite TCA
├── docs/
│   ├── BLOOMBERG_BLPAPI_DATA_INSTRUCTIONS.md
│   └── SYSTEM_ARCHITECTURE_AND_VALIDATION.md   # denna fil
├── research/
│   ├── 06_marginal_contribution.py   # Shapley-Ablation (ergodisk validering)
│   └── shapley_results.csv           # Output från 06
└── test_*.py
```

---

## 8. Sammanfattning

| Komponent | Roll | Kapital | Output |
|-----------|------|---------|--------|
| Trinity Stack | Kärnan | 80–100 % (rest) | Target weights |
| Supply-Chain | Spjutet | 0–20 % (dynamisk) | Tactical weights |
| Debt Wall | Skölden | Veto | Blacklist |
| Master Allocator | Orkestrering | — | Slutlig orderefil (Nordnet) |

Alla tre strategier kan forskas på, optimeras och stresstestas i isolering. Master Allocator slår ihop deras output på en hög abstraktionsnivå utan att slå ihop källkoden.

---

## 9. Institutional Architecture Fixes (v9.1)

**Tre kritiska fixar:**

1. **Continuous Regime Scaling (Layer 2):** Ersatt diskret `predict()` med `predict_proba()`. Kontinuerlig `regime_exposure_scalar` (0.0–1.0) = dot product av state-sannolikheter och target exposures (Normal=1.0, Uncertain=0.5, Crisis=0.0).
2. **Dynamic Tactical Allocation (Meta):** `tactical_share` är inte längre konstant. Beräknas dynamiskt: 4 % per aktiv supply-chain-ticker, max 20 %. Tom → 0 %. `core_share` = 1.0 - `tactical_share`.
3. **Removal of Mechanical Stop-Loss (Layer 6):** Expected Shortfall och market stop-loss borttagna. Risk hanteras ex-ante via positionsgränser, regime-skalär och HRP. Limit-logik (MidPrice - X × Spread) behålls.

---

## 10. Ergodisk Riskhantering och Shapley-Ablation

**Grundprincip:** I ändligt kapital med ruinrisk är σ (volatilitet) irrelevant om den inte kopplas till G (geometrisk tillväxt). Varje lager är en transformation L_k: w^(k-1) → w^(k). Om transformationen sänker volatiliteten men sänker G ännu mer, köper man falsk trygghet till priset av framtida köpkraft.

**Metrik:** G = E[ln(1+R)] (geometrisk tillväxt) ersätter aritmetiskt medelvärde μ som primär måttstock.

### 10.1 Shapley-värde för marginalbidrag

Varje lagers bidrag φ_k mäts över alla 2^n kombinationer (inkl. interaktionseffekter, t.ex. HMM vs Vol-Target):

```
φ_k = Σ_{S⊆N\{k}} [|S|!(n-|S|-1)!/n!] · [G(S∪{k}) - G(S)]
```

Implementering: `research/06_marginal_contribution.py`

### 10.2 Script 06: Shapley-Ablation (Högsta domstolen) — Inferential Hardness

| Steg | Beskrivning |
|------|-------------|
| **Block Bootstrap** | 12-månaders block (252 dagar), N=1000 iterationer. Full Shapley-matris per iteration. |
| **95% CI** | Konfidensintervall för φ_k^G och φ_k^ES (percentil 2.5, 97.5). |
| **Stability Score** | Andel bootstrap-iterationer där φ_k^G > 0 (%). Minst 65 % krävs för KEEP. |
| **Regime-Specific Shapley** | φ_k^G under Crisis, Bull, Low Liquidity. Ergodicity Penalty om signifikant negativ φ_k^G i Crisis/Low Liquidity. |
| **Refined Fractional Kelly** | f* med shrinkage på medelvärde och excess variance-straff för icke-normala fördelningar. |
| **Final Verdict** | **DELETION** om: CI inkluderar noll, Stability &lt;65 %, Ergodicity Penalty, eller redundant (Corr&gt;0,7 med högre-impact). |
| **Output** | `research/shapley_results.csv` (CI-bands, Stability, Regime-Shapley), `research/bootstrap_results.parquet` (fördelningar). |

### 10.3 Hypoteser (Brutal gallring)

| Lager | Förväntat resultat |
|-------|--------------------|
| **HRP & Vol-Target** | Hög korrelation i risk-residualer. En kan ryka. |
| **NLP** | φ_k^G ≈ 0. Ofta kosmetik som drunknar i brus för småbolag. |
| **Debt Wall** | Starkt negativ φ_k^MDD (minskar drawdown), positiv Skew. Skölden. |
| **Supply-Chain** | Högst φ_k^G men också högst bidrag till Vol-of-Vol. Motorn. |

### 10.4 Fractional Kelly (referens)

Slutallokering kalibreras efter realiserad geometrisk kant:

```
f* = (G_adj / σ²) · Margin of Safety (0,3)
```

### 10.5 Körning

```bash
python research/06_marginal_contribution.py
```
