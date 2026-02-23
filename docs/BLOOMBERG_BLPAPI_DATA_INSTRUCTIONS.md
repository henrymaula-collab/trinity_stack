# Trinity Stack — Bloomberg BLPAPI & Capital IQ Data Instructions

**Datum:** 2025-02  
**Period:** 2006-01-01 → 2026-12-31  
**Univers:** Nordic Small Caps (OMX Helsinki + OMX Stockholm)

> **Viktiga metodologiska krav:** Point-in-time universum (ingen survivorship bias), justerade priser (splits/utdelningar), äkta Breadth beräknad från eget universum.

---

## 0. Arbetsfördelning: CapIQ vs. Bloomberg

| Datatyp | Källa | Motivering |
|---------|-------|------------|
| **Pris & Marknad** | Bloomberg | Osagbar för OHLC, volym och makro |
| **SUE & Estimering** | Bloomberg | BBG:s konsensus-feed (ANR) är standard |
| **Debt Maturity** | Capital IQ | Mer detaljerad nedbrytning lån vs. obligationer |
| **Supply Chain** | Capital IQ | "Business Relationships"-databasen är djupare för micro-caps |

För nordiska småbolag är CapIQ objektivt överlägset för skuldväggar och leverantörsrelationer – analytiker manuellt rensar finansiella rapporter i högre grad än Bloombergs automatiska algoritmer för micro-caps.

---

## 1. Förutsättningar

- Bloomberg Terminal + B-PIPE (eller Data License) aktiv
- Starta Bloomberg Terminal innan du kör script (BLPAPI använder lokal session)

**Primär implementering:** R med Rblpapi (se sektion 6b). Scripten `data1_BLOOMBERG.R` och `data2_BLOOMBERG.R` är redo att köras i RStudio på Bloomberg Terminal.

**Alternativ (Python):**
```bash
pip install pdblp   # eller: pip install blpapi (officiell SDK)
```

---

## 2. Fil 1: prices.parquet

**Output-kolumner:** `date`, `ticker`, `PX_OPEN`, `PX_LAST`, `PX_HIGH`, `PX_LOW`, `PX_TURN_OVER`, `BID_ASK_SPREAD_PCT` (alla i EUR)

### BLPAPI / pdblp

| Trinity-kolumn       | Bloomberg Field    | Anteckning                          |
|----------------------|--------------------|-------------------------------------|
| PX_OPEN              | `PX_OPEN`          | **KRITISKT för T+1.** Utan PX_OPEN lider "Pessimistisk exekvering" av look-ahead bias – du handlar i praktiken på dagens stängning trots att du inte vet stängningskursen förrän marknaden stängt. |
| PX_LAST              | `PX_LAST` + overrides | **Måste vara justerad** för splits/utdelningar; i EUR (se FX nedan) |
| PX_HIGH              | `PX_HIGH`          | För dags-volatilitet och fill-sannolikhetsmodell |
| PX_LOW               | `PX_LOW`           | För dags-volatilitet |
| PX_TURN_OVER         | `PX_TURN_OVER`     | **Daglig omsättning i valuta** (inte antal aktier). Amihud + Capacity + Participation Cap. |
| VWAP_CP              | `VWAP_CP`          | **KRITISKT för Layer 6 TCA.** Volymviktad kurs — referenspris för slippage och Market Impact. PX_LAST i illikvida småbolag är brusigt/manipulerbart i slutauktioner. |
| BID_ASK_SPREAD_PCT   | Beräkna            | `(PX_ASK - PX_BID) / PX_MID * 100` (eller `(PX_ASK + PX_BID)/2`) |

**Bloomberg Fields att hämta:**
```python
fields = [
    "PX_OPEN",       # För T+1 entry – KRITISKT
    "PX_LAST",       # För beräkning av avkastning
    "PX_HIGH",       # För volatilitetsskalning
    "PX_LOW",        # För volatilitetsskalning
    "PX_TURN_OVER",  # För Amihud & Capacity
    "PX_BID",        # För spread-beräkning
    "PX_ASK",        # För spread-beräkning
    "VWAP_CP",       # Volymviktad kurs — Layer 6 TCA, slippage, Market Impact
]
```

> **Omsättning vs volym:** `PX_VOLUME` (antal aktier) är felaktigt i småbolag – 10k aktier à 0,10 EUR ≠ 10k aktier à 100 EUR. `PX_TURN_OVER` = Daily Traded Value i lokal valuta.

### KRITISKT: Justerade priser (undvik split-/utdelningseffekter)

BLPAPI returnerar som standard *råa, ojusterade* stängningskurser. En 5:1 split ger falskt –80 % kursfall och korrupt momentum-signal.

**Overrides som måste inkluderas i `bdh`-request:**

```python
overrides = [
    ("CshAdjNormal", True),   # Justering för normala utdelningar
    ("CshAdjAbnormal", True), # Justering för abnormala utdelningar
    ("CapChgExec", True)      # Justering för kapitaländringar (splits, reverse splits)
]
```

### KRITISKT: Point-in-time universum (eliminera survivorship bias)

**Fel:** Att hämta `INDEX_MEMBERS` idag ger endast bolag som överlevt fram till 2026. Konkurser, uppköp och avlistningar sedan 2006 saknas → ML-modellen tränas på ett universum med statistiskt noll konkursris.

**Implementerat i R-skripten:** `get_historical_universe()` använder `bds(..., INDX_MWEIGHT)` med **kvartalsvis** END_DATE_OVERRIDE (0331, 0630, 0930, 1231) för varje år 2006–2026. Fångar intra-år till-/avlistningar. Ticker-normalisering: använder Bloombergs "Member Ticker and Exchange Code" (FH/SS) oförändrad, lägger endast till " Equity". Batchad bdh (50 tickers/request) undviker Daily Data Limits.

### KRITISKT: Valuta-normalisering (FX – SEK → EUR)

OMX Helsinki = EUR, OMX Stockholm = SEK. Rådata utan valutakonvertering gör att LightGBM blandar SEK och EUR i samma matris → korrupta beräkningar (Amihud, momentum, fundamentals).

**Hämta `SEKEUR Curncy`** som separat tidsserie för att normalisera Stockholmstickers till EUR:

```python
# Efter bdh-hämtning
sek_tickers = [...]  # Lista över Stockholm-tickers
fx = con.bdh("SEKEUR Curncy", "PX_LAST", start, end)  # SEK per 1 EUR
# För varje rad där ticker in sek_tickers: PX_OPEN, PX_LAST, PX_HIGH, PX_LOW, PX_TURN_OVER *= fx.loc[date]
```

**Alternativ:** Bloomberg valuta-override i bdh (om tillgängligt). Separat valutakurs + multiplikation i pandas är ofta säkrast.

### pdblp-exempel (Historical)

```python
import pdblp
import pandas as pd

overrides = [
    ("CshAdjNormal", True),
    ("CshAdjAbnormal", True),
    ("CapChgExec", True)
]

con = pdblp.BCon(debug=False, port=8194, timeout=60000)
con.start()

tickers = [...]  # Point-in-time universum (se ovan)
fields = ["PX_OPEN", "PX_LAST", "PX_HIGH", "PX_LOW", "PX_TURN_OVER", "PX_BID", "PX_ASK"]
start, end = "20060101", "20261231"

df = con.bdh(tickers, fields, start, end, longdata=True, ovrds=overrides)
con.stop()

# 1) Konvertera SEK → EUR (se FX-sektionen ovan)
# 2) BID_ASK_SPREAD_PCT = (PX_ASK - PX_BID) / ((PX_ASK + PX_BID)/2) * 100
```

### Ungefärlig datapunkter

- ~5 000 handelsdagar × 30–80 tickers ≈ **150 000 – 400 000 rader**

---

## 3. Fil 2: fundamentals.parquet

**Output-kolumner:** `report_date`, `ticker`, `ACTUAL_EPS`, `CONSENSUS_EPS`, `EPS_STD`, `ROIC`, `Dilution`, `Accruals`

> **Schema-ändring:** `Piotroski_F` ersatt med `Dilution` (se optimering nedan). Layer 1 Quality composite måste uppdateras att använda Dilution.

### Bloomberg Fields

| Trinity-kolumn   | Bloomberg Field (eller alternativ)          | Anteckning                        |
|------------------|---------------------------------------------|-----------------------------------|
| report_date      | `EARN_ANN_DT_TIME_HIST_WITH_EPS` / `BEST_EPS_EST_DT` | Earnings announcement date   |
| ACTUAL_EPS       | `BEST_EPS` (actual) / `EARN_ANN_EPS_ACTUAL` | Actual reported EPS               |
| CONSENSUS_EPS    | `BEST_EPS` (pre-announcement)               | Se fallback nedan                 |
| EPS_STD          | `BEST_EPS_EST_STD_DEV`                      | Se fallback nedan                 |
| ROIC             | `RETURN_ON_INVESTED_CAPITAL`                | ROIC                              |
| Dilution         | `EQY_SH_OUT` (uteslående aktier)            | Se optimering nedan               |
| Dividend_Yield   | `BEST_DIV_YLD` (estimat) + `DIVIDEND_YIELD` (historisk) | **Båda krävs.** `coalesce(BEST_DIV_YLD, DIVIDEND_YIELD)` — nordiska småbolag saknar ofta analytikertäckning; endast BEST_DIV_YLD ger gles träningsmatris. |
| Accruals         | `ACCRUAL_RATIO` eller (NI - CFO) / TA       | Beräkna från fundamentals         |

### Fallback för SUE (nordiska småbolag saknar analytikertäckning)

Nordiska småbolag har ofta inga `BEST_EPS`/`CONSENSUS_EPS`-data. Att kräva dessa ger 50–70 % NaN → pipeline fail-fast och kraftigt krympt träningsdata.

**Naive Random Walk fallback om CONSENSUS_EPS saknas:**

- `CONSENSUS_EPS` = EPS samma kvartal föregående år (`EPS_{t-4}`)
- `EPS_STD` = standardavvikelsen av de senaste 8 kvartalens EPS

Implementera i datapreprocessering före sparande av fundamentals.parquet.

### Optimering: Ersätt Piotroski F med Dilution

Piotroski F kräver ROA, CFO, ΔROA, accruals, leverage, likviditet, equity offer → många BulkDataRequests, rate-limits och ofullständig data för småbolag 20 år tillbaka.

**Ersättning:** Hämta `EQY_SH_OUT` (uteslående aktier) och beräkna:

```
Dilution = EQY_SH_OUT_t / EQY_SH_OUT_{t-1 year}
```

Dilution straffar utspädning (emissioner), vilket är centralt för småbolagsöverlevnad. Samma överlevnadsinformation som F-score med minimal datacomplexitet.

### Service: ReferenceDataRequest / BulkDataRequest

Fundamentals hämtas som **reference/snapshot** per datum. FLDS <Go> visar tillgängliga fält.

### Ytterligare marginalvärde (valfritt): Insynshandel & sektorklassificering

I illikvida nordiska miljöer med hög informationsasymmetri ger dessa två datatyper stark förbättring:

| Datatyp | Betydelse | Bloomberg Fields |
|-------|-----------|------------------|
| **Insynshandel** | Ledningens/styrelsens nettoköp är stark signal när analytikertäckning saknas. Divergens: lågt momentum + stigande insynsköp → trendvändning/botten. | `NUM_INSIDER_BUYS`, `NUM_INSIDER_SELLS`, eller `NET_INSIDER_TRANS_VALUE` |
| **Sektorklassificering** | Utan sektorrisk riskerar momentum att följa branschbubblor (t.ex. enbart tech). Sektorneutralt momentum belönar starkaste aktierna *inom* varje sektor. | `GICS_SECTOR_NAME` eller `BICS_LEVEL_1_SECTOR_NAME` |

Implementering: Hämta som referensdata, lägg till som kolumner i fundamentals.parquet. FeatureEngineer slussar dem till träningsmatrisen utan arkitekturändring. Din nuvarande instruktion är tillräcklig för den initiala fasen; dessa kan aktiveras senare för högre IC.

### Ungefärlig datapunkter

- ~2–4 rapporter/år × tickers × 20 år ≈ **1 200 – 6 000 rader**

---

## 4. Fil 3: macro.parquet

**Output:** index = `date`, kolumner: `V2TX`, `Breadth`, `Rates_FI`, `Rates_SE`, `MOVE` (valfritt)

### Bloomberg Fields

| Trinity-kolumn | Bloomberg Ticker     | Bloomberg Field | Beskrivning          |
|----------------|----------------------|-----------------|----------------------|
| V2TX           | `V2TX Index`         | `PX_LAST`       | Euro Stoxx 50 Vol (regimskalning, spread stress) |
| Rates           | `GDBR10 Govt`        | `PX_LAST`       | **Tysk 10Y (Bund)** – HMM-input, undviker multikollinearitet |
| Rates_FI       | `GFFI10 Govt`        | `PX_LAST`       | 10Y ränta Finland (Local_Rate för FH-tickers) |
| Rates_SE       | `GSGB10YR Govt`      | `PX_LAST`       | 10Y ränta Sverige (Local_Rate för SS-tickers) |
| Breadth        | Hämta ej från BB     | —               | Beräkna från prices  |
| MOVE           | `MOVE Index`         | `PX_LAST`       | **(Valfritt)** Räntevolatilitet. Kritiskt för Debt Wall – refinansieringsrisk korrelerar med MOVE |

**Rates (HMM):** Tysk 10Y (Bund) är ortogonal och stabil för regimdetektering. Undvik räntespread (FI−SE) – den fångar valutadifferens, inte systematisk marknadsrisk.

**Local_Rate (Layer 1):** Rates_FI och Rates_SE mappas dynamiskt per ticker (FH → Rates_FI, SS → Rates_SE) för excess return och räntekostnadstäckning.

### Breadth (beräkna själv)

**KRITISKT:** Hämta inte Breadth från Bloomberg. Volatilitet ≠ Breadth. Beräkna Breadth i Layer 1: andel tickers i ditt universum som handlas över 200d SMA (per dag, med .shift(1)). Lägg till i macro.parquet efter att prices är laddad.

### pdblp-exempel

```python
macro_tickers = ["V2TX Index", "GFFI10 Govt", "GSGB10YR Govt", "MOVE Index"]  # MOVE valfritt
macro_fields = ["PX_LAST"]
df = con.bdh(macro_tickers, macro_fields, "20060101", "20261231")
# Rename till V2TX, Rates_FI, Rates_SE. Breadth beräknas från prices.parquet
```

### Ungefärlig datapunkter

- ~5 000 dagar ≈ **5 000 rader**

---

## 5. Fil 4: news.parquet

**Output-kolumner:** `ticker`, `date`, `text`

### Bloomberg

- **BN <Go>** / News API
- **BQNT** – News Quant
- **NAN <Go>** – News Analytics
- Event-type: `RN` (Press Release) eller motsvarande

Fält att hämta: `News headline` / `News text`, `News date`, `Related security`

BLPAPI: News retrieval görs via **IntradayBarRequest** eller **News**-service. Se Bloomberg API documentation för exakt service/request.

### Alternativ

Om BLPAPI-news är begränsad: exportera news manuellt från Terminal (NAN, BN) till CSV och konvertera till parquet med kolumnerna `ticker`, `date`, `text`.

### Ungefärlig datapunkter

- Varierar: **5 000 – 50 000 rader**

---

## 4a. Supply-Chain (CapIQ-fokus) – Spjutet

**Output:** `date`, `supplier_ticker`, `customer_ticker`, `revenue_dependency_pct`

Om CapIQ används: "Business Relationships"-databasen. Om Bloomberg används som fallback:

| Trinity-kolumn | Bloomberg/Alternativ | Anteckning |
|----------------|----------------------|------------|
| revenue_dependency_pct | `SPLC_REVENUE_DEPENDENCY_PCT` | Procentuell andel intäkter från kund |
| customer_ticker | `SPLC_CUSTOMER_TICKER` | Tickern för global kund (Mega-cap) |

---

## 4b. Debt Wall (CapIQ-fokus) – Skölden

**Output:** `date`, `ticker`, `market_cap`, `cash_equivalents`, `ttm_fcf`, `next_12m_debt_maturity`, …

CapIQ är överlägsen för skulddetaljer. Kravspecifikation för utdrag:

| Trinity-kolumn | CapIQ / Bloomberg Field | Anteckning |
|----------------|-------------------------|------------|
| cash_equivalents | `BS_CASH_NEAR_CASH_ITEM` | Aktuell likviditet |
| next_12m_debt_maturity | `DEBT_MATURITY_SCHEDULE_NEXT_12M` | Total skuld som förfaller inom ett år |
| ttm_fcf | `FREE_CASH_FLOW_TTM` | För Burn rate – räcker kassan till förfallet? |

---

## 6. Sammanfattning – exakta fält per dataset

```
PRICES (bdh + overrides CshAdjNormal, CshAdjAbnormal, CapChgExec):
  tickers:  Point-in-time universum (INDX_MWEIGHT + END_DATE_OVERRIDE per månad, eller dead_tickers.csv)
  fields:   PX_OPEN, PX_LAST, PX_HIGH, PX_LOW, PX_TURN_OVER, PX_BID, PX_ASK
  fx:       SEKEUR Curncy (separat tidsserie för Stockholm-tickers)
  period:   2006-01-01 → 2026-12-31
  → Output: date, ticker, PX_OPEN, PX_LAST, PX_HIGH, PX_LOW, PX_TURN_OVER, BID_ASK_SPREAD_PCT (alla i EUR)
  → KRITISKT: PX_OPEN krävs för T+1 Pessimistisk exekvering. Utan den → look-ahead bias.

FUNDAMENTALS:
  tickers:  [samma point-in-time som prices]
  fields:   EARN_ANN_*, BEST_EPS, BEST_EPS_EST_STD_DEV, RETURN_ON_INVESTED_CAPITAL,
            EQY_SH_OUT (för Dilution)
  → Output: report_date, ticker, ACTUAL_EPS, CONSENSUS_EPS, EPS_STD, ROIC, Dilution, Accruals
  → Fallback SUE: CONSENSUS_EPS = EPS_{t-4}, EPS_STD = std(8 kvartal) om BB saknar data
  → Dilution = EQY_SH_OUT_t / EQY_SH_OUT_{t-1year} (ersätter Piotroski_F)

MACRO (bdh):
  tickers:  V2TX Index, GDBR10 Govt (Bund), GFFI10 Govt, GSGB10YR Govt, MOVE Index (valfritt)
  fields:   PX_LAST
  → Output: date, V2TX, Breadth, Rates (Bund), Rates_FI, Rates_SE, MOVE
  → Breadth beräknas i Layer 1 från prices (% tickers > 200d SMA)

SUPPLY-CHAIN (CapIQ prioriterad):
  → Output: date, supplier_ticker, customer_ticker, revenue_dependency_pct
  → CapIQ: Business Relationships. BBG-fallback: SPLC_REVENUE_DEPENDENCY_PCT, SPLC_CUSTOMER_TICKER

DEBT WALL (CapIQ prioriterad):
  → Output: date, ticker, cash_equivalents, next_12m_debt_maturity, ttm_fcf, …
  → CapIQ/BBG: BS_CASH_NEAR_CASH_ITEM, DEBT_MATURITY_SCHEDULE_NEXT_12M, FREE_CASH_FLOW_TTM

NEWS:
  → Output: ticker, date, text
  → Via News API eller manuell export
```

---

## 6b. R-skript (Bloomberg i två delar)

För att undvika kvotgräns – kör datamängden i två separata körningar:

| Script | Innehåll | Output |
|--------|----------|--------|
| `docs/data1_BLOOMBERG.R` | Priser + FX (SEKEUR) | prices.parquet, fx_sekeur.parquet |
| `docs/data2_BLOOMBERG.R` | Fundamentals, Macro, News | fundamentals.parquet, macro.parquet, news.parquet |
| `docs/data_CAPITAL_IQ.R` | Supply Chain, Debt Wall | supply_chain.parquet, debt_wall.parquet |

**Körning:** Öppna i RStudio på en Bloomberg Terminal, sätt `OUTPUT_DIR` och `TICKERS`, kör data1 först, sedan data2. CapIQ-scriptet förutsätter CSV-export från CapIQ Excel Add-in.

---

## 7. Spara till Trinity format

Efter hämtning, spara i `data/raw/`:

```python
# Exempel
prices_df.to_parquet("data/raw/prices.parquet", index=False)
fundamentals_df.to_parquet("data/raw/fundamentals.parquet", index=False)
macro_df.to_parquet("data/raw/macro.parquet")  # index = date
news_df.to_parquet("data/raw/news.parquet", index=False)
```

**Viktigt:**

- `date` / `report_date` som datetime
- Inga NaN i kritiska kolumner (pipeline fail-fast)
- Prices: sortera på date, ticker
- Macro: index = DatetimeIndex (datum)

---

## 8. Port & koppling

- Standard BLPAPI-port: **8194**
- Terminal måste vara igång
- För `pdblp`: `BCon(port=8194)` – kontrollera att ingen annan process använder porten

---

## 9. Datavolym och modellträning

- **~400k rader** daglig prisdata är optimal volym för LightGBM (gradient boosting). Träd hittar djupa, olinjära interaktioner (t.ex. Amihud + SUE under olika makroregimer) utan RAM-problem.
- Deep Learning (LSTM/Transformers) skulle kräva miljoner rader + tick-data för att undvika överanpassning. Gradient Boosting excellerar i 100k–500k rader tabulär paneldata.
- Antal *oberoende* observationer är lägre p.g.a. seriell autokorrelation – **expanding walk-forward CV** i Layer 3 hanterar detta korrekt.
