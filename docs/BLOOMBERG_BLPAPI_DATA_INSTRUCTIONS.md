# Trinity Stack — Bloomberg BLPAPI Data Retrieval Instructions

**Datum:** 2025-02  
**Period:** 2006-01-01 → 2026-12-31  
**Univers:** Nordic Small Caps (OMX Helsinki + OMX Stockholm)

> **Viktiga metodologiska krav:** Point-in-time universum (ingen survivorship bias), justerade priser (splits/utdelningar), äkta Breadth beräknad från eget universum.

---

## 1. Förutsättningar

- Bloomberg Terminal + B-PIPE (eller Data License) aktiv
- Python med `blpapi` (eller `pdblp` som wrapper)
- Starta Bloomberg Terminal innan du kör script (BLPAPI använder lokal session)

### Installation
```bash
pip install pdblp
# Eller: pip install blpapi  (officiell SDK)
```

---

## 2. Fil 1: prices.parquet

**Output-kolumner:** `date`, `ticker`, `PX_LAST`, `PX_TURN_OVER`, `BID_ASK_SPREAD_PCT` (alla i EUR)

### BLPAPI / pdblp

| Trinity-kolumn       | Bloomberg Field    | Anteckning                          |
|----------------------|--------------------|-------------------------------------|
| PX_LAST              | `PX_LAST` + overrides | **Måste vara justerad** för splits/utdelningar; i EUR (se FX nedan) |
| PX_TURN_OVER         | `PX_TURN_OVER`     | **Daglig omsättning i valuta** (inte antal aktier). Amihud + Market Impact bygger på fiat. |
| BID_ASK_SPREAD_PCT   | Beräkna            | `(PX_ASK - PX_BID) / PX_MID * 100`  |

Om `PX_MID` saknas: använd `(PX_ASK - PX_BID) / PX_LAST * 100`.

**Bloomberg Fields att hämta:** `PX_LAST`, `PX_TURN_OVER`, `PX_BID`, `PX_ASK`

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

**Korrektion:** Hämta historiska indexmedlemmar per månad/kvartal.

- **Bloomberg:** `INDX_MWEIGHT` med `END_DATE_OVERRIDE` för varje månad, eller BQL (Bloomberg Query Language) för point-in-time universum.
- **Alternativ:** Manuell `dead_tickers.csv` med avlistade bolag (t.ex. från tidigare Portman-projekt) och slå ihop med aktuella constituents per tidsperiod.

### KRITISKT: Valuta-normalisering (FX – SEK → EUR)

OMX Helsinki = EUR, OMX Stockholm = SEK. Rådata utan valutakonvertering gör att LightGBM blandar SEK och EUR i samma matris → korrupta beräkningar (Amihud, momentum, fundamentals).

**Säkrast för Point-in-Time:** Hämta `SEKEUR Curncy` historiskt via bdh, multiplicera SEK-aktiernas PX_LAST och PX_TURN_OVER med valutakursen i Layer 1 (eller i datapreprocessering). Bevara PiT-noggrannhet.

**Alternativ:** Bloomberg valuta-override i bdh (om tillgängligt för dina tickers). Men separat valutakurs + multiplikation i pandas är ofta säkrast.

```python
# Pseudokod: efter bdh-hämtning
sek_tickers = [...]  # Lista över Stockholm-tickers
fx = con.bdh("SEKEUR Curncy", "PX_LAST", start, end)  # SEK per 1 EUR
# För varje rad där ticker in sek_tickers: PX_LAST *= fx.loc[date], PX_TURN_OVER *= fx.loc[date]
```

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
fields = ["PX_LAST", "PX_TURN_OVER", "PX_BID", "PX_ASK"]  # PX_TURN_OVER, inte PX_VOLUME
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

**Output:** index = `date`, kolumner: `V2TX`, `Breadth`, `Rates`

### Bloomberg Fields

| Trinity-kolumn | Bloomberg Ticker     | Bloomberg Field | Beskrivning          |
|----------------|----------------------|-----------------|----------------------|
| V2TX           | `V2TX Index`         | `PX_LAST`       | Euro Stoxx 50 Vol    |
| Rates          | `GFFI10 Govt` (FI) eller `GSGB10YR Govt` (SE) | `PX_LAST` | 10Y ränta |
| Breadth        | Hämta ej från BB     | —               | Beräkna från prices  |

### Breadth (beräkna själv)

**KRITISKT:** Hämta inte Breadth från Bloomberg. Volatilitet ≠ Breadth. Beräkna Breadth i Layer 1: andel tickers i ditt universum som handlas över 200d SMA (per dag, med .shift(1)). Lägg till i macro.parquet efter att prices är laddad.

### pdblp-exempel

```python
macro_tickers = ["V2TX Index", "GFFI10 Govt", "GSGB10YR Govt"]
macro_fields = ["PX_LAST"]
df = con.bdh(macro_tickers, macro_fields, "20060101", "20261231")
# Rename till V2TX, Rates. Breadth läggs till via beräkning från prices.parquet
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

## 6. Sammanfattning – exakta fält per dataset

```
PRICES (bdh + overrides CshAdjNormal, CshAdjAbnormal, CapChgExec):
  tickers:  Point-in-time universum (INDX_MWEIGHT + END_DATE_OVERRIDE per månad, eller dead_tickers.csv)
  fields:   PX_LAST, PX_TURN_OVER, PX_BID, PX_ASK
  period:   2006-01-01 → 2026-12-31
  → Output: date, ticker, PX_LAST, PX_TURN_OVER, BID_ASK_SPREAD_PCT (alla i EUR, SEK konverterade)

FUNDAMENTALS:
  tickers:  [samma point-in-time som prices]
  fields:   EARN_ANN_*, BEST_EPS, BEST_EPS_EST_STD_DEV, RETURN_ON_INVESTED_CAPITAL,
            EQY_SH_OUT (för Dilution)
  → Output: report_date, ticker, ACTUAL_EPS, CONSENSUS_EPS, EPS_STD, ROIC, Dilution, Accruals
  → Fallback SUE: CONSENSUS_EPS = EPS_{t-4}, EPS_STD = std(8 kvartal) om BB saknar data
  → Dilution = EQY_SH_OUT_t / EQY_SH_OUT_{t-1year} (ersätter Piotroski_F)

MACRO (bdh):
  tickers:  V2TX Index, GFFI10 Govt (eller GSGB10YR Govt)
  fields:   PX_LAST
  → Output: date, V2TX, Rates; Breadth beräknas i Layer 1 från prices (% tickers > 200d SMA)

NEWS:
  → Output: ticker, date, text
  → Via News API eller manuell export
```

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
