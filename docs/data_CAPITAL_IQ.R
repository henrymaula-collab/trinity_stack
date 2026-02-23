# =============================================================================
# Trinity Stack — Capital IQ DATA: Supply Chain + Debt Wall
# =============================================================================
# CapIQ är överlägset Bloomberg för skuldväggar och leverantörsrelationer
# hos nordiska småbolag (manuell analytikerrensning).
#
# Access: CapIQ Excel Add-in, Compustat/CapIQ API, eller export till CSV.
# Detta script förutsätter CSV-export från CapIQ – anpassa källor efter din
# tillgänglighet (Excel, REST API, WRDS).
# =============================================================================

library(dplyr)
library(arrow)
library(readr)

OUTPUT_DIR <- "data/raw"

# =============================================================================
# ALTERNATIV 1: Läs från CSV-export (CapIQ Excel → Spara som CSV)
# =============================================================================
# I CapIQ Excel Add-in eller Pro:
# 1. Supply Chain: Sök "Business Relationships" / "Key Customers"
# 2. Debt Wall: Sök "Balance Sheet", "Debt Maturity", "Cash Flow"
# 3. Exportera till CSV med kolumnerna nedan
# =============================================================================

# ---- Supply Chain: Krävda kolumner ----
# date | supplier_ticker | customer_ticker | revenue_dependency_pct
# CapIQ-fält: Supplier/Customer relationship, Revenue dependency %
# BBN-fallback: SPLC_REVENUE_DEPENDENCY_PCT, SPLC_CUSTOMER_TICKER

supply_chain_schema <- function(df) {
  required <- c("date", "supplier_ticker", "customer_ticker", "revenue_dependency_pct")
  missing <- setdiff(required, names(df))
  if (length(missing) > 0) {
    warning("Supply chain saknar kolumner: ", paste(missing, collapse = ", "))
    for (m in missing) df[[m]] <- NA
  }
  df %>%
    select(date, supplier_ticker, customer_ticker, revenue_dependency_pct) %>%
    mutate(
      date = as.Date(date),
      revenue_dependency_pct = as.numeric(revenue_dependency_pct)
    )
}

# ---- Debt Wall: Krävda kolumner ----
# date | ticker | market_cap | cash_equivalents | ttm_fcf | next_12m_debt_maturity
# CapIQ-fält:
#   BS_CASH_NEAR_CASH_ITEM  → cash_equivalents
#   DEBT_MATURITY_SCHEDULE_NEXT_12M → next_12m_debt_maturity
#   FREE_CASH_FLOW_TTM → ttm_fcf

debt_wall_schema <- function(df) {
  required <- c("date", "ticker", "cash_equivalents", "next_12m_debt_maturity", "ttm_fcf")
  for (r in required) {
    if (!r %in% names(df)) df[[r]] <- NA_real_
  }
  if (!"market_cap" %in% names(df)) df$market_cap <- NA_real_
  df %>%
    select(date, ticker, market_cap, cash_equivalents, ttm_fcf, next_12m_debt_maturity) %>%
    mutate(
      date = as.Date(date),
      across(c(market_cap, cash_equivalents, ttm_fcf, next_12m_debt_maturity), as.numeric)
    )
}

# =============================================================================
# EXEMPEL: Läs CSV och spara Parquet
# =============================================================================
# Uppdatera sökvägar till dina exporterade filer från CapIQ

run_capiq_import <- function(
  supply_chain_csv = NULL,   # t.ex. "exports/capiq_supply_chain.csv"
  debt_wall_csv = NULL       # t.ex. "exports/capiq_debt_wall.csv"
) {
  dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

  if (!is.null(supply_chain_csv) && file.exists(supply_chain_csv)) {
    sc <- read_csv(supply_chain_csv, show_col_types = FALSE)
    sc_out <- supply_chain_schema(as.data.frame(sc))
    write_parquet(sc_out, file.path(OUTPUT_DIR, "supply_chain.parquet"))
    message("Supply chain sparad: ", nrow(sc_out), " rader")
  } else {
    message("Supply chain: Skapa CSV från CapIQ Business Relationships och ange sökväg.")
    # Placeholder
    sc_placeholder <- data.frame(
      date = as.Date(character(0)),
      supplier_ticker = character(0),
      customer_ticker = character(0),
      revenue_dependency_pct = numeric(0)
    )
    write_parquet(sc_placeholder, file.path(OUTPUT_DIR, "supply_chain.parquet"))
  }

  if (!is.null(debt_wall_csv) && file.exists(debt_wall_csv)) {
    dw <- read_csv(debt_wall_csv, show_col_types = FALSE)
    dw_out <- debt_wall_schema(as.data.frame(dw))
    write_parquet(dw_out, file.path(OUTPUT_DIR, "debt_wall.parquet"))
    message("Debt wall sparad: ", nrow(dw_out), " rader")
  } else {
    message("Debt wall: Skapa CSV från CapIQ (BS_CASH_*, DEBT_MATURITY_*, FREE_CASH_FLOW_TTM) och ange sökväg.")
    dw_placeholder <- data.frame(
      date = as.Date(character(0)),
      ticker = character(0),
      market_cap = numeric(0),
      cash_equivalents = numeric(0),
      ttm_fcf = numeric(0),
      next_12m_debt_maturity = numeric(0)
    )
    write_parquet(dw_placeholder, file.path(OUTPUT_DIR, "debt_wall.parquet"))
  }
}

# Kör med dina sökvägar (ändra innan körning)
# run_capiq_import(
#   supply_chain_csv = "path/to/capiq_supply_chain.csv",
#   debt_wall_csv = "path/to/capiq_debt_wall.csv"
# )

# Om du bara vill skapa placeholders:
run_capiq_import()

# =============================================================================
# CapIQ EXCEL ADD-IN: Steg för manuell export
# =============================================================================
# Supply Chain (Spjutet):
# 1. Öppna CapIQ Excel, sök företag (supplier)
# 2. Välj "Key Customers" / "Business Relationships"
# 3. Exportera: Supplier, Customer, Revenue dependency %
# 4. Lägg till date-kolumn (rapportdatum)
#
# Debt Wall (Skölden):
# 1. Sök företag
# 2. Financials → Balance Sheet: Cash and Equivalents (BS_CASH_NEAR_CASH_ITEM)
# 3. Debt → Maturity Schedule: Summa 12M (DEBT_MATURITY_SCHEDULE_NEXT_12M)
# 4. Cash Flow: Free Cash Flow TTM (FREE_CASH_FLOW_TTM)
# 5. Market Data: Market Cap
# =============================================================================
