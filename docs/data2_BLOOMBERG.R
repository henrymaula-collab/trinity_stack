# =============================================================================
# Trinity Stack — Bloomberg DATA 2: FUNDAMENTALS, MACRO, NEWS
# =============================================================================
# Kör efter data1. Lättare datamängder — kvotvänlig.
# Output: fundamentals.parquet, macro.parquet, news.parquet (om tillgänglig)
#
# Krav: Bloomberg Terminal igång; TICKERS från data1 (nordic_historical_universe.csv)
#       eller PiT-universum via get_historical_universe
# =============================================================================

library(Rblpapi)
library(dplyr)
library(arrow)
library(tidyr)

# ---- Konfiguration ----
OUTPUT_DIR <- "data/raw"  # Samma som data1
START_DATE <- as.Date("2006-01-01")
END_DATE <- as.Date("2026-12-31")
FUND_BATCH_SIZE <- 50L
UNIVERSE_CACHE <- file.path(OUTPUT_DIR, "nordic_historical_universe.csv")

# Läs TICKERS från data1s cache (kör data1 först) eller fallback till manuell lista
if (file.exists(UNIVERSE_CACHE)) {
  TICKERS <- read.csv(UNIVERSE_CACHE, stringsAsFactors = FALSE)$ticker
  cat(sprintf("Universum från data1: %d tickers\n", length(TICKERS)))
} else {
  TICKERS <- c("NOKIA FH Equity", "UPM FH Equity", "ORNBV FH Equity",
               "VOLV-B SS Equity", "ERIC-B SS Equity", "ATCO-A SS Equity")
  cat("Varning: nordic_historical_universe.csv saknas. Kör data1 först. Använder fallback-lista.\n")
}

# ---- Anslut Bloomberg ----
blpConnect(host = "localhost", port = 8194L)

# =============================================================================
# 1. FUNDAMENTALS (batchad för stort universum)
# =============================================================================
cat("Hämtar fundamentals...\n")

fund_fields <- c(
  "EARN_ANN_DT_TIME_HIST_WITH_EPS",
  "BEST_EPS",
  "BEST_EPS_EST_STD_DEV",
  "RETURN_ON_INVESTED_CAPITAL",
  "EQY_SH_OUT",
  "ACCRUAL_RATIO"
)
opt <- c(periodicitySelection = "QUARTERLY")
fund_list <- list()
batches <- split(TICKERS, ceiling(seq_along(TICKERS) / FUND_BATCH_SIZE))
for (b in seq_along(batches)) {
  chunk <- batches[[b]]
  cat(sprintf("  Fundamentals batch %d/%d: %d tickers\n", b, length(batches), length(chunk)))
  for (ticker in chunk) {
    tryCatch({
      df <- bdh(ticker, fund_fields, START_DATE, END_DATE, options = opt)
      if (!is.null(df) && nrow(df) > 0) {
        df$ticker <- gsub(" Equity$", "", ticker)
        fund_list[[ticker]] <- df
      }
    }, error = function(e) invisible(NULL))
  }
}
fund_list_valid <- fund_list[!sapply(fund_list, is.null)]
if (length(fund_list_valid) == 0) {
  stop("Inga fundamentals hämtade. Kontrollera tickers och Bloomberg.", call. = FALSE)
}
fundamentals_raw <- bind_rows(fund_list_valid)

# Namngivning för Trinity
if (nrow(fundamentals_raw) > 0) {
  fundamentals_df <- fundamentals_raw %>%
    rename(
      report_date = date,
      ACTUAL_EPS = BEST_EPS,
      EPS_STD = BEST_EPS_EST_STD_DEV,
      ROIC = RETURN_ON_INVESTED_CAPITAL,
      Accruals = ACCRUAL_RATIO
    )
  # CONSENSUS_EPS: fallback = EPS t-4 (beräknas i preprocess)
  if (!"CONSENSUS_EPS" %in% names(fundamentals_df)) {
    fundamentals_df$CONSENSUS_EPS <- NA_real_
  }
  # Dilution = EQY_SH_OUT_t / EQY_SH_OUT_t-1y (beräknas i preprocess)
  fundamentals_df <- fundamentals_df %>%
    select(report_date, ticker, ACTUAL_EPS, CONSENSUS_EPS, EPS_STD, ROIC,
           EQY_SH_OUT, Accruals, everything())
  write_parquet(fundamentals_df, file.path(OUTPUT_DIR, "fundamentals.parquet"))
  cat(sprintf("Fundamentals: %d rader\n", nrow(fundamentals_df)))
} else {
  cat("Inga fundamentals hämtade. Skapa placeholder eller kontrollera tickers.\n")
}

# =============================================================================
# 2. MACRO (V2TX, Rates=Bund, Rates_FI, Rates_SE, MOVE)
# =============================================================================
# Rates = tysk 10Y (Bund) — ortogonal för HMM, undviker multikollinearitet FI/SE
# Rates_FI, Rates_SE — mappas till Local_Rate per bolag (FH/SS) i Layer 1
cat("Hämtar macro...\n")

macro_tickers <- c("V2TX Index", "GDBR10 Govt", "GFFI10 Govt", "GSGB10YR Govt", "MOVE Index")
macro_raw <- bdh(macro_tickers, "PX_LAST", START_DATE, END_DATE, include.non.trading.days = FALSE)

if (is.list(macro_raw) && !is.data.frame(macro_raw)) {
  # Wide till long och pivot
  macro_list <- lapply(names(macro_raw), function(n) {
    df <- as.data.frame(macro_raw[[n]])
    if (is.null(df) || nrow(df) == 0) return(NULL)
    col_name <- switch(n,
      "V2TX Index" = "V2TX",
      "GDBR10 Govt" = "Rates",
      "GFFI10 Govt" = "Rates_FI",
      "GSGB10YR Govt" = "Rates_SE",
      "MOVE Index" = "MOVE",
      n
    )
    names(df) <- c("date", col_name)
    df
  })
  valid_macro <- macro_list[!sapply(macro_list, is.null)]
  if (length(valid_macro) == 0) {
    stop("Ingen makrodata hämtad. Kontrollera Bloomberg-anslutning och tickers.", call. = FALSE)
  }
  macro_df <- Reduce(function(x, y) {
    if (is.null(y)) return(x)
    merge(x, y, by = "date", all = TRUE)
  }, valid_macro)
  # Behåll Rates_FI och Rates_SE separat — mappas per bolags hemvist (FH/SS) i Layer 1
} else {
  macro_df <- as.data.frame(macro_raw)
}

macro_df$date <- as.Date(macro_df$date)
if (nrow(macro_df) == 0) {
  stop("Makrodata tom. Kontrollera Bloomberg.", call. = FALSE)
}
macro_df$Breadth <- NA_real_  # Beräknas i Layer 1 från prices
# Välj kolumner: Rates = Bund (HMM), Rates_FI/Rates_SE = Local_Rate i Layer 1
macro_cols <- c("date", "V2TX", "Breadth", "Rates", "Rates_FI", "Rates_SE", "MOVE")
macro_cols <- macro_cols[macro_cols %in% names(macro_df)]
macro_df <- macro_df %>% select(all_of(macro_cols), everything())

write_parquet(macro_df, file.path(OUTPUT_DIR, "macro.parquet"))
cat(sprintf("Macro: %d rader\n", nrow(macro_df)))

# =============================================================================
# 3. NEWS (om BLPAPI News tillgänglig)
# =============================================================================
# Bloomberg News API kräver separat service. Ofta exporteras manuellt.
# Skapa placeholder om inte tillgängligt.

news_path <- file.path(OUTPUT_DIR, "news.parquet")
if (!file.exists(news_path)) {
  news_placeholder <- data.frame(
    ticker = character(0),
    date = as.Date(character(0)),
    text = character(0),
    stringsAsFactors = FALSE
  )
  write_parquet(news_placeholder, news_path)
  cat("News: placeholder skapad. Exportera manuellt från BN/NAN och ersätt.\n")
} else {
  cat("News.parquet finns redan.\n")
}

blpDisconnect()
cat("Klar. Sparat: fundamentals.parquet, macro.parquet\n")
