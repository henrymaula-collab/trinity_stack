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
  "ACCRUAL_RATIO",
  "BEST_DIV_YLD",    # Uppskattning; gles för nordiska småbolag utan analytikertäckning
  "DIVIDEND_YIELD",  # Historisk/trailing; fallback när BEST_DIV_YLD saknas
  "CUR_MKT_CAP"      # Marknadsvärde; använd aldrig PX_VOLUME för värdering
)
opt <- c(periodicitySelection = "QUARTERLY")

# ---- Tracker: Hoppa över Success ----
FUND_STATUS_LOG <- file.path(OUTPUT_DIR, "fundamentals_status_log.csv")
if (file.exists(FUND_STATUS_LOG)) {
  fund_status <- read.csv(FUND_STATUS_LOG, stringsAsFactors = FALSE)
  success_tick <- fund_status$Ticker[fund_status$Status == "Success"]
  TICKERS_PENDING <- setdiff(TICKERS, success_tick)
  cat(sprintf("Tracker: %d Success (hoppas över), %d kvar\n", length(success_tick), length(TICKERS_PENDING)))
} else {
  fund_status <- data.frame(Ticker = character(0), Status = character(0), stringsAsFactors = FALSE)
  TICKERS_PENDING <- TICKERS
}

fund_list <- list()
for (ticker in TICKERS_PENDING) {
  res <- tryCatch({
    df <- bdh(ticker, fund_fields, START_DATE, END_DATE, options = opt)
    if (is.null(df) || nrow(df) == 0) {
      list(status = "Failed", data = NULL)
    } else {
      df$ticker <- gsub(" Equity$", "", ticker)
      list(status = "Success", data = df)
    }
  }, error = function(e) {
    list(status = "Failed", data = NULL)
  })
  # Uppdatera tracker-loggen utan att skapa dubbletter
  idx <- which(fund_status$Ticker == ticker)
  if (length(idx) > 0) {
    fund_status$Status[idx] <- res$status
  } else {
    fund_status <- rbind(fund_status, data.frame(Ticker = ticker, Status = res$status, stringsAsFactors = FALSE))
  }
  if (!is.null(res$data)) fund_list[[ticker]] <- res$data
}
write.csv(fund_status, FUND_STATUS_LOG, row.names = FALSE)
# Slå ihop nya (bind_rows) + befintliga parquet
# Inkludera data för redan Success-tickers från tidigare körning
fund_list_valid <- fund_list[!sapply(fund_list, is.null)]
fund_path <- file.path(OUTPUT_DIR, "fundamentals.parquet")
fundamentals_new <- if (length(fund_list_valid) > 0) bind_rows(fund_list_valid) else NULL
if (!is.null(fundamentals_new) && file.exists(fund_path)) {
  fund_existing <- read_parquet(fund_path)
  ticker_col <- if ("ticker" %in% names(fund_existing)) "ticker" else "Ticker"
  # Säker filtrering för att undvika dubbletter vid uppdatering
  fund_existing <- fund_existing %>%
    filter(!(!!sym(ticker_col) %in% TICKERS_PENDING))
  fundamentals_raw <- bind_rows(fund_existing, fundamentals_new)
} else if (!is.null(fundamentals_new)) {
  fundamentals_raw <- fundamentals_new
} else if (file.exists(fund_path)) {
  fundamentals_raw <- read_parquet(fund_path)
} else {
  stop("Inga fundamentals hämtade och ingen befintlig fil.", call. = FALSE)
}

# Namngivning för Trinity
if (nrow(fundamentals_raw) > 0) {
  fundamentals_df <- fundamentals_raw
  if ("date" %in% names(fundamentals_df) && !"report_date" %in% names(fundamentals_df)) {
    fundamentals_df <- fundamentals_df %>% rename(report_date = date)
  }
  renames <- list(ACTUAL_EPS = "BEST_EPS", EPS_STD = "BEST_EPS_EST_STD_DEV",
                  ROIC = "RETURN_ON_INVESTED_CAPITAL", Accruals = "ACCRUAL_RATIO")
  for (nn in names(renames)) {
    if (renames[[nn]] %in% names(fundamentals_df)) {
      fundamentals_df <- fundamentals_df %>% rename(!!nn := !!sym(renames[[nn]]))
    }
  }
  # CONSENSUS_EPS: fallback = EPS t-4 (beräknas i preprocess)
  if (!"CONSENSUS_EPS" %in% names(fundamentals_df)) {
    fundamentals_df$CONSENSUS_EPS <- NA_real_
  }
  # Vektoriserad och säker coalesce-logik för utdelningar
  if (!"BEST_DIV_YLD" %in% names(fundamentals_df)) fundamentals_df$BEST_DIV_YLD <- NA_real_
  if (!"DIVIDEND_YIELD" %in% names(fundamentals_df)) fundamentals_df$DIVIDEND_YIELD <- NA_real_
  fundamentals_df <- fundamentals_df %>%
    mutate(Dividend_Yield = dplyr::coalesce(BEST_DIV_YLD, DIVIDEND_YIELD)) %>%
    select(-BEST_DIV_YLD, -DIVIDEND_YIELD)
  if ("CUR_MKT_CAP" %in% names(fundamentals_df)) {
    fundamentals_df <- fundamentals_df %>% rename(Market_Cap = CUR_MKT_CAP)
  } else {
    fundamentals_df$Market_Cap <- NA_real_
  }
  fundamentals_df <- fundamentals_df %>%
    select(report_date, ticker, ACTUAL_EPS, CONSENSUS_EPS, EPS_STD, ROIC,
           EQY_SH_OUT, Dividend_Yield, Market_Cap, Accruals, everything())
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
