# =============================================================================
# Trinity Stack — Bloomberg DATA 1: PRICES + FX
# =============================================================================
# Kör först. Största datamängden — hålls separat för att undvika kvot.
# Output: prices.parquet, fx_sekeur.parquet
#
# Krav: Bloomberg Terminal igång, Rblpapi installerat
# install.packages("Rblpapi")
# =============================================================================

library(Rblpapi)
library(dplyr)
library(arrow)
library(tidyr)

# ---- Konfiguration ----
OUTPUT_DIR <- "data/raw"  # Ändra till din Trinity Stack-sökväg
START_DATE <- as.Date("2006-01-01")
END_DATE <- as.Date("2026-12-31")

# Point-in-time universum (ändra till dina tickers)
# Exempel: OMX Helsinki + Stockholm. Använd INDX_MWEIGHT med END_DATE_OVERRIDE
# eller manuell lista + dead_tickers.csv för survivorship bias-frihet
TICKERS <- c(
  "NOKIA FH Equity", "UPM FH Equity", "ORNBV FH Equity",  # Helsinki
  "VOLV-B SS Equity", "ERIC-B SS Equity", "ATCO-A SS Equity"  # Stockholm
)

# Stockholm-tickers för FX-konvertering (suffix SS)
SEK_TICKERS <- TICKERS[grepl(" SS ", TICKERS)]

# Justerade priser (KRITISKT för splits/utdelningar)
OVERRIDES <- c(
  CshAdjNormal = "true",
  CshAdjAbnormal = "true",
  CapChgExec = "true"
)

FIELDS <- c(
  "PX_OPEN",      # T+1 entry – KRITISKT
  "PX_LAST",
  "PX_HIGH",
  "PX_LOW",
  "PX_TURN_OVER",
  "PX_BID",
  "PX_ASK"
)

# ---- Anslut Bloomberg ----
blpConnect(host = "localhost", port = 8194L)

# ---- 1. Hämta priser ----
cat("Hämtar priser... (kan ta flera minuter)\n")
prices_raw <- bdh(
  securities = TICKERS,
  fields = FIELDS,
  start.date = START_DATE,
  end.date = END_DATE,
  overrides = OVERRIDES,
  include.non.trading.days = FALSE
)

# Konvertera till long format – Rblpapi returnerar list per security
if (is.list(prices_raw) && !is.data.frame(prices_raw)) {
  prices_list <- lapply(names(prices_raw), function(ticker) {
    df <- as.data.frame(prices_raw[[ticker]])
    if (is.null(df) || nrow(df) == 0) return(NULL)
    df$ticker <- gsub(" Equity$", "", ticker)
    df
  })
  prices_df <- bind_rows(prices_list[!sapply(prices_list, is.null)])
} else {
  prices_df <- as.data.frame(prices_raw)
  if (!"ticker" %in% names(prices_df)) {
    prices_df$ticker <- gsub(" Equity$", "", TICKERS[1])
  } else {
    prices_df$ticker <- gsub(" Equity$", "", prices_df$ticker)
  }
}

# Ticker rensat direkt — undvik join-fel mot fundamentals
SEK_TICKERS_CLEAN <- gsub(" Equity$", "", SEK_TICKERS)

# Säkerställ kolumnnamn (Bloomberg returnerar date som radnamn ibland)
if (!"date" %in% names(prices_df)) {
  prices_df$date <- as.Date(rownames(prices_df))
}
prices_df$date <- as.Date(prices_df$date)

# ---- 2. Hämta SEKEUR för Stockholm-tickers ----
cat("Hämtar SEKEUR Curncy...\n")
fx_df <- bdh(
  securities = "SEKEUR Curncy",
  fields = "PX_LAST",
  start.date = START_DATE,
  end.date = END_DATE,
  include.non.trading.days = FALSE
)
if (is.list(fx_df) && !is.data.frame(fx_df) && length(fx_df) > 0) {
  fx_df <- as.data.frame(fx_df[[1]])
}
names(fx_df) <- c("date", "sekeur")
fx_df$date <- as.Date(fx_df$date)

# Spara FX separat
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
write_parquet(as.data.frame(fx_df), file.path(OUTPUT_DIR, "fx_sekeur.parquet"))

# ---- 3. Konvertera SEK → EUR för Stockholm-tickers ----
prices_df$is_sek <- prices_df$ticker %in% SEK_TICKERS_CLEAN
prices_df <- prices_df %>% left_join(
  fx_df %>% rename(sekeur_rate = sekeur),
  by = "date"
)
price_cols <- c("PX_OPEN", "PX_LAST", "PX_HIGH", "PX_LOW", "PX_TURN_OVER", "PX_BID", "PX_ASK")
for (col in price_cols) {
  if (col %in% names(prices_df)) {
    prices_df[[col]] <- as.numeric(prices_df[[col]])
    prices_df[[col]] <- ifelse(
      prices_df$is_sek & !is.na(prices_df$sekeur_rate) & prices_df$sekeur_rate > 0,
      prices_df[[col]] / prices_df$sekeur_rate,
      prices_df[[col]]
    )
  }
}
prices_df$is_sek <- prices_df$sekeur_rate <- NULL

# ---- 4. Beräkna BID_ASK_SPREAD_PCT ----
prices_df$PX_MID <- (as.numeric(prices_df$PX_BID) + as.numeric(prices_df$PX_ASK)) / 2
prices_df$BID_ASK_SPREAD_PCT <- ifelse(
  prices_df$PX_MID > 0,
  100 * (as.numeric(prices_df$PX_ASK) - as.numeric(prices_df$PX_BID)) / prices_df$PX_MID,
  NA_real_
)
prices_df$PX_MID <- NULL

# ---- 5. Output-kolumner för Trinity ----
out_cols <- c("date", "ticker", "PX_OPEN", "PX_LAST", "PX_HIGH", "PX_LOW",
              "PX_TURN_OVER", "BID_ASK_SPREAD_PCT")
out_cols <- out_cols[out_cols %in% names(prices_df)]
prices_out <- prices_df[, out_cols]

# Sortera
prices_out <- prices_out %>% arrange(date, ticker)

# Spara
write_parquet(prices_out, file.path(OUTPUT_DIR, "prices.parquet"))

cat(sprintf("Sparat: %s\n", file.path(OUTPUT_DIR, "prices.parquet")))
cat(sprintf("Rader: %d\n", nrow(prices_out)))

blpDisconnect()
