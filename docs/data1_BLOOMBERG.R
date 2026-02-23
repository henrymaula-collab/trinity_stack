# =============================================================================
# Trinity Stack — Bloomberg DATA 1: PRICES + FX
# =============================================================================
# Kör först. Största datamängden — hålls separat för att undvika kvot.
# Output: prices.parquet, fx_sekeur.parquet, nordic_historical_universe.csv
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
BDH_BATCH_SIZE <- 50L  # Undvik Daily Data Limits vid stort universum
UNIVERSE_CACHE <- file.path(OUTPUT_DIR, "nordic_historical_universe.csv")
USE_CACHED_UNIVERSE <- TRUE  # FALSE = hämta nytt PiT-universum från Bloomberg

# ---- Dynamiskt Point-in-Time Universum (eliminerar survivorship bias) ----
# Kvartalsvis sampling: fångar bolag som tillkommer/avlistas inom året
get_historical_universe <- function(indices, start_year, end_year) {
  all_tickers <- character(0)
  quarter_dates <- c("0331", "0630", "0930", "1231")  # Kvartalsvis
  for (idx in indices) {
    for (year in start_year:end_year) {
      for (q in quarter_dates) {
        date_str <- paste0(sprintf("%04d", year), q)
        tryCatch({
          members <- bds(idx, "INDX_MWEIGHT", overrides = c(END_DATE_OVERRIDE = date_str))
          if (!is.null(members) && nrow(members) > 0) {
            col_member <- names(members)[1]
            raw <- as.character(members[[col_member]])
            raw <- raw[!is.na(raw) & nchar(trimws(raw)) > 0]
            # Respektera befintlig börskod (FH/SS): endast lägg till " Equity" i slutet
            tickers <- ifelse(grepl(" Equity$", raw), raw, paste(trimws(raw), "Equity"))
            all_tickers <- c(all_tickers, tickers)
          }
        }, error = function(e) {
          warning(sprintf("Kunde inte hämta %s för %s: %s", idx, date_str, conditionMessage(e)))
        })
      }
    }
  }
  unique(all_tickers)
}

# ---- Anslut Bloomberg (krävs för PiT-universum) ----
blpConnect(host = "localhost", port = 8194L)

# Läs eller generera universum
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
if (USE_CACHED_UNIVERSE && file.exists(UNIVERSE_CACHE)) {
  TICKERS <- read.csv(UNIVERSE_CACHE, stringsAsFactors = FALSE)$ticker
  cat(sprintf("Universum från cache: %d tickers (%s)\n", length(TICKERS), UNIVERSE_CACHE))
} else {
  target_indices <- c("OMXSSC Index", "OMXHSC Index")  # Stockholm & Helsinki Small Cap
  TICKERS <- get_historical_universe(target_indices, 2006, 2026)
  if (length(TICKERS) == 0) {
    warning("PiT-universum tomt. Fallback till manuell lista.")
    TICKERS <- c("NOKIA FH Equity", "UPM FH Equity", "ORNBV FH Equity",
                 "VOLV-B SS Equity", "ERIC-B SS Equity", "ATCO-A SS Equity")
  } else {
    write.csv(data.frame(ticker = TICKERS), UNIVERSE_CACHE, row.names = FALSE)
    cat(sprintf("PiT-universum sparad: %d unika tickers\n", length(TICKERS)))
  }
}

# Stockholm-tickers för FX-konvertering (suffix SS)
SEK_TICKERS <- TICKERS[grepl(" SS ", TICKERS)]

# Justerade priser (KRITISKT för splits/utdelningar)
OVERRIDES <- c(
  CshAdjNormal = "true",
  CshAdjAbnormal = "true",
  CapChgExec = "true"
)

# Aldrig PX_VOLUME. VWAP_CP = volymviktad kurs — KRITISKT för Layer 6 TCA & slippage;
# PX_LAST i illikvida småbolag är brusigt/manipulerbart i slutauktion; VWAP ger empiriskt underlag för Market Impact.
FIELDS <- c("PX_OPEN", "PX_LAST", "PX_HIGH", "PX_LOW", "PX_TURN_OVER", "PX_BID", "PX_ASK", "VWAP_CP")

# ---- Tracker: Läs status_log, hoppa över Success ----
STATUS_LOG <- file.path(OUTPUT_DIR, "status_log.csv")
if (file.exists(STATUS_LOG)) {
  status_df <- read.csv(STATUS_LOG, stringsAsFactors = FALSE)
  success_tickers <- status_df$Ticker[status_df$Status == "Success"]
  TICKERS_PENDING <- setdiff(TICKERS, success_tickers)
  cat(sprintf("Tracker: %d Success (hoppas över), %d kvar att hämta\n", length(success_tickers), length(TICKERS_PENDING)))
} else {
  status_df <- data.frame(Ticker = character(0), Status = character(0), stringsAsFactors = FALSE)
  TICKERS_PENDING <- TICKERS
}

# ---- 1. Hämta priser (iterativ batching, checkpoint, robust felhantering) ----
all_batch_paths <- character(0)
if (length(TICKERS_PENDING) > 0) {
  batches <- split(TICKERS_PENDING, ceiling(seq_along(TICKERS_PENDING) / BDH_BATCH_SIZE))
  for (b in seq_along(batches)) {
    chunk <- batches[[b]]
    cat(sprintf("  Batch %d/%d: %d tickers\n", b, length(batches), length(chunk)))
    batch_data <- list()
    for (ticker in chunk) {
      res <- tryCatch({
        raw <- bdh(
          securities = ticker,
          fields = FIELDS,
          start.date = START_DATE,
          end.date = END_DATE,
          overrides = OVERRIDES,
          include.non.trading.days = FALSE
        )
        df <- if (is.data.frame(raw)) raw else as.data.frame(raw[[1]])
        if (is.null(df) || nrow(df) == 0) {
          list(status = "Failed", data = NULL)
        } else {
          df$ticker <- gsub(" Equity$", "", ticker)
          list(status = "Success", data = df)
        }
      }, error = function(e) {
        list(status = "Failed", data = NULL)
      })
      idx <- which(status_df$Ticker == ticker)
      if (length(idx) > 0) {
        status_df$Status[idx] <- res$status
      } else {
        status_df <- rbind(status_df, data.frame(Ticker = ticker, Status = res$status, stringsAsFactors = FALSE))
      }
      if (!is.null(res$data)) batch_data[[length(batch_data) + 1L]] <- res$data
    }
    if (length(batch_data) > 0) {
      batch_df <- bind_rows(batch_data)
      batch_path <- file.path(OUTPUT_DIR, sprintf("prices_batch_%d.parquet", b))
      write_parquet(batch_df, batch_path)
      all_batch_paths <- c(all_batch_paths, batch_path)
    }
    write.csv(status_df, STATUS_LOG, row.names = FALSE)
  }
}

# Slå ihop alla batchar + eventuella tidigare batchfiler
batch_files <- list.files(OUTPUT_DIR, pattern = "^prices_batch_.*\\.parquet$", full.names = TRUE)
if (length(batch_files) > 0) {
  prices_df <- bind_rows(lapply(batch_files, read_parquet))
} else {
  stop("Inga prismatalogar (batchar) tillgängliga.", call. = FALSE)
}
if (!"ticker" %in% names(prices_df)) prices_df$ticker <- character(0)

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
price_cols <- c("PX_OPEN", "PX_LAST", "PX_HIGH", "PX_LOW", "PX_TURN_OVER", "PX_BID", "PX_ASK", "VWAP_CP")
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
              "PX_TURN_OVER", "BID_ASK_SPREAD_PCT", "VWAP_CP")
out_cols <- out_cols[out_cols %in% names(prices_df)]
prices_out <- prices_df[, out_cols]

# Sortera
prices_out <- prices_out %>% arrange(date, ticker)

# Spara
write_parquet(prices_out, file.path(OUTPUT_DIR, "prices.parquet"))

cat(sprintf("Sparat: %s\n", file.path(OUTPUT_DIR, "prices.parquet")))
cat(sprintf("Rader: %d\n", nrow(prices_out)))

blpDisconnect()
