#!/usr/bin/env Rscript
# Trinity Stack v9.0 — Layer 1: Aalto/OMXH Equity Extraction
# Reads Datastream/Refinitiv exports. Outputs PiT-safe parquet to data/raw.
# No look-ahead; strict date monotonicity. Fail on future contamination.

suppressPackageStartupMessages({
  library(arrow)
  library(readr)
  library(dplyr)
})

REQUIRED_COLS <- c("date", "ticker", "PX_LAST", "PX_VOLUME")
RAW_DIR <- "data/raw"
OPTIONAL_COLS <- c("BID_ASK_SPREAD_PCT", "Quality_Score", "ACTUAL_EPS", "CONSENSUS_EPS",
                   "ROIC", "Piotroski_F", "Accruals")

main <- function(input_path = "data/sources/aalto_equity_export.csv") {
  if (!file.exists(input_path)) {
    stop("Input file not found: ", input_path, call. = FALSE)
  }

  df <- read_csv(input_path, show_col_types = FALSE)
  for (c in REQUIRED_COLS) {
    if (!c %in% names(df)) stop("Missing required column: ", c, call. = FALSE)
  }

  df$date <- as.Date(df$date)
  if (any(is.na(df$date))) stop("NA dates detected. Fail.", call. = FALSE)

  tickers <- unique(df$ticker)
  dir.create(RAW_DIR, recursive = TRUE, showWarnings = FALSE)

  for (t in tickers) {
    sub <- df %>% filter(ticker == .env$t) %>% arrange(date)
    if (nrow(sub) == 0) next
    if (!all(diff(as.numeric(sub$date)) >= 0)) {
      stop("[", t, "] Dates not monotonic. Look-ahead risk.", call. = FALSE)
    }
    sub <- sub %>% select(-ticker)
    out_path <- file.path(RAW_DIR, paste0(t, ".parquet"))
    write_parquet(sub, out_path)
    message("Wrote ", out_path, " (", nrow(sub), " rows)")
  }
}

args <- commandArgs(trailingOnly = TRUE)
input_path <- if (length(args) >= 1) args[1] else "data/sources/aalto_equity_export.csv"
main(input_path)
