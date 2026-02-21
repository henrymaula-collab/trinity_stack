#!/usr/bin/env Rscript
# Trinity Stack v9.0 — Layer 1: Hanken Macro Extraction
# Extracts V2TX, Breadth, Rates for Layer 2 regime detection.
# Output: data/raw/macro_regime.parquet. PiT-safe, no look-ahead.

suppressPackageStartupMessages({
  library(arrow)
  library(readr)
  library(dplyr)
})

REQUIRED_COLS <- c("date", "V2TX", "Breadth", "Rates")
OUT_PATH <- "data/raw/macro_regime.parquet"

main <- function(input_path = "data/sources/hanken_macro_export.csv") {
  if (!file.exists(input_path)) {
    stop("Input file not found: ", input_path, call. = FALSE)
  }

  df <- read_csv(input_path, show_col_types = FALSE)
  for (c in REQUIRED_COLS) {
    if (!c %in% names(df)) stop("Missing required column: ", c, call. = FALSE)
  }

  df$date <- as.Date(df$date)
  if (any(is.na(df$date))) stop("NA dates detected. Fail.", call. = FALSE)
  df <- df %>% arrange(date)
  if (!all(diff(as.numeric(df$date)) >= 0)) {
    stop("Dates not monotonic. Look-ahead risk.", call. = FALSE)
  }

  out <- df %>% select(all_of(REQUIRED_COLS))
  dir.create(dirname(OUT_PATH), recursive = TRUE, showWarnings = FALSE)
  write_parquet(out, OUT_PATH)
  message("Wrote ", OUT_PATH, " (", nrow(out), " rows)")
}

args <- commandArgs(trailingOnly = TRUE)
input_path <- if (length(args) >= 1) args[1] else "data/sources/hanken_macro_export.csv"
main(input_path)
