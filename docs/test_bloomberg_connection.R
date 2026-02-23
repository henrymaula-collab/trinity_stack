# =============================================================================
# Trinity Stack — Bloomberg API Connection Test
# =============================================================================
# Verifierar anslutning till B-PIPE/Terminal utan onödig datakvot.
# Kör: Rscript docs/test_bloomberg_connection.R
# =============================================================================

suppressPackageStartupMessages(library(Rblpapi))

TEST_SECURITY <- "VOLV-B SS Equity"
TEST_FIELDS <- "PX_LAST"
TEST_DAYS <- 5L
HOST <- "localhost"
PORT <- 8194L

cat("Trinity Stack — Bloomberg API Test\n")
cat("==================================\n")
cat(sprintf("Rblpapi version: %s\n", as.character(packageVersion("Rblpapi"))))
cat(sprintf("Mål: %s:%d\n", HOST, PORT))
cat("\n")

result <- tryCatch({
  blpConnect(host = HOST, port = PORT)
  cat("[OK] Ansluten till Bloomberg.\n\n")
  conn_ok <- TRUE
  NULL
}, error = function(e) {
  msg <- conditionMessage(e)
  if (grepl("connection refused|could not connect|connection", msg, ignore.case = TRUE)) {
    cat("[FEL] Anslutning misslyckades. Kontrollera:\n")
    cat("  - Är Bloomberg Terminal igång?\n")
    cat("  - Är B-PIPE aktiverat?\n")
    cat("  - Är port 8194 tillgänglig? (kan vara blockerad av brandvägg)\n")
  } else {
    cat(sprintf("[FEL] %s\n", msg))
  }
  list(conn_ok = FALSE, error = msg)
})

if (!is.null(result) && !result$conn_ok) {
  quit(save = "no", status = 1)
}

result2 <- tryCatch({
  end_date <- format(Sys.Date(), "%Y%m%d")
  start_date <- format(Sys.Date() - 14, "%Y%m%d")  # 2 veckor bakåt för att få 5 handelsdagar
  raw <- bdh(
    securities = TEST_SECURITY,
    fields = TEST_FIELDS,
    start.date = start_date,
    end.date = end_date,
    include.non.trading.days = FALSE
  )
  df <- if (is.data.frame(raw)) raw else as.data.frame(raw[[1]])
  if (is.null(df) || nrow(df) == 0) stop("Tomt svar från Bloomberg.")
  df <- tail(df, TEST_DAYS)  # Senaste 5 handelsdagar
  cat(sprintf("Hämtat %d rader för %s (PX_LAST, senaste %d handelsdagar):\n",
              nrow(df), TEST_SECURITY, TEST_DAYS))
  print(df)
  cat("\n[OK] Dataflöde bekräftat.\n")
  TRUE
}, error = function(e) {
  cat(sprintf("[FEL] Datahämtning misslyckades: %s\n", conditionMessage(e)))
  FALSE
})

blpDisconnect()
cat("\nAnslutningen kopplad från.\n")
cat("==================================\n")

if (!result2) quit(save = "no", status = 1)
cat("Test lyckades.\n")
