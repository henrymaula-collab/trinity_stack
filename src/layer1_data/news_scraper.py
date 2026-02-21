"""
Yahoo News Scraper for Nordic equities.
Fetches free RSS headlines. Output schema: [date, ticker, text].
Point-in-time safe: uses published date from feed only.
"""

from __future__ import annotations

import logging
from pathlib import Path

import feedparser
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

YAHOO_RSS_URL = "https://finance.yahoo.com/rss/headline?s={ticker}"


class YahooNewsScraper:
    """
    Fetches Yahoo Finance RSS headlines for Nordic tickers.
    Output conforms to Layer 1 schema expected by run_pipeline and Layer 4 NLP.
    """

    def fetch_news(self, tickers: list[str]) -> pd.DataFrame:
        """
        Fetch news headlines for each ticker from Yahoo RSS.

        Returns:
            DataFrame with columns ['date', 'ticker', 'text'].
            date: pandas Timestamp, UTC, tz-naive.
            text: combined title + summary.
            Sorted by date ascending.
        """
        rows: list[dict] = []

        for ticker in tickers:
            url = YAHOO_RSS_URL.format(ticker=ticker)
            try:
                feed = feedparser.parse(url)
            except Exception as e:
                logging.warning("Failed to fetch %s: %s", ticker, e)
                continue

            if not getattr(feed, "entries", None):
                logging.debug("No entries for %s", ticker)
                continue

            for entry in feed.entries:
                published = self._parse_published(entry)
                if published is None:
                    continue

                title = getattr(entry, "title", None) or ""
                summary = getattr(entry, "summary", None) or ""
                text = self._combine_text(title, summary)

                if not text.strip():
                    continue

                rows.append({
                    "date": published,
                    "ticker": ticker,
                    "text": text.strip(),
                })

        if not rows:
            return pd.DataFrame(columns=["date", "ticker", "text"])

        df = pd.DataFrame(rows)
        df = df[["date", "ticker", "text"]]
        df = df.sort_values("date", ascending=True).reset_index(drop=True)
        return df

    def _parse_published(self, entry) -> pd.Timestamp | None:
        """Convert feed published to pandas Timestamp, UTC, tz-naive."""
        try:
            raw = getattr(entry, "published_parsed", None)
            if raw is not None:
                ts = pd.Timestamp(*raw[:6], tz="UTC").tz_localize(None)
                return ts

            raw_str = getattr(entry, "published", None)
            if raw_str:
                ts = pd.to_datetime(raw_str, utc=True)
                return ts.tz_localize(None) if ts.tzinfo else ts
        except Exception:
            pass
        return None

    def _combine_text(self, title: str, summary: str) -> str:
        """Combine title and summary into a single text block."""
        parts = [t.strip() for t in (title, summary) if t and str(t).strip()]
        return " | ".join(parts)

    def update_parquet(
        self,
        tickers: list[str],
        file_path: str = "data/raw/news.parquet",
    ) -> None:
        """
        Fetch news, append to existing parquet (if present),
        drop duplicates on ['ticker', 'date', 'text'], overwrite file.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        new_df = self.fetch_news(tickers)
        if new_df.empty:
            logging.info("No new news fetched. Keeping existing file if present.")
            if path.exists():
                return
            new_df.to_parquet(path, index=False)
            return

        if path.exists():
            existing = pd.read_parquet(path)
            if not existing.empty and set(existing.columns) >= {"date", "ticker", "text"}:
                combined = pd.concat([existing, new_df], ignore_index=True)
            else:
                combined = new_df
        else:
            combined = new_df

        combined = combined.drop_duplicates(subset=["ticker", "date", "text"])
        combined = combined.sort_values("date", ascending=True).reset_index(drop=True)
        combined.to_parquet(path, index=False)
        logging.info("Saved news.parquet: %d rows", len(combined))
