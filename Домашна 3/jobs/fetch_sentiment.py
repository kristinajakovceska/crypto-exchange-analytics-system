# jobs/fetch_sentiment.py
from __future__ import annotations
import os, sys, time, sqlite3
from pathlib import Path
from typing import List, Dict, Any
import requests, feedparser, pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SENT_CACHE = DATA_DIR / "sentiment_cache.sqlite"

AN = SentimentIntensityAnalyzer()
HEADERS = {"User-Agent": "homework3/1.0 (+https://localhost)"}

SYMBOL_TO_TICKER = {
    "BTC": "BTC",
    "BTCUSD": "BTC",
    "BTCUSDT": "BTC",
    "ETH": "ETH",
    "ETHUSD": "ETH",
    "ETHUSDT": "ETH",
}

def score_text(t: str) -> float:
    return float(AN.polarity_scores(t or "")["compound"])

def fetch_cryptopanic(ticker: str, key: str, per_page: int = 50) -> List[Dict[str, Any]]:
    url = "https://cryptopanic.com/api/v1/posts/"
    params = {
        "auth_token": key,
        "currencies": ticker,
        "kind": "news",
        "regions": "en",
        "public": "true",
        "filter": "hot",
        "page": 1,
    }
    items: List[Dict[str, Any]] = []
    for _ in range(2):  # ~100 items
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            break
        js = r.json()
        for it in js.get("results", []):
            items.append({
                "time": it.get("published_at"),
                "title": it.get("title"),
                "source": (it.get("source") or {}).get("title"),
                "url": (it.get("url") or "") or (it.get("source") or {}).get("url"),
            })
        nxt = js.get("next")
        if not nxt:
            break
        params["page"] += 1
        time.sleep(0.6)
    return items

def fetch_rss_many(feeds: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for f in feeds:
        d = feedparser.parse(f)
        src = d.get("feed", {}).get("title", f)
        for e in d.get("entries", []):
            rows.append({
                "time": e.get("published") or e.get("updated") or "",
                "title": e.get("title") or "",
                "source": src,
                "url": e.get("link") or "",
            })
    return rows

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m jobs.fetch_sentiment <SYMBOL>")
        sys.exit(1)
    symbol = sys.argv[1].upper()
    ticker = SYMBOL_TO_TICKER.get(symbol, None)
    if not ticker:
        print(f"Unsupported symbol {symbol}. Try BTCUSDT or ETHUSDT.")
        sys.exit(2)

    # 1) CryptoPanic if key available
    key = os.getenv("CRYPTOPANIC_KEY", "").strip()
    if key:
        news = fetch_cryptopanic(ticker, key)
    else:
        # 2) RSS fallback
        feeds = [
            "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
            "https://cointelegraph.com/rss",
            "https://www.binance.com/en/feed/rss",  # may be limited, ok to skip if empty
        ]
        news = fetch_rss_many(feeds)

    if not news:
        # still write an empty table so the UI shows "no headlines"
        df = pd.DataFrame(columns=["time", "title", "source", "url", "compound"])
    else:
        df = pd.DataFrame(news)
        # Parse time best-effort
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        # Score with VADER
        df["compound"] = df["title"].fillna("").map(score_text)

    # Sort newest first and keep a few hundred
    df = df.sort_values("time", ascending=False).head(400)

    # Write to cache + CSV
    tbl = f"sent_{symbol}"
    outdir = DATA_DIR / "sentiment"
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{symbol.lower()}_sentiment.csv"
    df.to_csv(csv_path, index=False)

    with sqlite3.connect(SENT_CACHE) as con:
        df.to_sql(tbl, con, if_exists="replace", index=False)

    print(f"Wrote {len(df)} sentiment rows -> {SENT_CACHE} [{tbl}]")
    print(f"Also wrote CSV -> {csv_path}")
    if not key:
        print("Tip: set CRYPTOPANIC_KEY env var to improve coverage (free key).")

if __name__ == "__main__":
    main()
