"""
cc_10y_top1000_cryptocompare.py

Fetch top N crypto assets from CryptoCompare and download
~10 years of daily OHLCV data for each asset into an SQLite
database `cryptos.db`, using the same schema as Homework 1.

Prerequisites.
    - Python 3.12+ in your venv
    - requests, SQLAlchemy installed
    - CryptoCompare API key stored in env var CRYPTOCOMPARE_API_KEY

Run.
    python cc_10y_top1000_cryptocompare.py
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List

import requests
from sqlalchemy import (
    Column,
    Date,
    Float,
    Integer,
    String,
    create_engine,
    Index,
)
from sqlalchemy.orm import DeclarativeBase, Session


# ============================================================
# 1. Configuration
# ============================================================

API_BASE = "https://min-api.cryptocompare.com/data"
TOPLIST_ENDPOINT = f"{API_BASE}/top/mktcapfull"
HISTODAY_ENDPOINT = f"{API_BASE}/v2/histoday"

# How many coins and days.
MAX_COINS = 1000          # we will test with smaller later
DAYS = 2000               # CryptoCompare max limit, about 5.5 years
TSYM = "USD"


MAX_WORKERS = 16          # used for concurrency if you add it later
REQUEST_TIMEOUT = 30
SLEEP_BETWEEN_REQUESTS = 0.2  # seconds. be gentle with rate limits

DATABASE_URL = "sqlite:///cryptos.db"


def get_api_key() -> str:
    key = os.getenv("CRYPTOCOMPARE_API_KEY")
    if not key:
        raise RuntimeError(
            "Missing CRYPTOCOMPARE_API_KEY env variable. "
            "Create a free API key at CryptoCompare and set it in your environment."
        )
    return key


API_KEY = get_api_key()
HEADERS = {"authorization": f"Apikey {API_KEY}"}


# ============================================================
# 2. SQLAlchemy model. same schema as Homework1 candles table
# ============================================================

class Base(DeclarativeBase):
    pass


class Candle(Base):
    __tablename__ = "candles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    exchange = Column(String(50), nullable=False)
    base_asset = Column(String(50), nullable=False)
    quote_asset = Column(String(50), nullable=False)
    symbol = Column(String(50), nullable=False)
    date = Column(Date, nullable=False)

    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)        # base volume
    quote_volume = Column(Float, nullable=True)   # quote volume, if available

    last_price_24h = Column(Float, nullable=True)
    volume_24h = Column(Float, nullable=True)
    high_24h = Column(Float, nullable=True)
    low_24h = Column(Float, nullable=True)
    liquidity_score = Column(Float, nullable=True)


Index(
    "idx_candles_symbol_date",
    Candle.symbol,
    Candle.date,
    unique=True,
)


# ============================================================
# 3. Data structures and helpers
# ============================================================

@dataclass
class CandleRow:
    exchange: str
    base_asset: str
    quote_asset: str
    symbol: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float | None = None

    def to_dict(self) -> dict:
        return {
            "exchange": self.exchange,
            "base_asset": self.base_asset,
            "quote_asset": self.quote_asset,
            "symbol": self.symbol,
            "date": self.date.date(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "quote_volume": self.quote_volume,
            "last_price_24h": None,
            "volume_24h": None,
            "high_24h": None,
            "low_24h": None,
            "liquidity_score": None,
        }


def unix_to_date(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


# ============================================================
# 4. API calls
# ============================================================

def fetch_top_coins(limit: int = MAX_COINS, tsym: str = TSYM) -> List[str]:
    """
    Get top coins by market cap from CryptoCompare.

    CryptoCompare enforces a max limit per request (often 100).
    So we page through results using the `page` parameter until
    we either reach `limit` coins or the API returns no more data.
    """
    per_request = 100  # safe maximum for this endpoint on most keys
    coins: List[str] = []
    page = 0

    while len(coins) < limit:
        remaining = limit - len(coins)
        this_limit = min(per_request, remaining)

        params = {
            "tsym": tsym,
            "limit": this_limit,
            "page": page,
        }

        resp = requests.get(
            TOPLIST_ENDPOINT,
            params=params,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        body = resp.json()

        print(
            f"DEBUG: /top/mktcapfull page={page}, limit={this_limit}, "
            f"Response={body.get('Response')}, Message={body.get('Message')}"
        )

        if body.get("Response") == "Error":
            msg = body.get("Message", "Unknown CryptoCompare error")
            raise RuntimeError(f"CryptoCompare error in top list call: {msg}")

        data = body.get("Data") or []
        if not data:
            # No more coins available. stop paging
            print("DEBUG: No more data returned from top list. stopping pagination.")
            break

        for item in data:
            info = item.get("CoinInfo") or {}
            symbol = info.get("Name")
            if symbol:
                coins.append(symbol)
                if len(coins) >= limit:
                    break

        page += 1

    if not coins:
        raise RuntimeError(
            "No coins returned from CryptoCompare top list after paging. "
            f"Response={body.get('Response')}, Message={body.get('Message')}"
        )

    print(f"DEBUG: Collected {len(coins)} coin symbols from top list.")
    return coins




def fetch_histoday(symbol: str, tsym: str = TSYM, limit_days: int = DAYS) -> List[CandleRow]:
    # CryptoCompare maximum allowed limit for histoday is 2000
    effective_limit = min(limit_days, 2000)

    params = {
        "fsym": symbol,
        "tsym": tsym,
        "limit": effective_limit,
        "aggregate": 1,
    }
    resp = requests.get(
        HISTODAY_ENDPOINT,
        params=params,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()

    if data.get("Response") == "Error":
        message = data.get("Message", "Unknown error")
        print(f"[WARN] histoday error for {symbol}. {message}")
        return []

    payload = data.get("Data", {}).get("Data", [])
    rows: List[CandleRow] = []

    for entry in payload:
        t = entry.get("time")
        o = entry.get("open")
        h = entry.get("high")
        l = entry.get("low")
        c = entry.get("close")
        vol_from = entry.get("volumefrom")
        vol_to = entry.get("volumeto")

        if t is None or o is None or h is None or l is None or c is None:
            continue

        dt = unix_to_date(int(t))

        rows.append(
            CandleRow(
                exchange="cryptocompare",
                base_asset=symbol,
                quote_asset=tsym,
                symbol=f"{symbol}{tsym}",
                date=dt,
                open=float(o),
                high=float(h),
                low=float(l),
                close=float(c),
                volume=float(vol_from) if vol_from is not None else 0.0,
                quote_volume=float(vol_to) if vol_to is not None else None,
            )
        )

    return rows




# ============================================================
# 5. Database operations
# ============================================================

def init_db(url: str = DATABASE_URL) -> None:
    engine = create_engine(url, future=True)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def insert_candles(engine, candle_rows: Iterable[CandleRow]) -> int:
    count = 0
    with Session(engine) as session:
        # chunk insert for memory friendliness
        batch: List[dict] = []
        for row in candle_rows:
            batch.append(row.to_dict())
            if len(batch) >= 1000:
                session.bulk_insert_mappings(Candle, batch)
                session.commit()
                count += len(batch)
                batch.clear()
        if batch:
            session.bulk_insert_mappings(Candle, batch)
            session.commit()
            count += len(batch)
    return count


# ============================================================
# 6. Main orchestration
# ============================================================

def main() -> None:
    print("➡️ Initialising database cryptos.db ...")
    engine = create_engine(DATABASE_URL, future=True)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    print(f"➡️ Fetching top {MAX_COINS} coins by market cap from CryptoCompare ...")
    coins = fetch_top_coins(limit=MAX_COINS, tsym=TSYM)
    print(f"   Got {len(coins)} coins. example. {coins[:10]}")

    total_inserted = 0
    start = time.time()

    for idx, symbol in enumerate(coins, start=1):
        print(f"➡️ [{idx}/{len(coins)}] Downloading history for {symbol}{TSYM} ...")
        try:
            rows = fetch_histoday(symbol, tsym=TSYM, limit_days=DAYS)
        except Exception as exc:
            print(f"   [ERROR] Failed to fetch {symbol}. {exc}")
            continue

        if not rows:
            print(f"   [WARN] No data rows for {symbol}. skipping.")
            continue

        inserted = insert_candles(engine, rows)
        total_inserted += inserted
        print(f"   Inserted {inserted} rows for {symbol}. total so far. {total_inserted}")

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    duration = time.time() - start
    print(f"✅ Done. Inserted {total_inserted} candles in {duration:.1f} seconds.")
    print("   Output database. cryptos.db")


if __name__ == "__main__":
    main()
