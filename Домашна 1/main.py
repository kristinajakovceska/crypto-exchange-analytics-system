from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import List, Optional, Iterable

import requests
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Date,
    UniqueConstraint,
    Index,
    select,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# Configuration
# ============================================================


COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
BINANCE_BASE_URL = "https://api.binance.com"

# Up to 1000 coins requested from CoinGecko
TOP_N_COINS = 1000

# Database. SQLite file in project folder
DATABASE_URL = "sqlite:///crypto.db"

# Thread pool size. Adjust based on your machine and API rate limits
MAX_WORKERS = 12


class Settings:
    """
    Minimal settings wrapper.
    If you want environment variables, you can replace this with Pydantic BaseSettings.
    """
    top_n: int = TOP_N_COINS
    quote_asset: str = "USDT"
    exchange_name: str = "binance"


settings = Settings()


# ============================================================
# Database models
# ============================================================


class Base(DeclarativeBase):
    pass


class Candle(Base):
    __tablename__ = "candles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    exchange = Column(String(32), nullable=False, index=True)
    base_asset = Column(String(32), nullable=False, index=True)
    quote_asset = Column(String(32), nullable=False, index=True)
    symbol = Column(String(64), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)

    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    quote_volume = Column(Float, nullable=True)

    last_price_24h = Column(Float, nullable=True)
    volume_24h = Column(Float, nullable=True)
    high_24h = Column(Float, nullable=True)
    low_24h = Column(Float, nullable=True)
    liquidity_score = Column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint("exchange", "symbol", "date", name="uq_exchange_symbol_date"),
        Index("ix_symbol_date", "symbol", "date"),
    )


engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(engine)


# ============================================================
# Data classes
# ============================================================


@dataclass
class SymbolInfo:
    """
    Represents a tradeable symbol on Binance together with the CoinGecko id.
    """
    coingecko_id: str
    coingecko_symbol: str
    name: str
    base_asset: str
    quote_asset: str
    binance_symbol: str
    rank: int  # position in top N list

    def __str__(self) -> str:
        return f"{self.binance_symbol} (rank {self.rank})"


# ============================================================
# Helpers. REST wrappers
# ============================================================


def get_json(url: str, params: Optional[dict] = None) -> dict | list:
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                # rate limited. wait and retry
                time.sleep(2.0 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == 2:
                raise
            time.sleep(1.0 * (attempt + 1))
    # Should not reach here
    raise RuntimeError(f"Failed to GET {url}")


# ============================================================
# Filter 1. Get top symbols and map to Binance
# ============================================================


def filter1_get_top_symbols(limit: int) -> List[SymbolInfo]:
    """
    Filter 1.
    1. Get top `limit` coins by market cap from CoinGecko.
    2. Fetch Binance exchangeInfo.
    3. Keep only symbols that are actively traded on Binance with USDT as quote asset.
    4. Exclude delisted or duplicate pairs.

    Result. ordered list of SymbolInfo, sorted by market cap rank.
    """
    print(f"[Filter 1] Loading top {limit} coins from CoinGecko...")
    per_page = 250
    pages = math.ceil(limit / per_page)
    coingecko_coins: list[dict] = []

    for page in range(1, pages + 1):
        remaining = limit - len(coingecko_coins)
        if remaining <= 0:
            break

        page_size = min(per_page, remaining)
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": page_size,
            "page": page,
            "sparkline": "false",
        }
        data = get_json(f"{COINGECKO_BASE_URL}/coins/markets", params=params)
        if not data:
            break
        coingecko_coins.extend(data)

    print(f"[Filter 1] Got {len(coingecko_coins)} coins from CoinGecko.")

    print("[Filter 1] Loading Binance exchangeInfo...")
    exchange_info = get_json(f"{BINANCE_BASE_URL}/api/v3/exchangeInfo")
    symbols = exchange_info["symbols"]

    # Build lookup of base asset to Binance symbol where quote == USDT and status == TRADING
    valid_pairs: dict[str, dict] = {}
    for s in symbols:
        if s.get("status") != "TRADING":
            continue
        base = s["baseAsset"].upper()
        quote = s["quoteAsset"].upper()
        if quote != settings.quote_asset:
            continue
        symbol_code = s["symbol"].upper()
        # keep first occurrence per baseAsset
        if base not in valid_pairs:
            valid_pairs[base] = {
                "symbol": symbol_code,
                "base": base,
                "quote": quote,
            }

    print(f"[Filter 1] Binance USDT trading pairs found for {len(valid_pairs)} base assets.")

    mapped: List[SymbolInfo] = []
    rank = 1
    for coin in coingecko_coins:
        cg_id = coin["id"]
        cg_symbol = coin["symbol"].upper()
        name = coin["name"]
        base_asset = cg_symbol

        # Some symbols differ between CoinGecko and Binance. simple mapping by exact symbol
        if base_asset in valid_pairs:
            p = valid_pairs[base_asset]
            info = SymbolInfo(
                coingecko_id=cg_id,
                coingecko_symbol=cg_symbol,
                name=name,
                base_asset=p["base"],
                quote_asset=p["quote"],
                binance_symbol=p["symbol"],
                rank=rank,
            )
            mapped.append(info)

        rank += 1
        if rank > limit:
            break

    print(f"[Filter 1] Mapped {len(mapped)} coins to Binance USDT trading pairs.")
    return mapped


# ============================================================
# Filter 2. Check last stored date
# ============================================================


def filter2_check_last_date(db: Session, symbol: SymbolInfo) -> date:
    """
    Filter 2.
    For a given symbol find the last stored date in the database.

    If no data exists.
        - return today minus 10 years.

    If data exists.
        - return last_date + 1 day (start point for missing data).
    """
    stmt = (
        select(Candle)
        .where(
            Candle.exchange == settings.exchange_name,
            Candle.symbol == symbol.binance_symbol,
        )
        .order_by(Candle.date.desc())
        .limit(1)
    )
    result = db.execute(stmt).scalars().first()
    if result is None:
        start = date.today() - timedelta(days=365 * 10)
        return start
    else:
        return result.date + timedelta(days=1)


# ============================================================
# Filter 3. Download and store missing data
# ============================================================


def _download_klines(symbol_code: str, start: datetime, end: datetime) -> List[list]:
    """
    Low level helper for Binance klines, interval 1d.
    It handles pagination with limit 1000.
    Returns list of raw kline arrays from Binance.
    """
    ms_in_day = 24 * 60 * 60 * 1000
    start_ts = int(start.timestamp() * 1000)
    end_ts = int(end.timestamp() * 1000)
    all_rows: list[list] = []

    while True:
        params = {
            "symbol": symbol_code,
            "interval": "1d",
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": 1000,
        }
        data = get_json(f"{BINANCE_BASE_URL}/api/v3/klines", params=params)
        if not data:
            break

        all_rows.extend(data)
        last_open_time = data[-1][0]
        next_start = last_open_time + ms_in_day
        if next_start >= end_ts:
            break
        start_ts = next_start

    return all_rows


def _get_24h_stats(symbol_code: str) -> dict:
    """
    Helper to load 24h statistics from Binance.
    """
    data = get_json(f"{BINANCE_BASE_URL}/api/v3/ticker/24hr", params={"symbol": symbol_code})
    return data


def filter3_fill_missing_data(symbol: SymbolInfo, start_date: date) -> int:
    """
    Filter 3.
    Download missing OHLCV data for the given symbol and store it in the database.
    Returns number of inserted rows.
    """
    end_date = date.today()
    if start_date > end_date:
        # nothing to do
        return 0

    print(f"[Filter 3] {symbol.binance_symbol}. downloading data from {start_date} to {end_date}...")

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.min.time())

    klines = _download_klines(symbol.binance_symbol, start_dt, end_dt)
    if not klines:
        print(f"[Filter 3] {symbol.binance_symbol}. no klines returned.")
        return 0

    stats_24h = {}
    try:
        stats_24h = _get_24h_stats(symbol.binance_symbol)
    except Exception:
        stats_24h = {}

    last_price_24h = float(stats_24h.get("lastPrice")) if "lastPrice" in stats_24h else None
    vol_24h = float(stats_24h.get("volume")) if "volume" in stats_24h else None
    high_24h = float(stats_24h.get("highPrice")) if "highPrice" in stats_24h else None
    low_24h = float(stats_24h.get("lowPrice")) if "lowPrice" in stats_24h else None

    # simple liquidity proxy. quoteVolume / number_of_days
    liquidity_score = None
    try:
        qv = float(stats_24h.get("quoteVolume"))
        liquidity_score = qv / max(len(klines), 1)
    except Exception:
        liquidity_score = None

    inserted = 0
    db: Session = SessionLocal()
    try:
        for row in klines:
            # Kline format.
            # 0  open time
            # 1  open
            # 2  high
            # 3  low
            # 4  close
            # 5  volume
            # 6  close time
            # 7  quote asset volume
            open_time_ms = row[0]
            open_price = float(row[1])
            high_price = float(row[2])
            low_price = float(row[3])
            close_price = float(row[4])
            base_vol = float(row[5])
            quote_vol = float(row[7])

            candle_date = datetime.utcfromtimestamp(open_time_ms / 1000).date()
            # skip if outside requested range (safety)
            if candle_date < start_date or candle_date > end_date:
                continue

            candle = Candle(
                exchange=settings.exchange_name,
                base_asset=symbol.base_asset,
                quote_asset=symbol.quote_asset,
                symbol=symbol.binance_symbol,
                date=candle_date,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=base_vol,
                quote_volume=quote_vol,
                last_price_24h=last_price_24h,
                volume_24h=vol_24h,
                high_24h=high_24h,
                low_24h=low_24h,
                liquidity_score=liquidity_score,
            )
            db.merge(candle)
            inserted += 1

        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    print(f"[Filter 3] {symbol.binance_symbol}. inserted or updated {inserted} candles.")
    return inserted


# ============================================================
# Pipeline orchestration
# ============================================================


def process_symbol(symbol: SymbolInfo) -> tuple[str, int]:
    """
    Full pipeline for a single symbol.
        - Filter 2: find start date
        - Filter 3: download and store data
    """
    db = SessionLocal()
    try:
        start_date = filter2_check_last_date(db, symbol)
    finally:
        db.close()

    inserted = filter3_fill_missing_data(symbol, start_date)
    return symbol.binance_symbol, inserted


def run_full_pipeline() -> None:
    """
    Run complete pipeline.
        - Initialize database
        - Filter 1: get symbols
        - For each symbol in parallel run process_symbol
        - Print timing statistics for the empty database case
    """
    init_db()

    t0 = time.perf_counter()
    symbols = filter1_get_top_symbols(settings.top_n)
    if not symbols:
        print("No symbols mapped. check APIs or internet connection.")
        return

    print(f"[Pipeline] Processing {len(symbols)} symbols with up to {MAX_WORKERS} workers...")

    total_inserted = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_symbol = {executor.submit(process_symbol, s): s for s in symbols}
        for future in as_completed(future_to_symbol):
            sym = future_to_symbol[future]
            try:
                symbol_code, inserted = future.result()
                total_inserted += inserted
                print(f"[Pipeline] {symbol_code}. done. rows added. {inserted}")
            except Exception as exc:
                print(f"[Pipeline] {sym.binance_symbol}. error. {exc}")

    t1 = time.perf_counter()
    elapsed = t1 - t0
    print("=" * 60)
    print(f"[Pipeline] Completed for {len(symbols)} symbols.")
    print(f"[Pipeline] Total candles inserted or updated. {total_inserted}")
    print(f"[Pipeline] Total time. {elapsed:.2f} seconds")
    print("=" * 60)


# ============================================================
# Entry point
# ============================================================


if __name__ == "__main__":
    run_full_pipeline()
