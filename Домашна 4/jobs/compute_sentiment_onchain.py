# jobs/compute_sentiment_onchain.py
from __future__ import annotations
import sys
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

# Read candles from your database
from analytics_api.db import SessionLocal, Candle
from sqlalchemy import select, asc

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SENT_CACHE = DATA_DIR / "sentiment_cache.sqlite"
ONCHAIN_CACHE = DATA_DIR / "onchain_cache.sqlite"

def load_candles(symbol: str) -> pd.DataFrame:
    with SessionLocal() as s:
        rows = s.execute(
            select(
                Candle.date, Candle.open, Candle.high, Candle.low, Candle.close, Candle.volume
            )
            .where(Candle.symbol == symbol)
            .order_by(asc(Candle.date))
        ).all()
    if not rows:
        raise SystemExit(f"No candles for {symbol}. Check symbol name exactly as shown on /symbols.")
    df = pd.DataFrame(rows, columns=["date","open","high","low","close","volume"])
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna().sort_values("date").reset_index(drop=True)

def build_proxy_onchain(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["date"] = df["date"]
    vol = df["volume"].replace(0, np.nan)
    vol_scaled = vol / (vol.median() if np.isfinite(vol.median()) else 1.0)
    out["nvt"] = (df["close"] / vol_scaled).replace([np.inf, -np.inf], np.nan)
    rol = df["close"].rolling(180, min_periods=30).mean()
    out["mvrv"] = (df["close"] / rol).replace([np.inf, -np.inf], np.nan)
    for col in ["active_addresses","tx_count","exchange_in","exchange_out","whale_txs","hash_rate","tvl"]:
        out[col] = np.nan
    return out.dropna(how="all")

def build_synthetic_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    ret = df["close"].pct_change().fillna(0.0)
    comp = np.tanh(ret * 3.0)  # map returns to [-1,1]
    return pd.DataFrame({"time": df["date"], "compound": comp.astype(float)})

def write_sqlite(df: pd.DataFrame, db_path: Path, table: str):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        df.to_sql(table, con, if_exists="replace", index=False)

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m jobs.compute_sentiment_onchain <SYMBOL>")
        sys.exit(1)
    symbol = sys.argv[1].upper()
    candles = load_candles(symbol)
    onchain = build_proxy_onchain(candles)
    sentiment = build_synthetic_sentiment(candles)
    write_sqlite(onchain, ONCHAIN_CACHE, f"onchain_{symbol}")
    write_sqlite(sentiment, SENT_CACHE, f"sent_{symbol}")
    print(f"Wrote {len(onchain)} on-chain rows -> {ONCHAIN_CACHE} (onchain_{symbol})")
    print(f"Wrote {len(sentiment)} sentiment rows -> {SENT_CACHE} (sent_{symbol})")

if __name__ == "__main__":
    main()
