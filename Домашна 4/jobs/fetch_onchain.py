# jobs/fetch_onchain.py
from __future__ import annotations
import sys, sqlite3, time
from pathlib import Path
from typing import Dict, Any, List
import requests
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ONCHAIN_CACHE = DATA_DIR / "onchain_cache.sqlite"

# Symbol → asset ids
ASSET_MAP = {
    "BTC": {"cg": "bitcoin", "bc": "btc"},
    "BTCUSDT": {"cg": "bitcoin", "bc": "btc"},
    "BTCUSD": {"cg": "bitcoin", "bc": "btc"},
    "ETH": {"cg": "ethereum", "bc": "eth"},
    "ETHUSDT": {"cg": "ethereum", "bc": "eth"},
    "ETHUSD": {"cg": "ethereum", "bc": "eth"},
}

HEADERS = {"User-Agent": "homework3/1.0 (+https://localhost)"}

def get_json(url: str, params: Dict[str, Any] | None = None, retries: int = 3, sleep: float = 0.8):
    for i in range(retries):
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        if r.status_code == 200:
            return r.json()
        time.sleep(sleep)
    raise RuntimeError(f"HTTP {r.status_code} for {url}")

def fetch_blockchain_chart(chart: str) -> pd.DataFrame:
    # https://api.blockchain.info/charts/<chart>?timespan=all&format=json
    data = get_json(f"https://api.blockchain.info/charts/{chart}", params={"timespan": "all", "format": "json"})
    pts = data.get("values", [])
    if not pts:
        return pd.DataFrame(columns=["date", chart])
    df = pd.DataFrame(pts)
    df["x"] = pd.to_datetime(df["x"], unit="s", utc=True).dt.tz_convert(None)
    df = df.rename(columns={"x": "date", "y": chart})
    return df[["date", chart]].sort_values("date").reset_index(drop=True)

def fetch_market_cap_coingecko(cg_id: str) -> pd.DataFrame:
    # https://api.coingecko.com/api/v3/coins/<id>/market_chart?vs_currency=usd&days=max
    js = get_json(f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart",
                  params={"vs_currency": "usd", "days": "max"})
    mcap = js.get("market_caps", [])
    if not mcap:
        return pd.DataFrame(columns=["date", "market_cap"])
    df = pd.DataFrame(mcap, columns=["ts_ms", "market_cap"])
    df["date"] = pd.to_datetime(df["ts_ms"], unit="ms")
    df = df[["date", "market_cap"]].sort_values("date").reset_index(drop=True)
    return df

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m jobs.fetch_onchain <SYMBOL>  (e.g., BTCUSDT or BTC)")
        sys.exit(1)
    sym = sys.argv[1].upper()
    if sym not in ASSET_MAP:
        print(f"Symbol {sym} not supported. Try one of: {', '.join(ASSET_MAP.keys())}")
        sys.exit(2)

    cg_id = ASSET_MAP[sym]["cg"]

    # Blockchain.com charts (daily series)
    # Active addresses
    df_aa = fetch_blockchain_chart("active-addresses")
    # Transactions count
    df_tx = fetch_blockchain_chart("n-transactions")
    # Hash rate
    df_hr = fetch_blockchain_chart("hash-rate")
    # Estimated tx volume USD (proxy for on-chain volume $)
    df_vusd = fetch_blockchain_chart("estimated-transaction-volume-usd")

    # Market cap from CoinGecko
    df_mcap = fetch_market_cap_coingecko(cg_id)

    # Optional: realized cap (for MVRV) — if not available we leave MVRV as NaN.
    try:
        df_rcap = fetch_blockchain_chart("realized-cap")
        have_rcap = not df_rcap.empty
    except Exception:
        df_rcap = pd.DataFrame(columns=["date", "realized-cap"])
        have_rcap = False

    # Merge on date
    df = df_mcap.merge(df_vusd, on="date", how="outer") \
                .merge(df_aa, on="date", how="outer") \
                .merge(df_tx, on="date", how="outer") \
                .merge(df_hr, on="date", how="outer")
    if have_rcap:
        df = df.merge(df_rcap.rename(columns={"realized-cap": "realized_cap"}), on="date", how="outer")

    df = df.sort_values("date").reset_index(drop=True)

    # Compute NVT = Market Cap / On-chain TX Volume (USD proxy)
    df["nvt"] = np.where(df["estimated-transaction-volume-usd"].gt(0),
                         df["market_cap"] / df["estimated-transaction-volume-usd"], np.nan)

    # Compute MVRV if realized cap exists
    df["mvrv"] = np.where(df.get("realized_cap", pd.Series(index=df.index)).gt(0),
                          df["market_cap"] / df["realized_cap"], np.nan)

    # Canonical column names expected by your page
    out = pd.DataFrame({
        "date": df["date"],
        "nvt": df["nvt"],
        "mvrv": df["mvrv"],
        "active_addresses": df.get("active-addresses"),
        "tx_count": df.get("n-transactions"),
        "exchange_in": np.nan,   # not available from free sources
        "exchange_out": np.nan,  # not available from free sources
        "whale_txs": np.nan,     # not available from free sources
        "hash_rate": df.get("hash-rate"),
        "tvl": np.nan,           # general TVL needs DeFi APIs; left NaN
    })

    # Drop rows with no date
    out = out.dropna(subset=["date"]).copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date

    # Write to cache sqlite and CSV
    tbl = f"onchain_{sym}"
    (DATA_DIR / "onchain" / sym).mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "onchain" / sym / "onchain_daily.csv"
    out.to_csv(csv_path, index=False)

    with sqlite3.connect(ONCHAIN_CACHE) as con:
        out.to_sql(tbl, con, if_exists="replace", index=False)

    print(f"Wrote {len(out)} on-chain rows -> {ONCHAIN_CACHE} [{tbl}]")
    print(f"Also wrote CSV -> {csv_path}")

if __name__ == "__main__":
    main()
