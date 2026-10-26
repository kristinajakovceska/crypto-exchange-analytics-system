# analytics_api/sentiment_onchain.py
from __future__ import annotations
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

# Paths for the two caches
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SENT_CACHE = DATA_DIR / "sentiment_cache.sqlite"
ONCHAIN_CACHE = DATA_DIR / "onchain_cache.sqlite"


# ---------- SQLite helpers ----------
def _sql_get(*, table: str, cache_path: str | Path) -> pd.DataFrame | None:
    """Read entire table into a DataFrame. Return None if db/table missing."""
    path = Path(cache_path)
    if not path.exists():
        return None
    try:
        with sqlite3.connect(path) as con:
            return pd.read_sql_query(f"SELECT * FROM {table}", con, parse_dates=["ts", "date"])
    except Exception:
        return None


def _sql_upsert_df(*, df: pd.DataFrame, table: str, cache_path: str | Path, if_exists: str = "replace") -> None:
    """Upsert helper used by the job script."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        df.to_sql(table, con, index=False, if_exists=if_exists)


# ---------- Sentiment ----------
def sentiment_score(sent_df: pd.DataFrame) -> float:
    """
    Return mean compound VADER score in [-1, 1].
    Input schema (by job): ts, source, title, compound, neg, neu, pos
    """
    if sent_df is None or sent_df.empty:
        return float("nan")
    return float(np.nanmean(sent_df["compound"]))


def sentiment_breakdown(sent_df: pd.DataFrame) -> dict:
    """Return counts of positive/neutral/negative for pie charts."""
    if sent_df is None or sent_df.empty:
        return {"pos": 0, "neu": 0, "neg": 0}
    pos = (sent_df["compound"] > 0.05).sum()
    neg = (sent_df["compound"] < -0.05).sum()
    neu = len(sent_df) - pos - neg
    return {"pos": int(pos), "neu": int(neu), "neg": int(neg)}


# ---------- On-chain helpers ----------
def _safe_ratio(a: float, b: float) -> float:
    return float(a) / float(b) if (b is not None and float(b) != 0.0) else float("nan")


def enrich_onchain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expect columns (all optional but recommended):
    date, active_addresses, tx_count, exchange_in, exchange_out,
    whale_txs, hash_rate, tvl, market_cap, tx_volume_onchain, realized_cap

    Compute NVT and MVRV if inputs available.
    """
    if df is None or df.empty:
        return df

    d = df.copy()
    # Ensure date is datetime and sorted
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"]).sort_values("date")

    # Derived metrics
    if "market_cap" in d.columns and "tx_volume_onchain" in d.columns:
        d["nvt"] = d.apply(lambda r: _safe_ratio(r.get("market_cap"), r.get("tx_volume_onchain")), axis=1)
    if "market_cap" in d.columns and "realized_cap" in d.columns:
        d["mvrv"] = d.apply(lambda r: _safe_ratio(r.get("market_cap"), r.get("realized_cap")), axis=1)

    return d


# ---------- Simple fusion ----------
def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    m, sd = s.mean(skipna=True), s.std(skipna=True)
    if sd and not np.isnan(sd):
        return (s - m) / sd
    return pd.Series(np.zeros(len(s)), index=s.index)


def combine_signal(
    *,
    ta_score: float = 0.0,
    sent_score: float | None = None,
    nvt: float | None = None,
    mvrv: float | None = None,
) -> float:
    """
    Heuristic fusion to get a combined [-1, 1] score:
    - sentiment: already in [-1, 1]
    - NVT: inverted & z-scored (lower NVT often more bullish)
    - MVRV: z-scored and inverted when very high (overheated)
    - TA score: user-provided in [-1, 1]

    Weights can be tuned. Here: TA 0.35, Sent 0.35, NVT 0.15, MVRV 0.15.
    """
    s = 0.0
    w_ta, w_sent, w_nvt, w_mvrv = 0.35, 0.35, 0.15, 0.15

    # clamp ta_score to [-1,1]
    try:
        ta_score = max(-1.0, min(1.0, float(ta_score)))
    except Exception:
        ta_score = 0.0
    s += w_ta * ta_score

    # sentiment in [-1, 1]
    try:
        sent_val = float(sent_score)
        sent_val = max(-1.0, min(1.0, sent_val))
    except Exception:
        sent_val = 0.0
    s += w_sent * sent_val

    # NVT + MVRV scalarization (if provided)
    nv = None if nvt is None or np.isnan(nvt) else float(nvt)
    mv = None if mvrv is None or np.isnan(mvrv) else float(mvrv)

    # Simple transforms: map high NVT to bearish (negative), high MVRV to bearish
    if nv is not None and np.isfinite(nv):
        # squash to [-1,1] via 1/(1+x) then invert
        nv_s = 1.0 / (1.0 + nv)
        nv_s = 2 * (nv_s - 0.5)  # [-1,1], higher becomes positive -> invert
        nv_s = -nv_s
        s += w_nvt * nv_s

    if mv is not None and np.isfinite(mv):
        # Map around 1.0 (fair value) -> near 0. Elevated MVRV -> negative
        mv_s = -np.tanh(mv - 1.0)  # above 1 => negative, below 1 => positive
        s += w_mvrv * float(mv_s)

    # Final clamp
    s = max(-1.0, min(1.0, s))
    return float(s)
