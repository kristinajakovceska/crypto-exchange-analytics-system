from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI

from app.sentiment_onchain import _sql_get, combine_signal

app = FastAPI(title="Sentiment Service", version="1.0.0")


def data_dir() -> Path:
    p = os.getenv("DATA_DIR", "")
    if p:
        return Path(p)
    return Path(__file__).resolve().parent.parent.parent.parent / "data"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/sentiment/{symbol}")
def sentiment_api(symbol: str) -> dict[str, Any]:
    ddir = data_dir()

    sdf = _sql_get(table=f"sent_{symbol}", cache_path=str(ddir / "sentiment_cache.sqlite"))
    cdf = _sql_get(table=f"onchain_{symbol}", cache_path=str(ddir / "onchain_cache.sqlite"))

    if sdf is None or sdf.empty:
        return {
            "symbol": symbol,
            "sentiment": None,
            "combined": 0.0,
            "nvt": None,
            "mvrv": None,
            "news": [],
            "pie": {"pos": 0, "neu": 0, "neg": 0},
        }

    sent = sdf.copy()
    if "time" in sent.columns:
        sent["ts"] = pd.to_datetime(sent["time"], errors="coerce")
    elif "ts" in sent.columns:
        sent["ts"] = pd.to_datetime(sent["ts"], errors="coerce")
    else:
        sent["ts"] = pd.to_datetime(sent.index, errors="coerce")

    sent["compound"] = pd.to_numeric(sent["compound"], errors="coerce")
    s_avg = float(sent["compound"].dropna().mean())

    pie = {
        "pos": int((sent["compound"] > 0.05).sum()),
        "neu": int(((sent["compound"] >= -0.05) & (sent["compound"] <= 0.05)).sum()),
        "neg": int((sent["compound"] < -0.05).sum()),
    }

    news = (
        sent.sort_values("ts", ascending=False)
        .head(20)[["ts", "compound"]]
        .assign(title="News sentiment")
        .to_dict(orient="records")
    )

    nvt = None
    mvrv = None
    if cdf is not None and not cdf.empty:
        if "nvt" in cdf.columns:
            nvt = float(pd.to_numeric(cdf["nvt"], errors="coerce").dropna().iloc[-1])
        if "mvrv" in cdf.columns:
            mvrv = float(pd.to_numeric(cdf["mvrv"], errors="coerce").dropna().iloc[-1])

    combined = combine_signal(
        ta_score=0.0,
        sent_score=s_avg,
        nvt=nvt if nvt is not None else np.nan,
        mvrv=mvrv if mvrv is not None else np.nan,
    )

    return {
        "symbol": symbol,
        "sentiment": s_avg,
        "combined": float(combined),
        "nvt": nvt,
        "mvrv": mvrv,
        "news": news,
        "pie": pie,
    }
