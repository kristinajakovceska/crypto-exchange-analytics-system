from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, Query
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.utils import load_symbol_df
from app.indicators import compute_indicators, resample_tf
from app.signals import generate_signals

app = FastAPI(title="TA Service", version="1.1.0")


def get_db_session() -> Session:
    return SessionLocal()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/ta/{symbol}")
def ta_api(
    symbol: str,
    timeframe: str = Query("1d", pattern="^(1d|1w|1m)$"),
    limit: int = Query(100, ge=10, le=500),
) -> dict[str, Any]:
    db = get_db_session()
    try:
        df = load_symbol_df(db, symbol)

        if df is None or df.empty:
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "last": None,
                "table": [],
                "error": "No data returned for symbol",
            }

        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"]).set_index("date")

        if timeframe != "1d":
            df = resample_tf(df, timeframe)

        sig = generate_signals(compute_indicators(df))
        if sig is None or sig.empty:
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "last": None,
                "table": [],
                "error": "Not enough data after indicators",
            }

        last_row = sig.iloc[-1]

        table_df = sig.tail(limit).reset_index()
        table_df = table_df.rename(columns={table_df.columns[0]: "date"})
        table_df["date"] = pd.to_datetime(table_df["date"], errors="coerce").dt.strftime("%d/%m/%Y")
        table = table_df.to_dict(orient="records")

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "last": {
                "signal": str(last_row.get("signal", "N/A")),
                "score": float(last_row.get("score", 0.0)),
                **{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in last_row.to_dict().items()},
            },
            "table": table,
        }
    except Exception as e:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "last": None,
            "table": [],
            "error": f"{type(e).__name__}: {e}",
        }
    finally:
        db.close()
