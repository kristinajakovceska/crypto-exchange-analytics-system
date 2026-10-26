from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Query
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.utils import load_symbol_df
from app.lstm_model import train_lstm

app = FastAPI(title="Forecast Service", version="1.0.0")


def get_db_session() -> Session:
    return SessionLocal()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/predict/{symbol}")
def predict_api(
    symbol: str,
    lookback: int = Query(30, ge=10, le=365),
    horizon: int = Query(30, ge=1, le=365),
) -> dict[str, Any]:
    db = get_db_session()
    try:
        df = load_symbol_df(db, symbol)[["open", "high", "low", "close", "volume"]]
        res = train_lstm(df, lookback=lookback)
        preds = res["forecast_next"](n_steps=horizon)
        return {
            "symbol": symbol,
            "lookback": int(lookback),
            "horizon": int(horizon),
            "metrics": res.get("metrics", {}),
            "forecast": list(map(float, preds)),
        }
    finally:
        db.close()
