# analytics_api/main.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import asc, desc, func, select
from sqlalchemy.orm import Session

# Absolute imports
from analytics_api.db import DB_PATH, Candle, SessionLocal
from analytics_api.indicators import compute_indicators, resample_tf
from analytics_api.lstm_model import train_lstm
from analytics_api.sentiment_onchain import _sql_get, combine_signal
from analytics_api.signals import generate_signals
from analytics_api.utils import load_symbol_df


# ------------------------------------------------------------------------------
# App init
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Crypto Exchange Analytics System",
    description="Technical indicators. LSTM forecast. Sentiment and on-chain blend.",
    version="3.0.0",
)

# Templates / Static
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ------------------------------------------------------------------------------
# DB session dependency
# ------------------------------------------------------------------------------
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _fmt_date(d: datetime | None) -> str:
    return d.strftime("%d/%m/%Y") if d else ""


# ------------------------------------------------------------------------------
# Core pages
# ------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
@app.get("/home", response_class=HTMLResponse)
def home(request: Request, db: Session = Depends(get_db)):
    total_candles = db.scalar(select(func.count()).select_from(Candle)) or 0
    total_symbols = db.scalar(select(func.count(func.distinct(Candle.symbol)))) or 0

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "total_candles": total_candles,
            "total_symbols": total_symbols,
            "title": "Home",
        },
    )


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "title": "Login"},
    )


@app.get("/download/db")
def download_db():
    return FileResponse(
        path=str(DB_PATH),
        filename="cryptos.db",
        media_type="application/octet-stream",
    )


# ------------------------------------------------------------------------------
# Symbols list + detail pages
# ------------------------------------------------------------------------------
@app.get("/symbols", response_class=HTMLResponse)
def symbols_page(
    request: Request,
    q: str | None = Query(None),
    db: Session = Depends(get_db),
):
    stmt = (
        select(
            Candle.symbol,
            Candle.base_asset,
            Candle.quote_asset,
            func.min(Candle.date).label("first_date"),
        )
        .group_by(Candle.symbol, Candle.base_asset, Candle.quote_asset)
        .order_by(asc(Candle.symbol))
    )

    if q:
        stmt = stmt.where(Candle.symbol.contains(q.upper()))

    rows = db.execute(stmt).all()

    items = []
    for r in rows:
        m = r._mapping
        items.append(
            {
                "symbol": m["symbol"],
                "base": m["base_asset"],
                "quote": m["quote_asset"],
                "first_date": _fmt_date(m["first_date"]),
            }
        )

    return templates.TemplateResponse(
        "symbols.html",
        {
            "request": request,
            "items": items,
            "q": q or "",
            "title": "Symbols",
        },
    )


@app.get("/symbols/{symbol}", response_class=HTMLResponse)
def symbol_detail_page(symbol: str, request: Request, db: Session = Depends(get_db)):
    header = db.execute(
        select(
            Candle.base_asset,
            Candle.quote_asset,
            func.min(Candle.date),
            func.max(Candle.date),
        ).where(Candle.symbol == symbol)
    ).one_or_none()

    if not header:
        raise HTTPException(status_code=404, detail="Symbol not found")

    base, quote, min_d, max_d = header

    rows = db.execute(
        select(
            Candle.date,
            Candle.open,
            Candle.high,
            Candle.low,
            Candle.close,
            Candle.volume,
        )
        .where(Candle.symbol == symbol)
        .order_by(desc(Candle.date))
        .limit(50)
    ).all()

    table = []
    for r in rows:
        m = r._mapping
        table.append(
            {
                "date": _fmt_date(m["date"]),
                "open": m["open"],
                "high": m["high"],
                "low": m["low"],
                "close": m["close"],
                "volume": m["volume"],
            }
        )

    return templates.TemplateResponse(
        "symbol_detail.html",
        {
            "request": request,
            "symbol": symbol,
            "base": base,
            "quote": quote,
            "min_d": _fmt_date(min_d),
            "max_d": _fmt_date(max_d),
            "table": table,
            "title": f"{symbol} details",
        },
    )


# ------------------------------------------------------------------------------
# Technical Analysis pages (SAFE)
# ------------------------------------------------------------------------------
@app.get("/ta/{symbol}", response_class=HTMLResponse)
def ta_html(
    request: Request,
    symbol: str,
    timeframe: str = Query("1d", pattern="^(1d|1w|1m)$"),
    db: Session = Depends(get_db),
):
    df = load_symbol_df(db, symbol)

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).set_index("date")

    if timeframe != "1d":
        df = resample_tf(df, timeframe)

    try:
        ind = compute_indicators(df)
        sig = generate_signals(ind)
    except Exception:
        sig = None

    if sig is None or sig.empty:
        return templates.TemplateResponse(
            "ta.html",
            {
                "request": request,
                "symbol": symbol,
                "timeframe": timeframe,
                "last": {},
                "table": [],
                "empty_msg": "Not enough data for this timeframe.",
                "title": f"{symbol} TA",
            },
        )

    last = sig.iloc[-1]

    table = sig.tail(100).reset_index()
    table = table.rename(columns={table.columns[0]: "date"})
    table["date"] = pd.to_datetime(table["date"], errors="coerce") \
        .dt.strftime("%d/%m/%Y")

    table = table.to_dict(orient="records")

    return templates.TemplateResponse(
        "ta.html",
        {
            "request": request,
            "symbol": symbol,
            "timeframe": timeframe,
            "last": last.to_dict(),
            "table": table,
            "title": f"{symbol} TA",
        },
    )


@app.get("/api/ta/{symbol}")
def ta_api(
    symbol: str,
    timeframe: str = Query("1d", pattern="^(1d|1w|1m)$"),
    db: Session = Depends(get_db),
):
    df = load_symbol_df(db, symbol)

    if timeframe != "1d":
        df = resample_tf(df, timeframe)

    try:
        sig = generate_signals(compute_indicators(df))
        if sig is None or sig.empty:
            raise ValueError
        last = sig.iloc[-1]
    except Exception:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "signal": "N/A",
            "score": 0.0,
        }

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "signal": last["signal"],
        "score": float(last["score"]),
    }


# ------------------------------------------------------------------------------
# LSTM forecast
# ------------------------------------------------------------------------------
@app.get("/predict/{symbol}", response_class=HTMLResponse)
def predict_html(
    request: Request,
    symbol: str,
    lookback: int = 30,
    horizon: int = 30,
    db: Session = Depends(get_db),
):
    df = load_symbol_df(db, symbol)[["open", "high", "low", "close", "volume"]]

    res = train_lstm(df, lookback=lookback)
    preds = res["forecast_next"](n_steps=horizon)

    return templates.TemplateResponse(
        "predict.html",
        {
            "request": request,
            "symbol": symbol,
            "metrics": res["metrics"],
            "forecast": list(map(float, preds)),
            "lookback": lookback,
            "horizon": horizon,
            "title": f"{symbol} forecast",
        },
    )


# ------------------------------------------------------------------------------
# Sentiment pages
# ------------------------------------------------------------------------------
@app.get("/sentiment/{symbol}", response_class=HTMLResponse)
def sentiment_html(request: Request, symbol: str):
    data_dir = Path(__file__).resolve().parent.parent / "data"

    # --- load sentiment ---
    sdf = _sql_get(
        table=f"sent_{symbol}",
        cache_path=str(data_dir / "sentiment_cache.sqlite"),
    )

    # --- load on-chain (optional, може да е празно) ---
    cdf = _sql_get(
        table=f"onchain_{symbol}",
        cache_path=str(data_dir / "onchain_cache.sqlite"),
    )

    if sdf is None or sdf.empty:
        return templates.TemplateResponse(
            "sentiment.html",
            {
                "request": request,
                "symbol": symbol,
                "sentiment": None,
                "combined": 0.0,
                "nvt": None,
                "mvrv": None,
                "news": [],
                "pie": {"pos": 0, "neu": 0, "neg": 0},
                "title": f"{symbol} sentiment",
            },
        )

    # -----------------------
    # SENTIMENT PROCESSING
    # -----------------------
    sent = sdf.copy()

    # normalize time column
    if "time" in sent.columns:
        sent["ts"] = pd.to_datetime(sent["time"], errors="coerce")
    elif "ts" in sent.columns:
        sent["ts"] = pd.to_datetime(sent["ts"], errors="coerce")
    else:
        sent["ts"] = pd.to_datetime(sent.index, errors="coerce")

    sent["compound"] = pd.to_numeric(sent["compound"], errors="coerce")

    # average sentiment
    s_avg = float(sent["compound"].dropna().mean())

    # pie counts
    pie = {
        "pos": int((sent["compound"] > 0.05).sum()),
        "neu": int(((sent["compound"] >= -0.05) & (sent["compound"] <= 0.05)).sum()),
        "neg": int((sent["compound"] < -0.05).sum()),
    }

    # headlines (latest 20)
    news = (
        sent.sort_values("ts", ascending=False)
        .head(20)[["ts", "compound"]]
        .assign(title="News sentiment")
        .to_dict(orient="records")
    )

    # -----------------------
    # ON-CHAIN (optional)
    # -----------------------
    nvt = None
    mvrv = None

    if cdf is not None and not cdf.empty:
        if "nvt" in cdf.columns:
            nvt = float(pd.to_numeric(cdf["nvt"], errors="coerce").dropna().iloc[-1])
        if "mvrv" in cdf.columns:
            mvrv = float(pd.to_numeric(cdf["mvrv"], errors="coerce").dropna().iloc[-1])

    # -----------------------
    # COMBINED SCORE
    # -----------------------
    combined = combine_signal(
        ta_score=0.0,
        sent_score=s_avg,
        nvt=nvt if nvt is not None else np.nan,
        mvrv=mvrv if mvrv is not None else np.nan,
    )

    return templates.TemplateResponse(
        "sentiment.html",
        {
            "request": request,
            "symbol": symbol,
            "sentiment": s_avg,
            "combined": float(combined),
            "nvt": nvt,
            "mvrv": mvrv,
            "news": news,
            "pie": pie,
            "title": f"{symbol} sentiment",
        },
    )
