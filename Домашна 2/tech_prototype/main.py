from __future__ import annotations

from collections.abc import Generator
from datetime import date, datetime
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import (
    Column,
    Date,
    Float,
    Integer,
    String,
    create_engine,
    func,
    select,
    Index,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


# ============================================================
# 1. Database configuration
# ============================================================

# Make sure cryptos.db is in this same folder
DATABASE_URL = "sqlite:///cryptos.db"

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


class Candle(Base):
    __tablename__ = "candles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    exchange: Mapped[str] = mapped_column(String(50), nullable=False)
    base_asset: Mapped[str] = mapped_column(String(50), nullable=False)
    quote_asset: Mapped[str] = mapped_column(String(50), nullable=False)
    symbol: Mapped[str] = mapped_column(String(50), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)

    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)
    quote_volume: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    last_price_24h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    volume_24h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    high_24h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    low_24h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    liquidity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


Index(
    "idx_candles_symbol_date",
    Candle.symbol,
    Candle.date,
    unique=True,
)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================
# 2. Pydantic schemas
# ============================================================

class SymbolInfo(BaseModel):
    symbol: str
    base_asset: str
    quote_asset: str
    first_date: date
    last_date: date
    num_rows: int


class CandleOut(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: Optional[float]


class AnalyticsOut(BaseModel):
    symbol: str
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    count: int = Field(..., description="Number of candles in the period")
    min_close: Optional[float] = None
    max_close: Optional[float] = None
    avg_close: Optional[float] = None


# ============================================================
# 3. FastAPI app and templates
# ============================================================

app = FastAPI(
    title="Crypto Exchange Analytics API",
    description="Homework 2 technical prototype on top of cryptos.db.",
    version="2.0.0",
)

templates = Jinja2Templates(directory="templates")


# ============================================================
# 4. Date parsing helper (built-in only)
# ============================================================

def parse_iso_date(value: Optional[str]) -> Optional[date]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        # expects YYYY-MM-DD
        return datetime.fromisoformat(value).date()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format. expected YYYY-MM-DD, got '{value}'",
        )


# ============================================================
# 5. HTML views
# ============================================================

@app.get("/", response_class=HTMLResponse)
def html_home(request: Request, db: Session = Depends(get_db)):
    total_candles = db.scalar(select(func.count()).select_from(Candle)) or 0
    total_symbols = db.scalar(select(func.count(func.distinct(Candle.symbol)))) or 0
    first_date = db.scalar(select(func.min(Candle.date))) or None
    last_date = db.scalar(select(func.max(Candle.date))) or None

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "total_candles": total_candles,
            "total_symbols": total_symbols,
            "first_date": first_date,
            "last_date": last_date,
        },
    )


@app.get("/symbols", response_class=HTMLResponse)
def html_symbols(request: Request, db: Session = Depends(get_db)):
    stmt = (
        select(
            Candle.symbol,
            Candle.base_asset,
            Candle.quote_asset,
            func.min(Candle.date).label("first_date"),
            func.max(Candle.date).label("last_date"),
            func.count().label("num_rows"),
        )
        .group_by(Candle.symbol, Candle.base_asset, Candle.quote_asset)
        .order_by(Candle.symbol)
    )
    rows = db.execute(stmt).all()

    return templates.TemplateResponse(
        "symbols.html",
        {
            "request": request,
            "rows": rows,
        },
    )


@app.get("/symbols/{symbol}", response_class=HTMLResponse)
def html_symbol_detail(symbol: str, request: Request, db: Session = Depends(get_db)):
    stmt_meta = (
        select(
            Candle.symbol,
            Candle.base_asset,
            Candle.quote_asset,
            func.min(Candle.date),
            func.max(Candle.date),
            func.count(),
        )
        .where(Candle.symbol == symbol)
    )
    meta_row = db.execute(stmt_meta).first()
    if not meta_row:
        raise HTTPException(status_code=404, detail="Symbol not found")

    stmt_recent = (
        select(Candle)
        .where(Candle.symbol == symbol)
        .order_by(Candle.date.desc())
        .limit(100)
    )
    recent = db.execute(stmt_recent).scalars().all()

    return templates.TemplateResponse(
        "symbol_detail.html",
        {
            "request": request,
            "meta": {
                "symbol": meta_row[0],
                "base_asset": meta_row[1],
                "quote_asset": meta_row[2],
                "first_date": meta_row[3],
                "last_date": meta_row[4],
                "num_rows": meta_row[5],
            },
            "candles": recent,
        },
    )


# ============================================================
# 6. JSON API
# ============================================================

@app.get("/api/symbols", response_model=List[SymbolInfo])
def api_list_symbols(db: Session = Depends(get_db)):
    stmt = (
        select(
            Candle.symbol,
            Candle.base_asset,
            Candle.quote_asset,
            func.min(Candle.date).label("first_date"),
            func.max(Candle.date).label("last_date"),
            func.count().label("num_rows"),
        )
        .group_by(Candle.symbol, Candle.base_asset, Candle.quote_asset)
        .order_by(Candle.symbol)
    )
    rows = db.execute(stmt).all()

    return [
        SymbolInfo(
            symbol=row.symbol,
            base_asset=row.base_asset,
            quote_asset=row.quote_asset,
            first_date=row.first_date,
            last_date=row.last_date,
            num_rows=row.num_rows,
        )
        for row in rows
    ]


@app.get("/api/candles/{symbol}", response_model=List[CandleOut])
def api_get_candles(
    symbol: str,
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD, inclusive"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD, inclusive"),
    db: Session = Depends(get_db),
):
    s_date = parse_iso_date(start_date)
    e_date = parse_iso_date(end_date)

    stmt = select(Candle).where(Candle.symbol == symbol)

    if s_date is not None:
        stmt = stmt.where(Candle.date >= s_date)
    if e_date is not None:
        stmt = stmt.where(Candle.date <= e_date)

    stmt = stmt.order_by(Candle.date.asc())
    candles = db.execute(stmt).scalars().all()

    if not candles:
        exists_stmt = select(func.count()).select_from(Candle).where(Candle.symbol == symbol)
        exists = db.scalar(exists_stmt)
        if not exists:
            raise HTTPException(status_code=404, detail="Symbol not found")

    return [
        CandleOut(
            date=c.date,
            open=c.open,
            high=c.high,
            low=c.low,
            close=c.close,
            volume=c.volume,
            quote_volume=c.quote_volume,
        )
        for c in candles
    ]


@app.get("/api/analytics/{symbol}", response_model=AnalyticsOut)
def api_get_analytics(
    symbol: str,
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD, inclusive"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD, inclusive"),
    db: Session = Depends(get_db),
):
    s_date = parse_iso_date(start_date)
    e_date = parse_iso_date(end_date)

    stmt = select(
        func.count().label("count"),
        func.min(Candle.close).label("min_close"),
        func.max(Candle.close).label("max_close"),
        func.avg(Candle.close).label("avg_close"),
    ).where(Candle.symbol == symbol)

    if s_date is not None:
        stmt = stmt.where(Candle.date >= s_date)
    if e_date is not None:
        stmt = stmt.where(Candle.date <= e_date)

    result = db.execute(stmt).one()
    count = result.count or 0

    if count == 0:
        exists_stmt = select(func.count()).select_from(Candle).where(Candle.symbol == symbol)
        exists = db.scalar(exists_stmt) or 0
        if not exists:
            raise HTTPException(status_code=404, detail="Symbol not found")

    return AnalyticsOut(
        symbol=symbol,
        start_date=s_date,
        end_date=e_date,
        count=count,
        min_close=result.min_close,
        max_close=result.max_close,
        avg_close=result.avg_close,
    )
