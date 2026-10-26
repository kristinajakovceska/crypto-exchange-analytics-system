from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import asc, desc, func, select
from sqlalchemy.orm import Session

from analytics_api.db import Candle


@dataclass(frozen=True)
class SymbolRow:
    symbol: str
    base_asset: str
    quote_asset: str
    first_date: datetime | None


@dataclass(frozen=True)
class SymbolHeader:
    base_asset: str
    quote_asset: str
    min_date: datetime | None
    max_date: datetime | None


@dataclass(frozen=True)
class CandleRow:
    date: datetime | None
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: float | None


class CandleRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_totals(self) -> tuple[int, int]:
        total_candles = self.db.scalar(select(func.count()).select_from(Candle)) or 0
        total_symbols = self.db.scalar(select(func.count(func.distinct(Candle.symbol)))) or 0
        return int(total_candles), int(total_symbols)

    def list_symbols(self, q: str | None = None) -> list[SymbolRow]:
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

        rows = self.db.execute(stmt).all()

        out: list[SymbolRow] = []
        for r in rows:
            m = r._mapping
            out.append(
                SymbolRow(
                    symbol=m["symbol"],
                    base_asset=m["base_asset"],
                    quote_asset=m["quote_asset"],
                    first_date=m["first_date"],
                )
            )
        return out

    def get_symbol_header(self, symbol: str) -> SymbolHeader | None:
        row = self.db.execute(
            select(
                Candle.base_asset,
                Candle.quote_asset,
                func.min(Candle.date).label("min_date"),
                func.max(Candle.date).label("max_date"),
            ).where(Candle.symbol == symbol)
        ).one_or_none()

        if not row:
            return None

        m = row._mapping
        return SymbolHeader(
            base_asset=m["base_asset"],
            quote_asset=m["quote_asset"],
            min_date=m["min_date"],
            max_date=m["max_date"],
        )

    def get_recent_candles(self, symbol: str, limit: int = 50) -> list[CandleRow]:
        rows = self.db.execute(
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
            .limit(limit)
        ).all()

        out: list[CandleRow] = []
        for r in rows:
            m = r._mapping
            out.append(
                CandleRow(
                    date=m["date"],
                    open=m["open"],
                    high=m["high"],
                    low=m["low"],
                    close=m["close"],
                    volume=m["volume"],
                )
            )
        return out
