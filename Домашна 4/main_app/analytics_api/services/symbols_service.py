from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from analytics_api.repositories.candles_repo import CandleRepository


def _fmt_date(d: datetime | None) -> str:
    return d.strftime("%d/%m/%Y") if d else ""


class SymbolsService:
    def __init__(self, db: Session):
        self.repo = CandleRepository(db)

    def get_home_overview(self) -> tuple[int, int]:
        return self.repo.get_totals()

    def list_symbols(self, q: str | None) -> list[dict]:
        rows = self.repo.list_symbols(q=q)
        return [
            {
                "symbol": r.symbol,
                "base": r.base_asset,
                "quote": r.quote_asset,
                "first_date": _fmt_date(r.first_date),
            }
            for r in rows
        ]

    def get_symbol_detail(self, symbol: str) -> dict | None:
        header = self.repo.get_symbol_header(symbol)
        if not header:
            return None

        candles = self.repo.get_recent_candles(symbol, limit=50)

        table = [
            {
                "date": _fmt_date(r.date),
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in candles
        ]

        return {
            "symbol": symbol,
            "base": header.base_asset,
            "quote": header.quote_asset,
            "min_d": _fmt_date(header.min_date),
            "max_d": _fmt_date(header.max_date),
            "table": table,
        }
