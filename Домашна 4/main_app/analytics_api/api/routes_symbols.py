from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from analytics_api.core.db import get_db
from analytics_api.core.settings import templates
from analytics_api.services.symbols_service import SymbolsService

router = APIRouter()


@router.get("/symbols", response_class=HTMLResponse)
def symbols_page(
    request: Request,
    q: str | None = Query(None),
    db: Session = Depends(get_db),
):
    svc = SymbolsService(db)
    items = svc.list_symbols(q=q)

    return templates.TemplateResponse(
        "symbols.html",
        {"request": request, "items": items, "q": q or "", "title": "Symbols"},
    )


@router.get("/symbols/{symbol}", response_class=HTMLResponse)
def symbol_detail_page(
    symbol: str,
    request: Request,
    db: Session = Depends(get_db),
):
    svc = SymbolsService(db)
    data = svc.get_symbol_detail(symbol)

    if not data:
        raise HTTPException(status_code=404, detail="Symbol not found")

    return templates.TemplateResponse(
        "symbol_detail.html",
        {
            "request": request,
            "symbol": data["symbol"],
            "base": data["base"],
            "quote": data["quote"],
            "min_d": data["min_d"],
            "max_d": data["max_d"],
            "table": data["table"],
            "title": f"{symbol} details",
        },
    )
