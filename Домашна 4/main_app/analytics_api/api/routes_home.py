from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy.orm import Session

from analytics_api.core.db import get_db
from analytics_api.core.settings import templates
from analytics_api.db import DB_PATH
from analytics_api.services.symbols_service import SymbolsService

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
@router.get("/home", response_class=HTMLResponse)
def home(request: Request, db: Session = Depends(get_db)):
    svc = SymbolsService(db)
    total_candles, total_symbols = svc.get_home_overview()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "total_candles": total_candles,
            "total_symbols": total_symbols,
            "title": "Home",
        },
    )


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "title": "Login"},
    )


@router.get("/download/db")
def download_db():
    return FileResponse(
        path=str(DB_PATH),
        filename="cryptos.db",
        media_type="application/octet-stream",
    )
