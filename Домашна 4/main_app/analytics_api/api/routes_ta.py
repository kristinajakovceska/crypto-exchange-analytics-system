from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from analytics_api.core.settings import templates
from analytics_api.services.ta_service import TechnicalAnalysisService

router = APIRouter()


@router.get("/ta/{symbol}", response_class=HTMLResponse)
def ta_html(
    request: Request,
    symbol: str,
    timeframe: str = Query("1d", pattern="^(1d|1w|1m)$"),
):
    svc = TechnicalAnalysisService()
    payload = svc.get_ta_payload(symbol=symbol, timeframe=timeframe, limit=100)

    last = payload.get("last") or {}
    table = payload.get("table") or []
    empty_msg = payload.get("error")

    return templates.TemplateResponse(
        "ta.html",
        {
            "request": request,
            "symbol": symbol,
            "timeframe": timeframe,
            "last": last,
            "table": table,
            "empty_msg": empty_msg,
            "title": f"{symbol} TA",
        },
    )


@router.get("/api/ta/{symbol}")
def ta_api(
    symbol: str,
    timeframe: str = Query("1d", pattern="^(1d|1w|1m)$"),
):
    svc = TechnicalAnalysisService()
    return svc.get_ta_payload(symbol=symbol, timeframe=timeframe, limit=100)
