from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from analytics_api.core.settings import templates
from analytics_api.services.sentiment_service import SentimentService

router = APIRouter()


@router.get("/sentiment/{symbol}", response_class=HTMLResponse)
def sentiment_html(request: Request, symbol: str):
    svc = SentimentService()
    data = svc.load(symbol)

    return templates.TemplateResponse(
        "sentiment.html",
        {
            "request": request,
            "symbol": symbol,
            "sentiment": data["sentiment"],
            "combined": data["combined"],
            "nvt": data["nvt"],
            "mvrv": data["mvrv"],
            "news": data["news"],
            "pie": data["pie"],
            "title": f"{symbol} sentiment",
        },
    )
