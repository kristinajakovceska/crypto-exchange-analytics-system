from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from analytics_api.core.settings import templates
from analytics_api.services.forecast_service import ForecastService

router = APIRouter()


@router.get("/predict/{symbol}", response_class=HTMLResponse)
def predict_html(
    request: Request,
    symbol: str,
    lookback: int = Query(30, ge=10, le=365),
    horizon: int = Query(30, ge=1, le=365),
):
    svc = ForecastService()
    data = svc.forecast(symbol=symbol, lookback=lookback, horizon=horizon)

    # Microservice returns: symbol, lookback, horizon, metrics, forecast
    return templates.TemplateResponse(
        "predict.html",
        {
            "request": request,
            "symbol": data.get("symbol", symbol),
            "metrics": data.get("metrics", {}),
            "forecast": data.get("forecast", []),
            "lookback": data.get("lookback", lookback),
            "horizon": data.get("horizon", horizon),
            "title": f"{symbol} forecast",
        },
    )
