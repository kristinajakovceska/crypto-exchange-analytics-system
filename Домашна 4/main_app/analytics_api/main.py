from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from analytics_api.core.settings import STATIC_DIR

from analytics_api.api.routes_home import router as home_router
from analytics_api.api.routes_symbols import router as symbols_router
from analytics_api.api.routes_ta import router as ta_router
from analytics_api.api.routes_predict import router as predict_router
from analytics_api.api.routes_sentiment import router as sentiment_router

app = FastAPI(
    title="Crypto Exchange Analytics System",
    description="Technical indicators. LSTM forecast. Sentiment and on-chain blend.",
    version="4.0.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(home_router)
app.include_router(symbols_router)
app.include_router(ta_router)
app.include_router(predict_router)
app.include_router(sentiment_router)
