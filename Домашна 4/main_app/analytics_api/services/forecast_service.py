from __future__ import annotations

import os
import httpx


class ForecastService:
    def __init__(self):
        self.base_url = os.getenv("FORECAST_URL", "http://127.0.0.1:8002")

    def forecast(self, symbol: str, lookback: int, horizon: int) -> dict:
        r = httpx.get(
            f"{self.base_url}/api/predict/{symbol}",
            params={"lookback": lookback, "horizon": horizon},
            timeout=60.0,
        )
        r.raise_for_status()
        return r.json()
