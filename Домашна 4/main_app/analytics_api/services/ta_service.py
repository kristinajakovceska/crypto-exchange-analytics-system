from __future__ import annotations

import os
import httpx


class TechnicalAnalysisService:
    def __init__(self):
        self.base_url = os.getenv("TA_URL", "http://127.0.0.1:8001")

    def get_ta_payload(self, symbol: str, timeframe: str, limit: int = 100) -> dict:
        r = httpx.get(
            f"{self.base_url}/api/ta/{symbol}",
            params={"timeframe": timeframe, "limit": limit},
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json()
