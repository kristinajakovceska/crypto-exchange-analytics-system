from __future__ import annotations

import os
import httpx


class SentimentService:
    def __init__(self):
        self.base_url = os.getenv("SENTIMENT_URL", "http://127.0.0.1:8003")

    def load(self, symbol: str) -> dict:
        r = httpx.get(f"{self.base_url}/api/sentiment/{symbol}", timeout=20.0)
        r.raise_for_status()
        return r.json()
