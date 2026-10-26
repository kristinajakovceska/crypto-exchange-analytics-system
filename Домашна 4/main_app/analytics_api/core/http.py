from __future__ import annotations

import httpx


def get_http_client(timeout_s: float = 10.0) -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(timeout_s))
