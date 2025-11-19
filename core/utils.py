import httpx
from typing import Optional


def get_http_client(timeout: float = 15.0, headers: Optional[dict] = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout, headers=headers)
