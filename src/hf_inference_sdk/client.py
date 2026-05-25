from __future__ import annotations

from typing import Any

import httpx


class InferenceClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        response = httpx.get(f"{self.base_url}/healthz", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def catalog(self) -> list[dict[str, Any]]:
        response = httpx.get(f"{self.base_url}/v1/catalog", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def generate_text(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = httpx.post(f"{self.base_url}/v1/generate/text", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def generate_image(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = httpx.post(f"{self.base_url}/v1/generate/image", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
