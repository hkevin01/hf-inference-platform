from __future__ import annotations

from fastapi.testclient import TestClient

from hf_inference_platform.config import Settings
from hf_inference_platform.main import create_app
from hf_inference_platform.schemas import ImageGenerationResponse, TextGenerationResponse
from hf_inference_platform.service import InferenceService


class StubTextBackend:
    def capabilities(self) -> list[object]:
        return []

    def generate_text(self, request) -> TextGenerationResponse:
        return TextGenerationResponse(
            model_id=request.model_id,
            backend="stub-text",
            text="ok",
            latency_seconds=0.12,
            metadata={"device_metrics": {"device": "cpu"}, "cache_hit": True, "model_load_seconds": 0.0},
        )

    def warmup(self, model_id: str, prompt: str) -> None:
        return None


class StubImageBackend:
    def capabilities(self) -> list[object]:
        return []

    def generate_image(self, request) -> ImageGenerationResponse:
        return ImageGenerationResponse(
            model_id=request.model_id,
            backend="stub-image",
            image_base64="abc",
            latency_seconds=0.34,
            metadata={"device_metrics": {"device": "cpu"}, "cache_hit": True, "model_load_seconds": 0.0},
        )

    def warmup(self, model_id: str, prompt: str) -> None:
        return None


def build_test_client(settings: Settings | None = None) -> TestClient:
    resolved_settings = settings or Settings()
    service = InferenceService(
        resolved_settings,
        text_backend=StubTextBackend(),
        image_backend=StubImageBackend(),
    )
    app = create_app(resolved_settings, service)
    return TestClient(app)


def test_text_endpoint_includes_tenant_and_queue_metadata() -> None:
    client = build_test_client()
    response = client.post(
        "/v1/generate/text",
        json={"model_id": "distilgpt2", "prompt": "hello", "max_new_tokens": 4},
        headers={"X-Tenant-ID": "public"},
    )
    response.raise_for_status()
    payload = response.json()
    assert payload["metadata"]["tenant_id"] == "public"
    assert "queue_wait_seconds" in payload["metadata"]


def test_image_endpoint_includes_tenant_and_queue_metadata() -> None:
    client = build_test_client()
    response = client.post(
        "/v1/generate/image",
        json={"model_id": "runwayml/stable-diffusion-v1-5", "prompt": "hello", "num_inference_steps": 2},
        headers={"X-Tenant-ID": "public"},
    )
    response.raise_for_status()
    payload = response.json()
    assert payload["metadata"]["tenant_id"] == "public"
    assert "queue_wait_seconds" in payload["metadata"]


def test_observability_and_metrics_endpoints_exist() -> None:
    client = build_test_client()
    observability = client.get("/v1/observability")
    observability.raise_for_status()
    assert "admission" in observability.json()
    metrics = client.get("/metrics")
    metrics.raise_for_status()
    assert "hf_inflight_requests" in metrics.text


def test_missing_required_tenant_header_is_rejected() -> None:
    client = build_test_client(Settings(REQUIRE_TENANT_HEADER=True))
    response = client.post(
        "/v1/generate/text",
        json={"model_id": "distilgpt2", "prompt": "hello", "max_new_tokens": 4},
    )
    assert response.status_code == 403