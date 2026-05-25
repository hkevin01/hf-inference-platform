from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from .backends import ImageBackend, TextBackend
from .config import Settings
from .hf_backends import DiffusersImageBackend, build_text_backend
from .platform import PlatformController
from .schemas import ImageGenerationRequest, ImageGenerationResponse, TextGenerationRequest, TextGenerationResponse


@dataclass(slots=True)
class InferenceService:
    settings: Settings
    platform: PlatformController | None = None
    text_backend: TextBackend | None = None
    image_backend: ImageBackend | None = None

    def __post_init__(self) -> None:
        if self.platform is None:
            self.platform = PlatformController(self.settings)
        if self.text_backend is None:
            self.text_backend = build_text_backend(self.settings, self.platform.metrics)
        if self.image_backend is None:
            self.image_backend = DiffusersImageBackend(self.settings, self.platform.metrics)

    def catalog(self) -> list[dict[str, str]]:
        capabilities = self.text_backend.capabilities() + self.image_backend.capabilities()
        return [cap.model_dump() for cap in capabilities]

    def generate_text(self, request: TextGenerationRequest, tenant_id: str | None = None) -> TextGenerationResponse:
        policy = self.platform.resolve_tenant(tenant_id)
        self.platform.ensure_model_access(policy, "text", request.model_id)
        ticket = self.platform.admission.acquire(policy)
        try:
            response = cast(TextBackend, self.text_backend).generate_text(request)
            response.metadata = {
                **response.metadata,
                "tenant_id": policy.tenant_id,
                "queue_wait_seconds": ticket.queue_wait_seconds,
            }
            self.platform.metrics.record_request(
                operation="text",
                latency_seconds=response.latency_seconds,
                queue_wait_seconds=ticket.queue_wait_seconds,
                device_metrics=response.metadata.get("device_metrics"),
            )
            return response
        except Exception:
            self.platform.metrics.record_failure("text")
            raise
        finally:
            self.platform.admission.release(policy.tenant_id)

    def generate_image(self, request: ImageGenerationRequest, tenant_id: str | None = None) -> ImageGenerationResponse:
        policy = self.platform.resolve_tenant(tenant_id)
        self.platform.ensure_model_access(policy, "image", request.model_id)
        ticket = self.platform.admission.acquire(policy)
        try:
            response = cast(ImageBackend, self.image_backend).generate_image(request)
            response.metadata = {
                **response.metadata,
                "tenant_id": policy.tenant_id,
                "queue_wait_seconds": ticket.queue_wait_seconds,
            }
            self.platform.metrics.record_request(
                operation="image",
                latency_seconds=response.latency_seconds,
                queue_wait_seconds=ticket.queue_wait_seconds,
                device_metrics=response.metadata.get("device_metrics"),
            )
            return response
        except Exception:
            self.platform.metrics.record_failure("image")
            raise
        finally:
            self.platform.admission.release(policy.tenant_id)

    def observability_snapshot(self) -> dict[str, object]:
        return self.platform.observability_snapshot()

    def prometheus_metrics(self) -> str:
        return self.platform.prometheus_text()

    def warmup(self) -> None:
        if not self.settings.enable_model_warmup:
            return
        if self.settings.warmup_text_model_id and hasattr(self.text_backend, "warmup"):
            self.text_backend.warmup(self.settings.warmup_text_model_id, self.settings.warmup_text_prompt)
            self.platform.metrics.record_warmup("text")
        if self.settings.warmup_image_model_id and hasattr(self.image_backend, "warmup"):
            self.image_backend.warmup(self.settings.warmup_image_model_id, self.settings.warmup_image_prompt)
            self.platform.metrics.record_warmup("image")
