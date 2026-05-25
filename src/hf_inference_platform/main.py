from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Header
from fastapi.responses import PlainTextResponse

from .config import Settings, get_settings
from .platform import AdmissionRejectedError, TenantBoundaryError
from .schemas import (
    ImageGenerationRequest,
    ImageGenerationResponse,
    TextGenerationRequest,
    TextGenerationResponse,
)
from .service import InferenceService

def create_app(settings: Settings | None = None, service: InferenceService | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    resolved_service = service or InferenceService(resolved_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        resolved_service.warmup()
        yield

    app = FastAPI(title="HF Inference Platform", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict[str, str | bool]:
        return {
            "status": "ok",
            "allow_nsfw": resolved_settings.allow_nsfw,
            "device": resolved_settings.default_device,
            "text_backend_runtime": resolved_settings.text_backend_runtime,
        }


    @app.get("/v1/catalog")
    def catalog() -> list[dict[str, str]]:
        return resolved_service.catalog()


    @app.post("/v1/generate/text", response_model=TextGenerationResponse)
    def generate_text(
        request: TextGenerationRequest,
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
    ) -> TextGenerationResponse:
        try:
            return resolved_service.generate_text(request, x_tenant_id)
        except (AdmissionRejectedError, TenantBoundaryError) as exc:
            from fastapi import HTTPException

            raise HTTPException(status_code=429 if isinstance(exc, AdmissionRejectedError) else 403, detail=str(exc)) from exc


    @app.post("/v1/generate/image", response_model=ImageGenerationResponse)
    def generate_image(
        request: ImageGenerationRequest,
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
    ) -> ImageGenerationResponse:
        try:
            return resolved_service.generate_image(request, x_tenant_id)
        except (AdmissionRejectedError, TenantBoundaryError) as exc:
            from fastapi import HTTPException

            raise HTTPException(status_code=429 if isinstance(exc, AdmissionRejectedError) else 403, detail=str(exc)) from exc


    @app.get("/v1/observability")
    def observability() -> dict[str, object]:
        return resolved_service.observability_snapshot()


    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        return resolved_service.prometheus_metrics()

    return app


settings = get_settings()
service = InferenceService(settings)
app = create_app(settings, service)


def run() -> None:
    uvicorn.run("hf_inference_platform.main:app", host="0.0.0.0", port=settings.service_port, reload=False)
