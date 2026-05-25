from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from .config import get_settings
from .schemas import (
    ImageGenerationRequest,
    ImageGenerationResponse,
    TextGenerationRequest,
    TextGenerationResponse,
)
from .service import InferenceService

settings = get_settings()
service = InferenceService(settings)
app = FastAPI(title="HF Inference Platform", version="0.1.0")


@app.get("/healthz")
def healthz() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "allow_nsfw": settings.allow_nsfw,
        "device": settings.default_device,
    }


@app.get("/v1/catalog")
def catalog() -> list[dict[str, str]]:
    return service.catalog()


@app.post("/v1/generate/text", response_model=TextGenerationResponse)
def generate_text(request: TextGenerationRequest) -> TextGenerationResponse:
    return service.text_backend.generate_text(request)


@app.post("/v1/generate/image", response_model=ImageGenerationResponse)
def generate_image(request: ImageGenerationRequest) -> ImageGenerationResponse:
    return service.image_backend.generate_image(request)


def run() -> None:
    uvicorn.run("hf_inference_platform.main:app", host="0.0.0.0", port=settings.service_port, reload=False)
