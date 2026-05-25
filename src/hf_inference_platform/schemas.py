from typing import Any

from pydantic import BaseModel, Field


class TextGenerationRequest(BaseModel):
    model_id: str = Field(default="distilgpt2")
    prompt: str
    max_new_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=0.8, ge=0.0, le=5.0)
    do_sample: bool = True
    attn_implementation: str | None = None


class TextGenerationResponse(BaseModel):
    model_id: str
    backend: str
    text: str
    latency_seconds: float
    metadata: dict[str, Any]


class ImageGenerationRequest(BaseModel):
    model_id: str = Field(default="runwayml/stable-diffusion-v1-5")
    prompt: str
    negative_prompt: str | None = None
    height: int = Field(default=512, ge=256, le=2048)
    width: int = Field(default=512, ge=256, le=2048)
    num_inference_steps: int = Field(default=20, ge=1, le=100)
    guidance_scale: float = Field(default=7.5, ge=0.0, le=30.0)


class ImageGenerationResponse(BaseModel):
    model_id: str
    backend: str
    image_base64: str
    latency_seconds: float
    metadata: dict[str, Any]


class ModelCapability(BaseModel):
    task: str
    backend: str
    recommended_runtime: str
    notes: str
