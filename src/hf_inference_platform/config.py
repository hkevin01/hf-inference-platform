from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    allow_nsfw: bool = Field(default=True, alias="ALLOW_NSFW")
    default_device: Literal["auto", "cpu", "cuda"] = Field(default="auto", alias="DEFAULT_DEVICE")
    default_torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] = Field(
        default="auto",
        alias="DEFAULT_TORCH_DTYPE",
    )
    enable_torch_compile: bool = Field(default=False, alias="ENABLE_TORCH_COMPILE")
    enable_regional_compile: bool = Field(default=True, alias="ENABLE_REGIONAL_COMPILE")
    enable_cpu_offload: bool = Field(default=False, alias="ENABLE_CPU_OFFLOAD")
    enable_model_warmup: bool = Field(default=False, alias="ENABLE_MODEL_WARMUP")
    text_backend_runtime: Literal["transformers", "vllm", "tgi"] = Field(
        default="transformers",
        alias="TEXT_BACKEND_RUNTIME",
    )
    remote_text_base_url: str | None = Field(default=None, alias="REMOTE_TEXT_BASE_URL")
    remote_text_model_id: str | None = Field(default=None, alias="REMOTE_TEXT_MODEL_ID")
    remote_text_api_key: str | None = Field(default=None, alias="REMOTE_TEXT_API_KEY")
    remote_text_timeout_seconds: float = Field(default=120.0, alias="REMOTE_TEXT_TIMEOUT_SECONDS")
    max_concurrent_requests: int = Field(default=4, alias="MAX_CONCURRENT_REQUESTS")
    max_queue_size: int = Field(default=32, alias="MAX_QUEUE_SIZE")
    max_queue_wait_seconds: float = Field(default=30.0, alias="MAX_QUEUE_WAIT_SECONDS")
    default_tenant_id: str = Field(default="public", alias="DEFAULT_TENANT_ID")
    require_tenant_header: bool = Field(default=False, alias="REQUIRE_TENANT_HEADER")
    tenant_policies_json: str = Field(
        default='{"public":{"allowed_text_models":["*"],"allowed_image_models":["*"]}}',
        alias="TENANT_POLICIES_JSON",
    )
    warmup_text_model_id: str | None = Field(default=None, alias="WARMUP_TEXT_MODEL_ID")
    warmup_text_prompt: str = Field(default="Return the single word ready.", alias="WARMUP_TEXT_PROMPT")
    warmup_image_model_id: str | None = Field(default=None, alias="WARMUP_IMAGE_MODEL_ID")
    warmup_image_prompt: str = Field(default="simple color gradient background", alias="WARMUP_IMAGE_PROMPT")
    service_port: int = Field(default=8000, alias="SERVICE_PORT")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
