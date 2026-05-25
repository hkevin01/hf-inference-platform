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
    service_port: int = Field(default=8000, alias="SERVICE_PORT")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
