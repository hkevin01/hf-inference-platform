from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .schemas import (
    ImageGenerationRequest,
    ImageGenerationResponse,
    ModelCapability,
    TextGenerationRequest,
    TextGenerationResponse,
)


class InferenceBackend(ABC):
    name: str

    @abstractmethod
    def capabilities(self) -> list[ModelCapability]:
        raise NotImplementedError


class TextBackend(InferenceBackend, ABC):
    @abstractmethod
    def generate_text(self, request: TextGenerationRequest) -> TextGenerationResponse:
        raise NotImplementedError


class ImageBackend(InferenceBackend, ABC):
    @abstractmethod
    def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        raise NotImplementedError


BackendMap = dict[str, InferenceBackend]


def safe_metadata(metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return metadata or {}
