from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings
from .hf_backends import DiffusersImageBackend, TransformersTextBackend


@dataclass(slots=True)
class InferenceService:
    settings: Settings
    text_backend: TransformersTextBackend = field(init=False)
    image_backend: DiffusersImageBackend = field(init=False)

    def __post_init__(self) -> None:
        self.text_backend = TransformersTextBackend(self.settings)
        self.image_backend = DiffusersImageBackend(self.settings)

    def catalog(self) -> list[dict[str, str]]:
        capabilities = self.text_backend.capabilities() + self.image_backend.capabilities()
        return [cap.model_dump() for cap in capabilities]
