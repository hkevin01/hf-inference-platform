from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

from PIL import Image

from .config import Settings

if TYPE_CHECKING:
    import torch


def torch_module() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install the project with the '[inference]' extra to enable model execution."
        ) from exc
    return torch


def resolve_device(settings: Settings) -> str:
    if settings.default_device != "auto":
        return settings.default_device
    torch = torch_module()
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(settings: Settings, device: str) -> Any:
    torch = torch_module()
    if settings.default_torch_dtype == "float16":
        return torch.float16
    if settings.default_torch_dtype == "bfloat16":
        return torch.bfloat16
    if settings.default_torch_dtype == "float32":
        return torch.float32
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def timer() -> float:
    return perf_counter()


def elapsed(start: float) -> float:
    return round(perf_counter() - start, 4)


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@dataclass(slots=True)
class LoadedModel:
    model: Any
    metadata: dict[str, Any]
