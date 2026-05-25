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


def snapshot_device_metrics(device: str) -> dict[str, Any]:
    if device != "cuda":
        return {
            "device": device,
            "utilization_percent": None,
            "memory_allocated_bytes": None,
            "memory_reserved_bytes": None,
            "memory_total_bytes": None,
            "memory_utilization_ratio": None,
        }

    torch = torch_module()
    if not torch.cuda.is_available():
        return {
            "device": "cuda",
            "utilization_percent": None,
            "memory_allocated_bytes": None,
            "memory_reserved_bytes": None,
            "memory_total_bytes": None,
            "memory_utilization_ratio": None,
        }

    device_index = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device_index).total_memory
    allocated = torch.cuda.memory_allocated(device_index)
    reserved = torch.cuda.memory_reserved(device_index)
    utilization = None
    if hasattr(torch.cuda, "utilization"):
        try:
            utilization = float(torch.cuda.utilization(device_index))
        except Exception:
            utilization = None

    return {
        "device": "cuda",
        "utilization_percent": utilization,
        "memory_allocated_bytes": allocated,
        "memory_reserved_bytes": reserved,
        "memory_total_bytes": total_memory,
        "memory_utilization_ratio": round(allocated / total_memory, 4) if total_memory else 0.0,
    }


@dataclass(slots=True)
class LoadedModel:
    model: Any
    metadata: dict[str, Any]
