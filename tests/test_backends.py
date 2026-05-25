from __future__ import annotations

import sys
from types import SimpleNamespace

from PIL import Image

from hf_inference_platform.config import Settings
from hf_inference_platform.hf_backends import DiffusersImageBackend, TransformersTextBackend
from hf_inference_platform.platform import MetricsCollector
from hf_inference_platform.schemas import ImageGenerationRequest, TextGenerationRequest


class FakeTokenizer:
    @classmethod
    def from_pretrained(cls, model_id: str, token: str | None = None):
        return cls()

    def __call__(self, prompt: str, return_tensors: str = "pt") -> dict[str, list[int]]:
        return {"input_ids": [1, 2, 3]}

    def decode(self, *_args, **_kwargs) -> str:
        return "generated text"


class FakeTextModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def generate(self, **kwargs):
        return [[1, 2, 3]]


class FakePipeline:
    def __call__(self, **kwargs):
        return SimpleNamespace(images=[Image.new("RGB", (32, 32), color="white")])


class FakePipelineFactory:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return FakePipeline()


def test_transformers_backend_metadata_tracks_cache_and_load(monkeypatch) -> None:
    monkeypatch.setattr("hf_inference_platform.hf_backends.resolve_device", lambda settings: "cpu")
    monkeypatch.setattr("hf_inference_platform.hf_backends.resolve_dtype", lambda settings, device: "float32")
    monkeypatch.setattr(
        "hf_inference_platform.hf_backends.snapshot_device_metrics",
        lambda device: {"device": device, "utilization_percent": None},
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForCausalLM=FakeTextModel, AutoTokenizer=FakeTokenizer),
    )

    backend = TransformersTextBackend(Settings(), MetricsCollector())
    request = TextGenerationRequest(model_id="distilgpt2", prompt="hello", max_new_tokens=4)

    first = backend.generate_text(request)
    second = backend.generate_text(request)

    assert first.metadata["runtime"] == "transformers"
    assert first.metadata["cache_hit"] is False
    assert second.metadata["cache_hit"] is True
    assert first.metadata["device_metrics"]["device"] == "cpu"


def test_diffusers_backend_metadata_tracks_cache_and_load(monkeypatch) -> None:
    monkeypatch.setattr("hf_inference_platform.hf_backends.resolve_device", lambda settings: "cpu")
    monkeypatch.setattr("hf_inference_platform.hf_backends.resolve_dtype", lambda settings, device: "float32")
    monkeypatch.setattr("hf_inference_platform.hf_backends.torch_module", lambda: SimpleNamespace())
    monkeypatch.setattr(
        "hf_inference_platform.hf_backends.snapshot_device_metrics",
        lambda device: {"device": device, "utilization_percent": None},
    )
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(AutoPipelineForText2Image=FakePipelineFactory),
    )

    backend = DiffusersImageBackend(Settings(), MetricsCollector())
    request = ImageGenerationRequest(model_id="runwayml/stable-diffusion-v1-5", prompt="hello")

    first = backend.generate_image(request)
    second = backend.generate_image(request)

    assert first.metadata["runtime"] == "diffusers"
    assert first.metadata["cache_hit"] is False
    assert second.metadata["cache_hit"] is True
    assert first.metadata["device_metrics"]["device"] == "cpu"