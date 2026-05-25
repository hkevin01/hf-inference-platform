import pytest

from hf_inference_platform.config import Settings
from hf_inference_platform.runtime import resolve_device


def test_settings_default_nsfw_is_enabled() -> None:
    settings = Settings()
    assert settings.allow_nsfw is True


def test_resolve_device_explicit_cpu() -> None:
    settings = Settings(DEFAULT_DEVICE="cpu")
    assert resolve_device(settings) == "cpu"


def test_resolve_device_without_torch_when_explicit_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import() -> None:
        raise RuntimeError("should not import torch for explicit device")

    monkeypatch.setattr("hf_inference_platform.runtime.torch_module", fail_import)
    settings = Settings(DEFAULT_DEVICE="cpu")
    assert resolve_device(settings) == "cpu"
