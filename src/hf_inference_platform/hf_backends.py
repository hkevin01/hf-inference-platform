from __future__ import annotations

import httpx
from typing import cast

from .backends import ImageBackend, ModelCapability, TextBackend
from .config import Settings
from .platform import MetricsCollector
from .runtime import (
    LoadedModel,
    elapsed,
    image_to_base64,
    resolve_device,
    resolve_dtype,
    snapshot_device_metrics,
    timer,
    torch_module,
)
from .schemas import (
    ImageGenerationRequest,
    ImageGenerationResponse,
    TextGenerationRequest,
    TextGenerationResponse,
)


class TransformersTextBackend(TextBackend):
    name = "transformers"

    def __init__(self, settings: Settings, metrics: MetricsCollector | None = None):
        self.settings = settings
        self.metrics = metrics
        self._cache: dict[str, tuple[object, LoadedModel]] = {}

    def capabilities(self) -> list[ModelCapability]:
        return [
            ModelCapability(
                task="text-generation",
                backend=self.name,
                recommended_runtime="transformers",
                notes="Good for single-node serving, debugging, and parity with Hugging Face APIs.",
            ),
            ModelCapability(
                task="high-throughput-llm",
                backend=self.name,
                recommended_runtime="vllm-or-tgi",
                notes="Move to vLLM or TGI when continuous batching and KV-cache scheduling become the bottleneck.",
            ),
        ]

    def _load(self, model_id: str, attn_implementation: str | None) -> tuple[object, LoadedModel, bool, float]:
        cache_key = f"{model_id}:{attn_implementation or 'default'}"
        if cache_key in self._cache:
            if self.metrics is not None:
                self.metrics.record_cache(self.name, model_id, hit=True)
            tokenizer, loaded = self._cache[cache_key]
            return tokenizer, loaded, True, 0.0

        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_start = timer()
        device = resolve_device(self.settings)
        dtype = resolve_dtype(self.settings, device)
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=self.settings.hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            token=self.settings.hf_token,
            attn_implementation=attn_implementation or "sdpa",
        )
        if device == "cuda":
            model = model.to(device)

        loaded = LoadedModel(
            model=model,
            metadata={
                "device": device,
                "dtype": str(dtype).replace("torch.", ""),
                "attn_implementation": attn_implementation or "sdpa",
                "compile_enabled": False,
                "device_metrics": snapshot_device_metrics(device),
            },
        )
        self._cache[cache_key] = (tokenizer, loaded)
        load_seconds = elapsed(load_start)
        if self.metrics is not None:
            self.metrics.record_cache(self.name, model_id, hit=False)
            self.metrics.record_model_load(self.name, model_id, load_seconds)
        return tokenizer, loaded, False, load_seconds

    def generate_text(self, request: TextGenerationRequest) -> TextGenerationResponse:
        start = timer()
        tokenizer, loaded, cache_hit, load_seconds = self._load(request.model_id, request.attn_implementation)
        model = loaded.model
        inputs = tokenizer(request.prompt, return_tensors="pt")
        if loaded.metadata["device"] == "cuda":
            inputs = {key: value.to("cuda") for key, value in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=request.do_sample,
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        metadata = dict(loaded.metadata)
        metadata.update(
            {
                "cache_hit": cache_hit,
                "model_load_seconds": load_seconds,
                "runtime": self.name,
            }
        )
        return TextGenerationResponse(
            model_id=request.model_id,
            backend=self.name,
            text=text,
            latency_seconds=elapsed(start),
            metadata=metadata,
        )

    def warmup(self, model_id: str, prompt: str) -> None:
        self.generate_text(
            TextGenerationRequest(
                model_id=model_id,
                prompt=prompt,
                max_new_tokens=4,
                do_sample=False,
                temperature=0.0,
            )
        )


class VllmTextBackend(TextBackend):
    name = "vllm"

    def __init__(self, settings: Settings, metrics: MetricsCollector | None = None):
        if not settings.remote_text_base_url:
            raise RuntimeError("REMOTE_TEXT_BASE_URL is required when TEXT_BACKEND_RUNTIME=vllm")
        self.settings = settings
        self.metrics = metrics
        self.base_url = settings.remote_text_base_url.rstrip("/")

    def capabilities(self) -> list[ModelCapability]:
        return [
            ModelCapability(
                task="high-throughput-llm",
                backend=self.name,
                recommended_runtime="vllm",
                notes="Routes text generation to a remote vLLM OpenAI-compatible server for batching-oriented serving.",
            )
        ]

    def generate_text(self, request: TextGenerationRequest) -> TextGenerationResponse:
        headers = {"content-type": "application/json"}
        if self.settings.remote_text_api_key:
            headers["authorization"] = f"Bearer {self.settings.remote_text_api_key}"

        response = httpx.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": request.model_id or self.settings.remote_text_model_id,
                "prompt": request.prompt,
                "max_tokens": request.max_new_tokens,
                "temperature": request.temperature,
            },
            headers=headers,
            timeout=self.settings.remote_text_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        text = payload["choices"][0]["text"]
        return TextGenerationResponse(
            model_id=request.model_id,
            backend=self.name,
            text=text,
            latency_seconds=0.0,
            metadata={
                "cache_hit": None,
                "model_load_seconds": 0.0,
                "runtime": self.name,
                "remote": True,
                "upstream_url": self.base_url,
                "device_metrics": {},
            },
        )

    def warmup(self, model_id: str, prompt: str) -> None:
        self.generate_text(
            TextGenerationRequest(
                model_id=model_id,
                prompt=prompt,
                max_new_tokens=4,
                do_sample=False,
                temperature=0.0,
            )
        )


class TGITextBackend(TextBackend):
    name = "tgi"

    def __init__(self, settings: Settings, metrics: MetricsCollector | None = None):
        if not settings.remote_text_base_url:
            raise RuntimeError("REMOTE_TEXT_BASE_URL is required when TEXT_BACKEND_RUNTIME=tgi")
        self.settings = settings
        self.metrics = metrics
        self.base_url = settings.remote_text_base_url.rstrip("/")

    def capabilities(self) -> list[ModelCapability]:
        return [
            ModelCapability(
                task="high-throughput-llm",
                backend=self.name,
                recommended_runtime="tgi",
                notes="Routes text generation to a remote TGI endpoint using the OpenAI-compatible chat API.",
            )
        ]

    def generate_text(self, request: TextGenerationRequest) -> TextGenerationResponse:
        headers = {"content-type": "application/json"}
        if self.settings.remote_text_api_key:
            headers["authorization"] = f"Bearer {self.settings.remote_text_api_key}"

        response = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": request.model_id or self.settings.remote_text_model_id,
                "messages": [{"role": "user", "content": request.prompt}],
                "max_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "stream": False,
            },
            headers=headers,
            timeout=self.settings.remote_text_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, list):
            text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        else:
            text = content
        return TextGenerationResponse(
            model_id=request.model_id,
            backend=self.name,
            text=text,
            latency_seconds=0.0,
            metadata={
                "cache_hit": None,
                "model_load_seconds": 0.0,
                "runtime": self.name,
                "remote": True,
                "upstream_url": self.base_url,
                "device_metrics": {},
            },
        )

    def warmup(self, model_id: str, prompt: str) -> None:
        self.generate_text(
            TextGenerationRequest(
                model_id=model_id,
                prompt=prompt,
                max_new_tokens=4,
                do_sample=False,
                temperature=0.0,
            )
        )


class DiffusersImageBackend(ImageBackend):
    name = "diffusers"

    def __init__(self, settings: Settings, metrics: MetricsCollector | None = None):
        self.settings = settings
        self.metrics = metrics
        self._cache: dict[str, LoadedModel] = {}

    def capabilities(self) -> list[ModelCapability]:
        return [
            ModelCapability(
                task="text-to-image",
                backend=self.name,
                recommended_runtime="diffusers",
                notes="Keep diffusion pipelines in PyTorch until you have model-specific reason to move lower-level.",
            ),
            ModelCapability(
                task="image-or-video-optimization",
                backend=self.name,
                recommended_runtime="diffusers+torch.compile",
                notes="Compile the denoiser or repeated transformer blocks, not the entire pipeline, to reduce cold start.",
            ),
        ]

    def _load(self, model_id: str) -> tuple[LoadedModel, bool, float]:
        if model_id in self._cache:
            if self.metrics is not None:
                self.metrics.record_cache(self.name, model_id, hit=True)
            return self._cache[model_id], True, 0.0

        torch = torch_module()
        from diffusers import AutoPipelineForText2Image

        load_start = timer()
        device = resolve_device(self.settings)
        dtype = resolve_dtype(self.settings, device)
        pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
            token=self.settings.hf_token,
        )

        if device == "cuda":
            pipeline = pipeline.to(device)
            if hasattr(pipeline, "unet"):
                pipeline.unet.to(memory_format=torch.channels_last)
            if self.settings.enable_cpu_offload:
                pipeline.enable_model_cpu_offload()
            if self.settings.enable_torch_compile and hasattr(pipeline, "transformer"):
                if self.settings.enable_regional_compile and hasattr(pipeline.transformer, "compile_repeated_blocks"):
                    pipeline.transformer.compile_repeated_blocks(fullgraph=True, dynamic=True)
                else:
                    pipeline.transformer = torch.compile(
                        pipeline.transformer,
                        fullgraph=True,
                        dynamic=True,
                        mode="max-autotune",
                    )

        loaded = LoadedModel(
            model=pipeline,
            metadata={
                "device": device,
                "dtype": str(dtype).replace("torch.", ""),
                "compile_enabled": self.settings.enable_torch_compile,
                "regional_compile_enabled": self.settings.enable_regional_compile,
                "cpu_offload_enabled": self.settings.enable_cpu_offload,
                "allow_nsfw": self.settings.allow_nsfw,
                "device_metrics": snapshot_device_metrics(device),
            },
        )
        self._cache[model_id] = loaded
        load_seconds = elapsed(load_start)
        if self.metrics is not None:
            self.metrics.record_cache(self.name, model_id, hit=False)
            self.metrics.record_model_load(self.name, model_id, load_seconds)
        return loaded, False, load_seconds

    def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        start = timer()
        loaded, cache_hit, load_seconds = self._load(request.model_id)
        pipeline = loaded.model
        call_kwargs = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "height": request.height,
            "width": request.width,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
        }
        result = pipeline(**call_kwargs)
        image = result.images[0]
        metadata = dict(loaded.metadata)
        metadata.update(
            {
                "cache_hit": cache_hit,
                "model_load_seconds": load_seconds,
                "runtime": self.name,
            }
        )
        return ImageGenerationResponse(
            model_id=request.model_id,
            backend=self.name,
            image_base64=image_to_base64(image),
            latency_seconds=elapsed(start),
            metadata=metadata,
        )

    def warmup(self, model_id: str, prompt: str) -> None:
        self.generate_image(
            ImageGenerationRequest(
                model_id=model_id,
                prompt=prompt,
                num_inference_steps=1,
                height=256,
                width=256,
                guidance_scale=1.0,
            )
        )


def build_text_backend(settings: Settings, metrics: MetricsCollector | None = None) -> TextBackend:
    if settings.text_backend_runtime == "vllm":
        return VllmTextBackend(settings, metrics)
    if settings.text_backend_runtime == "tgi":
        return TGITextBackend(settings, metrics)
    return TransformersTextBackend(settings, metrics)
