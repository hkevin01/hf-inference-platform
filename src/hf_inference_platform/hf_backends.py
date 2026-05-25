from __future__ import annotations

from typing import cast

from .backends import ImageBackend, ModelCapability, TextBackend
from .config import Settings
from .runtime import LoadedModel, elapsed, image_to_base64, resolve_device, resolve_dtype, timer, torch_module
from .schemas import (
    ImageGenerationRequest,
    ImageGenerationResponse,
    TextGenerationRequest,
    TextGenerationResponse,
)


class TransformersTextBackend(TextBackend):
    name = "transformers"

    def __init__(self, settings: Settings):
        self.settings = settings
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

    def _load(self, model_id: str, attn_implementation: str | None) -> tuple[object, LoadedModel]:
        cache_key = f"{model_id}:{attn_implementation or 'default'}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        from transformers import AutoModelForCausalLM, AutoTokenizer

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
            },
        )
        self._cache[cache_key] = (tokenizer, loaded)
        return tokenizer, loaded

    def generate_text(self, request: TextGenerationRequest) -> TextGenerationResponse:
        start = timer()
        tokenizer, loaded = self._load(request.model_id, request.attn_implementation)
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
        return TextGenerationResponse(
            model_id=request.model_id,
            backend=self.name,
            text=text,
            latency_seconds=elapsed(start),
            metadata=loaded.metadata,
        )


class DiffusersImageBackend(ImageBackend):
    name = "diffusers"

    def __init__(self, settings: Settings):
        self.settings = settings
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

    def _load(self, model_id: str) -> LoadedModel:
        if model_id in self._cache:
            return self._cache[model_id]

        torch = torch_module()
        from diffusers import AutoPipelineForText2Image

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
            },
        )
        self._cache[model_id] = loaded
        return loaded

    def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        start = timer()
        loaded = self._load(request.model_id)
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
        return ImageGenerationResponse(
            model_id=request.model_id,
            backend=self.name,
            image_base64=image_to_base64(image),
            latency_seconds=elapsed(start),
            metadata=loaded.metadata,
        )
