# Optimization Guide

This scaffold is built around current guidance from Hugging Face and PyTorch for inference hosting.

## Diffusers

- Prefer `bfloat16` or `float16` on CUDA, depending on hardware support.
- Keep scaled dot product attention enabled and prefer the best available backend on your GPU.
- Change modules to `channels_last` where appropriate.
- Compile the denoiser or repeated transformer blocks, not the entire pipeline, to reduce cold-start pain.
- Use `dynamic=True` if your platform serves multiple image sizes and you want to avoid recompiles.
- Be careful with CPU offload. It helps memory pressure but often hurts latency.
- Benchmark scheduler choices and step counts per hosted model instead of assuming default settings are acceptable.

## Transformers

- Default to `attn_implementation="sdpa"` unless the target model and GPU clearly benefit from `flash_attention_2` or kernels from the Hugging Face kernels registry.
- Treat plain Transformers serving as the correctness baseline, not automatically the throughput winner.
- Once request volume grows, evaluate vLLM or TGI for continuous batching and better KV-cache scheduling.

## Immediate audit checklist for AI-generated inference code

- Look for model reloads inside request handlers.
- Look for accidental CPU tensor creation in the hot path.
- Look for dtype mismatches that upcast tensors to float32.
- Look for whole-pipeline `torch.compile` use where regional compile would be better.
- Look for image size changes that force recompilation.
- Look for per-request tokenizer loads, config loads, or scheduler construction.
- Look for hidden sync points around `.item()`, `.cpu()`, and logging.
