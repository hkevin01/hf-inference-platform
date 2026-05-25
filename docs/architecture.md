# Architecture

## Serving shape

The scaffold separates model-serving concerns from transport concerns.

- FastAPI handles the HTTP surface.
- `InferenceService` coordinates backend selection.
- `TransformersTextBackend` keeps plain Transformers serving simple and debuggable.
- `DiffusersImageBackend` focuses on Diffusers-native image generation paths.
- `hf_inference_sdk` gives the platform team a stable client contract.

## Why this layout

For diffusion workloads, the main optimization surface is usually inside the denoiser or transformer block, not the API layer. For text workloads, the API layer matters less than batching policy and KV-cache management. This layout keeps those concerns separate so you can swap runtimes later without rewriting your service contract.

## Runtime strategy

- Use Diffusers and PyTorch for image and video generation prototypes, model bring-up, and parity testing.
- Use `torch.compile` selectively on the denoiser or repeated transformer blocks instead of whole-pipeline compilation.
- Leave room to route LLM and VLM generation to vLLM or TGI when continuous batching becomes necessary.
- Treat NSFW enablement as deployment policy, not a hidden model-side side effect.
