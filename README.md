# HF Inference Platform

This project is a Python-first scaffold for hosting Hugging Face Diffusers and Transformers workloads with PyTorch. It is structured around three needs:

- a service layer for model inference
- a reusable Python SDK for your platform clients and internal tooling
- an audit and benchmarking surface for finding slow paths before they harden into production debt

The default design keeps image and video generation on Diffusers and PyTorch, while leaving room to route text-heavy workloads to purpose-built runtimes such as vLLM or TGI when throughput matters more than a single Python process abstraction.

## What is included

- FastAPI service with health, catalog, text generation, and image generation endpoints
- pluggable backend adapters for Diffusers and Transformers
- runtime settings for NSFW policy, dtype, device placement, torch.compile, and CPU offload
- Python SDK package for talking to the service
- benchmark script for baseline latency measurement
- optimization guide tailored to Hugging Face inference hosting

## NSFW handling

This scaffold does not enforce a hardcoded content block. Instead, it exposes an explicit `ALLOW_NSFW` setting so the platform policy is a deliberate deployment choice. If your deployment must allow adult media, keep `ALLOW_NSFW=true` and use models that are licensed and intended for that use case. If a model ships a safety checker or equivalent content filter, treat that behavior as model-specific and not as a platform invariant.

## Quick start

```bash
cd /home/kevin/Projects/hf-inference-platform
cp .env.example .env
uv venv
source .venv/bin/activate
uv pip install -e '.[dev,inference]'
uvicorn hf_inference_platform.main:app --reload
```

Open `http://127.0.0.1:8000/docs` after the server starts.

If you only want the control-plane code, SDK, and tests without the heavyweight model stack, use `uv pip install -e '.[dev]'`.

## Example requests

Text generation:

```bash
curl -X POST http://127.0.0.1:8000/v1/generate/text \
  -H 'content-type: application/json' \
  -d '{
    "model_id": "distilgpt2",
    "prompt": "Write a short deployment checklist for an LLM API.",
    "max_new_tokens": 64
  }'
```

Image generation:

```bash
curl -X POST http://127.0.0.1:8000/v1/generate/image \
  -H 'content-type: application/json' \
  -d '{
    "model_id": "runwayml/stable-diffusion-v1-5",
    "prompt": "studio product photography, camera on tripod, tungsten highlights",
    "num_inference_steps": 20,
    "height": 512,
    "width": 512
  }'
```

## Project layout

- `src/hf_inference_platform`: service code, runtime settings, and backend adapters
- `src/hf_inference_sdk`: Python client library
- `scripts`: benchmarking and operational helpers
- `docs`: architecture and optimization notes
- `tests`: focused unit tests for the scaffolded logic

## Recommended next steps

1. Add your real scheduler, queueing, and tenancy model.
2. Split text serving onto vLLM or TGI once you start chasing batch throughput.
3. Add GPU-specific benchmarks for the models you actually intend to host.
4. Audit any AI-generated code against the checklist in `docs/audit-playbook.md`.
