# Audit Playbook

Use this when reviewing AI-written inference code before it goes deeper into production.

## Code review targets

- hot path allocations
- model lifecycle and cache discipline
- device placement consistency
- dtype consistency
- error handling around model load failures
- separation between request validation and model execution
- benchmark coverage for p50, p95, cold start, and warm start

## Refactor priorities

1. Remove duplicate model-loading code and centralize caches.
2. Split transport models from runtime models.
3. Add deterministic benchmark scripts before changing optimization flags.
4. Introduce runtime adapters so the same API can target Diffusers, Transformers, vLLM, or TGI.
5. Add production telemetry around queueing time, generation time, and model load time.
