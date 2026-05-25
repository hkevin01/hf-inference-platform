from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Condition, Lock
from time import perf_counter
from typing import Any

from .config import Settings


class TenantBoundaryError(RuntimeError):
    pass


class AdmissionRejectedError(RuntimeError):
    pass


@dataclass(slots=True)
class TenantPolicy:
    tenant_id: str
    allowed_text_models: set[str] = field(default_factory=lambda: {"*"})
    allowed_image_models: set[str] = field(default_factory=lambda: {"*"})
    max_concurrent_requests: int | None = None

    def allows(self, operation: str, model_id: str) -> bool:
        allowed_models = self.allowed_text_models if operation == "text" else self.allowed_image_models
        return "*" in allowed_models or model_id in allowed_models


@dataclass(slots=True)
class AdmissionTicket:
    tenant_id: str
    queue_wait_seconds: float


class AdmissionController:
    def __init__(self, max_concurrent_requests: int, max_queue_size: int, max_queue_wait_seconds: float) -> None:
        self.max_concurrent_requests = max_concurrent_requests
        self.max_queue_size = max_queue_size
        self.max_queue_wait_seconds = max_queue_wait_seconds
        self._condition = Condition()
        self._inflight = 0
        self._queue_depth = 0
        self._tenant_inflight: dict[str, int] = defaultdict(int)

    def acquire(self, policy: TenantPolicy) -> AdmissionTicket:
        start = perf_counter()
        with self._condition:
            if self._can_run(policy):
                self._reserve(policy.tenant_id)
                return AdmissionTicket(tenant_id=policy.tenant_id, queue_wait_seconds=0.0)

            if self._queue_depth >= self.max_queue_size:
                raise AdmissionRejectedError("request queue is full")

            self._queue_depth += 1
            while True:
                remaining = self.max_queue_wait_seconds - (perf_counter() - start)
                if remaining <= 0:
                    self._queue_depth -= 1
                    raise AdmissionRejectedError("request queue wait time exceeded")
                self._condition.wait(timeout=remaining)
                if self._can_run(policy):
                    self._queue_depth -= 1
                    self._reserve(policy.tenant_id)
                    return AdmissionTicket(
                        tenant_id=policy.tenant_id,
                        queue_wait_seconds=round(perf_counter() - start, 4),
                    )

    def release(self, tenant_id: str) -> None:
        with self._condition:
            self._inflight = max(0, self._inflight - 1)
            self._tenant_inflight[tenant_id] = max(0, self._tenant_inflight[tenant_id] - 1)
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "max_concurrent_requests": self.max_concurrent_requests,
                "max_queue_size": self.max_queue_size,
                "queue_depth": self._queue_depth,
                "inflight_requests": self._inflight,
                "tenant_inflight": dict(self._tenant_inflight),
            }

    def _can_run(self, policy: TenantPolicy) -> bool:
        if self._inflight >= self.max_concurrent_requests:
            return False
        tenant_limit = policy.max_concurrent_requests
        if tenant_limit is None:
            return True
        return self._tenant_inflight[policy.tenant_id] < tenant_limit

    def _reserve(self, tenant_id: str) -> None:
        self._inflight += 1
        self._tenant_inflight[tenant_id] += 1


class MetricsCollector:
    def __init__(self) -> None:
        self._lock = Lock()
        self._requests_total: dict[str, int] = defaultdict(int)
        self._request_failures: dict[str, int] = defaultdict(int)
        self._request_latency_seconds: dict[str, float] = defaultdict(float)
        self._queue_wait_seconds: dict[str, float] = defaultdict(float)
        self._model_load_seconds: dict[str, float] = defaultdict(float)
        self._model_load_count: dict[str, int] = defaultdict(int)
        self._cache_hits: dict[str, int] = defaultdict(int)
        self._cache_misses: dict[str, int] = defaultdict(int)
        self._warmup_runs: dict[str, int] = defaultdict(int)
        self._last_device_metrics: dict[str, Any] = {}

    def record_request(
        self,
        operation: str,
        latency_seconds: float,
        queue_wait_seconds: float,
        device_metrics: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._requests_total[operation] += 1
            self._request_latency_seconds[operation] += latency_seconds
            self._queue_wait_seconds[operation] += queue_wait_seconds
            if device_metrics is not None:
                self._last_device_metrics = device_metrics

    def record_failure(self, operation: str) -> None:
        with self._lock:
            self._request_failures[operation] += 1

    def record_model_load(self, backend: str, model_id: str, load_seconds: float) -> None:
        key = f"{backend}:{model_id}"
        with self._lock:
            self._model_load_seconds[key] += load_seconds
            self._model_load_count[key] += 1

    def record_cache(self, backend: str, model_id: str, hit: bool) -> None:
        key = f"{backend}:{model_id}"
        with self._lock:
            if hit:
                self._cache_hits[key] += 1
            else:
                self._cache_misses[key] += 1

    def record_warmup(self, operation: str) -> None:
        with self._lock:
            self._warmup_runs[operation] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            cache: dict[str, dict[str, Any]] = {}
            for key in sorted(set(self._cache_hits) | set(self._cache_misses)):
                hits = self._cache_hits.get(key, 0)
                misses = self._cache_misses.get(key, 0)
                total = hits + misses
                cache[key] = {
                    "hits": hits,
                    "misses": misses,
                    "hit_rate": round(hits / total, 4) if total else 0.0,
                }

            loads: dict[str, dict[str, Any]] = {}
            for key, total_seconds in self._model_load_seconds.items():
                count = self._model_load_count[key]
                loads[key] = {
                    "count": count,
                    "total_seconds": round(total_seconds, 4),
                    "average_seconds": round(total_seconds / count, 4) if count else 0.0,
                }

            operations: dict[str, dict[str, Any]] = {}
            for operation, total in self._requests_total.items():
                operations[operation] = {
                    "requests_total": total,
                    "request_failures": self._request_failures.get(operation, 0),
                    "average_latency_seconds": round(self._request_latency_seconds[operation] / total, 4)
                    if total
                    else 0.0,
                    "average_queue_wait_seconds": round(self._queue_wait_seconds[operation] / total, 4)
                    if total
                    else 0.0,
                }

            return {
                "operations": operations,
                "model_load": loads,
                "cache": cache,
                "warmups": dict(self._warmup_runs),
                "device": dict(self._last_device_metrics),
            }

    def prometheus_text(self, admission_snapshot: dict[str, Any]) -> str:
        snapshot = self.snapshot()
        lines = [
            "# HELP hf_requests_total Total completed requests by operation.",
            "# TYPE hf_requests_total counter",
        ]
        for operation, payload in snapshot["operations"].items():
            lines.append(f'hf_requests_total{{operation="{operation}"}} {payload["requests_total"]}')
            lines.append(f'hf_request_failures_total{{operation="{operation}"}} {payload["request_failures"]}')
            lines.append(
                f'hf_average_latency_seconds{{operation="{operation}"}} {payload["average_latency_seconds"]}'
            )
            lines.append(
                f'hf_average_queue_wait_seconds{{operation="{operation}"}} {payload["average_queue_wait_seconds"]}'
            )

        lines.extend(
            [
                "# HELP hf_inflight_requests Current inflight requests.",
                "# TYPE hf_inflight_requests gauge",
                f'hf_inflight_requests {admission_snapshot["inflight_requests"]}',
                "# HELP hf_queue_depth Current queued requests.",
                "# TYPE hf_queue_depth gauge",
                f'hf_queue_depth {admission_snapshot["queue_depth"]}',
            ]
        )
        return "\n".join(lines) + "\n"


class PlatformController:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.tenants = self._parse_tenant_policies(settings)
        self.admission = AdmissionController(
            max_concurrent_requests=settings.max_concurrent_requests,
            max_queue_size=settings.max_queue_size,
            max_queue_wait_seconds=settings.max_queue_wait_seconds,
        )
        self.metrics = MetricsCollector()

    def resolve_tenant(self, tenant_id: str | None) -> TenantPolicy:
        resolved_id = tenant_id or self.settings.default_tenant_id
        if tenant_id is None and self.settings.require_tenant_header:
            raise TenantBoundaryError("tenant header is required")
        if resolved_id not in self.tenants:
            raise TenantBoundaryError(f"unknown tenant: {resolved_id}")
        return self.tenants[resolved_id]

    def ensure_model_access(self, policy: TenantPolicy, operation: str, model_id: str) -> None:
        if not policy.allows(operation, model_id):
            raise TenantBoundaryError(f"tenant {policy.tenant_id} cannot access model {model_id}")

    def observability_snapshot(self) -> dict[str, Any]:
        return {
            "admission": self.admission.snapshot(),
            "metrics": self.metrics.snapshot(),
            "tenants": {
                tenant_id: {
                    "allowed_text_models": sorted(policy.allowed_text_models),
                    "allowed_image_models": sorted(policy.allowed_image_models),
                    "max_concurrent_requests": policy.max_concurrent_requests,
                }
                for tenant_id, policy in self.tenants.items()
            },
        }

    def prometheus_text(self) -> str:
        return self.metrics.prometheus_text(self.admission.snapshot())

    def _parse_tenant_policies(self, settings: Settings) -> dict[str, TenantPolicy]:
        raw = json.loads(settings.tenant_policies_json)
        tenants: dict[str, TenantPolicy] = {}
        for tenant_id, payload in raw.items():
            tenants[tenant_id] = TenantPolicy(
                tenant_id=tenant_id,
                allowed_text_models=set(payload.get("allowed_text_models", ["*"])),
                allowed_image_models=set(payload.get("allowed_image_models", ["*"])),
                max_concurrent_requests=payload.get("max_concurrent_requests"),
            )
        if settings.default_tenant_id not in tenants:
            tenants[settings.default_tenant_id] = TenantPolicy(tenant_id=settings.default_tenant_id)
        return tenants