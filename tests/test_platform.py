from __future__ import annotations

import threading
from time import sleep

import pytest

from hf_inference_platform.config import Settings
from hf_inference_platform.platform import AdmissionRejectedError, PlatformController, TenantBoundaryError


def test_platform_controller_resolves_default_tenant() -> None:
    controller = PlatformController(Settings())
    policy = controller.resolve_tenant(None)
    assert policy.tenant_id == "public"


def test_platform_controller_rejects_unknown_tenant() -> None:
    controller = PlatformController(Settings())
    with pytest.raises(TenantBoundaryError):
        controller.resolve_tenant("missing")


def test_platform_controller_enforces_model_access() -> None:
    controller = PlatformController(
        Settings(
            TENANT_POLICIES_JSON=(
                '{"public":{"allowed_text_models":["distilgpt2"],"allowed_image_models":["*"]}}'
            )
        )
    )
    policy = controller.resolve_tenant("public")
    controller.ensure_model_access(policy, "text", "distilgpt2")
    with pytest.raises(TenantBoundaryError):
        controller.ensure_model_access(policy, "text", "gpt2")


def test_admission_controller_rejects_when_queue_is_full() -> None:
    controller = PlatformController(Settings(MAX_CONCURRENT_REQUESTS=1, MAX_QUEUE_SIZE=0))
    policy = controller.resolve_tenant("public")
    ticket = controller.admission.acquire(policy)
    try:
        with pytest.raises(AdmissionRejectedError):
            controller.admission.acquire(policy)
    finally:
        controller.admission.release(ticket.tenant_id)


def test_admission_controller_respects_tenant_limit() -> None:
    controller = PlatformController(
        Settings(
            MAX_CONCURRENT_REQUESTS=2,
            TENANT_POLICIES_JSON='{"public":{"allowed_text_models":["*"],"allowed_image_models":["*"],"max_concurrent_requests":1}}',
        )
    )
    policy = controller.resolve_tenant("public")
    first = controller.admission.acquire(policy)
    result: list[float] = []

    def acquire_after_release() -> None:
        second = controller.admission.acquire(policy)
        result.append(second.queue_wait_seconds)
        controller.admission.release(second.tenant_id)

    thread = threading.Thread(target=acquire_after_release)
    thread.start()
    sleep(0.05)
    controller.admission.release(first.tenant_id)
    thread.join(timeout=1)
    assert result and result[0] > 0.0