import httpx
import pytest

from hf_inference_sdk import InferenceClient


@pytest.mark.parametrize(
    ("path", "method_name"),
    [
        ("/healthz", "health"),
        ("/v1/catalog", "catalog"),
        ("/v1/observability", "observability"),
    ],
)
def test_client_simple_gets(path: str, method_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, timeout: float) -> httpx.Response:
        request = httpx.Request("GET", url)
        return httpx.Response(status_code=200, json={"path": path}, request=request)

    monkeypatch.setattr(httpx, "get", fake_get)
    client = InferenceClient("http://example.com")
    result = getattr(client, method_name)()
    assert result == {"path": path}


def test_client_metrics_text(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, timeout: float) -> httpx.Response:
        request = httpx.Request("GET", url)
        return httpx.Response(status_code=200, text="hf_queue_depth 0\n", request=request)

    monkeypatch.setattr(httpx, "get", fake_get)
    client = InferenceClient("http://example.com")
    assert client.metrics_text() == "hf_queue_depth 0\n"
