"""API contract tests.

Run against the stub predictor, so the whole file executes in about a second with
no torch, no weights and no network. What is being tested is the *contract* —
status codes, schema shape, validation behaviour — which is exactly the part that
breaks when someone edits a response model.
"""

from __future__ import annotations

import pytest
from api.model_loader import reset_predictor, set_predictor
from fastapi.testclient import TestClient

from forgeml.inference.predictor import StubPredictor


@pytest.fixture
def client() -> TestClient:
    set_predictor(StubPredictor("pytest"))
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client
    reset_predictor()


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["version"]
    assert payload["uptime_seconds"] >= 0


def test_health_reports_healthy_even_without_a_model(client: TestClient) -> None:
    """Liveness is not readiness. A container still loading weights must not be killed."""
    payload = client.get("/health").json()
    assert payload["status"] == "healthy"
    assert payload["model_loaded"] is False


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def test_model_endpoint_describes_what_is_serving(client: TestClient) -> None:
    response = client.get("/model")
    assert response.status_code == 200

    payload = response.json()
    assert payload["name"] == "stub"
    assert payload["loaded"] is False
    assert "method" in payload


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def test_predict_returns_the_full_contract(client: TestClient) -> None:
    response = client.post("/predict", json={"prompt": "Explain overfitting."})
    assert response.status_code == 200

    payload = response.json()
    for key in ("model", "prediction", "latency_ms", "prompt_tokens",
                "completion_tokens", "finish_reason"):
        assert key in payload
    assert isinstance(payload["prediction"], str)
    assert payload["latency_ms"] >= 0


def test_predict_accepts_optional_context(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={"prompt": "Summarise this.", "context": "Some grounding text.",
              "max_new_tokens": 32},
    )
    assert response.status_code == 200


def test_empty_prompt_is_422_not_500(client: TestClient) -> None:
    """Validation belongs at the edge, not in a traceback from the model."""
    assert client.post("/predict", json={"prompt": ""}).status_code == 422


def test_missing_prompt_is_422(client: TestClient) -> None:
    assert client.post("/predict", json={}).status_code == 422


def test_out_of_range_parameters_are_422(client: TestClient) -> None:
    assert client.post(
        "/predict", json={"prompt": "hi", "max_new_tokens": 99999}
    ).status_code == 422
    assert client.post(
        "/predict", json={"prompt": "hi", "temperature": -1}
    ).status_code == 422
    assert client.post(
        "/predict", json={"prompt": "hi", "top_p": 0}
    ).status_code == 422


def test_malformed_json_is_422(client: TestClient) -> None:
    response = client.post(
        "/predict", content="not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_predict_failure_becomes_a_clean_500() -> None:
    """A traceback must never reach the caller, and neither must the error text.

    `raise_server_exceptions` is a TestClient constructor argument, so this test
    builds its own client instead of reusing the fixture.
    """

    class Broken(StubPredictor):
        def predict(self, prompt, context=None, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("internal detail that must not leak")

    set_predictor(Broken("broken"))
    from api.main import app

    with TestClient(app, raise_server_exceptions=False) as broken_client:
        response = broken_client.post("/predict", json={"prompt": "hi"})

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "RuntimeError" in detail
    assert "internal detail that must not leak" not in detail
    reset_predictor()


# ---------------------------------------------------------------------------
# metrics + misc
# ---------------------------------------------------------------------------


def test_metrics_counts_predictions(client: TestClient) -> None:
    before = client.get("/metrics").json()["requests_total"]
    client.post("/predict", json={"prompt": "one"})
    client.post("/predict", json={"prompt": "two"})
    after = client.get("/metrics").json()

    assert after["requests_total"] == before + 2
    assert after["latency_ms_mean"] >= 0
    assert after["tokens_generated_total"] > 0


def test_root_lists_the_endpoints(client: TestClient) -> None:
    payload = client.get("/").json()
    assert "/predict" in payload["endpoints"]


def test_openapi_schema_is_generated(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/predict" in schema["paths"]
    assert "/health" in schema["paths"]


def test_stub_response_is_clearly_labelled() -> None:
    """Nobody should mistake a stub answer for a model answer."""
    result = StubPredictor("test").predict("hello")
    assert result.text.startswith("[stub]")
    assert result.finish_reason == "stub"
