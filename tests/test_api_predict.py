
# ============================================================
# MILESTONE 09 - PREDICTION API INTEGRATION TEST
# Step 09.44 - Synthetic Inference Engine
# ============================================================

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app


# ------------------------------------------------------------
# 1. Load the approved feature configuration.
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "preprocessing_config.json"
)

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    FEATURE_CONFIG = json.load(file)


# ------------------------------------------------------------
# 2. Generate synthetic bank application data.
# ------------------------------------------------------------

def make_synthetic_application():

    application = {}

    for feature in FEATURE_CONFIG["numeric_features"]:
        application[feature] = 1.0

    for feature in FEATURE_CONFIG["categorical_features"]:
        application[feature] = "synthetic_category"

    for feature in FEATURE_CONFIG["binary_features"]:
        application[feature] = 0

    for feature in FEATURE_CONFIG["sentinel_features"]:
        application[feature] = -1.0

    return application


# ------------------------------------------------------------
# 3. Define a synthetic inference engine.
# ------------------------------------------------------------

class FakeFraudInference:
    """
    Simulate the existing FraudInference interface.

    Does not load the real PyTorch model or private data.
    """

    def __init__(self):

        self.policy = {
            "status": "FROZEN_PROVISIONAL_CANDIDATE",
            "production_approved": False,
            "model": {
                "checkpoint_sha256": "a" * 64,
            },
        }

        self.received_count = 0

    def predict_proba(self, applications):

        self.received_count = len(applications)

        assert self.received_count == 2

        assert len(applications.columns) == 29

        return [0.125, 0.875]


# ------------------------------------------------------------
# 4. Test the complete prediction API workflow.
# ------------------------------------------------------------

def test_predict_endpoint_with_synthetic_engine():

    # Create a controlled inference engine.
    fake_engine = FakeFraudInference()

    # Inject it into an isolated FastAPI application.
    test_app = create_app(
        inference_factory=lambda: fake_engine
    )

    # Prepare two valid synthetic applications.
    applications = [
        make_synthetic_application(),
        make_synthetic_application(),
    ]

    # Start the API and submit the prediction request.
    with TestClient(test_app) as client:

        response = client.post(
            "/predict",
            json={"applications": applications},
        )

    # --------------------------------------------------------
    # 5. Verify the HTTP response.
    # --------------------------------------------------------

    assert response.status_code == 200, response.text

    result = response.json()

    assert len(result["predictions"]) == 2

    assert fake_engine.received_count == 2

    # --------------------------------------------------------
    # 6. Verify probability values and application order.
    # --------------------------------------------------------

    assert result["predictions"][0] == {
        "application_index": 0,
        "fraud_probability": 0.125,
    }

    assert result["predictions"][1] == {
        "application_index": 1,
        "fraud_probability": 0.875,
    }

    # --------------------------------------------------------
    # 7. Verify model metadata and governance.
    # --------------------------------------------------------

    assert result["model_version"] == "sha256:" + "a" * 64

    assert (
        result["policy_status"]
        == "FROZEN_PROVISIONAL_CANDIDATE"
    )

    assert result["production_approved"] is False

    assert result["decision_type"] == "risk_score_only"

    # No automatic rejection or investigation decision.
    assert "automatic_decision" not in result
    assert "investigate" not in result

    # --------------------------------------------------------
    # 8. Report successful verification.
    # --------------------------------------------------------

    print("HTTP prediction endpoint: VERIFIED")
    print("Synthetic applications processed: 2")
    print("Probability ordering: VERIFIED")
    print("Response metadata: VERIFIED")
    print("Research-only policy: PRESERVED")
    print("Private model artifacts: NOT ACCESSED")
    print("Step 09.44: PASS")



# ============================================================
# Step 09.48 - API Error-Handling Tests
# ============================================================


# ------------------------------------------------------------
# 1. Verify model execution errors remain private.
# ------------------------------------------------------------

def test_predict_model_failure_returns_safe_error(caplog):
    """
    Verify that internal inference failures produce
    a generic HTTP 500 response.

    No private application details should be returned.
    """

    class FailingFraudInference(FakeFraudInference):

        def predict_proba(self, applications):
            raise RuntimeError(
                "SYNTHETIC_PRIVATE_ERROR_MARKER"
            )

    # Create an isolated API with a failing engine.
    test_app = create_app(
        inference_factory=FailingFraudInference
    )

    application = make_synthetic_application()

    with TestClient(test_app) as client:
        response = client.post(
            "/predict",
            json={"applications": [application]},
        )

    # Confirm the service reports an internal error.
    assert response.status_code == 500

    # Confirm no internal exception details are returned.
    assert response.json() == {
        "detail": "Fraud scoring failed.",
    }

    assert (
        "SYNTHETIC_PRIVATE_ERROR_MARKER"
        not in response.text
    )

    # --------------------------------------------------------
    # Step 09.84 - Verify API-to-logger integration.
    # --------------------------------------------------------

    failure_logs = [
        record
        for record in caplog.records
        if record.name == "fraud_detection.api"
        and record.levelname == "ERROR"
        and record.getMessage().startswith(
            "event=fraud_inference_failed exception_type="
        )
    ]

    assert len(failure_logs) == 1
    assert failure_logs[0].exc_info is None

    print("API-to-logger integration: VERIFIED")
    print("Sanitized failure event: VERIFIED")

    print("Inference failure handling: VERIFIED")
    print("Sensitive error disclosure: PREVENTED")
    print("Step 09.48A: PASS")


# ------------------------------------------------------------
# 2. Verify unavailable models return HTTP 503.
# ------------------------------------------------------------

def test_predict_model_unavailable_returns_503():
    """
    Verify that prediction requests fail safely
    when the inference engine is not ready.
    """

    fake_engine = FakeFraudInference()

    test_app = create_app(
        inference_factory=lambda: fake_engine
    )

    application = make_synthetic_application()

    with TestClient(test_app) as client:

        # Simulate an unavailable inference service.
        test_app.state.model_ready = False

        response = client.post(
            "/predict",
            json={"applications": [application]},
        )

    # Confirm a clear service-unavailable response.
    assert response.status_code == 503

    assert response.json() == {
        "detail": "Fraud inference engine is unavailable.",
    }

    # Confirm no inference was performed.
    assert fake_engine.received_count == 0

    print("Model readiness enforcement: VERIFIED")
    print("Unavailable model response: HTTP 503")
    print("Step 09.48B: PASS")


# ============================================================
# Step 09.55 - HTTP Batch-Limit Validation
# ============================================================


def test_predict_http_batch_limits():
    """
    Verify invalid batch sizes return HTTP 422.

    Confirm the inference engine is never called.
    """

    # 1. Create a synthetic inference engine.
    fake_engine = FakeFraudInference()

    # 2. Build an isolated FastAPI application.
    test_app = create_app(
        inference_factory=lambda: fake_engine
    )

    # 3. Generate a synthetic bank application.
    application = make_synthetic_application()

    # 4. Test invalid batch sizes over HTTP.
    with TestClient(test_app) as client:

        empty_response = client.post(
            "/predict",
            json={"applications": []},
        )

        oversized_response = client.post(
            "/predict",
            json={"applications": [application] * 101},
        )

    # 5. Verify both requests are rejected.
    assert empty_response.status_code == 422
    assert oversized_response.status_code == 422

    # 6. Verify the correct validation errors.
    assert (
        empty_response.json()["detail"][0]["type"]
        == "too_short"
    )

    assert (
        oversized_response.json()["detail"][0]["type"]
        == "too_long"
    )

    # 7. Confirm inference never executed.
    assert fake_engine.received_count == 0

    # 8. Report successful verification.
    print("Empty batch: HTTP 422")
    print("Oversized batch: HTTP 422")
    print("Inference execution: PREVENTED")
    print("Step 09.55: PASS")
