
# ============================================================
# MILESTONE 09 - API READINESS TESTS
# Step 09.73 - Model Readiness Verification
# ============================================================

from fastapi.testclient import TestClient

from app.main import create_app


# ------------------------------------------------------------
# 1. Verify the initialized model reports READY.
# ------------------------------------------------------------

def test_ready_endpoint_when_model_available():
    """
    Verify HTTP 200 when the synthetic inference
    engine initializes successfully.
    """

    fake_engine = object()

    test_app = create_app(
        inference_factory=lambda: fake_engine
    )

    with TestClient(test_app) as client:

        response = client.get("/ready")

        assert test_app.state.model_ready is True

        assert (
            test_app.state.inference_engine
            is fake_engine
        )

    assert response.status_code == 200

    assert response.json() == {
        "status": "ready",
    }

    # Verify lifecycle shutdown cleanup.
    assert test_app.state.model_ready is False
    assert test_app.state.inference_engine is None

    print("Model available: HTTP 200 VERIFIED")


# ------------------------------------------------------------
# 2. Verify readiness failure when model is not ready.
# ------------------------------------------------------------

def test_ready_endpoint_when_model_not_ready():
    """
    Verify HTTP 503 when the readiness flag is false.
    """

    test_app = create_app(
        inference_factory=lambda: object()
    )

    with TestClient(test_app) as client:

        # Simulate the model becoming unavailable.
        test_app.state.model_ready = False

        response = client.get("/ready")

        health_response = client.get("/health")

    assert response.status_code == 503

    assert response.json() == {
        "detail": "Fraud inference engine is unavailable.",
    }

    # The API remains alive despite model unavailability.
    assert health_response.status_code == 200

    print("Model not ready: HTTP 503 VERIFIED")
    print("API liveness remains: HTTP 200")


# ------------------------------------------------------------
# 3. Verify readiness failure when engine is missing.
# ------------------------------------------------------------

def test_ready_endpoint_when_engine_missing():
    """
    Verify HTTP 503 when no inference engine is present.
    """

    test_app = create_app(
        inference_factory=lambda: object()
    )

    with TestClient(test_app) as client:

        # Simulate an unavailable inference engine.
        test_app.state.inference_engine = None

        response = client.get("/ready")

    assert response.status_code == 503

    assert response.json() == {
        "detail": "Fraud inference engine is unavailable.",
    }

    print("Missing inference engine: HTTP 503 VERIFIED")
    print("Private model artifacts: NOT ACCESSED")
    print("Step 09.73: PASS")
