
# ============================================================
# MILESTONE 09 - FASTAPI AUTOMATED TESTING
# Step 09.36 - Application Factory Health Test
# ============================================================

from fastapi.testclient import TestClient

from app.main import create_app


# ------------------------------------------------------------
# 1. Test the health endpoint using a synthetic engine.
# ------------------------------------------------------------

def test_health_endpoint_returns_alive():
    """
    Verify API liveness using dependency injection.

    No private model artifacts are loaded.
    """

    # --------------------------------------------------------
    # 2. Create a synthetic inference factory.
    # --------------------------------------------------------

    initialization_count = 0

    def fake_inference_factory():
        nonlocal initialization_count

        initialization_count += 1

        return object()

    # --------------------------------------------------------
    # 3. Build an isolated FastAPI application.
    # --------------------------------------------------------

    test_app = create_app(
        inference_factory=fake_inference_factory
    )

    # --------------------------------------------------------
    # 4. Start the application and call the health endpoint.
    # --------------------------------------------------------

    with TestClient(test_app) as client:

        response = client.get("/health")

        # Verify the synthetic engine was initialized.
        assert initialization_count == 1
        assert test_app.state.model_ready is True

    # --------------------------------------------------------
    # 5. Verify the HTTP response.
    # --------------------------------------------------------

    assert response.status_code == 200

    expected_response = {
        "status": "alive",
    }

    assert response.json() == expected_response

    assert (
        "application/json"
        in response.headers["content-type"]
    )

    # --------------------------------------------------------
    # 6. Verify shutdown cleanup.
    # --------------------------------------------------------

    assert test_app.state.model_ready is False

    assert test_app.state.inference_engine is None

    # --------------------------------------------------------
    # 7. Report successful verification.
    # --------------------------------------------------------

    print("Health endpoint: VERIFIED")
    print("Synthetic model initialization: VERIFIED")
    print("Shutdown cleanup: VERIFIED")
    print("Private model artifacts: NOT ACCESSED")
    print("Step 09.36: PASS")
