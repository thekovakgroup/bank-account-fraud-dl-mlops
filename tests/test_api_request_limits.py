
# ============================================================
# MILESTONE 09 - REQUEST SIZE PROTECTION TEST
# Step 09.90 - Verify 256 KiB HTTP Body Limit
# ============================================================

from fastapi.testclient import TestClient

from app.main import create_app
from app.request_limits import PredictionBodyLimitMiddleware


def test_prediction_request_body_size_boundary():
    """
    Verify that:
    1. A request at the byte limit reaches validation.
    2. A request exceeding the limit receives HTTP 413.

    No real model artifacts or customer records are used.
    """

    # --------------------------------------------------------
    # 1. Build an isolated API using a synthetic engine.
    # --------------------------------------------------------

    test_app = create_app(
        inference_factory=lambda: object()
    )

    body_limit = (
        PredictionBodyLimitMiddleware.MAX_BODY_BYTES
    )

    assert body_limit == 262_144

    with TestClient(test_app) as client:

        # ----------------------------------------------------
        # 2. Send exactly 256 KiB of invalid JSON.
        #    This should reach FastAPI validation.
        # ----------------------------------------------------

        accepted_response = client.post(
            "/predict",
            content=b" " * body_limit,
            headers={"content-type": "application/json"},
        )

        assert accepted_response.status_code == 422

        # ----------------------------------------------------
        # 3. Send one byte beyond the permitted limit.
        #    Middleware must reject before validation.
        # ----------------------------------------------------

        rejected_response = client.post(
            "/predict",
            content=b" " * (body_limit + 1),
            headers={"content-type": "application/json"},
        )

        assert rejected_response.status_code == 413

        assert rejected_response.json() == {
            "detail": "Request body too large."
        }

    # --------------------------------------------------------
    # 4. Report verified protection.
    # --------------------------------------------------------

    print("256 KiB boundary: HTTP 422 VERIFIED")
    print("256 KiB + 1 byte: HTTP 413 VERIFIED")
    print("Frozen model artifacts: NOT ACCESSED")
    print("Step 09.90: PASS")
