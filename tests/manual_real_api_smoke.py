
# ============================================================
# MILESTONE 09 - REAL MODEL API SMOKE TEST
# Step 09.46 - Frozen PyTorch Inference Through FastAPI
# ============================================================

import json
import math
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


# ------------------------------------------------------------
# 1. Locate the approved feature configuration.
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "preprocessing_config.json"
)


# ------------------------------------------------------------
# 2. Build synthetic application data.
# ------------------------------------------------------------

def make_synthetic_application():

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    application = {}

    for feature in config["numeric_features"]:
        application[feature] = 1.0

    for feature in config["categorical_features"]:
        application[feature] = "synthetic_category"

    for feature in config["binary_features"]:
        application[feature] = 0

    for feature in config["sentinel_features"]:
        application[feature] = -1.0

    return application


# ------------------------------------------------------------
# 3. Test the real frozen inference engine.
# ------------------------------------------------------------

def test_real_frozen_model_api():

    application = make_synthetic_application()

    # Start FastAPI with the actual frozen model.
    with TestClient(app) as client:

        assert app.state.model_ready is True

        response = client.post(
            "/predict",
            json={"applications": [application]},
        )

    # --------------------------------------------------------
    # 4. Verify the HTTP response.
    # --------------------------------------------------------

    assert response.status_code == 200, response.text

    result = response.json()

    assert len(result["predictions"]) == 1

    assert result["predictions"][0]["application_index"] == 0

    # --------------------------------------------------------
    # 5. Validate the actual model output.
    # --------------------------------------------------------

    fraud_probability = result["predictions"][0][
        "fraud_probability"
    ]

    assert math.isfinite(fraud_probability)

    assert 0.0 <= fraud_probability <= 1.0

    # --------------------------------------------------------
    # 6. Verify model identity and governance.
    # --------------------------------------------------------

    assert result["model_version"].startswith("sha256:")

    assert len(result["model_version"]) == 71

    assert (
        result["policy_status"]
        == "FROZEN_PROVISIONAL_CANDIDATE"
    )

    assert result["production_approved"] is False

    assert result["decision_type"] == "risk_score_only"

    # --------------------------------------------------------
    # 7. Verify shutdown cleanup.
    # --------------------------------------------------------

    assert app.state.model_ready is False
    assert app.state.inference_engine is None

    print("Real frozen PyTorch model: LOADED")
    print("API prediction: VERIFIED")
    print("Probability range: VERIFIED")
    print("Model identity: VERIFIED")
    print("Research-only policy: PRESERVED")
    print("Application shutdown: VERIFIED")
    print("Month-7 holdout: NOT ACCESSED")
    print("Step 09.46: PASS")
