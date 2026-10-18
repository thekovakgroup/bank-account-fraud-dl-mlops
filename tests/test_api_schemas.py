
# ============================================================
# MILESTONE 09 — PYDANTIC SCHEMA TESTS
# Step 09.24 — Input Validation Testing
# ============================================================

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas import (
    FraudApplication,
    PredictionRequest,
)


# ------------------------------------------------------------
# 1. Load the frozen preprocessing feature configuration.
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "preprocessing_config.json"
)

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    FEATURE_CONFIG = json.load(file)


# ------------------------------------------------------------
# 2. Create synthetic application data for testing.
# ------------------------------------------------------------

def make_valid_application():
    """
    Generate a synthetic application matching the raw
    feature contract.

    No private dataset or final holdout is accessed.
    """

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
# 3. Verify valid input is accepted.
# ------------------------------------------------------------

def test_valid_application():

    application = make_valid_application()

    validated = FraudApplication.model_validate(
        application
    )

    assert len(validated.model_dump()) == 29


# ------------------------------------------------------------
# 4. Verify missing required features are rejected.
# ------------------------------------------------------------

def test_missing_feature_rejected():

    application = make_valid_application()

    del application["income"]

    with pytest.raises(ValidationError):
        FraudApplication.model_validate(application)


# ------------------------------------------------------------
# 5. Verify unexpected fields are rejected.
# ------------------------------------------------------------

def test_excluded_feature_rejected():

    application = make_valid_application()

    application["fraud_bool"] = 1

    with pytest.raises(ValidationError):
        FraudApplication.model_validate(application)


# ------------------------------------------------------------
# 6. Verify invalid binary values are rejected.
# ------------------------------------------------------------

def test_invalid_binary_rejected():

    application = make_valid_application()

    application["email_is_free"] = 2

    with pytest.raises(ValidationError):
        FraudApplication.model_validate(application)


# ------------------------------------------------------------
# 7. Verify invalid numeric values are rejected.
# ------------------------------------------------------------

@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_nonfinite_numeric_rejected(invalid_value):

    application = make_valid_application()

    application["income"] = invalid_value

    with pytest.raises(ValidationError):
        FraudApplication.model_validate(application)


# ------------------------------------------------------------
# 8. Verify empty categorical strings are rejected.
# ------------------------------------------------------------

def test_empty_category_rejected():

    application = make_valid_application()

    application["payment_type"] = ""

    with pytest.raises(ValidationError):
        FraudApplication.model_validate(application)


# ------------------------------------------------------------
# 9. Verify legitimate sentinel values are accepted.
# ------------------------------------------------------------

def test_sentinel_values_accepted():

    application = make_valid_application()

    validated = FraudApplication.model_validate(
        application
    )

    for feature in FEATURE_CONFIG["sentinel_features"]:
        assert getattr(validated, feature) == -1.0


# ------------------------------------------------------------
# 10. Verify empty prediction batches are rejected.
# ------------------------------------------------------------

def test_empty_prediction_request_rejected():

    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(
            {"applications": []}
        )


# ------------------------------------------------------------
# 11. Verify a valid prediction request.
# ------------------------------------------------------------

def test_valid_prediction_request():

    application = make_valid_application()

    request = PredictionRequest.model_validate(
        {"applications": [application]}
    )

    assert len(request.applications) == 1
    assert isinstance(
        request.applications[0],
        FraudApplication,
    )


# ============================================================
# Step 09.53 - Prediction Batch Limit Regression Tests
# ============================================================


# ------------------------------------------------------------
# 1. Test minimum and maximum batch boundaries.
# ------------------------------------------------------------

@pytest.mark.parametrize(
    "batch_size, expected_error",
    [
        (0, "too_short"),
        (1, None),
        (100, None),
        (101, "too_long"),
    ],
)
def test_prediction_request_batch_limits(
    batch_size,
    expected_error,
):
    """
    Verify the API accepts 1-100 applications.

    Reject empty batches and batches larger than 100.

    Uses synthetic application records only.
    """

    # 1. Generate an approved synthetic application.
    application = make_valid_application()

    # 2. Construct a batch of the requested size.
    request_payload = {
        "applications": [
            application
            for _ in range(batch_size)
        ],
    }

    # 3. Verify invalid batch sizes are rejected.
    if expected_error is not None:

        with pytest.raises(ValidationError) as error_info:
            PredictionRequest.model_validate(
                request_payload
            )

        errors = error_info.value.errors()

        assert any(
            error["type"] == expected_error
            and error["loc"] == ("applications",)
            for error in errors
        )

        return

    # 4. Verify valid batch sizes are accepted.
    validated_request = PredictionRequest.model_validate(
        request_payload
    )

    assert len(validated_request.applications) == batch_size
