
# ============================================================
# MILESTONE 09 - API RESPONSE VALIDATION TESTS
# Step 09.40 - Automated Response Schema Testing
# ============================================================

import pytest

from pydantic import ValidationError

from app.response_schemas import PredictionResponse


# ------------------------------------------------------------
# 1. Create synthetic response data.
# ------------------------------------------------------------

def make_valid_response():
    """
    Create a valid synthetic prediction response.

    No trained model or private dataset is accessed.
    """

    return {
        "model_version": "schema-test-only",
        "policy_status": "FROZEN_PROVISIONAL_CANDIDATE",
        "production_approved": False,
        "decision_type": "risk_score_only",
        "predictions": [
            {
                "application_index": 0,
                "fraud_probability": 0.25,
            }
        ],
    }


# ------------------------------------------------------------
# 2. Verify valid responses are accepted.
# ------------------------------------------------------------

def test_valid_prediction_response():

    response = PredictionResponse.model_validate(
        make_valid_response()
    )

    assert response.predictions[0].fraud_probability == 0.25
    assert response.decision_type == "risk_score_only"
    assert response.production_approved is False


# ------------------------------------------------------------
# 3. Reject probabilities outside [0, 1] or non-finite.
# ------------------------------------------------------------

@pytest.mark.parametrize(
    "invalid_probability",
    [
        -0.01,
        1.01,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_invalid_probability_rejected(
    invalid_probability,
):

    response = make_valid_response()

    response["predictions"][0][
        "fraud_probability"
    ] = invalid_probability

    with pytest.raises(ValidationError):
        PredictionResponse.model_validate(response)


# ------------------------------------------------------------
# 4. Reject negative application indices.
# ------------------------------------------------------------

def test_negative_application_index_rejected():

    response = make_valid_response()

    response["predictions"][0]["application_index"] = -1

    with pytest.raises(ValidationError):
        PredictionResponse.model_validate(response)


# ------------------------------------------------------------
# 5. Prevent unauthorized production approval.
# ------------------------------------------------------------

def test_production_approval_rejected():

    response = make_valid_response()

    response["production_approved"] = True

    with pytest.raises(ValidationError):
        PredictionResponse.model_validate(response)


# ------------------------------------------------------------
# 6. Preserve the frozen provisional policy.
# ------------------------------------------------------------

def test_invalid_policy_status_rejected():

    response = make_valid_response()

    response["policy_status"] = "PRODUCTION_APPROVED"

    with pytest.raises(ValidationError):
        PredictionResponse.model_validate(response)


# ------------------------------------------------------------
# 7. Reject unsupported decision types.
# ------------------------------------------------------------

def test_automatic_rejection_decision_rejected():

    response = make_valid_response()

    response["decision_type"] = "automatic_rejection"

    with pytest.raises(ValidationError):
        PredictionResponse.model_validate(response)


# ------------------------------------------------------------
# 8. Reject empty prediction responses.
# ------------------------------------------------------------

def test_empty_predictions_rejected():

    response = make_valid_response()

    response["predictions"] = []

    with pytest.raises(ValidationError):
        PredictionResponse.model_validate(response)


# ------------------------------------------------------------
# 9. Require a nonempty model version.
# ------------------------------------------------------------

def test_empty_model_version_rejected():

    response = make_valid_response()

    response["model_version"] = ""

    with pytest.raises(ValidationError):
        PredictionResponse.model_validate(response)


# ------------------------------------------------------------
# 10. Reject unexpected response fields.
# ------------------------------------------------------------

def test_unexpected_response_field_rejected():

    response = make_valid_response()

    response["automatic_decision"] = "reject"

    with pytest.raises(ValidationError):
        PredictionResponse.model_validate(response)
