
# ============================================================
# MILESTONE 09 - FASTAPI RESPONSE SCHEMAS
# Step 09.38 - Prediction Response Contract
# ============================================================

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ------------------------------------------------------------
# 1. Define one calibrated fraud risk score.
# ------------------------------------------------------------

class FraudPrediction(BaseModel):
    """
    Represent one application's calibrated fraud probability.

    A fraud probability is NOT an automatic rejection
    or a cohort investigation recommendation.
    """

    model_config = ConfigDict(extra="forbid")

    application_index: int = Field(
        ge=0,
        description="Zero-based position in the request batch.",
    )

    fraud_probability: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Calibrated fraud probability from 0 to 1.",
    )


# ------------------------------------------------------------
# 2. Define the complete API prediction response.
# ------------------------------------------------------------

class PredictionResponse(BaseModel):
    """
    Return calibrated fraud risk scores and
    frozen model metadata.
    """

    model_config = ConfigDict(extra="forbid")

    model_version: str = Field(
        min_length=1,
    )

    policy_status: Literal[
        "FROZEN_PROVISIONAL_CANDIDATE"
    ]

    production_approved: Literal[False]

    decision_type: Literal[
        "risk_score_only"
    ] = "risk_score_only"

    predictions: list[FraudPrediction] = Field(
        min_length=1,
    )
