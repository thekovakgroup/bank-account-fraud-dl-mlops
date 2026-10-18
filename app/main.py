
# ============================================================
# MILESTONE 09 - BANK ACCOUNT FRAUD DETECTION API
# Step 09.42 - Frozen Model Prediction Integration
# ============================================================

from collections.abc import Callable

import pandas as pd

from fastapi import FastAPI, HTTPException, Request

from app.model_lifecycle import build_model_lifespan
from app.schemas import PredictionRequest
from app.response_schemas import (
    FraudPrediction,
    PredictionResponse,
)

from src.fraud_inference import FraudInference
from app.safe_logging import log_inference_failure
from app.request_limits import PredictionBodyLimitMiddleware




# ------------------------------------------------------------
# 1. Define the FastAPI application factory.
# ------------------------------------------------------------

def create_app(
    inference_factory: Callable[[], FraudInference] = FraudInference,
) -> FastAPI:
    """
    Create a research-only fraud risk scoring API.

    The application supports dependency injection for
    testing without private model artifacts.
    """

    api = FastAPI(
        title="Bank Account Fraud Detection API",
        description=(
            "Research-only fraud risk scoring API. "
            "Predictions support human investigations "
            "and must not trigger automatic application rejection."
        ),
        version="0.1.0",
        lifespan=build_model_lifespan(
            inference_factory
        ),
    )

    # --------------------------------------------------------
    # Step 09.88 - Protect prediction request resources.
    # --------------------------------------------------------

    api.add_middleware(
        PredictionBodyLimitMiddleware
    )

    # --------------------------------------------------------
    # 2. API liveness endpoint.
    # --------------------------------------------------------

    @api.get("/health", tags=["Operations"])
    def health_check() -> dict[str, str]:
        """
        Confirm the API is responding.

        This does not confirm model readiness.
        """

        return {
            "status": "alive",
        }

    # --------------------------------------------------------
    # Step 09.71 - Model readiness endpoint.
    # --------------------------------------------------------

    @api.get("/ready", tags=["Operations"])
    def readiness_check(request: Request) -> dict[str, str]:
        """
        Verify that the frozen inference engine is ready.

        This endpoint does not execute predictions.
        """

        model_ready = getattr(
            request.app.state,
            "model_ready",
            False,
        )

        inference_engine = getattr(
            request.app.state,
            "inference_engine",
            None,
        )

        if not model_ready or inference_engine is None:
            raise HTTPException(
                status_code=503,
                detail="Fraud inference engine is unavailable.",
            )

        return {
            "status": "ready",
        }

    # --------------------------------------------------------
    # 3. Calibrated fraud prediction endpoint.
    # --------------------------------------------------------

    @api.post(
        "/predict",
        response_model=PredictionResponse,
        tags=["Predictions"],
    )
    def predict_fraud(
        payload: PredictionRequest,
        request: Request,
    ) -> PredictionResponse:
        """
        Return calibrated fraud probabilities.

        Predictions are risk scores only.
        No automatic application decisions are made.
        """

        # ----------------------------------------------------
        # 4. Verify model readiness.
        # ----------------------------------------------------

        model_ready = getattr(
            request.app.state,
            "model_ready",
            False,
        )

        inference_engine = getattr(
            request.app.state,
            "inference_engine",
            None,
        )

        if not model_ready or inference_engine is None:
            raise HTTPException(
                status_code=503,
                detail="Fraud inference engine is unavailable.",
            )

        # ----------------------------------------------------
        # 5. Convert validated requests into a DataFrame.
        # ----------------------------------------------------

        application_records = [
            application.model_dump()
            for application in payload.applications
        ]

        applications = pd.DataFrame(
            application_records
        )

        # ----------------------------------------------------
        # 6. Execute frozen inference and validate output.
        # ----------------------------------------------------

        try:
            probabilities = (
                inference_engine.predict_proba(
                    applications
                )
            )

            if len(probabilities) != len(applications):
                raise ValueError(
                    "Prediction count mismatch"
                )

            prediction_results = [
                FraudPrediction(
                    application_index=index,
                    fraud_probability=float(probability),
                )
                for index, probability in enumerate(
                    probabilities
                )
            ]

            # -----------------------------------------------
            # 7. Retrieve frozen model metadata.
            # -----------------------------------------------

            policy = inference_engine.policy

            checkpoint_hash = policy[
                "model"
            ]["checkpoint_sha256"]

            # Use the actual checkpoint fingerprint as
            # an immutable model identification value.
            model_identifier = (
                f"sha256:{checkpoint_hash}"
            )

            # -----------------------------------------------
            # 8. Return the validated API response.
            # -----------------------------------------------

            return PredictionResponse(
                model_version=model_identifier,
                policy_status=policy["status"],
                production_approved=policy[
                    "production_approved"
                ],
                predictions=prediction_results,
            )


        except Exception as error:

            # Record the failure without exposing sensitive data.
            log_inference_failure(error)

            # Preserve the existing generic client response.
            raise HTTPException(
                status_code=500,
                detail="Fraud scoring failed.",
            )

    return api


# ------------------------------------------------------------
# 9. Create the default serving application.
# ------------------------------------------------------------

app = create_app()
