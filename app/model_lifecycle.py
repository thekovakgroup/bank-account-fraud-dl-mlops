
# ============================================================
# MILESTONE 09 - FASTAPI MODEL LIFECYCLE
# Step 09.30 - Frozen Model Lifecycle Manager
# ============================================================

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.fraud_inference import FraudInference


# ------------------------------------------------------------
# 1. Define a reusable model lifecycle factory.
# ------------------------------------------------------------

def build_model_lifespan(
    inference_factory: Callable[[], FraudInference],
):
    """
    Create a FastAPI lifespan manager.

    The inference factory is injectable so automated
    API tests can use a synthetic test double without
    loading private model artifacts.
    """

    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ) -> AsyncIterator[None]:

        # ----------------------------------------------------
        # 2. Initialize model readiness state.
        # ----------------------------------------------------

        app.state.inference_engine = None
        app.state.model_ready = False

        # ----------------------------------------------------
        # 3. Load the frozen inference engine once.
        # ----------------------------------------------------

        try:
            engine = inference_factory()

        except Exception as error:
            raise RuntimeError(
                "Frozen inference engine failed to initialize."
            ) from error

        app.state.inference_engine = engine
        app.state.model_ready = True

        # ----------------------------------------------------
        # 4. Keep the engine available during API operation.
        # ----------------------------------------------------

        try:
            yield

        # ----------------------------------------------------
        # 5. Clear references during application shutdown.
        # ----------------------------------------------------

        finally:
            app.state.model_ready = False
            app.state.inference_engine = None

    return lifespan
