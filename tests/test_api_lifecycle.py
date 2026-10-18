
# ============================================================
# MILESTONE 09 - MODEL LIFECYCLE TESTING
# Step 09.32 - Synthetic Inference Engine
# ============================================================

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.model_lifecycle import build_model_lifespan


# ------------------------------------------------------------
# 1. Test model startup and shutdown.
# ------------------------------------------------------------

def test_model_lifecycle_startup_shutdown():
    """
    Verify that the inference engine:

    1. Loads once during startup.
    2. Remains available during API operation.
    3. Is released during shutdown.

    No real model artifacts are loaded.
    """

    # --------------------------------------------------------
    # 2. Track synthetic engine initialization.
    # --------------------------------------------------------

    created_engines = []

    def fake_inference_factory():
        """
        Create a lightweight synthetic inference engine.
        """

        engine = object()

        created_engines.append(engine)

        return engine

    # --------------------------------------------------------
    # 3. Create an isolated FastAPI application.
    # --------------------------------------------------------

    test_app = FastAPI(
        lifespan=build_model_lifespan(
            fake_inference_factory
        )
    )

    # --------------------------------------------------------
    # 4. Verify startup and model availability.
    # --------------------------------------------------------

    with TestClient(test_app):

        assert len(created_engines) == 1

        assert test_app.state.model_ready is True

        assert (
            test_app.state.inference_engine
            is created_engines[0]
        )

    # --------------------------------------------------------
    # 5. Verify shutdown cleanup.
    # --------------------------------------------------------

    assert test_app.state.model_ready is False

    assert test_app.state.inference_engine is None

    # --------------------------------------------------------
    # 6. Verify initialization happened exactly once.
    # --------------------------------------------------------

    assert len(created_engines) == 1

    print("Model initialized: ONCE")
    print("Startup readiness: VERIFIED")
    print("Shutdown cleanup: VERIFIED")
    print("Private model artifacts: NOT ACCESSED")
    print("Step 09.32: PASS")


# ============================================================
# Step 09.76 - Model Initialization Failure Testing
# ============================================================


def test_model_initialization_failure():
    """
    Verify that an inference engine initialization
    failure prevents FastAPI startup.

    No real model artifacts are accessed.
    """

    from pytest import raises

    # --------------------------------------------------------
    # 1. Simulate a missing or corrupted model artifact.
    # --------------------------------------------------------

    def failing_inference_factory():
        raise FileNotFoundError(
            "SYNTHETIC_MISSING_MODEL_ARTIFACT"
        )

    # --------------------------------------------------------
    # 2. Create an isolated FastAPI application.
    # --------------------------------------------------------

    test_app = FastAPI(
        lifespan=build_model_lifespan(
            failing_inference_factory
        )
    )

    # --------------------------------------------------------
    # 3. Verify startup fails with a controlled error.
    # --------------------------------------------------------

    with raises(
        RuntimeError,
        match="Frozen inference engine failed to initialize",
    ) as error_info:

        with TestClient(test_app):
            pass

    # --------------------------------------------------------
    # 4. Verify the public error message is generic.
    # --------------------------------------------------------

    assert str(error_info.value) == (
        "Frozen inference engine failed to initialize."
    )

    # --------------------------------------------------------
    # 5. Verify the model was never marked ready.
    # --------------------------------------------------------

    assert test_app.state.model_ready is False

    assert test_app.state.inference_engine is None

    # --------------------------------------------------------
    # 6. Report successful verification.
    # --------------------------------------------------------

    print("Model initialization failure: DETECTED")
    print("Application startup: PREVENTED")
    print("Model readiness: FALSE")
    print("Private model artifacts: NOT ACCESSED")
    print("Step 09.76: PASS")
