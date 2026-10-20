
# ============================================================
# MILESTONE 10 - MLFLOW MODEL REGISTRY
# Step 10.41 - Frozen Fraud Inference PyFunc Wrapper
# ============================================================

from pathlib import Path

import mlflow.pyfunc
import pandas as pd

from src.fraud_inference import FraudInference


# ------------------------------------------------------------
# 1. Define an MLflow-compatible inference model.
# ------------------------------------------------------------

class FrozenFraudPyFunc(mlflow.pyfunc.PythonModel):
    """
    Wrap the frozen fraud inference pipeline for MLflow.

    Includes:
        - Fitted preprocessing
        - Frozen PyTorch MLP
        - Frozen Platt calibration

    No training, refitting, or threshold optimization.
    """

    # --------------------------------------------------------
    # 2. Load and verify the frozen inference pipeline.
    # --------------------------------------------------------

    def load_context(self, context):
        """
        Initialize inference from the packaged artifact root.

        FraudInference verifies required artifact hashes.
        """

        bundle_root = Path(
            context.artifacts["frozen_bundle"]
        ).resolve()

        self.engine = FraudInference(
            project_root=bundle_root
        )

        assert (
            self.engine.policy["production_approved"]
            is False
        ), "Production approval must remain false"

    # --------------------------------------------------------
    # 3. Return calibrated fraud probabilities.
    # --------------------------------------------------------

    def predict(
        self,
        context,
        model_input: pd.DataFrame,
        params=None,
    ) -> pd.DataFrame:

        """
        Score a DataFrame of raw application features.

        Returns calibrated fraud probabilities only.

        Does not:
            - Reject bank account applications
            - Select investigations
            - Apply cohort investigation limits
            - Retrain or recalibrate the model
        """

        if not isinstance(model_input, pd.DataFrame):
            raise TypeError(
                "Expected a pandas DataFrame"
            )

        probabilities = self.engine.predict_proba(
            model_input
        )

        if len(probabilities) != len(model_input):
            raise ValueError(
                "Prediction count mismatch"
            )

        return pd.DataFrame(
            {
                "fraud_probability": probabilities,
            },
            index=model_input.index,
        )
