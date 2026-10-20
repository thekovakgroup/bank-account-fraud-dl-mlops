
# ============================================================
# MILESTONE 10 - MLFLOW MODEL SIGNATURE
# Step 10.46 - Explicit Input and Output Schema
# ============================================================

import json
from pathlib import Path

from mlflow.models import ModelSignature
from mlflow.types import ColSpec, DataType, Schema


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "preprocessing_config.json"
)


def build_fraud_model_signature(
    config_path=CONFIG_PATH,
) -> ModelSignature:
    """
    Build the MLflow signature from the frozen feature config.

    Expected inputs:
        18 numeric
        5 categorical
        6 binary

    Expected output:
        fraud_probability

    No model fitting or prediction occurs.
    """

    # --------------------------------------------------------
    # 1. Load the authoritative feature configuration.
    # --------------------------------------------------------

    with Path(config_path).open(
        "r", encoding="utf-8"
    ) as file:
        config = json.load(file)

    numeric = config["numeric_features"]
    categorical = config["categorical_features"]
    binary = config["binary_features"]

    # --------------------------------------------------------
    # 2. Verify the expected frozen feature counts.
    # --------------------------------------------------------

    assert len(numeric) == 18
    assert len(categorical) == 5
    assert len(binary) == 6

    all_features = numeric + categorical + binary

    assert len(all_features) == 29
    assert len(set(all_features)) == 29

    # --------------------------------------------------------
    # 3. Prevent excluded features from entering the model.
    # --------------------------------------------------------

    excluded = set(config["excluded_features"])

    assert not excluded.intersection(all_features)

    assert set(config["sentinel_features"]).issubset(
        set(numeric)
    )

    # --------------------------------------------------------
    # 4. Define MLflow input column types.
    # --------------------------------------------------------

    input_columns = (
        [
            ColSpec(DataType.double, name)
            for name in numeric
        ]
        +
        [
            ColSpec(DataType.string, name)
            for name in categorical
        ]
        +
        [
            ColSpec(DataType.long, name)
            for name in binary
        ]
    )

    # --------------------------------------------------------
    # 5. Define the calibrated probability output.
    # --------------------------------------------------------

    output_columns = [
        ColSpec(
            DataType.double,
            "fraud_probability",
        )
    ]

    # --------------------------------------------------------
    # 6. Return the complete MLflow model signature.
    # --------------------------------------------------------

    return ModelSignature(
        inputs=Schema(input_columns),
        outputs=Schema(output_columns),
    )
