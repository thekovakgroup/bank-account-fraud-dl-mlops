import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_PREPROCESSING_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "preprocessing_config.json"
)


def load_preprocessing_config(
    config_path=DEFAULT_PREPROCESSING_CONFIG_PATH,
):
    """Load preprocessing configuration from JSON."""
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def sentinel_indicator_names(config):
    """Return deterministic names for sentinel indicator features."""
    return [
        f"{column}__is_minus_one"
        for column in config["sentinel_features"]
    ]


def validate_preprocessing_config(config):
    """Validate that preprocessing feature groups are internally consistent."""
    numeric = config["numeric_features"]
    categorical = config["categorical_features"]
    binary = config["binary_features"]
    sentinel = config["sentinel_features"]

    primary_features = numeric + categorical + binary

    if len(primary_features) != len(set(primary_features)):
        raise ValueError(
            "A feature appears in more than one primary preprocessing group."
        )

    unknown_sentinel_features = (
        set(sentinel) - set(numeric)
    )

    if unknown_sentinel_features:
        raise ValueError(
            "Sentinel features must also be numeric features: "
            f"{sorted(unknown_sentinel_features)}"
        )


def prepare_model_features(dataframe, config=None):
    """
    Select model inputs, add sentinel indicators, and enforce
    float32 for numeric/binary model inputs.
    """
    if config is None:
        config = load_preprocessing_config()

    validate_preprocessing_config(config)

    numeric = config["numeric_features"]
    categorical = config["categorical_features"]
    binary = config["binary_features"]
    sentinel = config["sentinel_features"]
    sentinel_value = config["sentinel_value"]

    required_columns = numeric + categorical + binary

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required model features: {missing_columns}"
        )

    features = dataframe[required_columns].copy()

    for column in sentinel:
        indicator_column = f"{column}__is_minus_one"

        features[indicator_column] = (
            features[column] == sentinel_value
        ).astype(np.float32)

    float32_columns = (
        numeric
        + binary
        + sentinel_indicator_names(config)
    )

    features[float32_columns] = (
        features[float32_columns]
        .astype(np.float32)
    )

    return features


def build_preprocessor(config=None):
    """
    Build an unfitted sklearn preprocessing pipeline.

    Numeric:
        StandardScaler

    Categorical:
        OneHotEncoder with unknown-category protection

    Binary and sentinel indicators:
        Passthrough
    """
    if config is None:
        config = load_preprocessing_config()

    validate_preprocessing_config(config)

    numeric = config["numeric_features"]
    categorical = config["categorical_features"]
    binary = config["binary_features"]

    passthrough = (
        binary
        + sentinel_indicator_names(config)
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float32,
                ),
                categorical,
            ),
            (
                "passthrough",
                "passthrough",
                passthrough,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )