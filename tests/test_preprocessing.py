import numpy as np
import pandas as pd
import pytest

from src.features.preprocessing import (
    build_preprocessor,
    load_preprocessing_config,
    prepare_model_features,
    sentinel_indicator_names,
    validate_preprocessing_config,
)


def make_sample_dataframe():
    config = load_preprocessing_config()

    data = {}

    for column in config["numeric_features"]:
        data[column] = [1.0, 2.0, 3.0]

    for column in config["sentinel_features"]:
        data[column] = [-1.0, 2.0, 3.0]

    for column in config["categorical_features"]:
        data[column] = ["A", "B", "A"]

    for column in config["binary_features"]:
        data[column] = [0, 1, 0]

    return pd.DataFrame(data)


def test_preprocessing_config_is_valid():
    config = load_preprocessing_config()

    validate_preprocessing_config(config)

    assert len(config["numeric_features"]) == 18
    assert len(config["categorical_features"]) == 5
    assert len(config["binary_features"]) == 6
    assert len(config["sentinel_features"]) == 5


def test_prepare_model_features_adds_sentinel_indicators():
    config = load_preprocessing_config()
    dataframe = make_sample_dataframe()

    features = prepare_model_features(
        dataframe,
        config,
    )

    indicator_columns = sentinel_indicator_names(config)

    assert len(features.columns) == 34

    for indicator_column in indicator_columns:
        assert indicator_column in features.columns
        assert features[indicator_column].dtype == np.float32
        assert features[indicator_column].tolist() == [
            1.0,
            0.0,
            0.0,
        ]


def test_prepare_model_features_rejects_missing_column():
    config = load_preprocessing_config()
    dataframe = make_sample_dataframe()

    dataframe = dataframe.drop(
        columns=[config["numeric_features"][0]]
    )

    with pytest.raises(
        ValueError,
        match="Missing required model features",
    ):
        prepare_model_features(
            dataframe,
            config,
        )


def test_preprocessor_handles_unknown_category():
    config = load_preprocessing_config()

    train_dataframe = make_sample_dataframe()

    validation_dataframe = (
        make_sample_dataframe()
        .iloc[[0]]
        .copy()
    )

    validation_dataframe.loc[
        validation_dataframe.index[0],
        "payment_type",
    ] = "UNSEEN_CATEGORY"

    train_features = prepare_model_features(
        train_dataframe,
        config,
    )

    validation_features = prepare_model_features(
        validation_dataframe,
        config,
    )

    preprocessor = build_preprocessor(config)

    X_train = preprocessor.fit_transform(
        train_features
    )

    X_validation = preprocessor.transform(
        validation_features
    )

    assert X_train.dtype == np.float32
    assert X_validation.dtype == np.float32

    assert X_train.shape[1] == X_validation.shape[1]

    assert np.isfinite(X_train).all()
    assert np.isfinite(X_validation).all()